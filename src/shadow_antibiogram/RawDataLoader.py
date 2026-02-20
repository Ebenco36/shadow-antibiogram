# parallel_data_loader_dask.py
from __future__ import annotations

import os
import re
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Iterable, List, Dict, Callable, Optional, Union, Any, Tuple

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    import pyarrow  # noqa: F401
    import pyarrow.feather as _pa_feather
    import pyarrow.dataset as _pa_ds  # noqa: F401
except Exception:
    _pa_feather = None
    _pa_ds = None


import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*DataFrame concatenation with empty or all-NA.*",
)


try:
    import dask.dataframe as dd
    from dask import delayed
except Exception:
    dd = None
    delayed = None

PathLike = Union[str, os.PathLike]

# --------------------------
# Utilities
# --------------------------
_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")  # matches 'Unnamed: 0', 'Unnamed: 1', etc.

def _default_csv_engine():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        return "c"

def _drop_unnamed_cols_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names look like 'Unnamed: 0', etc."""
    drop = [c for c in df.columns if _UNNAMED_RE.match(str(c))]
    if drop:
        df = df.drop(columns=drop)
    return df

def _clean_pdf(
    df: pd.DataFrame,
    *,
    drop_all_na_columns: bool,
    drop_unnamed_columns: bool,
) -> pd.DataFrame:
    """Per-DataFrame cleanup before concatenation."""
    if drop_unnamed_columns:
        df = _drop_unnamed_cols_pdf(df)
    if drop_all_na_columns:
        # Remove columns that are entirely NA in this piece
        df = df.dropna(axis=1, how="all")
    return df

def _align_schema_pdfs(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align all pandas DataFrames to the union of columns (outer join semantics)."""
    all_cols = pd.Index([])
    for d in dfs:
        all_cols = all_cols.union(d.columns)
    return [d.reindex(columns=all_cols) for d in dfs]

# --------------------------
# Workers (pandas backend)
# --------------------------
def _read_one(
    path: str,
    ext: str,
    add_source: bool,
    reader_options: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Top-level worker function for pandas backend (required for spawn)."""
    p = Path(path)
    ext = ext.lower()
    opts = {k.lower(): v for k, v in (reader_options or {}).items()}
    ro = opts.get(ext, {})

    if ext == ".feather":
        df = pd.read_feather(p, **ro)
    elif ext == ".parquet":
        df = pd.read_parquet(p, **ro)
    elif ext == ".csv":
        ro = {"engine": ro.pop("engine", _default_csv_engine()), **ro}
        df = pd.read_csv(p, **ro)
    elif ext == ".json":
        ro = {"lines": ro.pop("lines", True), **ro}
        df = pd.read_json(p, **ro)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, **ro)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (file: {p})")

    if add_source:
        df.insert(0, "__source__", str(p))
    return df

def _run_custom_reader(path: str, reader_key: str, add_source: bool, registry: Dict[str, Callable[[str], pd.DataFrame]]) -> pd.DataFrame:
    """
    Top-level wrapper to call a custom reader stored in a registry by key.
    NOTE: Custom reader functions should be top-level functions (NOT lambdas) to be picklable under spawn.
    """
    reader = registry[reader_key]
    df = reader(path)
    if add_source and "__source__" not in df.columns:
        df.insert(0, "__source__", path)
    return df


class ParallelDataLoader:
    """
    Fast, robust loader for large, mixed-format tabular data.

    Backends:
      - backend="pandas": multiprocessing with pandas (good for medium→large files when RAM is OK).
      - backend="dask": out-of-core + parallel I/O with Dask (preferred for very large datasets).
      - backend="auto": uses Dask if available; else pandas.

    Formats:
      - Built-in: .feather, .parquet, .csv, .json, .xlsx/.xls
      - Custom via `register_reader(".ext", func)` (pandas backend).
    """

    def __init__(
        self,
        workers: Optional[int] = None,
        fail_fast: bool = False,
        chunksize: int = 1,
        *,
        drop_all_na_columns: bool = True,
        drop_unnamed_columns: bool = True,
        align_columns_before_concat: bool = True,
        executor: str = "process",  # "process" | "thread" | "auto"
        mp_start_method: str = "spawn",  # good default for macOS/Win; linux can be "fork"
    ):
        """
        Args:
            workers: number of workers for pandas backend (defaults to os.cpu_count()).
            fail_fast: raise on first file error if True; otherwise collect and print errors.
            chunksize: reserved for future batching; not used directly here.
            drop_all_na_columns: drop per-DF columns that are entirely NA (prevents concat FutureWarning).
            drop_unnamed_columns: drop columns named like 'Unnamed: 0' (CSV index artifacts).
            align_columns_before_concat: align all DataFrames to the union of columns before concat.
            executor: choose "process" (default), "thread", or "auto" (threads if custom readers found).
            mp_start_method: multiprocessing start method for ProcessPool ("spawn" recommended cross-platform).
        """
        self.workers = workers or os.cpu_count() or 1
        self.fail_fast = fail_fast
        self.chunksize = chunksize
        self.drop_all_na_columns = drop_all_na_columns
        self.drop_unnamed_columns = drop_unnamed_columns
        self.align_columns_before_concat = align_columns_before_concat
        self.executor = executor
        self.mp_start_method = mp_start_method
        self._custom_readers: Dict[str, Callable[[str], pd.DataFrame]] = {}

    def register_reader(self, extension: str, func: Callable[[str], pd.DataFrame]) -> None:
        """
        Register a custom reader for an extension. The function MUST be a top-level function
        (no lambdas/closures) if you plan to use the process pool.
        """
        if not extension.startswith("."):
            extension = "." + extension
        self._custom_readers[extension.lower()] = func

    # ---------- helpers ----------
    def _expand_inputs(
        self,
        paths: Iterable[PathLike],
        recursive: bool = True,
        patterns: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        pats = list(patterns or ["*.feather", "*.parquet", "*.csv", "*.json", "*.xlsx", "*.xls"])
        out: List[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_file():
                out.append(p)
            elif p.is_dir():
                it = p.rglob if recursive else p.glob
                for pat in pats:
                    out.extend(it(pat))
            else:
                out.extend([Path(x) for x in Path().glob(str(p))])
        uniq = sorted({x.resolve() for x in out if x.exists() and x.is_file()})
        return uniq

    def _split_by_ext(self, files: List[Path]) -> Dict[str, List[Path]]:
        groups: Dict[str, List[Path]] = {}
        for f in files:
            groups.setdefault(f.suffix.lower(), []).append(f)
        return groups

    # ---------- Dask readers ----------
    def _dask_read_group(
        self,
        ext: str,
        files: List[Path],
        add_source: bool,
        reader_options: Dict[str, Dict[str, Any]],
    ):
        if dd is None:
            raise ImportError("Dask is not installed. `pip install dask[dataframe]`")

        opts = {k.lower(): v for k, v in (reader_options or {}).items()}
        ro = opts.get(ext, {})

        def _drop_unnamed_cols_dd(pdf: pd.DataFrame) -> pd.DataFrame:
            return _drop_unnamed_cols_pdf(pdf)

        def _add_source_meta(d, src):
            if add_source and "__source__" not in d.columns:
                d.insert(0, "__source__", src)
            return d

        if ext == ".parquet":
            pattern = self._to_glob(files)
            ddf = dd.read_parquet(pattern, **ro)
            if add_source and len(files) == 1:
                ddf = ddf.map_partitions(_add_source_meta, str(files[0]))
            ddf = ddf.map_partitions(_drop_unnamed_cols_dd)
            return ddf

        if ext == ".csv":
            pattern = self._to_glob(files)
            csv_ro = {
                "blocksize": ro.pop("blocksize", "128MB"),
                "assume_missing": ro.pop("assume_missing", True),
                **ro,
            }
            ddf = dd.read_csv(pattern, **csv_ro)
            if add_source and len(files) == 1:
                ddf = ddf.map_partitions(_add_source_meta, str(files[0]))
            ddf = ddf.map_partitions(_drop_unnamed_cols_dd)
            return ddf

        if ext == ".json":
            pattern = self._to_glob(files)
            json_ro = {
                "blocksize": ro.pop("blocksize", "128MB"),
                "lines": ro.pop("lines", True),
                **ro,
            }
            ddf = dd.read_json(pattern, **json_ro)
            if add_source and len(files) == 1:
                ddf = ddf.map_partitions(_add_source_meta, str(files[0]))
            ddf = ddf.map_partitions(_drop_unnamed_cols_dd)
            return ddf

        if ext == ".feather":
            if _pa_feather is None:
                raise ImportError("pyarrow.feather is required to read .feather with Dask.")

            # 1) Read ONE file to infer schema *after* cleanup
            first_path = str(files[0])
            _tbl0 = _pa_feather.read_table(first_path, **ro)
            _pdf0 = _tbl0.to_pandas(types_mapper=None)

            # Always drop Unnamed:* everywhere
            _pdf0 = _pdf0.loc[:, ~_pdf0.columns.astype(str).str.match(r"^Unnamed(?::\s*\d+)?$")]
            if add_source:
                # If you keep __source__, ensure it exists in ALL partitions
                _pdf0.insert(0, "__source__", first_path)

            # 2) Build meta: same columns, force your desired dtypes (e.g., keep everything as string)
            # If you only want to coerce a known problem set, replace the dict comp below accordingly.
            meta = pd.DataFrame({c: pd.Series(dtype="string") for c in _pdf0.columns}).head(0)

            delayed_dfs = []
            cols_meta = list(meta.columns)

            for f in files:
                @delayed
                def _read_feather_one(path=str(f), ro=ro, add_source=add_source, cols_meta=cols_meta):
                    tbl = _pa_feather.read_table(path, **ro)
                    pdf = tbl.to_pandas(types_mapper=None)

                    # Drop Unnamed:* consistently
                    pdf = pdf.loc[:, ~pdf.columns.astype(str).str.match(r"^Unnamed(?::\s*\d+)?$")]

                    if add_source:
                        if "__source__" not in pdf.columns:
                            pdf.insert(0, "__source__", path)

                    # Ensure ALL meta columns exist (add missing as NA)
                    for c in cols_meta:
                        if c not in pdf.columns:
                            pdf[c] = pd.NA

                    # Reorder to meta columns only (drop any extras)
                    pdf = pdf[cols_meta]

                    # Coerce to string (or your chosen dtypes) to match meta exactly
                    for c in cols_meta:
                        pdf[c] = pdf[c].astype("string")

                    return pdf

                delayed_dfs.append(_read_feather_one())

            # 3) Create Dask DF with explicit meta so Dask doesn’t infer a stale schema
            ddf = dd.from_delayed(delayed_dfs, meta=meta)
            return ddf

        if ext in (".xlsx", ".xls"):
            delayed_dfs = []
            for f in files:
                @delayed
                def _read_excel_one(path=str(f), ro=ro, add_source=add_source):
                    pdf = pd.read_excel(path, **ro)
                    if add_source:
                        pdf.insert(0, "__source__", path)
                    pdf = _drop_unnamed_cols_pdf(pdf)
                    return pdf
                delayed_dfs.append(_read_excel_one())
            ddf = dd.from_delayed(delayed_dfs)
            return ddf

        raise ValueError(f"Dask backend: unsupported extension {ext}")

    def _to_glob(self, files: List[Path]) -> Union[str, List[str]]:
        parents = {f.parent for f in files}
        exts = {f.suffix for f in files}
        if len(parents) == 1 and len(exts) == 1:
            parent = next(iter(parents))
            ext = next(iter(exts))
            return str(parent / f"*{ext}")
        return [str(f) for f in files]

    # ---------- public API ----------
    def load(
        self,
        paths: Iterable[PathLike],
        *,
        backend: str = "auto",          # "dask" | "pandas" | "auto"
        recursive: bool = True,
        patterns: Optional[Iterable[str]] = None,
        concat: bool = True,
        add_source: bool = True,
        reader_options: Optional[Dict[str, Dict[str, Any]]] = None,
        concat_sort_columns: bool = False,
        ignore_index: bool = True,
    ):
        """
        Load many files with either Dask (out-of-core) or pandas (multiprocessing).

        Returns:
            - backend="dask"  -> dask.dataframe.DataFrame (if concat=True) or dict[ext]->dask DF
            - backend="pandas"-> pandas.DataFrame (if concat=True) or list[pd.DataFrame]
        """
        files = self._expand_inputs(paths, recursive=recursive, patterns=patterns)
        if not files:
            raise FileNotFoundError("No matching input files.")

        if backend == "auto":
            backend = "dask" if dd is not None else "pandas"

        if backend == "dask":
            return self._load_dask(files, add_source, reader_options, concat)
        elif backend == "pandas":
            return self._load_pandas(files, add_source, reader_options, concat, concat_sort_columns, ignore_index)
        else:
            raise ValueError("backend must be one of {'auto','dask','pandas'}")

    def _load_dask(
        self,
        files: List[Path],
        add_source: bool,
        reader_options: Optional[Dict[str, Dict[str, Any]]],
        concat: bool,
    ):
        groups = self._split_by_ext(files)
        ddf_parts: List[Tuple[str, Any]] = []
        for ext, group in groups.items():
            ddf = self._dask_read_group(ext, group, add_source, reader_options or {})
            ddf_parts.append((ext, ddf))

        if not concat:
            return {ext: d for ext, d in ddf_parts}

        dd_items = [d for _, d in ddf_parts]
        if len(dd_items) == 1:
            return dd_items[0]
        # Safer concat across formats with outer join (schema alignment)
        return dd.concat(dd_items, interleave_partitions=True, axis=0, join="outer")

    def _load_pandas(
        self,
        files: List[Path],
        add_source: bool,
        reader_options: Optional[Dict[str, Dict[str, Any]]],
        concat: bool,
        concat_sort_columns: bool,
        ignore_index: bool,
    ):
        # Build tasks. For custom readers we store the key (extension) and call via top-level helper.
        tasks: List[Tuple[str, str, Optional[Callable[..., pd.DataFrame]]]] = []
        has_custom = False
        for f in files:
            ext = f.suffix.lower()
            if ext in self._custom_readers:
                has_custom = True
                fn = partial(_run_custom_reader, str(f), ext, add_source, self._custom_readers)
                tasks.append(("__custom__", str(f), fn))
            else:
                tasks.append((ext, str(f), None))

        # Choose executor
        exec_mode = self.executor
        if exec_mode == "auto":
            exec_mode = "thread" if has_custom else "process"

        dfs: List[pd.DataFrame] = []
        errors: List[str] = []

        if exec_mode == "process":
            ctx = mp.get_context(self.mp_start_method)
            Executor = partial(ProcessPoolExecutor, mp_context=ctx)
        elif exec_mode == "thread":
            Executor = ThreadPoolExecutor
        else:
            raise ValueError("executor must be one of {'process','thread','auto'}")

        with Executor(max_workers=self.workers) as ex:
            future_map = {}
            for ext, path, custom in tasks:
                fut = ex.submit(custom) if custom else ex.submit(_read_one, path, ext, add_source, reader_options or {})
                future_map[fut] = path

            for fut in as_completed(future_map):
                path = future_map[fut]
                try:
                    df = fut.result()
                    # Per-DF cleanup BEFORE concat (fixes the deprecation warning)
                    df = _clean_pdf(
                        df,
                        drop_all_na_columns=self.drop_all_na_columns,
                        drop_unnamed_columns=self.drop_unnamed_columns,
                    )
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    msg = f"[ERROR] {path}: {e}"
                    if self.fail_fast:
                        raise RuntimeError(msg) from e
                    errors.append(msg)

        if errors:
            print("\n".join(errors))
        if not dfs:
            raise RuntimeError("All file reads failed.")

        if not concat:
            return dfs

        # Optional: align schemas (outer union) to avoid dtype issues/missing columns
        if self.align_columns_before_concat:
            dfs = _align_schema_pdfs(dfs)

        # Final concat is now stable & warning-free
        df = pd.concat(dfs, ignore_index=ignore_index, sort=concat_sort_columns)
        return df


# ---------------------------
# CSV -> Feather chunking utilities
# ---------------------------
def _needs_string(series: pd.Series) -> bool:
    # Heuristic: if any value starts with '0' or contains non-digits, keep as text
    sample = series.dropna().astype(str).head(1000)
    if sample.empty:
        return False
    return any(s.startswith("0") for s in sample) or not sample.str.fullmatch(r"\d+").all()

def split_csv_to_feather_chunks_safe(
    csv_path: str,
    output_dir: str,
    chunksize: int = 100_000,
    force_string_cols: tuple[str, ...] = ("KreisZiffer",),  # seed with known columns
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype={c: "string" for c in force_string_cols},
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        # Ensure forced columns exist and are string
        for c in force_string_cols:
            if c in chunk.columns:
                chunk[c] = chunk[c].astype("string")

        # Auto-detect other risky columns (object dtype)
        for col in chunk.select_dtypes(include=["object"]).columns:
            if _needs_string(chunk[col]):
                chunk[col] = chunk[col].astype("string")

        out_file = output_dir / f"{csv_path.stem}_part{i}.feather"
        chunk.reset_index(drop=True).to_feather(out_file)
        print(f"Wrote {len(chunk)} rows -> {out_file}")

    print("Done splitting!")


# ---------------------------
# Convenience loader for your runner
# ---------------------------
from typing import List as _List  # avoid shadowing
from dask.dataframe.utils import make_meta
def load_data_feather(paths=["./datasets-archive"]):
    loader = ParallelDataLoader(
        workers=None,
        drop_all_na_columns=False,
        align_columns_before_concat=False,
        drop_unnamed_columns=True,
        executor="process",
        mp_start_method="spawn",
    )

    # 1) Load as Dask
    ddf = loader.load(
        paths=paths,
        backend="dask",
        patterns=["*.feather"],
        concat=True,
        add_source=False,
        reader_options={".feather": {}},
    )

    # 2) Build a dtype map: every column -> pandas "string" dtype
    dtype_map = {c: "string" for c in ddf.columns}

    # 3) Meta MUST be a DataFrame with those dtypes (not a dict of Series)
    meta = make_meta(dtype_map)              # empty df with the right dtypes
    meta = meta.reindex(columns=ddf.columns) # preserve original column order

    # 4) Cast EVERY partition to those dtypes
    ddf = ddf.map_partitions(pd.DataFrame.astype, dtype_map, meta=meta)

    # (Optional) sanity check: this should now be all 'string'
    # print(ddf._meta.dtypes)

    # 5) Reasonable file sizes
    ddf = ddf.repartition(partition_size="256MB")

    # 6) Ensure output dir exists, then write
    out_dir = Path("./datasets/dataset_parquet/")
    out_dir.mkdir(parents=True, exist_ok=True)

    ddf.to_parquet(
        out_dir,
        engine="pyarrow",
        write_index=False,
        compression="zstd",
    )

    return ddf



PARQUET_DIR = Path("./datasets/dataset_parquet")

def parquet_ready(path: Path = PARQUET_DIR) -> bool:
    if path.is_file() and path.suffix == ".parquet":
        return True
    if path.is_dir():
        # dask/pyarrow dataset: either part files or a global _metadata file
        if (path / "_metadata").exists():
            return True
        if any(path.glob("*.parquet")) or any(path.glob("*/*.parquet")):
            return True
    return False