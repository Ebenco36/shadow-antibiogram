import os
import re
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Union, Iterable, Tuple

import pandas as pd

_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")  # e.g. 'Unnamed: 0'

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if _UNNAMED_RE.match(str(c))]
    return df.drop(columns=cols) if cols else df

def _slug(s: object) -> str:
    s = "" if pd.isna(s) else str(s)
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s).strip("_")

def _iter_row_chunks(df: pd.DataFrame, rows_per_file: Optional[int]) -> Iterable[Tuple[int, pd.DataFrame]]:
    if not rows_per_file or rows_per_file <= 0:
        yield 0, df
        return
    n = len(df)
    i = 0
    part = 0
    while i < n:
        j = min(i + rows_per_file, n)
        yield part, df.iloc[i:j]
        i = j
        part += 1


class DataPreprocessorCleaning:
    """
    Preprocesses a dataset and saves results with optimal, partitioned layouts.

    Stable output paths (no version folders):
      Parquet:
        - single:   datasets/<stage>/<name>/dataset_parquet/part-00000.parquet
        - dataset:  datasets/<stage>/<name>/dataset_parquet/col=val/.../part-00001.parquet
      Feather:
        - single file: datasets/<stage>/<name>/<name>.feather
        - dataset:     datasets/<stage>/<name>/dataset_feather/[col=val/...]/part-00001.feather
      CSV:
        - single file: datasets/<stage>/<name>/<name>.csv(.gz)
        - dataset:     datasets/<stage>/<name>/dataset_csv/[col=val/...]/part-00001.csv(.gz)
    """

    def __init__(
        self,
        csv_path: str,
        cat_columns: List[str],
        antibiotic_pattern: str = " - ",
        missing_threshold: float = 0.6,
        *,
        base_dir: Union[str, Path] = "./datasets/",
    ):
        self.csv_path = csv_path
        self.cat_columns = cat_columns
        self.antibiotic_pattern = antibiotic_pattern
        self.missing_threshold = missing_threshold
        self.base_dir = Path(base_dir)

        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.cat_df: pd.DataFrame = pd.DataFrame()
        self.anti_df: pd.DataFrame = pd.DataFrame()
        self.merged_df: pd.DataFrame = pd.DataFrame()
        self.final_df: pd.DataFrame = pd.DataFrame()

    # ---------- loading & preprocessing ----------
    def load_data(self) -> pd.DataFrame:
        p = Path(self.csv_path)
        if p.is_dir():
            self.raw_df = pd.read_parquet(p, engine="pyarrow")
        else:
            suf = p.suffix.lower()
            if suf == ".parquet":
                self.raw_df = pd.read_parquet(p, engine="pyarrow")
            elif suf in (".feather", ".ft"):
                self.raw_df = pd.read_feather(p)
            else:
                self.raw_df = pd.read_csv(p, low_memory=False)
        self.raw_df = _drop_unnamed(self.raw_df)
        return self.raw_df

    def select_categorical(self) -> pd.DataFrame:
        self.cat_df = self.raw_df[self.cat_columns].copy()
        return self.cat_df

    def select_antibiotic(self) -> pd.DataFrame:
        cols = [c for c in self.raw_df.columns if self.antibiotic_pattern in c]
        df = self.raw_df[cols].dropna(axis=1, how="all")
        self.anti_df = self._drop_high_missing_columns(df)
        return self.anti_df

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_ratio = df.isna().mean()
        keep = missing_ratio[missing_ratio <= self.missing_threshold].index
        return df[keep].copy()

    def merge_subsets(self) -> pd.DataFrame:
        self.merged_df = pd.concat([self.cat_df, self.anti_df], axis=1)
        return self.merged_df

    def drop_rows_without_antibiotic(self) -> pd.DataFrame:
        subset_cols = [c for c in self.merged_df.columns if self.antibiotic_pattern in c]
        mask = ~self.merged_df[subset_cols].isna().all(axis=1)
        self.final_df = self.merged_df.loc[mask].reset_index(drop=True)
        return self.final_df

    def preprocess(self) -> pd.DataFrame:
        self.load_data()
        self.select_categorical()
        self.select_antibiotic()
        self.merge_subsets()
        return self.drop_rows_without_antibiotic()

    # ---------- saving ----------
    def _dataset_root(self, stage: str, name: str) -> Path:
        return self.base_dir / stage / _slug(name)

    def _write_metadata(self, where: Path, meta: Dict):
        if where.is_dir():
            where.mkdir(parents=True, exist_ok=True)
            mpath = where / "_metadata.json"
        else:
            where.parent.mkdir(parents=True, exist_ok=True)
            mpath = where.with_suffix(where.suffix + ".metadata.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _atomic_replace_dir(self, tmp_dir: Path, out_dir: Path):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.move(str(tmp_dir), str(out_dir))

    def _csv_ext(self, compression: Optional[str]) -> str:
        if (compression or "").lower() in {"gzip", "gz"}:
            return ".csv.gz"
        return ".csv"

    def save_data(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        stage: str = "processed",
        format: str = "parquet",                  # parquet|feather|csv
        compression: Optional[str] = None,
        partition_on: Optional[List[str]] = None, # list of columns to directory-partition by
        rows_per_file: Optional[int] = None,      # split files by row count (within each partition)
        overwrite: bool = True,
        write_index: bool = False,
    ) -> Path:
        """
        Save as single file (default) or as a partitioned dataset.

        Partition options:
          - partition_on=["colA", "colB"]  -> directory layout colA=.../colB=.../
          - rows_per_file=200_000          -> split into chunks per partition
          - both                           -> files split inside each partition directory
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty; nothing to save.")

        format = format.lower()
        if format not in {"parquet", "feather", "csv"}:
            raise ValueError("format must be one of {'parquet','feather','csv'}")

        if compression is None:
            compression = {"parquet": "zstd", "feather": "zstd", "csv": "gzip"}[format]

        root = self._dataset_root(stage, name)
        root.mkdir(parents=True, exist_ok=True)
        pdf = _drop_unnamed(df)

        is_partitioned = bool(partition_on) or (rows_per_file and rows_per_file > 0)

        # ---------- Parquet (best for large data) ----------
        if format == "parquet":
            out_dir = root / "dataset_parquet"
            if not is_partitioned:
                # single-file parquet inside a dataset directory for consistency
                tmp_dir = root / f".tmp-{uuid.uuid4().hex}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                pdf.to_parquet(
                    tmp_dir / "part-00000.parquet",
                    engine="pyarrow",
                    compression=compression,
                    index=write_index,
                )
                self._atomic_replace_dir(tmp_dir, out_dir)
                num_files = 1
            else:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except Exception as e:
                    raise ImportError(
                        "Partitioned Parquet requires 'pyarrow'. Install with `pip install pyarrow`."
                    ) from e

                tmp_dir = root / f".tmp-{uuid.uuid4().hex}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                if partition_on:
                    # Let pyarrow do directory partitioning by columns
                    table = pa.Table.from_pandas(pdf, preserve_index=write_index)
                    pq.write_to_dataset(
                        table,
                        root_path=str(tmp_dir),
                        partition_cols=partition_on,
                        compression=compression,
                        use_dictionary=True,
                    )
                    # Optionally split large leaf files further by rows_per_file (rarely needed)
                    if rows_per_file and rows_per_file > 0:
                        # Re-chunk each leaf file into row-chunks:
                        # Load each leaf back and re-split (keeps it simple & robust).
                        # For very large datasets prefer writing chunks per partition directly (below).
                        pass
                else:
                    # No column partitioning; split by rows_per_file
                    for i, chunk in _iter_row_chunks(pdf, rows_per_file):
                        fn = tmp_dir / f"part-{i:05d}.parquet"
                        chunk.to_parquet(
                            fn,
                            engine="pyarrow",
                            compression=compression,
                            index=write_index,
                        )

                # Publish atomically
                self._atomic_replace_dir(tmp_dir, out_dir)
                # Count files
                num_files = sum(1 for _ in out_dir.rglob("*.parquet"))

            meta = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "format": "parquet",
                "compression": compression,
                "stage": stage,
                "name": name,
                "rows": int(pdf.shape[0]),
                "cols": int(pdf.shape[1]),
                "dtypes": {c: str(t) for c, t in pdf.dtypes.items()},
                "partition_on": partition_on or [],
                "rows_per_file": rows_per_file or 0,
                "files": num_files,
            }
            self._write_metadata(out_dir, meta)
            return out_dir

        # ---------- Feather ----------
        elif format == "feather":
            if not is_partitioned:
                out_file = root / f"{_slug(name)}.feather"
                tmp_file = out_file.with_suffix(".feather.tmp")
                pdf.to_feather(tmp_file, compression=compression)
                os.replace(tmp_file, out_file)
                meta_target = out_file
                num_files = 1
            else:
                out_dir = root / "dataset_feather"
                tmp_dir = root / f".tmp-{uuid.uuid4().hex}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                if partition_on:
                    # write a file per partition (and rows_per_file inside)
                    gb = pdf.groupby(partition_on, dropna=False, sort=False)
                    file_count = 0
                    for keys, g in gb:
                        # keys can be a scalar or a tuple
                        sub = tmp_dir
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        for col, key in zip(partition_on, keys):
                            sub = sub / f"{_slug(col)}={_slug(key)}"
                        sub.mkdir(parents=True, exist_ok=True)
                        for i, chunk in _iter_row_chunks(g, rows_per_file):
                            (sub / f"part-{i:05d}.feather").write_bytes(b"")  # touch to reserve name
                            g2 = chunk  # (keep columns intact)
                            g2.to_feather(sub / f"part-{i:05d}.feather", compression=compression)
                            file_count += 1
                else:
                    # no column partitioning; split by rows
                    file_count = 0
                    for i, chunk in _iter_row_chunks(pdf, rows_per_file):
                        fn = tmp_dir / f"part-{i:05d}.feather"
                        chunk.to_feather(fn, compression=compression)
                        file_count += 1

                self._atomic_replace_dir(tmp_dir, out_dir)
                meta_target = out_dir
                num_files = file_count

            meta = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "format": "feather",
                "compression": compression,
                "stage": stage,
                "name": name,
                "rows": int(pdf.shape[0]),
                "cols": int(pdf.shape[1]),
                "dtypes": {c: str(t) for c, t in pdf.dtypes.items()},
                "partition_on": partition_on or [],
                "rows_per_file": rows_per_file or 0,
                "files": num_files,
            }
            self._write_metadata(meta_target, meta)
            return meta_target

        # ---------- CSV ----------
        else:
            ext = self._csv_ext(compression)
            if not is_partitioned:
                out_file = root / f"{_slug(name)}{ext}"
                tmp_file = out_file.with_suffix(ext + ".tmp")
                pdf.to_csv(
                    tmp_file,
                    index=write_index,
                    encoding="utf-8",
                    lineterminator="\n",
                    compression=compression,
                )
                os.replace(tmp_file, out_file)
                meta_target = out_file
                num_files = 1
            else:
                out_dir = root / "dataset_csv"
                tmp_dir = root / f".tmp-{uuid.uuid4().hex}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                if partition_on:
                    gb = pdf.groupby(partition_on, dropna=False, sort=False)
                    file_count = 0
                    for keys, g in gb:
                        sub = tmp_dir
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        for col, key in zip(partition_on, keys):
                            sub = sub / f"{_slug(col)}={_slug(key)}"
                        sub.mkdir(parents=True, exist_ok=True)
                        for i, chunk in _iter_row_chunks(g, rows_per_file):
                            fn = sub / f"part-{i:05d}{ext}"
                            chunk.to_csv(
                                fn,
                                index=write_index,
                                encoding="utf-8",
                                lineterminator="\n",
                                compression=compression,
                            )
                            file_count += 1
                else:
                    file_count = 0
                    for i, chunk in _iter_row_chunks(pdf, rows_per_file):
                        fn = tmp_dir / f"part-{i:05d}{ext}"
                        chunk.to_csv(
                            fn,
                            index=write_index,
                            encoding="utf-8",
                            lineterminator="\n",
                            compression=compression,
                        )
                        file_count += 1

                self._atomic_replace_dir(tmp_dir, out_dir)
                meta_target = out_dir
                num_files = file_count

            meta = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "format": "csv",
                "compression": compression,
                "stage": stage,
                "name": name,
                "rows": int(pdf.shape[0]),
                "cols": int(pdf.shape[1]),
                "dtypes": {c: str(t) for c, t in pdf.dtypes.items()},
                "partition_on": partition_on or [],
                "rows_per_file": rows_per_file or 0,
                "files": num_files,
            }
            self._write_metadata(meta_target, meta)
            return meta_target

    # ---------- convenience ----------
    def latest_output(self, name: str, stage: str = "processed") -> Optional[Path]:
        root = self._dataset_root(stage, name)
        for dirname in ("dataset_parquet", "dataset_feather", "dataset_csv"):
            p = root / dirname
            if p.exists():
                return p
        for suf in (".feather", ".csv", ".csv.gz", ".parquet"):
            f = root / f"{_slug(name)}{suf}"
            if f.exists():
                return f
        return None
