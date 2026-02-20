# src/runners/dataset_profiler.py
from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shadow_antibiogram.controllers.DatabaseOverview import AMRSummary
import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def _safe_float(x: Any) -> Optional[float]:
    """Convert numeric values to float safely, return None if not valid."""
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None

def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
        return f"{int(x)}"
    return f"{x:.4g}"

def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_') or "column"

def _try_parse_datetime_quiet(s: pd.Series) -> pd.Series:
    """
    Best-effort parse to datetime *without warnings*.
    Uses format='mixed' when available (Pandas ≥2.0), otherwise suppresses the
    'Could not infer format...' warning for older versions.
    """
    try:
        return pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually",
                category=UserWarning,
                module="pandas"
            )
            return pd.to_datetime(s, errors="coerce", utc=False)


# ----------------------------- data classes -----------------------------

@dataclass
class ColumnProfile:
    column: str
    dtype: str
    role: str  # numeric | categorical | datetime | text | boolean | other
    non_null: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_constant: bool
    is_id_like: bool
    sample_values: List[Any]

    # Numeric stats
    n_zeros: Optional[int] = None
    min: Optional[float] = None
    p5: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    # Categorical/text stats
    top: Optional[Any] = None
    top_freq: Optional[int] = None
    top_pct: Optional[float] = None

    # Datetime stats
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, (np.generic,)):
                d[k] = v.item()
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
        return d


@dataclass
class DatasetSummary:
    name: Optional[str]
    n_rows: int
    n_cols: int
    memory_mb: float
    duplicate_rows: int
    duplicate_pct: float
    constant_columns: List[str]
    primary_key_candidates: List[str]
    warnings: List[str]
    columns: List[ColumnProfile]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": round(self.memory_mb, 3),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 3),
            "constant_columns": self.constant_columns,
            "primary_key_candidates": self.primary_key_candidates,
            "warnings": self.warnings,
            "columns": [c.to_dict() for c in self.columns],
        }

    def to_markdown(self) -> str:
        lines = []
        title = f"### Dataset Summary{f' – {self.name}' if self.name else ''}"
        lines.append(title)
        lines.append("")
        lines.append(f"- **Rows**: {self.n_rows}")
        lines.append(f"- **Columns**: {self.n_cols}")
        lines.append(f"- **Memory**: {self.memory_mb:.3f} MB")
        lines.append(f"- **Duplicate rows**: {self.duplicate_rows} ({self.duplicate_pct:.2f}%)")
        lines.append(f"- **Constant columns**: {', '.join(self.constant_columns) if self.constant_columns else '—'}")
        lines.append(f"- **Primary key candidates**: {', '.join(self.primary_key_candidates) if self.primary_key_candidates else '—'}")
        if self.warnings:
            lines.append(f"- **Warnings**: {'; '.join(self.warnings)}")
        lines.append("")
        header = "| column | dtype | role | non_null | missing% | unique% | top | range/quantiles |"
        sep = "|---|---|---:|---:|---:|---:|---|---|"
        lines.extend([header, sep])
        for c in self.columns:
            top_repr = "—" if c.top is None else f"{str(c.top)[:30]} ({c.top_pct:.1f}%)"
            rng = "—"
            if c.role == "numeric":
                rng = f"[{_fmt(c.min)}, {_fmt(c.max)}], p50={_fmt(c.p50)}"
            elif c.role == "datetime":
                rng = f"{c.date_min} → {c.date_max}"
            row = f"| {c.column} | {c.dtype} | {c.role} | {c.non_null} | {c.missing_pct:.1f} | {c.unique_pct:.1f} | {top_repr} | {rng} |"
            lines.append(row)
        return "\n".join(lines)


# ----------------------------- profiler -----------------------------

class DatasetProfiler:
    """
    Lightweight profiler for pandas DataFrames intended for transparent handoffs to domain experts.
    """

    def __init__(
        self,
        *,
        sample_for_uniques: Optional[int] = None,
        id_like_regex: str = r"(?i)^(id|.*_id|id_.*|.*id.*|guid|uuid|ssn|mrn)$",
        categorical_threshold: int = 40,
        treat_bool_as_categorical: bool = True,
    ):
        self.sample_for_uniques = sample_for_uniques
        self.id_like_re = re.compile(id_like_regex)
        self.categorical_threshold = categorical_threshold
        self.treat_bool_as_categorical = treat_bool_as_categorical

    # ----------------------------- Public API -----------------------------

    def profile(self, df: pd.DataFrame, *, name: Optional[str] = None) -> DatasetSummary:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("profile() expects a pandas DataFrame")

        n_rows, n_cols = df.shape
        mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)

        dup_rows = int(df.duplicated().sum())
        dup_pct = (dup_rows / n_rows * 100.0) if n_rows else 0.0

        columns: List[ColumnProfile] = []
        constant_cols: List[str] = []
        warnings_list: List[str] = []

        for col in df.columns:
            s = df[col]
            role = self._infer_role(s)

            non_null = int(s.notna().sum())
            missing = int(s.isna().sum())
            missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0

            unique, _ = self._unique_count(s, n_rows)
            is_constant = (unique == 1) if n_rows > 0 else False
            if is_constant:
                constant_cols.append(col)

            is_id_like = bool(self.id_like_re.match(str(col))) or (unique == n_rows and missing == 0)

            cp = ColumnProfile(
                column=str(col),
                dtype=str(s.dtype),
                role=role,
                non_null=non_null,
                missing=missing,
                missing_pct=missing_pct,
                unique=unique,
                unique_pct=(unique / n_rows * 100.0) if n_rows else 0.0,
                is_constant=is_constant,
                is_id_like=is_id_like,
                sample_values=self._sample_values(s),
            )

            if role == "numeric":
                s_num = pd.to_numeric(s, errors="coerce")
                cp.n_zeros = int((s_num == 0).sum(skipna=True))
                quantiles = s_num.quantile([0.05, 0.5, 0.95])
                cp.min = _safe_float(s_num.min(skipna=True))
                cp.p5 = _safe_float(quantiles.get(0.05, np.nan))
                cp.p50 = _safe_float(quantiles.get(0.5, np.nan))
                cp.p95 = _safe_float(quantiles.get(0.95, np.nan))
                cp.max = _safe_float(s_num.max(skipna=True))
                cp.mean = _safe_float(s_num.mean(skipna=True))
                cp.std = _safe_float(s_num.std(skipna=True))
            elif role == "datetime":
                s_dt = _try_parse_datetime_quiet(s)  # <- warning-free
                if not s_dt.dropna().empty:
                    cp.date_min = s_dt.min(skipna=True)
                    cp.date_max = s_dt.max(skipna=True)
            else:
                vc = s.value_counts(dropna=True)
                if not vc.empty:
                    cp.top = vc.index[0]
                    cp.top_freq = int(vc.iloc[0])
                    cp.top_pct = (vc.iloc[0] / non_null * 100.0) if non_null else 0.0

            columns.append(cp)

        primary_key_candidates = sorted(
            [c.column for c in columns if c.unique == n_rows and c.missing == 0]
        )

        if dup_rows > 0:
            warnings_list.append("Dataset contains duplicate rows.")
        if constant_cols:
            warnings_list.append("Some columns are constant and may be dropped.")
        high_missing = [c.column for c in columns if c.missing_pct > 30]
        if high_missing:
            warnings_list.append(
                "Columns with >30% missing: "
                + ", ".join(high_missing[:8])
                + ("..." if len(high_missing) > 8 else "")
            )

        return DatasetSummary(
            name=name,
            n_rows=n_rows,
            n_cols=n_cols,
            memory_mb=mem_mb,
            duplicate_rows=dup_rows,
            duplicate_pct=dup_pct,
            constant_columns=constant_cols,
            primary_key_candidates=primary_key_candidates,
            warnings=warnings_list,
            columns=columns,
        )

    # --------------------------- Helper methods ---------------------------

    def _infer_role(self, s: pd.Series) -> str:
        dt = s.dtype
        if pd.api.types.is_bool_dtype(dt):
            return "categorical" if self.treat_bool_as_categorical else "boolean"
        if pd.api.types.is_numeric_dtype(dt):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(dt):
            return "datetime"

        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt):
            try_dt = _try_parse_datetime_quiet(s)
            if try_dt.notna().mean() > 0.6:
                return "datetime"
            nunique = s.nunique(dropna=True)
            if nunique <= self.categorical_threshold or nunique <= max(20, int(0.02 * len(s))):
                return "categorical"
            if s.dropna().astype(str).str.len().median() > 30:
                return "text"
            return "categorical"

        return "other"

    def _unique_count(self, s: pd.Series, n_rows: int) -> Tuple[int, float]:
        if self.sample_for_uniques and n_rows > self.sample_for_uniques:
            sample = s.sample(self.sample_for_uniques, random_state=42)
            u_sample = int(sample.nunique(dropna=True))
            unique_est = int(min(n_rows, u_sample * (n_rows / self.sample_for_uniques) ** 0.7))
            return unique_est, unique_est / n_rows * 100.0
        u = int(s.nunique(dropna=True))
        return u, (u / n_rows * 100.0) if n_rows else 0.0

    def _sample_values(self, s: pd.Series, k: int = 3) -> List[Any]:
        vals = pd.unique(s.dropna())[:k]
        out = []
        for v in vals:
            if isinstance(v, (np.floating, float)):
                out.append(round(float(v), 6))
            elif isinstance(v, (np.integer, int)):
                out.append(int(v))
            else:
                sv = str(v)
                out.append(sv if len(sv) <= 80 else sv[:77] + "…")
        return list(map(_json_safe, out))

    # --------------------------- Exports ---------------------------

    def export_to_folder(
        self,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        out_dir: Union[str, os.PathLike],
        topk: int = 50,
        overwrite: bool = True,
        write_value_counts: bool = True,
    ) -> Path:
        """
        Generate a folder with CSV artifacts:
          - dataset_summary.csv
          - columns_overview.csv
          - numeric_stats.csv
          - datetime_ranges.csv
          - categorical_overview.csv
          - value_counts/<col>.csv (if write_value_counts=True)
        """
        summary = self.profile(df, name=name)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "value_counts").mkdir(exist_ok=True)

        def _write_csv(path: Path, df_to_write: pd.DataFrame):
            if path.exists() and not overwrite:
                return
            df_to_write.to_csv(path, index=False)

        # dataset_summary.csv
        ds_row = pd.DataFrame([{
            "name": summary.name or "",
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "memory_mb": round(summary.memory_mb, 3),
            "duplicate_rows": summary.duplicate_rows,
            "duplicate_pct": round(summary.duplicate_pct, 3),
            "constant_columns": ", ".join(summary.constant_columns),
            "primary_key_candidates": ", ".join(summary.primary_key_candidates),
            "warnings": "; ".join(summary.warnings),
        }])
        _write_csv(out_dir / "dataset_summary.csv", ds_row)

        # columns_overview.csv
        cols_df = pd.DataFrame([c.to_dict() for c in summary.columns])
        preferred_order = [
            "column","dtype","role","non_null","missing","missing_pct","unique","unique_pct",
            "is_constant","is_id_like","sample_values","n_zeros","min","p5","p50","p95","max",
            "mean","std","top","top_freq","top_pct","date_min","date_max"
        ]
        cols_df = cols_df[[c for c in preferred_order if c in cols_df.columns]]
        _write_csv(out_dir / "columns_overview.csv", cols_df)

        # numeric_stats.csv
        num_cols = [c for c in summary.columns if c.role == "numeric"]
        if num_cols:
            num_df = pd.DataFrame([{
                "column": c.column,
                "non_null": c.non_null,
                "missing_pct": c.missing_pct,
                "n_zeros": c.n_zeros,
                "min": c.min, "p5": c.p5, "p50": c.p50, "p95": c.p95, "max": c.max,
                "mean": c.mean, "std": c.std
            } for c in num_cols])
            _write_csv(out_dir / "numeric_stats.csv", num_df)

        # datetime_ranges.csv
        dt_cols = [c for c in summary.columns if c.role == "datetime"]
        if dt_cols:
            dt_df = pd.DataFrame([{
                "column": c.column,
                "non_null": c.non_null,
                "missing_pct": c.missing_pct,
                "date_min": c.date_min,
                "date_max": c.date_max
            } for c in dt_cols])
            _write_csv(out_dir / "datetime_ranges.csv", dt_df)

        # categorical_overview.csv
        cat_cols = [c for c in summary.columns if c.role in ("categorical","text","boolean")]
        if cat_cols:
            cat_df = pd.DataFrame([{
                "column": c.column,
                "non_null": c.non_null,
                "missing_pct": c.missing_pct,
                "unique": c.unique,
                "unique_pct": c.unique_pct,
                "top": c.top,
                "top_freq": c.top_freq,
                "top_pct": c.top_pct
            } for c in cat_cols])
            _write_csv(out_dir / "categorical_overview.csv", cat_df)

        # value_counts/<col>.csv
        if write_value_counts and len(df):
            for c in cat_cols:
                colname = c.column
                try:
                    vc = df[colname].value_counts(dropna=False).head(topk).rename_axis(colname).reset_index(name="count")
                    vc[colname] = vc[colname].astype(object).where(pd.notna(vc[colname]), "NaN")
                    vc["share_pct"] = (vc["count"] / len(df) * 100).round(3)
                    fname = _sanitize_filename(colname) + ".csv"
                    _write_csv(out_dir / "value_counts" / fname, vc)
                except Exception:
                    continue

        readme = (
            f"# Data Descriptor – {summary.name or 'dataset'}\n\n"
            f"- Rows: {summary.n_rows}\n"
            f"- Columns: {summary.n_cols}\n"
            f"- Memory: {summary.memory_mb:.3f} MB\n"
            f"- Duplicate rows: {summary.duplicate_rows} ({summary.duplicate_pct:.2f}%)\n"
            f"- Primary key candidates: {', '.join(summary.primary_key_candidates) or '—'}\n"
            f"- Constant columns: {', '.join(summary.constant_columns) or '—'}\n"
            f"- Warnings: {('; '.join(summary.warnings)) or '—'}\n\n"
            "Artifacts:\n"
            "- dataset_summary.csv\n"
            "- columns_overview.csv\n"
            "- numeric_stats.csv\n"
            "- datetime_ranges.csv\n"
            "- categorical_overview.csv\n"
            "- value_counts/*.csv\n"
        )
        (out_dir / "README.txt").write_text(readme, encoding="utf-8")

        return out_dir


# ----------------------------- convenience -----------------------------

def profile_dataframe(df: pd.DataFrame, *, name: Optional[str] = None, **profiler_kwargs) -> Dict[str, Any]:
    profiler = DatasetProfiler(**profiler_kwargs)
    return profiler.profile(df, name=name).to_dict()



# src/runners/DataDescriptor.py
# from __future__ import annotations
import os
from pathlib import Path
from typing import Dict

import pandas as pd


def run_full_descriptor(
    df: pd.DataFrame,
    *,
    name: str = "dataset",
    out_root: str = "data_descriptor",
    profiler_topk: int = 50,
) -> Dict[str, Path]:
    """
    Runs:
      1) Generic dataset profiling (folder of CSVs)
      2) AMR-specific overview (CSV subfolders + one Excel)

    Returns dict with paths for easy reference.
    """
    root = Path(out_root)
    profiler_dir = root / "01_dataset_profiler"
    amr_dir = root / "02_amr_overview"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    amr_dir.mkdir(parents=True, exist_ok=True)

    # 1) DatasetProfiler
    profiler = DatasetProfiler(sample_for_uniques=1_000_000_000_000)
    profiler.export_to_folder(
        df, name=name, out_dir=profiler_dir, topk=profiler_topk, overwrite=True, write_value_counts=True
    )

    # 2) AMR Summary
    amr = AMRSummary(df)
    amr.run_all(output_dir=str(amr_dir), excel_name="amr_summary.xlsx")

    # 3) Root README
    readme = f"""# Data Descriptor Bundle – {name}

        This folder contains two complementary views:

        - **01_dataset_profiler/** – generic dataset transparency for domain experts:
        - dataset_summary.csv, columns_overview.csv, numeric_stats.csv, datetime_ranges.csv, categorical_overview.csv
        - value_counts/*.csv (per-column top K)
        - **02_amr_overview/** – ARS-style isolate-specific report:
        - overview.csv, counts_by_dimension/*, antibiotics/*, time/*, crosstabs/*, top/*
        - amr_summary.xlsx (single workbook with all sheets)

        Timestamps, formats, and ID-like columns are inferred with conservative, warning-free logic.
    """
    (root / "README.txt").write_text(readme, encoding="utf-8")

    return {
        "root": root,
        "profiler": profiler_dir,
        "amr": amr_dir,
    }
