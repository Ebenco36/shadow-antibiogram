from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import re
import time
import logging

import pandas as pd


# -------------------------
# Logging setup (safe default)
# -------------------------
def _default_logger(name: str = "MonthlyCoTestingAggregator") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@dataclass(frozen=True)
class AggregationSchema:
    """
    Schema for monthly-year aggregation that preserves co-testing patterns by
    aggregating *_Tested columns using logical OR (max over 0/1).

    Publication robustness goals:
      - deterministic output
      - strict schema validation
      - clear error messages
      - auditable aggregation rules
    """
    group_keys: Tuple[str, ...] = (
        "Pathogen",
        "PathogengroupL1",
        "GramType",
        "Sex",
        "PathogenGenus",
        "TextMaterialgroupRkiL0",
        "AgeGroup",
        "AgeRange",
        "CareType",
        "ARS_HospitalLevelManual",
        "ARS_WardType",
        "ARS_Region",
        "YearMonth",
        "Care_Complexity",
        "Hospital_Priority",
        "Facility_Function",
        "SeasonName",
        "MonthName",
        "Year",
        "Month"
    )

    antibiotic_tested_suffix: str = "_Tested"

    # Optional: allow additional boolean-like columns to be OR-aggregated
    extra_or_columns: Tuple[str, ...] = (
        "IsSpecificlyExcluded_Screening",
        "IsSpecificlyExcluded_Pathogen",
        "IsSpecificlyExcluded_PathogenevidenceNegative",
    )

    # Missing handling for required group keys (publication-safe, deterministic)
    allow_missing_group_keys: bool = True
    fill_missing_group_keys: bool = True
    missing_group_key_label: str = "Unknown"

    # Drop rows where any group key is missing/empty (before aggregation)
    drop_rows_with_missing_group_keys: bool = False
    treat_empty_string_as_missing: bool = True  # applies to string/object columns

    # Drop groups with small sample size (isolate count) before aggregation
    min_group_size: int = 1  # set e.g. 5/10/20 for public release

    # Add canonical monthly date derived from YearMonth
    add_month_date: bool = True
    month_date_colname: str = "Date"

    # Optional: retain a group size column for QC (can be disabled for public release)
    add_group_count: bool = False
    group_count_colname: str = "n_isolates_in_group"


class SchemaError(ValueError):
    """Raised when the input dataframe does not match required schema."""


class DataAggregationError(RuntimeError):
    """Raised when aggregation fails for unexpected reasons."""


@dataclass
class MonthlyCoTestingAggregator:
    """
    Aggregate isolate-level AMR testing data to monthly strata while keeping
    antibiotic *_Tested columns as binary indicators (0/1) using logical OR.

    Add-ons:
      - optionally drop rows with incomplete group keys
      - optionally drop strata with isolate count < min_group_size
      - optionally add Date (first day of month) derived from YearMonth

    Logging & summary:
      - logs progress of each stage
      - stores last_summary dict for programmatic access
      - optionally returns summary alongside the dataframe
    """
    schema: AggregationSchema = field(default_factory=AggregationSchema)
    tested_col_regex: Optional[str] = None
    logger: logging.Logger = field(default_factory=_default_logger)

    # Populated after running aggregate()/make_public_release()
    last_summary: Dict[str, Any] = field(default_factory=dict, init=False)

    # -------------------------
    # Utility: timing + summaries
    # -------------------------
    def _start_step(self, name: str) -> float:
        self.logger.info(f"[START] {name}")
        return time.perf_counter()

    def _end_step(self, name: str, t0: float, **metrics: Any) -> None:
        dt = time.perf_counter() - t0
        payload = {"seconds": round(dt, 3), **metrics}
        self.logger.info(f"[END] {name} | {payload}")
        self.last_summary.setdefault("steps", {})[name] = payload

    # -------------------------
    # Column detection / coercion
    # -------------------------
    def _detect_tested_columns(self, df: pd.DataFrame) -> List[str]:
        if self.tested_col_regex:
            pat = re.compile(self.tested_col_regex)
            cols = [c for c in df.columns if pat.search(c)]
        else:
            cols = [c for c in df.columns if c.endswith(self.schema.antibiotic_tested_suffix)]
        return cols

    @staticmethod
    def _is_booleanish(series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series):
            return True
        if pd.api.types.is_integer_dtype(series):
            return True
        if pd.api.types.is_float_dtype(series):
            return True
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            return True
        return False

    @staticmethod
    def _coerce_to_binary01(s: pd.Series, colname: str) -> pd.Series:
        """
        Coerce a column to pandas nullable integer (Int8) with values {0,1} (or <NA> if missing).
        Raises SchemaError if values are outside {0,1} after coercion.
        """
        if pd.api.types.is_bool_dtype(s):
            out = s.astype("Int8")

        elif pd.api.types.is_numeric_dtype(s):
            out = pd.to_numeric(s, errors="coerce")
            invalid = out.dropna().loc[~out.dropna().isin([0, 1])]
            if len(invalid) > 0:
                sample = invalid.iloc[:5].tolist()
                raise SchemaError(
                    f"Column '{colname}' contains non-binary numeric values (expected 0/1). "
                    f"Examples: {sample}"
                )
            out = out.astype("Int8")

        else:
            normalized = s.astype("string").str.strip().str.lower()
            mapping = {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}
            out = normalized.map(mapping)
            mask_unmapped = normalized.notna() & out.isna()
            if mask_unmapped.any():
                bad_vals = normalized[mask_unmapped].unique()[:5].tolist()
                raise SchemaError(
                    f"Column '{colname}' contains non-binary string values (expected 0/1/true/false). "
                    f"Examples: {bad_vals}"
                )
            out = out.astype("Int8")

        return out

    # -------------------------
    # Row & group filtering helpers
    # -------------------------
    def _drop_rows_with_missing_group_keys(self, work: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where ANY grouping key is missing (NA) and optionally empty string / whitespace.
        This happens BEFORE any filling policy and BEFORE aggregation.
        """
        if not self.schema.drop_rows_with_missing_group_keys:
            return work

        keys = list(self.schema.group_keys)

        # NA in any group key
        drop_mask = work[keys].isna().any(axis=1)

        # Empty-string in any group key (for string/object columns)
        empty_cols_checked = 0
        if self.schema.treat_empty_string_as_missing:
            empty_masks = []
            for c in keys:
                col = work[c]
                if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
                    empty_cols_checked += 1
                    empty_masks.append(col.astype("string").str.strip().eq(""))
            if empty_masks:
                empty_mask = pd.concat(empty_masks, axis=1).any(axis=1)
                drop_mask = drop_mask | empty_mask

        dropped = int(drop_mask.sum())
        self.last_summary.setdefault("filters", {})["drop_rows_with_missing_group_keys"] = {
            "enabled": True,
            "treat_empty_string_as_missing": self.schema.treat_empty_string_as_missing,
            "empty_cols_checked": empty_cols_checked,
            "rows_dropped": dropped,
        }

        return work.loc[~drop_mask].copy(deep=False)

    def _apply_group_key_missing_policy(self, work: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing group keys deterministically (if enabled).
        If drop_rows_with_missing_group_keys=True, this is usually a no-op.
        """
        if not self.schema.fill_missing_group_keys:
            return work

        label = self.schema.missing_group_key_label
        filled_total = 0

        for c in self.schema.group_keys:
            if work[c].isna().any():
                n = int(work[c].isna().sum())
                filled_total += n
                work[c] = work[c].astype("string").fillna(label)

        self.last_summary.setdefault("fills", {})["fill_missing_group_keys"] = {
            "enabled": True,
            "label": label,
            "values_filled": filled_total,
        }
        return work

    def _remove_unused_categories(self, work: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory by removing unused categories in grouping keys (important at scale).
        """
        removed_info: Dict[str, Dict[str, int]] = {}
        for c in self.schema.group_keys:
            if c in work.columns and pd.api.types.is_categorical_dtype(work[c]):
                before = len(work[c].cat.categories)
                work[c] = work[c].cat.remove_unused_categories()
                after = len(work[c].cat.categories)
                if after != before:
                    removed_info[c] = {"categories_before": before, "categories_after": after}

        if removed_info:
            self.last_summary.setdefault("memory", {})["removed_unused_categories"] = removed_info
        return work

    def _filter_small_groups(self, work: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows belonging to strata with isolate count < min_group_size.
        Done BEFORE aggregation to save memory/time.
        """
        k = int(self.schema.min_group_size)
        if k <= 1:
            self.last_summary.setdefault("filters", {})["min_group_size"] = {"enabled": False, "min_group_size": k}
            return work

        keys = list(self.schema.group_keys)

        sizes = (
            work.groupby(keys, dropna=False, sort=False, observed=True)
                .size()
                .rename("_group_n")
                .reset_index()
        )

        n_groups_before = int(len(sizes))
        keep = sizes.loc[sizes["_group_n"] >= k, keys]
        n_groups_after = int(len(keep))

        filtered = work.merge(keep, on=keys, how="inner")

        self.last_summary.setdefault("filters", {})["min_group_size"] = {
            "enabled": True,
            "min_group_size": k,
            "groups_before": n_groups_before,
            "groups_after": n_groups_after,
            "groups_dropped": n_groups_before - n_groups_after,
            "rows_after_filter": int(len(filtered)),
        }
        return filtered

    # -------------------------
    # Validation
    # -------------------------
    def validate_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise SchemaError("Input must be a pandas DataFrame.")

        missing = [c for c in self.schema.group_keys if c not in df.columns]
        if missing:
            raise SchemaError("Missing required grouping columns: " + ", ".join(missing))

        tested_cols = self._detect_tested_columns(df)
        if len(tested_cols) == 0:
            raise SchemaError(
                f"No antibiotic tested columns detected. Expected columns ending with "
                f"'{self.schema.antibiotic_tested_suffix}' (or matching tested_col_regex)."
            )

        if self.schema.add_month_date and "YearMonth" not in df.columns:
            raise SchemaError("YearMonth column is required to derive the monthly Date column.")

        # If user disallows missing keys AND we are not dropping/filling -> error
        null_key_cols = [c for c in self.schema.group_keys if df[c].isna().any()]
        if (
            null_key_cols
            and not self.schema.allow_missing_group_keys
            and not self.schema.fill_missing_group_keys
            and not self.schema.drop_rows_with_missing_group_keys
        ):
            raise SchemaError(
                "Grouping columns contain missing values (NA), which is disallowed by schema: "
                + ", ".join(null_key_cols)
            )

        for c in tested_cols:
            if not self._is_booleanish(df[c]):
                raise SchemaError(f"Column '{c}' is not boolean-like; expected 0/1 or True/False.")

        for c in self.schema.extra_or_columns:
            if c in df.columns and not self._is_booleanish(df[c]):
                raise SchemaError(f"Column '{c}' is not boolean-like; expected 0/1 or True/False.")

    # -------------------------
    # Main aggregation
    # -------------------------
    def aggregate(self, df: pd.DataFrame, return_summary: bool = False):
        """
        Perform the monthly-year aggregation.

        If return_summary=True, returns (out_df, summary_dict).
        Otherwise returns out_df and stores summary in self.last_summary.
        """
        self.last_summary = {}  # reset for this run

        t0 = self._start_step("validate_input")
        self.validate_input(df)
        self._end_step("validate_input", t0, rows_in=int(len(df)), cols_in=int(df.shape[1]))

        tested_cols = self._detect_tested_columns(df)
        extra_or_cols = [c for c in self.schema.extra_or_columns if c in df.columns]
        self.last_summary["columns"] = {
            "group_keys": list(self.schema.group_keys),
            "tested_cols_count": len(tested_cols),
            "tested_cols_sample": tested_cols[:10],
            "extra_or_cols_present": extra_or_cols,
        }

        t1 = self._start_step("prepare_working_frame")
        work = df.copy(deep=False)
        self._end_step("prepare_working_frame", t1, rows=int(len(work)))

        t2 = self._start_step("drop_rows_with_missing_group_keys")
        before = int(len(work))
        work = self._drop_rows_with_missing_group_keys(work)
        after = int(len(work))
        self._end_step("drop_rows_with_missing_group_keys", t2, rows_before=before, rows_after=after, rows_dropped=before - after)

        t3 = self._start_step("fill_missing_group_keys")
        before = int(len(work))
        work = self._apply_group_key_missing_policy(work)
        self._end_step("fill_missing_group_keys", t3, rows=int(len(work)), enabled=bool(self.schema.fill_missing_group_keys))

        t4 = self._start_step("remove_unused_categories")
        work = self._remove_unused_categories(work)
        self._end_step("remove_unused_categories", t4)

        t5 = self._start_step("filter_small_groups")
        before = int(len(work))
        work = self._filter_small_groups(work)
        after = int(len(work))
        self._end_step("filter_small_groups", t5, rows_before=before, rows_after=after, rows_dropped=before - after, min_group_size=int(self.schema.min_group_size))

        t6 = self._start_step("coerce_boolean_columns")
        # Coerce tested + extra columns to Int8 {0,1,<NA>}
        for c in tested_cols:
            work[c] = self._coerce_to_binary01(work[c], c)
        for c in extra_or_cols:
            work[c] = self._coerce_to_binary01(work[c], c)
        self._end_step("coerce_boolean_columns", t6, tested_cols=len(tested_cols), extra_or_cols=len(extra_or_cols))

        agg_dict: Dict[str, str] = {c: "max" for c in tested_cols}
        agg_dict.update({c: "max" for c in extra_or_cols})

        try:
            t7 = self._start_step("groupby_aggregate")
            gb = work.groupby(
                list(self.schema.group_keys),
                dropna=False,
                sort=False,
                observed=True,  # critical for categorical keys
            )
            out = gb.agg(agg_dict).reset_index()
            self._end_step("groupby_aggregate", t7, rows_out=int(len(out)))

            if self.schema.add_group_count:
                t8 = self._start_step("add_group_count")
                counts = gb.size().reset_index(name=self.schema.group_count_colname)
                out = out.merge(counts, on=list(self.schema.group_keys), how="left")
                self._end_step("add_group_count", t8, col_added=self.schema.group_count_colname)

        except Exception as e:
            self.logger.exception("Aggregation failed")
            raise DataAggregationError(f"Aggregation failed: {e}") from e

        t9 = self._start_step("add_month_date")
        if self.schema.add_month_date:
            out[self.schema.month_date_colname] = pd.to_datetime(out["YearMonth"], format="%Y-%m", errors="raise")
        self._end_step("add_month_date", t9, enabled=bool(self.schema.add_month_date), col=self.schema.month_date_colname)

        t10 = self._start_step("finalize_columns")
        ordered_cols = list(self.schema.group_keys)

        if self.schema.add_month_date:
            if "YearMonth" in ordered_cols and self.schema.month_date_colname in out.columns:
                ym_idx = ordered_cols.index("YearMonth")
                ordered_cols = (
                    ordered_cols[: ym_idx + 1]
                    + [self.schema.month_date_colname]
                    + ordered_cols[ym_idx + 1 :]
                )
            elif self.schema.month_date_colname in out.columns:
                ordered_cols.append(self.schema.month_date_colname)

        ordered_cols += extra_or_cols + tested_cols
        if self.schema.add_group_count:
            ordered_cols.append(self.schema.group_count_colname)

        ordered_cols = [c for c in ordered_cols if c in out.columns]
        out = out.loc[:, ordered_cols]

        for c in extra_or_cols + tested_cols:
            out[c] = out[c].astype("Int8")

        self._end_step("finalize_columns", t10, cols_out=int(out.shape[1]))

        # Run-level summary
        self.last_summary["run"] = {
            "rows_in": int(len(df)),
            "rows_working_final": int(len(work)),
            "rows_out": int(len(out)),
            "cols_out": int(out.shape[1]),
            "tested_cols": int(len(tested_cols)),
            "min_group_size": int(self.schema.min_group_size),
            "drop_rows_with_missing_group_keys": bool(self.schema.drop_rows_with_missing_group_keys),
            "fill_missing_group_keys": bool(self.schema.fill_missing_group_keys),
        }

        if return_summary:
            return out, self.last_summary
        return out

    def make_public_release(
        self,
        df: pd.DataFrame,
        drop_all_zero_rows: bool = False,
        keep_qc_count: bool = False,
        return_summary: bool = False,
    ):
        """
        Convenience wrapper for creating a public-release table.

        If return_summary=True, returns (out_df, summary_dict).
        """
        original = self.schema

        if keep_qc_count and not self.schema.add_group_count:
            self.schema = AggregationSchema(
                group_keys=original.group_keys,
                antibiotic_tested_suffix=original.antibiotic_tested_suffix,
                extra_or_columns=original.extra_or_columns,
                allow_missing_group_keys=original.allow_missing_group_keys,
                fill_missing_group_keys=original.fill_missing_group_keys,
                missing_group_key_label=original.missing_group_key_label,
                drop_rows_with_missing_group_keys=original.drop_rows_with_missing_group_keys,
                treat_empty_string_as_missing=original.treat_empty_string_as_missing,
                min_group_size=original.min_group_size,
                add_month_date=original.add_month_date,
                month_date_colname=original.month_date_colname,
                add_group_count=True,
                group_count_colname=original.group_count_colname,
            )

        try:
            out, summary = self.aggregate(df, return_summary=True)
        finally:
            self.schema = original

        tested_cols = self._detect_tested_columns(out)

        if drop_all_zero_rows and tested_cols:
            t0 = self._start_step("drop_all_zero_rows")
            before = int(len(out))
            mask_any_tested = out[tested_cols].fillna(0).sum(axis=1) > 0
            out = out.loc[mask_any_tested].reset_index(drop=True)
            after = int(len(out))
            self._end_step("drop_all_zero_rows", t0, rows_before=before, rows_after=after, rows_dropped=before - after)
            summary["post_filters"] = {"drop_all_zero_rows": {"rows_before": before, "rows_after": after}}

        if return_summary:
            return out, summary
        return out


# -------------------------
# Example usage (minimal)
# -------------------------
if __name__ == "__main__":
    from src.controllers.DataLoader import DataLoader
    from src.runners.DataProcessing import save_parquet_flat

    loader = DataLoader("./datasets/WHO_Aware_data_")
    df = loader.get_combined()

    aggregator = MonthlyCoTestingAggregator(
        schema=AggregationSchema(
            drop_rows_with_missing_group_keys=True,
            treat_empty_string_as_missing=True,
            min_group_size=5,
            add_month_date=True,
            month_date_colname="Date",
        )
    )

    monthly_public, summary = aggregator.make_public_release(
        df,
        drop_all_zero_rows=False,
        keep_qc_count=False,
        return_summary=True,
    )

    # You can print or persist the summary for reproducibility
    aggregator.logger.info(f"RUN SUMMARY | {summary['run']}")

    save_parquet_flat(
        df=monthly_public,
        out_dir="./datasets/WHO_Aware_data",
        rows_per_file=250_000
    )
