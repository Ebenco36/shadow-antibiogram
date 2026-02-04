from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import pandas as pd

Value = Union[str, int, float, bool]
Values = Union[Value, Sequence[Value]]


def _as_list(v: Values) -> List[Value]:
    if isinstance(v, (list, tuple, set, pd.Index)):
        return list(v)
    return [v]


@dataclass
class RobustIsolateCuration:
    """
    Robust hierarchical selector + reporting + file generation.

    Rules implemented:
    - Pathogen & PathogenGenus: assumed already pre-selected (not constrained), but can still be filtered if you pass them.
    - GramType allowed: Gram-negative, Gram-positive
    - Sex allowed: Woman, Man
    - CareType allowed: In-Patient, Out-Patient
    - TextMaterialgroupRkiL0 allowed:
        Urine, Wound, Swab, Respiratory, Blood Culture, Urogenital Swab
    - ARS_WardType: kept, but not required to filter.
    - Keep-as-is columns (no transforms implied):
        BroadAgeGroup, HighLevelAgeRange, Hospital_Priority, Care_Complexity, Facility_Function
    - Report includes:
        * overview (rows kept/removed)
        * distribution comparison for filtered columns
        * list of columns that have no filter (other columns)

    Output files:
    - <basename>__selected.csv
    - <basename>__report_overview.csv
    - <basename>__report_by_column.csv
    - <basename>__report_metadata.json
    - <basename>__non_filtered_columns.txt
    - <basename>__report.xlsx  (multiple sheets)
    """

    df: pd.DataFrame

    # Hierarchy (selection order)
    hierarchy: List[str] = field(default_factory=lambda: [
        "GramType", "Sex", "CareType", "TextMaterialgroupRkiL0"
    ])

    # Columns you said should remain as-is (we don't change anything; this is just documented)
    keep_as_is_cols: Tuple[str, ...] = (
        "BroadAgeGroup",
        "HighLevelAgeRange",
        "Hospital_Priority",
        "Care_Complexity",
        "Facility_Function",
    )

    # Allowed values (curation)
    allowed_values: Dict[str, List[str]] = field(default_factory=lambda: {
        "GramType": ["Gram-negative", "Gram-positive"],
        "Sex": ["Woman", "Man"],
        "CareType": ["In-Patient", "Out-Patient"],
        "TextMaterialgroupRkiL0": [
            "Urine", "Wound", "Swab", "Respiratory", "Blood Culture", "Urogenital Swab"
        ],
        # Pathogen / PathogenGenus intentionally not constrained
    })

    # Robust matching
    case_insensitive: bool = True
    strip_whitespace: bool = True

    label_aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # Ensure hierarchy columns exist
        missing = [c for c in self.hierarchy if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required hierarchy columns: {missing}")

        # Ensure allowed-values columns exist
        missing_allowed = [c for c in self.allowed_values if c not in self.df.columns]
        if missing_allowed:
            raise ValueError(f"allowed_values references columns not in df: {missing_allowed}")

    # -------------------------
    # Core selection (hierarchy)
    # -------------------------
    def select(
        self,
        choices: Dict[str, Values],
        *,
        dropna_in_filters: bool = True,
        validate_allowed: bool = True,
        return_copy: bool = True,
    ) -> pd.DataFrame:
        """
        Apply hierarchical filtering in order of self.hierarchy.
        choices can include Pathogen / PathogenGenus too (even though you say they’re already pre-selected).
        """
        # Validate columns exist
        unknown = [k for k in choices if k not in self.df.columns]
        if unknown:
            raise KeyError(f"Unknown choice columns: {unknown}")

        if validate_allowed:
            self._validate_choices_against_allowed(choices)

        cur = self.df
        # Apply hierarchy first
        for col in self.hierarchy:
            if col in choices:
                cur = self._filter(cur, col, choices[col], dropna_in_filters=dropna_in_filters)

        # Apply any extra filters (e.g., Pathogen, PathogenGenus) afterward (still “within” the hierarchy result)
        extras = [c for c in choices if c not in self.hierarchy]
        for col in extras:
            cur = self._filter(cur, col, choices[col], dropna_in_filters=dropna_in_filters)

        return cur.copy() if return_copy else cur

    # -------------------------
    # Comparison report
    # -------------------------
    def compare_report(
        self,
        selected_df: pd.DataFrame,
        *,
        report_columns: Optional[List[str]] = None,
        include_na: bool = True,
        percent_base: str = "global",  # "global" or "parent" (parent not used here; global is standard)
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare ORIGINAL vs SELECTED/filtered.

        Returns:
          - overview: one-row summary
          - by_column: long table with counts/%/delta per value, per column
          - other_columns: list of non-filtered columns (as a DataFrame)
        """
        if not isinstance(selected_df, pd.DataFrame):
            raise TypeError("selected_df must be a pandas DataFrame")

        n0 = len(self.df)
        n1 = len(selected_df)
        kept_pct = (n1 / n0 * 100) if n0 else 0.0

        overview = pd.DataFrame([{
            "original_rows": n0,
            "selected_rows": n1,
            "rows_removed": n0 - n1,
            "kept_percent": round(kept_pct, 6),
        }])

        cols = report_columns or list(dict.fromkeys(self.hierarchy))  # unique, keep order
        # also allow Pathogen/PathogenGenus if present in report_columns
        missing = [c for c in cols if c not in self.df.columns or c not in selected_df.columns]
        if missing:
            raise ValueError(f"Report columns missing from df/selected_df: {missing}")

        # list “other columns” (no filter)
        filtered_cols_set = set(cols)
        other_cols = [c for c in self.df.columns if c not in filtered_cols_set]
        other_columns_df = pd.DataFrame({"non_filtered_columns": other_cols})

        dropna_flag = not include_na  # pandas uses dropna=True to exclude NA
        rows = []
        for col in cols:
            orig_counts = self.df[col].value_counts(dropna=dropna_flag)
            sel_counts = selected_df[col].value_counts(dropna=dropna_flag)

            all_vals = orig_counts.index.union(sel_counts.index)

            for v in all_vals:
                oc = int(orig_counts.get(v, 0))
                sc = int(sel_counts.get(v, 0))

                op = (oc / n0 * 100) if n0 else 0.0
                sp = (sc / n1 * 100) if n1 else 0.0

                rows.append({
                    "column": col,
                    "value": v,
                    "original_count": oc,
                    "selected_count": sc,
                    "count_delta": sc - oc,
                    "original_percent": round(op, 6),
                    "selected_percent": round(sp, 6),
                    "delta_pct_points": round(sp - op, 6),
                    "category_retention_percent": round((sc / oc * 100), 6) if oc else (0.0 if sc == 0 else 100.0),
                })

        by_column = pd.DataFrame(rows).sort_values(
            by=["column", "original_count"], ascending=[True, False], kind="mergesort"
        )

        return {
            "overview": overview,
            "by_column": by_column,
            "other_columns": other_columns_df,
        }

    # -------------------------
    # File generation (NEW)
    # -------------------------
    def export_report(
        self,
        *,
        choices: Dict[str, Values],
        out_dir: str = "curation_outputs",
        basename: str = "isolate_curation",
        include_na: bool = True,
        dropna_in_filters: bool = True,
        validate_allowed: bool = True,
        # save_selected_csv: bool = True,
        save_selected_parquet: bool = False,
    ) -> Dict[str, str]:
        """
        Runs selection + report, then writes files.

        Returns a dict of written file paths.
        """
        os.makedirs(out_dir, exist_ok=True)

        selected = self.select(
            choices,
            dropna_in_filters=dropna_in_filters,
            validate_allowed=validate_allowed,
            return_copy=True,
        )
        report = self.compare_report(
            selected,
            report_columns=list(dict.fromkeys(self.hierarchy)),
            include_na=include_na,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # base = f"{basename}__{ts}"
        base = f"{basename}"
        paths: Dict[str, str] = {}

        # Selected dataset
        # if save_selected_csv:
        #     p = os.path.join(out_dir, f"{base}__selected.csv")
        #     selected.to_csv(p, index=False)
        #     paths["selected_csv"] = p

        if save_selected_parquet:
            from src.runners.DataProcessing import save_parquet_flat
            save_parquet_flat(
                df=selected,
                out_dir="./reproducible/WHO_Aware_data",
                rows_per_file=250_000
            )
            paths["selected_parquet"] = "./reproducible/WHO_Aware_data"

        # CSV reports
        p_overview = os.path.join(out_dir, f"{base}__report_overview.csv")
        report["overview"].to_csv(p_overview, index=False)
        paths["report_overview_csv"] = p_overview

        p_bycol = os.path.join(out_dir, f"{base}__report_by_column.csv")
        report["by_column"].to_csv(p_bycol, index=False)
        paths["report_by_column_csv"] = p_bycol

        # Other columns (non-filtered) list
        p_other = os.path.join(out_dir, f"{base}__non_filtered_columns.txt")
        with open(p_other, "w", encoding="utf-8") as f:
            for c in report["other_columns"]["non_filtered_columns"].tolist():
                f.write(str(c) + "\n")
        paths["non_filtered_columns_txt"] = p_other

        # Metadata (choices + rules snapshot)
        meta = {
            "timestamp": ts,
            "basename": basename,
            "hierarchy": self.hierarchy,
            "allowed_values": self.allowed_values,
            "choices_used": self._json_safe_choices(choices),
            "include_na_in_report": include_na,
            "dropna_in_filters": dropna_in_filters,
            "validate_allowed": validate_allowed,
            "kept_as_is_columns": list(self.keep_as_is_cols),
            "note": "Pathogen & PathogenGenus assumed pre-selected (WHO priority pathogens).",
        }
        p_meta = os.path.join(out_dir, f"{base}__report_metadata.json")
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        paths["metadata_json"] = p_meta

        # Excel report (multi-sheet)
        p_xlsx = os.path.join(out_dir, f"{base}__report.xlsx")
        with pd.ExcelWriter(p_xlsx, engine="openpyxl") as writer:
            report["overview"].to_excel(writer, sheet_name="overview", index=False)
            report["by_column"].to_excel(writer, sheet_name="by_column", index=False)
            report["other_columns"].to_excel(writer, sheet_name="non_filtered_cols", index=False)

            # Optional: save the “choices” as a sheet for reproducibility
            pd.DataFrame([meta["choices_used"]]).to_excel(writer, sheet_name="choices_used", index=False)

            # Optional: quick top shifts
            top_shifts = (
                report["by_column"]
                .assign(abs_shift=lambda d: d["delta_pct_points"].abs())
                .sort_values("abs_shift", ascending=False)
                .head(50)
                .drop(columns=["abs_shift"])
            )
            top_shifts.to_excel(writer, sheet_name="top_shifts", index=False)

        paths["report_xlsx"] = p_xlsx

        return paths

    # -------------------------
    # Internals
    # -------------------------
    def _normalize(self, x: Any) -> Any:
        if isinstance(x, str):
            if self.strip_whitespace:
                x = x.strip()
            if self.case_insensitive:
                x = x.casefold()
        return x

    def _apply_alias(self, col: str, v: Any) -> Any:
        """Map label variants to canonical curated labels (optional)."""
        if not isinstance(v, str):
            return v
        aliases = self.label_aliases.get(col, {})
        # try exact + normalized key matching
        if v in aliases:
            return aliases[v]
        nv = self._normalize(v)
        for k, canonical in aliases.items():
            if self._normalize(k) == nv:
                return canonical
        return v

    def _filter(self, df: pd.DataFrame, col: str, values: Values, *, dropna_in_filters: bool) -> pd.DataFrame:
        s = df[col]
        na_mask = s.isna() if dropna_in_filters else pd.Series(False, index=s.index)

        vals = [_as_list(values)][0]
        # apply aliases to requested values
        vals = [self._apply_alias(col, v) for v in vals]

        if self.case_insensitive or self.strip_whitespace:
            s_norm = s.astype("string")
            if self.strip_whitespace:
                s_norm = s_norm.str.strip()
            if self.case_insensitive:
                s_norm = s_norm.str.casefold()

            v_norm = [self._normalize(v) for v in vals]
            mask = s_norm.isin(v_norm)
        else:
            mask = s.isin(vals)

        mask = mask & (~na_mask)
        return df.loc[mask.fillna(False)]

    def _validate_choices_against_allowed(self, choices: Dict[str, Values]) -> None:
        for col, v in choices.items():
            # only validate columns you curated allowed values for
            if col not in self.allowed_values:
                continue
            allowed = [self._normalize(self._apply_alias(col, x)) for x in self.allowed_values[col]]
            requested = [self._normalize(self._apply_alias(col, x)) for x in _as_list(v)]
            bad = [x for x in requested if x not in allowed]
            if bad:
                raise ValueError(
                    f"Choice(s) not allowed for '{col}': {bad}. Allowed: {self.allowed_values[col]}"
                )

    @staticmethod
    def _json_safe_choices(choices: Dict[str, Values]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for k, v in choices.items():
            if isinstance(v, (list, tuple, set, pd.Index)):
                safe[k] = list(v)
            else:
                safe[k] = v
        return safe



from src.controllers.DataLoader import DataLoader
from src.mappers.top_pathogens import ALL_PATHOGENS

parquet_dir = "./datasets/WHO_Aware_data"
loader = DataLoader(parquet_dir, pathogen_groups_regex=ALL_PATHOGENS)
df = loader.get_combined(return_which="tested")
curator = RobustIsolateCuration(df)

choices = {
    "GramType": ["Gram-negative", "Gram-positive"],
    "Sex": ["Woman", "Man"],
    "CareType": ["In-Patient", "Out-Patient"],
    "TextMaterialgroupRkiL0": ["Urine", "Blood Culture", "Wound", "Swab", "Respiratory", "Urogenital Swab"],
    # Pathogen / PathogenGenus optional (already pre-selected)
}

paths = curator.export_report(
    choices=choices,
    out_dir="curation_outputs",
    basename="who_priority_isolates",
    include_na=True,
    save_selected_parquet=True,
)
print(paths)
