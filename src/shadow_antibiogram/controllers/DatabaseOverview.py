# src/runners/DatabaseOverview.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class AMRSummary:
    """
    Create comprehensive summary statistics for ARS-style isolate data.

    Expected columns (case-insensitive, extra columns are fine):
      - Core: Pathogen, PathogengroupL1, GramType, Sex, PathogenGenus,
              TextMaterialgroupRkiL0, ARS_HospitalLevelManual, AgeGroup, AgeRange,
              CareType, ARS_HospitalLevelManual, ARS_WardType, ARS_Region,
              Year, Month, MonthName, YearMonth, SeasonName
      - Antibiotic test flags: any column ending with "_Tested" (0/1)
        *If not present*, we fall back to antibiotic result columns that contain " - "
        and infer "tested" as non-null result.

    Main outputs (returned as dict of DataFrames and saved as CSV/Excel):
      - overview                        : record counts & completeness
      - counts_by_dimension/*           : counts by each useful dimension
      - antibiotics/overall             : test coverage per antibiotic
      - antibiotics/by_pathogen         : coverage per antibiotic & pathogen
      - time/overall_monthly            : isolates by YearMonth
      - time/pathogen_monthly           : isolates by YearMonth & Pathogen
      - crosstabs/gram_by_material      : GramType x TextMaterialgroupRkiL0
      - top/top_pathogens               : top pathogens by isolate count
      - top/top_genera                  : top genera by isolate count
    """
    df: pd.DataFrame
    top_n_pathogens: int = 20
    min_abx_tested_display: int = 1  # hide abx never tested
    _df: pd.DataFrame = field(init=False, repr=False)
    _abx_cols: List[str] = field(init=False, repr=False)          # *_Tested columns (0/1)
    _abx_infer_cols: List[str] = field(init=False, repr=False)    # fallback result cols with " - "
    _use_inferred_tested: bool = field(init=False, repr=False)    # whether fallback is in use
    _dim_cols: List[str] = field(init=False, repr=False)

    def __post_init__(self):
        self._df = self._prep_dataframe(self.df.copy())
        self._abx_cols = self._find_antibiotic_columns(self._df.columns)

        # Fallback: infer tested from result columns that contain " - " when no *_Tested exists
        self._abx_infer_cols = []
        if not self._abx_cols:
            self._abx_infer_cols = [
                c for c in self._df.columns
                if isinstance(c, str) and (" - " in c) and (not c.endswith("_Tested"))
            ]
        self._use_inferred_tested = (len(self._abx_cols) == 0 and len(self._abx_infer_cols) > 0)

        self._dim_cols = self._infer_dimension_columns(self._df.columns)

    # ---------------------------- public API ---------------------------- #

    def run_all(self, output_dir: str = "amr_summary", excel_name: str = "amr_summary.xlsx"
                ) -> Dict[str, pd.DataFrame]:
        """
        Run all summaries and save outputs as CSVs and a single Excel workbook.
        Returns a dict mapping 'group/name' -> DataFrame.
        """
        os.makedirs(output_dir, exist_ok=True)

        results: Dict[str, pd.DataFrame] = {}

        # Overview
        results["overview"] = self._overview()

        # Counts by each dimension
        for dim in self._dim_cols:
            res = (self._df
                   .groupby(dim, dropna=False)
                   .size()
                   .rename("isolates")
                   .reset_index()
                   .sort_values("isolates", ascending=False))
            results[f"counts_by_dimension/{dim}"] = res

        # ----- Top lists -----
        key = "counts_by_dimension/Pathogen"
        if key in results:
            top_pathogens = results[key]
        else:
            top_pathogens = self._empty_df(["Pathogen", "isolates"])
        top_pathogens = top_pathogens.head(self.top_n_pathogens).reset_index(drop=True)
        results["top/top_pathogens"] = top_pathogens

        if "PathogenGenus" in self._df.columns:
            top_genera = (self._df.groupby("PathogenGenus", dropna=False)
                          .size().rename("isolates").reset_index()
                          .sort_values("isolates", ascending=False)
                          .head(self.top_n_pathogens).reset_index(drop=True))
            results["top/top_genera"] = top_genera

        # Antibiotics: overall coverage
        results["antibiotics/overall"] = self._antibiotic_overall_coverage()

        # Antibiotics by pathogen (top N for readability)
        abx_by_pathogen = self._antibiotic_by_pathogen(
            top_pathogens["Pathogen"].dropna().astype(str).tolist()
        )
        results["antibiotics/by_pathogen"] = abx_by_pathogen

        # Time series
        overall_monthly, pathogen_monthly = self._time_series()
        results["time/overall_monthly"] = overall_monthly
        results["time/pathogen_monthly"] = pathogen_monthly

        # Crosstabs
        results["crosstabs/gram_by_material"] = self._crosstab("GramType", "TextMaterialgroupRkiL0")

        # Save to CSVs
        self._save_csvs(results, output_dir)

        # Save to single Excel
        self._save_excel(results, os.path.join(output_dir, excel_name))

        return results

    # -------------------------- core computations ----------------------- #

    def _overview(self) -> pd.DataFrame:
        total_isolates = int(len(self._df))

        # completeness for key columns
        completeness = []
        for col in self._dim_cols:
            non_null = int(self._df[col].notna().sum())
            completeness.append({
                "column": col,
                "non_null": non_null,
                "missing": total_isolates - non_null,
                "pct_non_null": round(100 * non_null / max(total_isolates, 1), 2)
            })

        # antibiotic column completeness (how many tested >=1)
        abx_rows = []
        if self._use_inferred_tested:
            # infer tested as "non-null result present" per antibiotic result column
            for col in self._abx_infer_cols:
                tested = int(self._df[col].notna().sum())
                if tested < self.min_abx_tested_display:
                    continue
                abx_rows.append({
                    "antibiotic": self._pretty_abx(col),
                    "tested_count": tested,
                    "tested_pct": round(100 * tested / max(total_isolates, 1), 2)
                })
        else:
            for col in self._abx_cols:
                tested = int(self._df[col].fillna(0).astype(int).sum())
                if tested < self.min_abx_tested_display:
                    continue
                abx_rows.append({
                    "antibiotic": self._pretty_abx(col),
                    "tested_count": tested,
                    "tested_pct": round(100 * tested / max(total_isolates, 1), 2)
                })

        overview_top = pd.DataFrame([{
            "metric": "total_isolates",
            "value": total_isolates
        }])

        completeness_df = pd.DataFrame(completeness).sort_values("pct_non_null", ascending=False)
        abx_df = (pd.DataFrame(abx_rows)
                    .sort_values("tested_count", ascending=False)
                  if abx_rows else
                  pd.DataFrame(columns=["antibiotic", "tested_count", "tested_pct"]))

        # return as a single table (stacked with section labels)
        overview_top["section"] = "00_overview"
        completeness_df["section"] = "10_completeness_dimensions"
        abx_df["section"] = "20_abx_test_coverage"

        # Align columns
        overview_top = overview_top.rename(columns={"metric": "name", "value": "value"})
        completeness_df = completeness_df.rename(columns={"column": "name", "pct_non_null": "value"})
        abx_df = abx_df.rename(columns={"antibiotic": "name", "tested_pct": "value"})

        # Add helpful extras in 'details'
        completeness_df["details"] = completeness_df.apply(
            lambda r: f"non_null={r['non_null']}, missing={r['missing']}", axis=1
        )
        if not abx_df.empty and "tested_count" in abx_df.columns:
            abx_df["details"] = abx_df.apply(lambda r: f"tested_count={r['tested_count']}", axis=1)
        else:
            abx_df["details"] = []

        overview_top["details"] = ""

        ordered = pd.concat([overview_top[["section", "name", "value", "details"]],
                             completeness_df[["section", "name", "value", "details"]],
                             abx_df[["section", "name", "value", "details"]]],
                            ignore_index=True)
        return ordered

    def _antibiotic_overall_coverage(self) -> pd.DataFrame:
        n = len(self._df)
        out = []

        if self._use_inferred_tested:
            if not self._abx_infer_cols:
                return self._empty_df(["antibiotic", "tested_count", "tested_pct"])
            for col in self._abx_infer_cols:
                tested = int(self._df[col].notna().sum())
                if tested < self.min_abx_tested_display:
                    continue
                out.append({
                    "antibiotic": self._pretty_abx(col),
                    "tested_count": tested,
                    "tested_pct": round(100 * tested / max(n, 1), 2)
                })
        else:
            if not self._abx_cols:
                return self._empty_df(["antibiotic", "tested_count", "tested_pct"])
            for col in self._abx_cols:
                tested = int(self._df[col].fillna(0).astype(int).sum())
                if tested < self.min_abx_tested_display:
                    continue
                out.append({
                    "antibiotic": self._pretty_abx(col),
                    "tested_count": tested,
                    "tested_pct": round(100 * tested / max(n, 1), 2)
                })

        return (pd.DataFrame(out)
                .sort_values(["tested_count", "antibiotic"], ascending=[False, True])
                .reset_index(drop=True))

    def _antibiotic_by_pathogen(self, pathogens: List[str]) -> pd.DataFrame:
        if "Pathogen" not in self._df.columns or len(pathogens) == 0:
            return self._empty_df(["Pathogen", "antibiotic", "tested_count", "tested_pct"])

        df = self._df[self._df["Pathogen"].isin(pathogens)].copy()
        if df.empty:
            return self._empty_df(["Pathogen", "antibiotic", "tested_count", "tested_pct"])

        out_rows = []
        if self._use_inferred_tested:
            cols = self._abx_infer_cols
            for path, g in df.groupby("Pathogen", dropna=False):
                n = len(g)
                for col in cols:
                    tested = int(g[col].notna().sum())
                    if tested < self.min_abx_tested_display:
                        continue
                    out_rows.append({
                        "Pathogen": path,
                        "antibiotic": self._pretty_abx(col),
                        "tested_count": tested,
                        "tested_pct": round(100 * tested / max(n, 1), 2)
                    })
        else:
            cols = self._abx_cols
            if not cols:
                return self._empty_df(["Pathogen", "antibiotic", "tested_count", "tested_pct"])
            for path, g in df.groupby("Pathogen", dropna=False):
                n = len(g)
                for col in cols:
                    tested = int(g[col].fillna(0).astype(int).sum())
                    if tested < self.min_abx_tested_display:
                        continue
                    out_rows.append({
                        "Pathogen": path,
                        "antibiotic": self._pretty_abx(col),
                        "tested_count": tested,
                        "tested_pct": round(100 * tested / max(n, 1), 2)
                    })

        res = (pd.DataFrame(out_rows)
               .sort_values(["Pathogen", "tested_count", "antibiotic"],
                            ascending=[True, False, True])
               .reset_index(drop=True))
        return res

    def _time_series(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Normalize YearMonth to a sortable monthly key
        if "YearMonth" in self._df.columns:
            ym = self._coerce_yearmonth(self._df["YearMonth"])
        elif {"Year", "Month"}.issubset(self._df.columns):
            ym = self._df["Year"].astype(str).str.zfill(4) + "-" + \
                 self._df["Month"].astype(str).str.zfill(2)
        else:
            # Fall back to Year only
            if "Year" in self._df.columns:
                ym = self._df["Year"].astype(int).astype(str) + "-01"
            else:
                return self._empty_df(["YearMonth", "isolates"]), self._empty_df(
                    ["YearMonth", "Pathogen", "isolates"]
                )

        df = self._df.copy()
        df["_YM"] = ym
        overall = (df.groupby("_YM", dropna=False)
                   .size().rename("isolates").reset_index()
                   .rename(columns={"_YM": "YearMonth"})
                   .sort_values("YearMonth"))

        if "Pathogen" in df.columns:
            by_pathogen = (df.groupby(["_YM", "Pathogen"], dropna=False)
                           .size().rename("isolates").reset_index()
                           .rename(columns={"_YM": "YearMonth"})
                           .sort_values(["YearMonth", "isolates"], ascending=[True, False]))
        else:
            by_pathogen = self._empty_df(["YearMonth", "Pathogen", "isolates"])

        return overall, by_pathogen

    def _crosstab(self, row: str, col: str) -> pd.DataFrame:
        for c in (row, col):
            if c not in self._df.columns:
                return self._empty_df([row, col, "isolates"])
        ct = pd.crosstab(self._df[row], self._df[col], dropna=False)
        ct.index.name = row
        ct.columns.name = col
        ct = ct.reset_index()
        return ct

    # ------------------------------ saving ------------------------------ #

    def _save_csvs(self, results: Dict[str, pd.DataFrame], output_dir: str) -> None:
        for key, df in results.items():
            # Create subdirectories based on key prefix (e.g., "antibiotics/overall")
            parts = key.split("/")
            if len(parts) > 1:
                subdir = os.path.join(output_dir, *parts[:-1])
                os.makedirs(subdir, exist_ok=True)
            else:
                subdir = output_dir
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", parts[-1]) + ".csv"
            df.to_csv(os.path.join(subdir, safe_name), index=False)

    def _save_excel(self, results: Dict[str, pd.DataFrame], path: str) -> None:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
            used_names = set()
            for key, df in results.items():
                sheet = re.sub(r"[^A-Za-z0-9]", "_", key)[-31:] or "Sheet"
                base = sheet
                i = 1
                while sheet in used_names:
                    i += 1
                    sheet = (base[:29] + f"_{i}") if len(base) > 28 else f"{base}_{i}"
                used_names.add(sheet)
                df.to_excel(xw, index=False, sheet_name=sheet)

    # --------------------------- helpers/utils -------------------------- #

    @staticmethod
    def _prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]
        # Coerce antibiotic flags to numeric 0/1 where possible
        abx_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_Tested")]
        for c in abx_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        # Clean common categoricals
        for c in ["Pathogen", "PathogengroupL1", "GramType", "Sex", "PathogenGenus",
                  "TextMaterialgroupRkiL0", "AgeGroup", "AgeRange",
                  "CareType", "ARS_HospitalLevelManual", "ARS_WardType", "ARS_Region",
                  "MonthName", "SeasonName", "YearMonth"]:
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip()
        # Year/Month to numeric where present
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        if "Month" in df.columns:
            df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
        return df

    @staticmethod
    def _find_antibiotic_columns(columns: pd.Index) -> List[str]:
        return [c for c in columns if isinstance(c, str) and c.endswith("_Tested")]

    @staticmethod
    def _infer_dimension_columns(columns: pd.Index) -> List[str]:
        preferred = [
            "Pathogen", "PathogengroupL1", "GramType", "PathogenGenus", "Sex",
            "TextMaterialgroupRkiL0", "AgeGroup", "AgeRange",
            "CareType", "ARS_HospitalLevelManual", "ARS_WardType", "ARS_Region",
            "Year", "Month", "MonthName", "YearMonth", "SeasonName"
        ]
        return [c for c in preferred if c in columns]

    @staticmethod
    def _pretty_abx(col: str) -> str:
        # Transform "AMC - Amoxicillin/clavulanic acid_Tested" -> "AMC - Amoxicillin/clavulanic acid"
        return col[:-7] if col.endswith("_Tested") else col

    @staticmethod
    def _coerce_yearmonth(s: pd.Series) -> pd.Series:
        """
        Try to coerce YearMonth to 'YYYY-MM' string (keeps original if already OK).
        """
        s = s.astype("string").str.strip()
        # If contains format 'YYYY-MM' already, keep
        mask_ok = s.str.match(r"^\d{4}-\d{2}$", na=False)
        if mask_ok.all():
            return s
        # Try parsing common variants like 'YYYY-M', 'YYYY/MM', 'YYYY.MM'
        s2 = (s.str.extract(r"(?P<y>\d{4})\D(?P<m>\d{1,2})", expand=True)
                .assign(y=lambda d: d["y"].fillna("0000"),
                        m=lambda d: d["m"].fillna("1"))
                .assign(m=lambda d: d["m"].astype(int).clip(1, 12).astype(str).str.zfill(2))
              )
        fixed = s2["y"] + "-" + s2["m"]
        # Keep original if completely unparsable
        fixed = fixed.where(s.notna(), other=np.nan)
        return fixed

    @staticmethod
    def _empty_df(cols: List[str]) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def run_database_stats(df: pd.DataFrame, output_dir: str = "amr_summary") -> Dict[str, pd.DataFrame]:
    """
    Wrapper used in your notebook/script:
        from shadow_antibiogram.runners.DatabaseOverview import run_database_stats
        results = run_database_stats(df)
    """
    summarizer = AMRSummary(df)
    results = summarizer.run_all(output_dir=output_dir)
    return results
