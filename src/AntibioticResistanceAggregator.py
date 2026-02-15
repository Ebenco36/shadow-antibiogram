import pandas as pd
import numpy as np
import itertools
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.runners.DataProcessing import save_parquet_flat


class AntibioticResistanceAggregator:
    """
    Strict translation of the R script 'create_aggregated_data_publication.R'.
    Uses custom DataLoader for input and save_parquet_flat for output.
    Includes detailed logging to diagnose cohort emptiness.
    """

    DEFAULT_VARIABLES = [
        "Pathogen",
        "PathogenGenus",
        "GramType",
        "Sex",
        "CareType",
        "TextMaterialgroupRkiL0",
        "ARS_WardType",
        "BroadAgeGroup",
        "HighLevelAgeRange",
        "Hospital_Priority",
        "Care_Complexity",
        "FacilityFunction",
        "Year",
    ]

    DEFAULT_ANTIBIOTIC_PATTERN = r"[A-Z]{3}\s-"  # e.g., "AMC - Amoxicillin/clavulanic acid_Tested"
    DEFAULT_COMBINATIONS = [
        {"Pathogen": "Escherichia coli", "CareType": "In-Patient",            "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "CareType": "Out-Patient",           "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "ARS_WardType": "Normal Ward",       "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "ARS_WardType": "Intensive Care Unit","TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "CareType": "In-Patient",            "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "ARS_WardType": "Normal Ward",       "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "ARS_WardType": "Intensive Care Unit","TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "CareType": "Out-Patient",           "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2019",                      "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2020",                      "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2021",                      "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2022",                      "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2023",                      "TextMaterialgroupRkiL0": "Urine"},
        {"Pathogen": "Escherichia coli", "Year": "2019",                      "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "Year": "2020",                      "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "Year": "2021",                      "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "Year": "2022",                      "TextMaterialgroupRkiL0": "Blood Culture"},
        {"Pathogen": "Escherichia coli", "Year": "2023",                      "TextMaterialgroupRkiL0": "Blood Culture"},
    ]

    def __init__(
        self,
        input_path: str,
        output_path: str = "data_for_publication_updated.feather",
        variables: Optional[List[str]] = None,
        antibiotic_pattern: Optional[str] = None,
        combinations: Optional[List[Dict[str, Any]]] = None,
        log_file: Optional[str] = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.variables = variables or self.DEFAULT_VARIABLES
        self.antibiotic_pattern = antibiotic_pattern or self.DEFAULT_ANTIBIOTIC_PATTERN
        self.combinations = combinations or self.DEFAULT_COMBINATIONS

        # Set up logging
        self.logger = logging.getLogger("AntibioticResistanceAggregator")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # File handler if requested
            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            else:
                # If no file, still use console with a simple format
                formatter = logging.Formatter('%(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

        self.raw_df: Optional[pd.DataFrame] = None
        self.antibiotic_cols: List[str] = []
        self.cohort_df: Optional[pd.DataFrame] = None
        self.result_df: Optional[pd.DataFrame] = None

    def _cast_filter_value(self, col: str, val: Any) -> Any:
        """Convert filter value to the column's dtype for proper comparison."""
        if col not in self.raw_df.columns:
            return val
        dtype = self.raw_df[col].dtype
        # If the column is numeric (int/float) and the value is a string, try converting
        if pd.api.types.is_numeric_dtype(dtype) and isinstance(val, str):
            try:
                # Convert to int if possible, else float
                if val.isdigit():
                    return int(val)
                else:
                    return float(val)
            except ValueError:
                pass  # leave as is, comparison will be False
        # If the column is categorical or object, leave as string
        return val

    def read_data(self) -> None:
        """Load data using your custom DataLoader and keep relevant columns."""
        from src.controllers.DataLoader import DataLoader
        from src.mappers.top_pathogens import ALL_PATHOGENS

        self.logger.info("Loading data with DataLoader...")
        loader = DataLoader(self.input_path, pathogen_groups_regex=ALL_PATHOGENS)
        self.raw_df = loader.get_combined(return_which="tested")
        self.logger.info(f"Loaded data shape: {self.raw_df.shape}")

        # --- Clean Year column: remove trailing '.0' and convert to plain string ---
        if "Year" in self.raw_df.columns:
            # Convert to string, strip '.0' at end, and keep as string (or could convert to int)
            # This ensures that filter strings like "2020" match values like "2020.0"
            orig_vals = self.raw_df["Year"].dropna().unique()
            self.raw_df["Year"] = self.raw_df["Year"].astype(str).str.replace(r'\.0$', '', regex=True)
            new_vals = self.raw_df["Year"].dropna().unique()
            self.logger.info(f"Cleaned Year column: original unique: {orig_vals[:5]} → now: {new_vals[:5]}")

        # Identify antibiotic columns via regex
        self.antibiotic_cols = list(self.raw_df.filter(regex=self.antibiotic_pattern).columns)
        if not self.antibiotic_cols:
            self.logger.error(f"No columns matched the antibiotic pattern '{self.antibiotic_pattern}'.")
            self.logger.error(f"Available columns: {list(self.raw_df.columns)}")
            raise ValueError(
                f"No columns matched the antibiotic pattern '{self.antibiotic_pattern}'. "
                f"Available columns: {list(self.raw_df.columns)}"
            )
        self.logger.info(f"Found {len(self.antibiotic_cols)} antibiotic columns: {self.antibiotic_cols[:5]}...")

        # Keep only existing variable columns + antibiotic columns (like any_of in R)
        existing_vars = [col for col in self.variables if col in self.raw_df.columns]
        cols_to_keep = existing_vars + self.antibiotic_cols
        self.raw_df = self.raw_df[cols_to_keep]
        self.logger.info(f"Kept {len(cols_to_keep)} columns: {existing_vars} + antibiotics")

        # ---- Detailed diagnostic logging ----
        self.logger.info("Unique values in key columns:")
        for col in self.variables:
            if col in self.raw_df.columns:
                uniq = self.raw_df[col].dropna().unique()
                self.logger.info(f"  {col} (dtype={self.raw_df[col].dtype}): {uniq[:10]}")
            else:
                self.logger.warning(f"  Column '{col}' not found in data!")

    def build_cohorts(self) -> None:
        """
        For each combination:
          - Filter rows that match all conditions (semi_join equivalent).
          - Set all columns that are not in the filter and not antibiotics to "-".
        Concatenate all cohort subsets.
        """
        if self.raw_df is None:
            raise RuntimeError("Call read_data() first.")

        cohort_dfs = []
        self.logger.info("=" * 60)
        self.logger.info("BUILDING COHORTS")
        self.logger.info("=" * 60)

        for i, comb in enumerate(self.combinations):
            self.logger.info(f"Processing cohort {i}: {comb}")
            # Build filter mask with type handling
            mask = pd.Series([True] * len(self.raw_df))
            for col, val in comb.items():
                if col not in self.raw_df.columns:
                    self.logger.warning(f"  Column '{col}' not in data – cohort will be empty.")
                    mask = pd.Series([False] * len(self.raw_df))
                    break
                # Cast value to match column dtype
                cast_val = self._cast_filter_value(col, val)
                self.logger.debug(f"  Filter: {col} == {cast_val} (original {val})")
                mask &= (self.raw_df[col] == cast_val)

            filtered = self.raw_df[mask].copy()
            self.logger.info(f"  → {len(filtered)} rows matched")

            if filtered.empty:
                # Extra diagnostics: show counts for each filter individually
                self.logger.debug("  Breakdown of individual filters:")
                for col, val in comb.items():
                    if col in self.raw_df.columns:
                        cast_val = self._cast_filter_value(col, val)
                        count = (self.raw_df[col] == cast_val).sum()
                        uniq_vals = self.raw_df[col].dropna().unique()[:5]
                        self.logger.debug(f"    {col}: {count} rows with value == {cast_val} (unique in col: {uniq_vals})")
                continue

            # Columns used in the filter
            filter_cols = list(comb.keys())

            # Columns that are NOT filter columns AND NOT antibiotic columns → set to "-"
            other_cols = [c for c in filtered.columns
                          if c not in filter_cols and c not in self.antibiotic_cols]
            filtered[other_cols] = "-"
            self.logger.debug(f"  Set {len(other_cols)} non‑filter columns to '-'")

            cohort_dfs.append(filtered)

        if not cohort_dfs:
            self.logger.error("No cohorts produced any data. Check filter values against actual data.")
            raise ValueError("No cohorts produced any data. Check filter values against actual data.")

        self.cohort_df = pd.concat(cohort_dfs, ignore_index=True)
        self.logger.info(f"Total rows in combined cohort dataframe: {len(self.cohort_df)}")
        self.logger.info("=" * 60)

    def compute_pairwise(self) -> None:
        """
        For each cohort and each antibiotic pair, compute the 2x2 contingency table,
        exactly as R's table() on factors with levels 0,1 (rows with NA are excluded).
        """
        if self.cohort_df is None or self.cohort_df.empty:
            raise RuntimeError("Cohort data is empty. Call build_cohorts() first.")

        out_rows = []
        total_pairs = 0
        self.logger.info("=" * 60)
        self.logger.info("COMPUTING PAIRWISE CONTINGENCY TABLES")
        self.logger.info("=" * 60)

        for comb in self.combinations:
            # Re‑filter the combined dataframe to this specific cohort
            mask = pd.Series([True] * len(self.cohort_df))
            for col, val in comb.items():
                if col not in self.cohort_df.columns:
                    mask = pd.Series([False] * len(self.cohort_df))
                    break
                cast_val = self._cast_filter_value(col, val)
                mask &= (self.cohort_df[col] == cast_val)

            cohort = self.cohort_df[mask].copy()
            if cohort.empty:
                self.logger.debug(f"Skipping empty cohort: {comb}")
                continue

            self.logger.info(f"Processing cohort: {comb} – {len(cohort)} rows")

            # Constant part: all non‑antibiotic columns (should be identical for all rows in this cohort)
            constant_part = cohort.drop(columns=self.antibiotic_cols).drop_duplicates()
            if len(constant_part) != 1:
                self.logger.warning(f"  Constant part has {len(constant_part)} unique rows; using first.")
                constant_part = constant_part.iloc[[0]].reset_index(drop=True)

            # For each pair of antibiotics
            pair_count = 0
            for ab1, ab2 in itertools.combinations(self.antibiotic_cols, 2):
                s1 = pd.to_numeric(cohort[ab1], errors="coerce")
                s2 = pd.to_numeric(cohort[ab2], errors="coerce")

                # Keep only rows where both values are exactly 0 or 1 (exclude NA like R's factor)
                valid = s1.isin([0, 1]) & s2.isin([0, 1])
                s1_valid = s1[valid].astype(int)
                s2_valid = s2[valid].astype(int)

                a = ((s1_valid == 1) & (s2_valid == 1)).sum()
                b = ((s1_valid == 0) & (s2_valid == 1)).sum()
                c = ((s1_valid == 1) & (s2_valid == 0)).sum()
                d = ((s1_valid == 0) & (s2_valid == 0)).sum()

                if a + b + c + d > 0:  # only log non‑zero pairs if desired
                    pair_count += 1

                row = constant_part.copy()
                row["ab_1"] = ab1
                row["ab_2"] = ab2
                row["a"] = a
                row["b"] = b
                row["c"] = c
                row["d"] = d
                out_rows.append(row)
                total_pairs += 1

            self.logger.info(f"  Generated {pair_count} non‑zero pairs out of {len(list(itertools.combinations(self.antibiotic_cols, 2)))} total")

        if not out_rows:
            self.logger.error("No pairwise data generated.")
            raise ValueError("No pairwise data generated.")

        self.result_df = pd.concat(out_rows, ignore_index=True)

        # Add the two fixed columns from the R script
        self.result_df["PathogenGenus"] = "Escherichia"
        self.result_df["GramType"] = "Neg"

        self.logger.info(f"Total antibiotic pairs processed: {total_pairs}")
        self.logger.info(f"Result shape: {self.result_df.shape}")
        self.logger.info("=" * 60)

    def write_output(self) -> None:
        if self.result_df is None:
            raise RuntimeError("No result to write.")
        self.logger.info(f"Writing output to {self.output_path}")
        # Use your custom function to save as partitioned Parquet
        save_parquet_flat(df=self.result_df, out_dir=self.output_path, rows_per_file=250_000)
        self.logger.info("Output saved successfully.")

    def run(self) -> None:
        self.read_data()
        self.build_cohorts()
        self.compute_pairwise()
        self.write_output()


if __name__ == "__main__":
    # Optional: set up a log file
    log_file = "aggregation.log"
    processor = AntibioticResistanceAggregator(
        input_path="./datasets/WHO_Aware_data__",
        output_path="./datasets/WHO_Aware_data",   # save_parquet_flat expects a directory
        log_file=log_file
    )
    processor.run()