# src/controllers/AMR/data/preprocessor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class DataPreprocessor:
    """
    Handles filtering of the WHO_Aware dataframe and selection of antibiotic columns.

    - Filters rows by PathogenGenus and TextMaterialgroupRkiL0 if available.
    - Determines which columns to treat as antibiotic features.
    """
    genera: List[str]
    materials: List[str]
    antibiotic_columns: List[str]

    def __post_init__(self):
        # Normalize to lists (just in case someone passes tuples)
        self.genera = list(self.genera) if self.genera is not None else []
        self.materials = list(self.materials) if self.materials is not None else []
        self.antibiotic_columns = list(self.antibiotic_columns or [])
        self._antibiotic_columns_resolved: List[str] | None = None

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply genus and material filters (if corresponding columns exist),
        then resolve the antibiotic columns on the filtered dataframe.
        """
        mask = pd.Series(True, index=df.index)

        if self.genera and "PathogenGenus" in df.columns:
            mask &= df["PathogenGenus"].isin(self.genera)

        if self.materials and "TextMaterialgroupRkiL0" in df.columns:
            mask &= df["TextMaterialgroupRkiL0"].isin(self.materials)

        df_filtered = df.loc[mask].copy()

        # Resolve which columns to treat as antibiotic features
        self._resolve_antibiotic_columns(df_filtered)

        return df_filtered

    def _resolve_antibiotic_columns(self, df: pd.DataFrame) -> None:
        """
        Decide which columns are antibiotics:
          - If config provided antibiotic_columns (non-empty), intersect with df.
          - Else, infer as any column ending with '_Tested' or '_Outcome'.
        """
        if self.antibiotic_columns:
            cols = [c for c in self.antibiotic_columns if c in df.columns]
        else:
            cols = [
                c
                for c in df.columns
                if c.endswith("_Tested") or c.endswith("_Outcome")
            ]

        self._antibiotic_columns_resolved = sorted(cols)

    def get_antibiotic_columns(self) -> List[str]:
        """
        Return the resolved list of antibiotic feature columns.

        NOTE: filter_dataframe(...) must be called at least once before this.
        """
        if self._antibiotic_columns_resolved is None:
            raise RuntimeError(
                "Antibiotic columns have not been resolved yet. "
                "Call filter_dataframe(...) before get_antibiotic_columns()."
            )
        return self._antibiotic_columns_resolved
