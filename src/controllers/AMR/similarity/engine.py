# src/controllers/AMR/similarity/engine.py

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.controllers.CoTestAnalyzer import CoTestAnalyzer


class SimilarityEngine:
    """
    High-level interface to compute antibiotic–antibiotic similarity matrices
    using different metrics (jaccard, dice, cosine, phi) on top of CoTestAnalyzer.
    """

    def __init__(self, df: pd.DataFrame, antibiotic_columns: List[str]):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed WHO_Aware data frame.
        antibiotic_columns : list of str
            Column names representing antibiotic test results.
        """
        self.df = df
        self.antibiotic_columns = antibiotic_columns
        self._analyzer = CoTestAnalyzer(df, antibiotic_columns)

    def compute_all(self) -> Dict[str, pd.DataFrame]:
        """
        Compute all supported similarity matrices and return as:
            {metric_name: DataFrame}
        """
        return {
            "jaccard": self._analyzer.jaccard(),
            "dice": self._analyzer.dice(),
            "cosine": self._analyzer.cos(),
            "phi": self._analyzer.phi(),
        }

    def compute(self, metric_name: str) -> pd.DataFrame:
        """
        Compute a specific similarity matrix.

        Parameters
        ----------
        metric_name : str
            One of ["jaccard", "dice", "cosine", "phi"].

        Returns
        -------
        similarity_matrix : pd.DataFrame
            Square similarity matrix for the given metric.
        """
        name = metric_name.lower()
        if name == "jaccard":
            return self._analyzer.jaccard()
        if name == "dice":
            return self._analyzer.dice()
        if name == "cosine":
            return self._analyzer.cos()
        if name == "phi":
            return self._analyzer.phi()

        raise ValueError(f"Unsupported similarity metric: {metric_name}")
