# src/controllers/AMR/similarity/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd


PAIRWISE_CORE = {"ab_1", "ab_2", "a", "b", "c", "d"}


def _is_pairwise_df(df: pd.DataFrame) -> bool:
    return PAIRWISE_CORE.issubset(set(df.columns))


@dataclass(frozen=True)
class SimilarityRequest:
    """
    Optional request wrapper for future extensibility.
    Currently mostly useful for pairwise stratification.
    """
    metric: str
    strata: Optional[dict] = None  # only used for pairwise aggregated data


class SimilarityEngine:
    """
    High-level interface to compute antibiotic–antibiotic similarity matrices.

    This engine auto-detects the input dataframe layout and dispatches to:
      - OLD (WIDE isolate-level): src.controllers.CoTestAnalyzer.CoTestAnalyzer
      - NEW (PAIRWISE aggregated): src.controllers.CoTestAnalyzerAggregated.CoTestAnalyzerAggregated

    Public API remains stable:
      - compute(metric_name)
      - compute_all()

    Notes
    -----
    - WIDE mode requires antibiotic_columns (list of *_Tested columns).
    - PAIRWISE mode ignores antibiotic_columns; antibiotics are values in ab_1/ab_2.
    - PAIRWISE similarity metrics supported here: jaccard, dice, cosine, overlap, phi
      (depending on what your aggregated analyzer implements; jaccard/dice/cosine/overlap
       are definitely supported, phi if added).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        antibiotic_columns: Optional[List[str]] = None,
        *,
        force_mode: Optional[str] = None,  # "wide" | "pairwise" | None
    ):
        if df is None or df.empty:
            raise ValueError("SimilarityEngine received an empty dataframe.")

        self.df = df
        self.antibiotic_columns = antibiotic_columns or []

        detected_pairwise = _is_pairwise_df(df)

        if force_mode is not None:
            fm = force_mode.strip().lower()
            if fm not in {"wide", "pairwise"}:
                raise ValueError("force_mode must be one of: None, 'wide', 'pairwise'")
            self.is_pairwise = (fm == "pairwise")
        else:
            self.is_pairwise = detected_pairwise

        # Instantiate the right analyzer
        if self.is_pairwise:
            # NEW aggregated pairwise analyzer
            from src.controllers.CoTestAnalyzerAggregated import CoTestAnalyzer as CoTestAnalyzerAggregated

            # In pairwise mode, antibiotic_columns are not required
            self._analyzer = CoTestAnalyzerAggregated(self.df)

        else:
            # OLD isolate-level wide analyzer
            if not self.antibiotic_columns:
                # Best-effort auto-detect (backwards-friendly)
                self.antibiotic_columns = [
                    c for c in self.df.columns if str(c).endswith("_Tested")
                ]

            if not self.antibiotic_columns:
                raise ValueError(
                    "WIDE data detected but no antibiotic_columns provided and no *_Tested columns found."
                )

            from src.controllers.CoTestAnalyzer import CoTestAnalyzer

            self._analyzer = CoTestAnalyzer(self.df, self.antibiotic_columns)

    # ------------------------------------------------------------
    # Compute helpers
    # ------------------------------------------------------------

    def compute_all(self, *, strata: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute all supported similarity matrices.

        Parameters
        ----------
        strata : dict | None
            Only applies to PAIRWISE aggregated data (filters strata columns before computing).
            Ignored in WIDE mode.

        Returns
        -------
        Dict[str, pd.DataFrame]
        """
        if self.is_pairwise:
            # aggregated analyzer: method names in your class are:
            # jaccard(), dice(), cos(), overlap()
            # phi only if you add it there.
            out = {
                "jaccard": self._analyzer.compute_matrix("jaccard", strata=strata),
                "dice": self._analyzer.compute_matrix("dice", strata=strata),
                "cosine": self._analyzer.compute_matrix("cosine", strata=strata),
                "overlap": self._analyzer.compute_matrix("overlap", strata=strata),
            }

            # phi is optional depending on your aggregated analyzer implementation
            try:
                out["phi"] = self._analyzer.compute_matrix("phi", strata=strata)
            except Exception:
                pass

            return out

        # WIDE mode (existing behavior)
        return {
            "jaccard": self._analyzer.jaccard(),
            "dice": self._analyzer.dice(),
            "cosine": self._analyzer.cos(),
            "phi": self._analyzer.phi(),
        }

    def compute(
        self,
        metric_name: str,
        *,
        strata: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute a specific similarity matrix.

        Parameters
        ----------
        metric_name : str
            WIDE supports:  jaccard, dice, cosine, phi
            PAIRWISE supports: jaccard, dice, cosine, overlap (+phi if implemented)
        strata : dict | None
            Only applies to PAIRWISE aggregated data.

        Returns
        -------
        pd.DataFrame
        """
        name = metric_name.strip().lower()

        # WIDE mode (existing behavior)
        if name == "jaccard":
            return self._analyzer.jaccard()
        if name == "dice":
            return self._analyzer.dice()
        if name in {"cosine", "cos"}:
            return self._analyzer.cos()
        if name == "phi":
            return self._analyzer.phi()

        raise ValueError(f"Unsupported similarity metric: {metric_name}")
