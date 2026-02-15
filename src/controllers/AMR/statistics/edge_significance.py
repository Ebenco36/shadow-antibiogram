# src/controllers/AMR/statistics/edge_significance.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact  # make sure scipy is installed


@dataclass
class EdgeSignificancePruner:
    """
    Compute per-edge p-values (Fisher's exact test) and FDR-corrected q-values
    for antibiotic–antibiotic pairs, and provide a pruned similarity matrix.

    Assumptions:
      - df_binary has one row per isolate and one column per antibiotic.
      - Each antibiotic column is binary (0/1). If not, you can preprocess
        before passing in (e.g. convert R/others to 1/0).
    """

    df_binary: pd.DataFrame
    antibiotic_cols: List[str]
    alpha: float = 0.05
    min_total: int = 20         # minimum number of non-missing samples to test
    min_positive: int = 3       # minimum count of co-positives to bother testing
    alternative: str = "two-sided"  # "two-sided" | "greater" | "less"

    # Filled by fit()
    pval_df: Optional[pd.DataFrame] = None
    qval_df: Optional[pd.DataFrame] = None
    significant_mask: Optional[pd.DataFrame] = None

    def fit(self) -> "EdgeSignificancePruner":
        """
        Compute Fisher p-values and FDR-corrected q-values for all
        antibiotic–antibiotic pairs (upper triangle).
        """
        cols = [c for c in self.antibiotic_cols if c in self.df_binary.columns]
        if len(cols) < 2:
            raise ValueError("Need at least 2 antibiotic columns for pairwise tests.")

        # Work on a copy restricted to the relevant columns
        X = self.df_binary[cols].copy()

        # Try to coerce to numeric / binary 0-1
        X = X.apply(pd.to_numeric, errors="coerce")
        # Treat any non-zero as 1, zero or NaN as 0
        X = (X > 0).astype(float)

        n = len(cols)
        pvals = np.ones((n, n), dtype=float)
        pvals[:] = np.nan  # we'll only fill upper triangle

        pairs: List[Tuple[int, int]] = []
        p_list: List[float] = []

        for i in range(n):
            xi = X.iloc[:, i].to_numpy()
            for j in range(i + 1, n):
                xj = X.iloc[:, j].to_numpy()

                # Drop rows with NaN in either column
                mask = ~np.isnan(xi) & ~np.isnan(xj)
                xi2, xj2 = xi[mask], xj[mask]
                total = len(xi2)
                if total < self.min_total:
                    # Not enough data to test: keep NaN / non-significant
                    continue

                # 2x2 contingency table
                a11 = int(np.sum((xi2 == 1) & (xj2 == 1)))
                a10 = int(np.sum((xi2 == 1) & (xj2 == 0)))
                a01 = int(np.sum((xi2 == 0) & (xj2 == 1)))
                a00 = int(np.sum((xi2 == 0) & (xj2 == 0)))

                if a11 < self.min_positive:
                    # Very rare co-positives: treat as not significant
                    continue

                table = [[a11, a10], [a01, a00]]

                try:
                    _, p = fisher_exact(table, alternative=self.alternative)
                except Exception:
                    p = np.nan

                pvals[i, j] = p
                pairs.append((i, j))
                p_list.append(p)

        if not p_list:
            # No tests performed
            self.pval_df = pd.DataFrame(pvals, index=cols, columns=cols)
            self.qval_df = self.pval_df.copy()
            self.significant_mask = pd.DataFrame(
                False, index=cols, columns=cols, dtype=bool
            )
            return self

        p_array = np.array(p_list, dtype=float)

        # Benjamini–Hochberg FDR
        q_array = self._benjamini_hochberg(p_array)

        # Fill full symmetric matrices
        qvals = np.full_like(pvals, np.nan, dtype=float)
        sig = np.zeros_like(pvals, dtype=bool)

        for (i, j), p, q in zip(pairs, p_array, q_array):
            pvals[i, j] = p
            pvals[j, i] = p
            qvals[i, j] = q
            qvals[j, i] = q
            if q <= self.alpha:
                sig[i, j] = True
                sig[j, i] = True

        # Diagonal: no test, set p=q=0, significant=False (no edge)
        for k in range(n):
            pvals[k, k] = 0.0
            qvals[k, k] = 0.0
            sig[k, k] = False

        self.pval_df = pd.DataFrame(pvals, index=cols, columns=cols)
        self.qval_df = pd.DataFrame(qvals, index=cols, columns=cols)
        self.significant_mask = pd.DataFrame(sig, index=cols, columns=cols)

        return self

    @staticmethod
    def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
        """
        Benjamini–Hochberg FDR correction.

        p : 1D array of p-values (may contain NaNs).
        Returns an array of q-values (FDR-adjusted p-values).
        """
        m = len(p)
        # Treat NaN as 1.0 (never significant)
        p_clean = np.where(np.isnan(p), 1.0, p)
        order = np.argsort(p_clean)
        ranks = np.empty(m, dtype=int)
        ranks[order] = np.arange(1, m + 1)

        q = p_clean * m / ranks
        # Enforce monotonicity from largest to smallest
        q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
        q_corrected = np.empty_like(q_sorted)
        q_corrected[order] = np.minimum(q_sorted, 1.0)
        return q_corrected

    def prune_similarity(self, sim_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the significance mask to a similarity matrix:

        - Keep edges where q <= alpha (significant_mask == True)
        - Zero out non-significant edges (off-diagonal)
        - Always keep the diagonal as-is.

        Assumes sim_matrix is square with antibiotic names as index=columns.
        """
        if self.significant_mask is None:
            raise RuntimeError("You must call .fit() before prune_similarity().")

        sim = sim_matrix.copy()

        # Align mask to similarity matrix
        mask = self.significant_mask.reindex(
            index=sim.index, columns=sim.columns
        ).fillna(False)

        # Do not touch diagonal
        diag_idx = np.eye(len(sim), dtype=bool)
        keep = mask.to_numpy()
        keep[diag_idx] = True

        pruned_values = np.where(keep, sim.to_numpy(), 0.0)
        pruned = pd.DataFrame(
            pruned_values, index=sim.index, columns=sim.columns
        )
        return pruned

    # ------------------------------------------------------------------ #
    # NEW: list of significant pairs
    # ------------------------------------------------------------------ #
    def get_significant_pairs(
        self,
        return_qvalues: bool = False,
        upper_only: bool = True,
    ):
        """
        Return all antibiotic–antibiotic pairs that are significant after FDR.

        Parameters
        ----------
        return_qvalues : bool
            If True, return (abx_i, abx_j, q_value) triples.
            If False, return (abx_i, abx_j) pairs.

        upper_only : bool
            If True (default), only return pairs from the upper triangle
            (i < j) to avoid duplicates. If False, return all (i, j) where
            significant_mask[i, j] is True (including symmetric duplicates).

        Returns
        -------
        list
            - If return_qvalues=False:
                List[Tuple[str, str]]
            - If return_qvalues=True:
                List[Tuple[str, str, float]]
        """
        if self.significant_mask is None:
            raise RuntimeError("You must call .fit() before get_significant_pairs().")

        sig = self.significant_mask
        cols = list(sig.columns)

        pairs = []

        if upper_only:
            # use only upper triangle (i < j) to avoid duplicates
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if bool(sig.iat[i, j]):
                        ai, aj = cols[i], cols[j]
                        if return_qvalues and self.qval_df is not None:
                            q = float(self.qval_df.iat[i, j])
                            pairs.append((ai, aj, q))
                        else:
                            pairs.append((ai, aj))
        else:
            # full matrix (careful: symmetric duplicates)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i == j:
                        continue
                    if bool(sig.iat[i, j]):
                        ai, aj = cols[i], cols[j]
                        if return_qvalues and self.qval_df is not None:
                            q = float(self.qval_df.iat[i, j])
                            pairs.append((ai, aj, q))
                        else:
                            pairs.append((ai, aj))

        return pairs

    # ------------------------------------------------------------------ #
    # NEW: export full p/q/significant table
    # ------------------------------------------------------------------ #
    def to_long_dataframe(self, upper_only: bool = True) -> pd.DataFrame:
        """
        Convert pval/qval/significant matrices into a long-form DataFrame.

        Columns:
          - antibiotic_i
          - antibiotic_j
          - p_value
          - q_value
          - significant (bool)

        By default only i < j (upper triangle) to avoid duplicates.
        """
        if self.pval_df is None or self.qval_df is None or self.significant_mask is None:
            raise RuntimeError("You must call .fit() before to_long_dataframe().")

        cols = list(self.pval_df.columns)
        rows: List[Dict[str, object]] = []

        n = len(cols)
        for i in range(n):
            j_start = i + 1 if upper_only else 0
            for j in range(j_start, n):
                if i == j:
                    continue

                p = float(self.pval_df.iat[i, j])
                q = float(self.qval_df.iat[i, j])

                # skip completely untested entries (both NaN)
                if np.isnan(p) and np.isnan(q):
                    continue

                sig = bool(self.significant_mask.iat[i, j])

                rows.append(
                    {
                        "antibiotic_i": cols[i],
                        "antibiotic_j": cols[j],
                        "p_value": p,
                        "q_value": q,
                        "significant": sig,
                    }
                )

        return pd.DataFrame(rows)

    def save_edge_statistics(
        self,
        filepath,
        upper_only: bool = True,
        extra_columns: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Save p/q/significant table to CSV.

        Parameters
        ----------
        filepath : str or Path
            Where to write the CSV.

        upper_only : bool
            If True, only store i<j pairs (default).

        extra_columns : dict, optional
            Extra metadata columns to add to every row, e.g.
            {
              "genus": "Escherichia",
              "material": "Urine",
              "metric": "jaccard",
              "tau": 0.3,
              "gamma": 1.0,
            }
        """
        df = self.to_long_dataframe(upper_only=upper_only)
        if extra_columns:
            for k, v in extra_columns.items():
                df[k] = v

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
