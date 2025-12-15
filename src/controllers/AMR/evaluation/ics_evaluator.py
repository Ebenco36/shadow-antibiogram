# src/controllers/AMR/evaluation/ics_evaluator.py

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class ICSEvaluator:
    """
    ICSEvaluator: Intra-Cluster Similarity Enrichment (ICS)

    This class computes an internal cluster quality measure based purely on the
    antibiotic–antibiotic similarity matrix.

    Given:
      - a similarity matrix S (n x n, antibiotics x antibiotics)
      - a partition (mapping antibiotic -> cluster_id)

    ICS measures how much the within-cluster similarity is enriched compared
    to the global similarity background.

    Formally:

        Let P_all      = {(i, j): i < j} be all unordered antibiotic pairs.
        Let P_within   = {(i, j): i < j, C(i) == C(j)} pairs in same cluster.

        mu_global  = mean_{(i,j) in P_all}    S_ij
        mu_within  = mean_{(i,j) in P_within} S_ij

        ICS = mu_within / mu_global

    Interpretation:
      - ICS > 1 : clusters are more similar internally than the global average.
      - ICS ≈ 1: clustering does not improve over global similarity structure.
      - ICS < 1 : clusters are, on average, less similar than global background.

    The evaluator returns a dictionary with:
      - "ICS"                : ICS score (mu_within / mu_global)
      - "ICS_mu_within"      : within-cluster mean similarity
      - "ICS_mu_global"      : global mean similarity
      - "ICS_n_pairs_within" : number of within-cluster pairs used
      - "ICS_n_pairs_all"    : number of all pairs used
    """

    def __init__(self, similarity_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        similarity_matrix : pd.DataFrame
            Square, symmetric similarity matrix with antibiotics as both
            index and columns (same order).
        """
        if not isinstance(similarity_matrix, pd.DataFrame):
            raise TypeError("similarity_matrix must be a pandas DataFrame.")

        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square (n x n).")

        if not similarity_matrix.index.equals(similarity_matrix.columns):
            raise ValueError(
                "similarity_matrix index and columns must be identical and in the same order."
            )

        self.similarity_matrix = similarity_matrix
        self.nodes: List[str] = list(similarity_matrix.index)
        # store as numpy array for faster operations
        self._S: np.ndarray = similarity_matrix.values.astype(float)

    def _all_pairs_stats(self) -> Tuple[float, int]:
        """
        Compute global mean similarity over all i < j.

        Returns
        -------
        mu_global : float
            Mean similarity over all unordered pairs.
        n_pairs_all : int
            Number of pairs used in the global mean.
        """
        n = len(self.nodes)
        if n < 2:
            return np.nan, 0

        # upper triangle without diagonal
        iu, ju = np.triu_indices(n, k=1)
        vals = self._S[iu, ju]

        if vals.size == 0:
            return np.nan, 0

        mu_global = float(vals.mean())
        return mu_global, int(vals.size)

    def _within_cluster_stats(self, partition: Dict[str, int]) -> Tuple[float, int]:
        """
        Compute mean similarity over i < j such that C(i) == C(j).

        Parameters
        ----------
        partition : dict
            Mapping {node -> cluster_id}.

        Returns
        -------
        mu_within : float
            Mean similarity over within-cluster pairs.
        n_pairs_within : int
            Number of within-cluster pairs used.
        """
        n = len(self.nodes)
        if n < 2:
            return np.nan, 0

        # labels aligned with self.nodes
        try:
            labels = np.array([partition[node] for node in self.nodes])
        except KeyError as e:
            raise KeyError(
                f"Partition is missing node {e.args[0]!r} that exists in the similarity matrix."
            )

        iu, ju = np.triu_indices(n, k=1)
        same_cluster = labels[iu] == labels[ju]

        if not np.any(same_cluster):
            return np.nan, 0

        vals_within = self._S[iu, ju][same_cluster]
        if vals_within.size == 0:
            return np.nan, 0

        mu_within = float(vals_within.mean())
        return mu_within, int(vals_within.size)

    def evaluate(self, partition: Dict[str, int]) -> Dict[str, float]:
        """
        Compute ICS and related statistics for a given partition.

        Parameters
        ----------
        partition : dict
            Mapping {antibiotic_name -> cluster_id} for ALL nodes present in
            the similarity_matrix.

        Returns
        -------
        scores : dict
            {
              "ICS": float,                # mu_within / mu_global
              "ICS_mu_within": float,      # within-cluster mean similarity
              "ICS_mu_global": float,      # global mean similarity
              "ICS_n_pairs_within": int,   # number of within-cluster pairs
              "ICS_n_pairs_all": int,      # number of all pairs
            }
        """
        mu_global, n_pairs_all = self._all_pairs_stats()
        mu_within, n_pairs_within = self._within_cluster_stats(partition)

        if np.isnan(mu_global) or mu_global == 0.0:
            ics = np.nan
        else:
            ics = float(mu_within / mu_global) if not np.isnan(mu_within) else np.nan

        return {
            "ICS": ics,
            "ICS_mu_within": mu_within,
            "ICS_mu_global": mu_global,
            "ICS_n_pairs_within": n_pairs_within,
            "ICS_n_pairs_all": n_pairs_all,
        }
