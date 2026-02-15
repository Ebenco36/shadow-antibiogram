# src/controllers/AMR/evaluation/internal_evaluator.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


class InternalEvaluator:
    """
    Internal clustering quality based on silhouette score.

    Uses the antibiotic–antibiotic similarity matrix to derive a
    distance matrix D = 1 - S, and computes silhouette_score
    with 'precomputed' distances.

    This is label-free and complements ICS.
    """

    def __init__(self, similarity_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        similarity_matrix : pd.DataFrame
            Square similarity matrix with identical index and columns.
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
        self._distance_matrix: np.ndarray = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute a distance matrix from the similarity matrix.

        For now we use: distance = 1 - similarity
        """
        S = self.similarity_matrix.values.astype(float)
        D = 1.0 - S
        # Ensure non-negative (in case of numeric noise)
        D[D < 0] = 0.0
        return D

    def compute_silhouette(self, partition: Dict[str, int]) -> float:
        """
        Compute silhouette score for a given partition.

        Parameters
        ----------
        partition : dict
            Mapping {node -> cluster_id} for all nodes.

        Returns
        -------
        score : float
            Silhouette score in [-1, 1]. Returns NaN if not computable
            (e.g., all in one cluster or each point its own cluster).
        """
        # cluster labels aligned with self.nodes
        labels = [partition[n] for n in self.nodes]

        # silhouette_score with precomputed distance matrix
        # silhouette requires at least 2 clusters and at least 1 point in each
        unique_clusters = set(labels)
        if len(unique_clusters) < 2:
            return float("nan")

        try:
            score = silhouette_score(
                self._distance_matrix, labels,
            )
            # print("Score: ", score)
        except ValueError as e:
            # e.g. clusters with single points etc.
            # print("See error ooo: ", str(e))
            return float("nan")

        return float(score)
