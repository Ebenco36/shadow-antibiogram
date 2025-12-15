# src/controllers/AMR/evaluation/stability_analyzer.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import adjusted_rand_score
import itertools


class StabilityAnalyzer:
    """
    Computes clustering stability across multiple runs (seeds)
    using mean pairwise Adjusted Rand Index (ARI).

    Given a list of partitions for the same parameter setting
    (metric, tau, gamma) and node order, we compute:

        stability = mean_{i<j} ARI(partition_i, partition_j)

    Values near 1.0 indicate strong stability; values near 0 suggest
    high sensitivity to random initialization.
    """

    def __init__(self):
        pass

    def compute_stability(
        self,
        partitions: List[Dict[str, int]],
        nodes: List[str],
    ) -> float:
        """
        Parameters
        ----------
        partitions : list of dict
            Each dict is {node -> cluster_id} from one seed.
        nodes : list
            Ordered list of node names, common to all partitions.

        Returns
        -------
        stability : float
            Mean pairwise ARI across all pairs of partitions.
            Returns NaN if fewer than 2 partitions are provided.
        """
        if len(partitions) < 2:
            return float("nan")

        # convert partitions to label arrays aligned with nodes
        label_arrays = []
        for p in partitions:
            labels = [p[n] for n in nodes]
            label_arrays.append(np.array(labels))

        aris: List[float] = []
        for i, j in itertools.combinations(range(len(label_arrays)), 2):
            ari_ij = adjusted_rand_score(label_arrays[i], label_arrays[j])
            aris.append(float(ari_ij))

        if not aris:
            return float("nan")

        return float(np.mean(aris))
