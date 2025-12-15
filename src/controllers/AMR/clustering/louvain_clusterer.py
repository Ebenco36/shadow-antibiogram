# src/controllers/AMR/clustering/louvain_clusterer.py

from __future__ import annotations

from typing import Dict, List

import networkx as nx

# python-louvain library
import community as community_louvain


class LouvainClusterer:
    """
    Wrapper around the Louvain community detection algorithm
    (python-louvain package).

    Uses:
        community_louvain.best_partition(
            G,
            weight="weight",
            resolution=gamma,
            random_state=seed,
        )
    """

    def __init__(self, gamma: float):
        """
        Parameters
        ----------
        gamma : float
            Resolution parameter for Louvain. Smaller values favor fewer,
            larger communities; larger values favor more, smaller communities.
        """
        self.gamma = float(gamma)

    def run_single(self, G: nx.Graph, seed: int | None = None) -> Dict[str, int]:
        """
        Run Louvain once and return a partition.

        Parameters
        ----------
        G : nx.Graph
            Weighted graph with 'weight' edge attributes.
        seed : int or None
            Random seed passed to python-louvain (random_state).

        Returns
        -------
        partition : dict
            Mapping {node -> community_id}.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("G must be a networkx.Graph instance.")

        partition = community_louvain.best_partition(
            G,
            weight="weight",
            resolution=self.gamma,
            random_state=seed,
        )
        # partition is already a dict[node -> community_id]
        return partition

    def run_multiple(self, G: nx.Graph, seeds: List[int]) -> List[Dict[str, int]]:
        """
        Run Louvain multiple times over the same graph with different seeds.

        Parameters
        ----------
        G : nx.Graph
            Graph to cluster.
        seeds : list of int
            Seeds to use for random initialization.

        Returns
        -------
        partitions : list of dict
            List of {node -> community_id} mappings, one per seed.
        """
        partitions: List[Dict[str, int]] = []
        for s in seeds:
            partitions.append(self.run_single(G, seed=s))
        return partitions
