# src/controllers/AMR/clustering/graph_builder.py

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


class GraphBuilder:
    """
    Builds an undirected weighted graph from an antibiotic–antibiotic
    similarity matrix, using a threshold tau.

        - Nodes: antibiotics (matrix index)
        - Edge (i, j) exists if S_ij >= tau
        - Edge weight: S_ij (float)
    """

    def __init__(self, tau: float):
        """
        Parameters
        ----------
        tau : float
            Similarity threshold for including an edge.
        """
        self.tau = float(tau)

    def build_graph(self, similarity_matrix: pd.DataFrame) -> nx.Graph:
        """
        Construct a NetworkX Graph from a similarity matrix.

        Parameters
        ----------
        similarity_matrix : pd.DataFrame
            Square DataFrame with identical index and columns
            representing pairwise similarities.

        Returns
        -------
        G : nx.Graph
            Undirected weighted graph.
        """
        if not isinstance(similarity_matrix, pd.DataFrame):
            raise TypeError("similarity_matrix must be a pandas DataFrame.")

        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square (n x n).")

        if not similarity_matrix.index.equals(similarity_matrix.columns):
            raise ValueError(
                "similarity_matrix index and columns must be identical and in the same order."
            )

        nodes = list(similarity_matrix.index)
        S = similarity_matrix.values.astype(float)
        n = len(nodes)

        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Only iterate over upper triangle to avoid duplicates
        iu, ju = np.triu_indices(n, k=1)
        for i, j in zip(iu, ju):
            w = S[i, j]
            if w >= self.tau:
                G.add_edge(nodes[i], nodes[j], weight=float(w))

        return G
