from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr


MatrixLike = Union[pd.DataFrame, str, Path]


@dataclass
class CompareConfig:
    tau: float = 0.3
    n_clusters: int = 5
    align: str = "intersection"         # "intersection" or "union"
    diag_value: float = 1.0
    top_k_edges: int = 20
    top_k_nodes: int = 20
    random_state: int = 42
    weighted_graph: bool = False        # if True, graph uses weights (jaccard) above tau
    out_dir: str = "./network_compare"
    dpi: int = 350
    save_pdf: bool = True


class JaccardNetworkComparator:
    """
    Publication-ready comparator for two symmetric similarity matrices (e.g., Jaccard).

    Semantics are explicitly:
      - full dataset similarity matrix
      - aggregated dataset similarity matrix

    Typical usage:
        comp = JaccardNetworkComparator(CompareConfig(tau=0.3, n_clusters=5))
        report = comp.fit(sim_full, sim_aggregated)
        comp.save_all()
    """

    def __init__(self, config: CompareConfig):
        self.cfg = config
        self.out = Path(self.cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # populated after fit()
        self.full_mat: Optional[pd.DataFrame] = None
        self.aggregated_mat: Optional[pd.DataFrame] = None
        self.labels: Optional[List[str]] = None

        self.G_full: Optional[nx.Graph] = None
        self.G_aggregated: Optional[nx.Graph] = None
        self.pos: Optional[Dict[str, np.ndarray]] = None

        self.comm_full: Optional[np.ndarray] = None
        self.comm_aggregated: Optional[np.ndarray] = None

        self.results: Optional[Dict] = None
        self.tables: Dict[str, pd.DataFrame] = {}

    # -------------------------
    # IO + validation
    # -------------------------
    @staticmethod
    def _read_matrix(maybe_path_or_df: MatrixLike) -> pd.DataFrame:
        if isinstance(maybe_path_or_df, (str, Path)):
            df = pd.read_csv(maybe_path_or_df, index_col=0)
        elif isinstance(maybe_path_or_df, pd.DataFrame):
            df = maybe_path_or_df.copy()
        else:
            raise TypeError("Expected a CSV path (str/Path) or a pandas DataFrame")

        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def _load_align_clean(
        self,
        full_dataset: MatrixLike,
        aggregated_dataset: MatrixLike,
        labels: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        F = self._read_matrix(full_dataset)
        A = self._read_matrix(aggregated_dataset)

        # infer labels
        if labels is None:
            if self.cfg.align == "intersection":
                common = sorted(set(F.index) & set(A.index) & set(F.columns) & set(A.columns))
                labels = common
            elif self.cfg.align == "union":
                union = sorted(set(F.index) | set(A.index) | set(F.columns) | set(A.columns))
                labels = union
            else:
                raise ValueError("config.align must be 'intersection' or 'union'")

        # align
        F = F.reindex(index=labels, columns=labels).fillna(0.0)
        A = A.reindex(index=labels, columns=labels).fillna(0.0)

        # symmetrize
        F = (F + F.T) / 2.0
        A = (A + A.T) / 2.0

        # clamp to [0,1] if it looks like similarity
        F = F.clip(lower=0.0, upper=1.0)
        A = A.clip(lower=0.0, upper=1.0)

        # diag
        np.fill_diagonal(F.values, self.cfg.diag_value)
        np.fill_diagonal(A.values, self.cfg.diag_value)

        # sanity checks
        if F.shape != A.shape:
            raise ValueError(f"Shape mismatch: full {F.shape} vs aggregated {A.shape}")
        if F.shape[0] != len(labels):
            raise ValueError("Label length mismatch with matrix size")

        return F, A, labels

    # -------------------------
    # Graph construction
    # -------------------------
    def _build_graph(self, mat: pd.DataFrame, labels: List[str]) -> nx.Graph:
        W = mat.values.copy()
        np.fill_diagonal(W, 0.0)

        if self.cfg.weighted_graph:
            # keep weights above tau
            W = np.where(W > self.cfg.tau, W, 0.0)
            G = nx.from_numpy_array(W)
            for u, v, d in G.edges(data=True):
                if "weight" not in d:
                    d["weight"] = float(W[u, v])
        else:
            # unweighted thresholded adjacency
            B = (W > self.cfg.tau).astype(int)
            G = nx.from_numpy_array(B)

        G = nx.relabel_nodes(G, {i: labels[i] for i in range(len(labels))})
        return G

    @staticmethod
    def _edge_set_from_mat(mat: pd.DataFrame, labels: List[str], tau: float) -> set[tuple[str, str]]:
        M = mat.values
        upper = np.triu(M > tau, k=1)
        r, c = np.where(upper)
        return {(labels[int(i)], labels[int(j)]) for i, j in zip(r, c)}

    # -------------------------
    # Communities + modularity
    # -------------------------
    def _spectral_partition(self, mat: pd.DataFrame) -> np.ndarray:
        sc = SpectralClustering(
            n_clusters=self.cfg.n_clusters,
            affinity="precomputed",
            random_state=self.cfg.random_state,
        )
        return sc.fit_predict(mat.values)

    @staticmethod
    def _modularity_of_partition(G: nx.Graph, comm_labels: np.ndarray) -> float:
        if G.number_of_edges() == 0:
            return 0.0
        nodes = list(G.nodes())
        clusters: Dict[int, set] = {}
        for node, cid in zip(nodes, comm_labels):
            clusters.setdefault(int(cid), set()).add(node)
        from networkx.algorithms.community.quality import modularity
        return float(modularity(G, list(clusters.values()), weight="weight"))

    # -------------------------
    # Metrics
    # -------------------------
    @staticmethod
    def _upper_triangle_values(mat: pd.DataFrame) -> np.ndarray:
        M = mat.values
        ui, uj = np.triu_indices(M.shape[0], k=1)
        return M[ui, uj]

    def _global_graph_metrics(self, G: nx.Graph) -> Dict[str, float]:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n <= 1:
            return {
                "n_nodes": float(n),
                "n_edges": float(m),
                "density": 0.0,
                "avg_degree": 0.0,
                "avg_clustering": 0.0,
                "assortativity": np.nan,
                "n_components": float(n),
                "largest_component_frac": 1.0 if n == 1 else 0.0,
            }

        density = nx.density(G)
        degrees = np.array([d for _, d in G.degree()])
        avg_degree = float(degrees.mean()) if len(degrees) else 0.0
        avg_clust = nx.average_clustering(G, weight="weight" if self.cfg.weighted_graph else None)

        try:
            assort = nx.degree_assortativity_coefficient(G)
        except Exception:
            assort = np.nan

        comps = list(nx.connected_components(G)) if not G.is_directed() else list(nx.weakly_connected_components(G))
        n_components = len(comps)
        largest_frac = max(len(c) for c in comps) / n if comps else 0.0

        return {
            "n_nodes": float(n),
            "n_edges": float(m),
            "density": float(density),
            "avg_degree": float(avg_degree),
            "avg_clustering": float(avg_clust),
            "assortativity": float(assort) if assort == assort else np.nan,
            "n_components": float(n_components),
            "largest_component_frac": float(largest_frac),
        }

    def _node_change_table(self, G_full: nx.Graph, G_agg: nx.Graph) -> pd.DataFrame:
        nodes = sorted(set(G_full.nodes()) | set(G_agg.nodes()))
        deg_full = dict(G_full.degree())
        deg_agg = dict(G_agg.degree())

        df = pd.DataFrame({"node": nodes})
        df["deg_full"] = df["node"].map(deg_full).fillna(0).astype(float)
        df["deg_aggregated"] = df["node"].map(deg_agg).fillna(0).astype(float)
        df["deg_delta"] = df["deg_aggregated"] - df["deg_full"]
        df["deg_abs_delta"] = df["deg_delta"].abs()

        if self.cfg.weighted_graph:
            s_full = dict(G_full.degree(weight="weight"))
            s_agg = dict(G_agg.degree(weight="weight"))
            df["strength_full"] = df["node"].map(s_full).fillna(0).astype(float)
            df["strength_aggregated"] = df["node"].map(s_agg).fillna(0).astype(float)
            df["strength_delta"] = df["strength_aggregated"] - df["strength_full"]
            df["strength_abs_delta"] = df["strength_delta"].abs()

        df = df.sort_values("deg_abs_delta", ascending=False).reset_index(drop=True)
        return df

    def _top_edge_changes(self, F: pd.DataFrame, A: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
        # delta is Aggregated - Full
        diff = (A.values - F.values)
        absdiff = np.abs(diff)
        n = len(labels)
        ui, uj = np.triu_indices(n, k=1)
        flat_abs = absdiff[ui, uj]
        top_idx = np.argsort(-flat_abs)[: self.cfg.top_k_edges]

        rows = []
        for k in top_idx:
            i, j = int(ui[k]), int(uj[k])
            rows.append(
                {
                    "node1": labels[i],
                    "node2": labels[j],
                    "full": float(F.values[i, j]),
                    "aggregated": float(A.values[i, j]),
                    "delta": float(A.values[i, j] - F.values[i, j]),
                    "abs_delta": float(absdiff[i, j]),
                }
            )
        return pd.DataFrame(rows)

    # -------------------------
    # Fit + report
    # -------------------------
    def fit(self, full_dataset: MatrixLike, aggregated_dataset: MatrixLike, labels: Optional[List[str]] = None) -> Dict:
        F, A, labels = self._load_align_clean(full_dataset, aggregated_dataset, labels=labels)
        self.full_mat, self.aggregated_mat, self.labels = F, A, labels

        # Graphs
        self.G_full = self._build_graph(F, labels)
        self.G_aggregated = self._build_graph(A, labels)

        # Shared layout
        Gunion = nx.compose(self.G_full, self.G_aggregated)
        self.pos = nx.spring_layout(Gunion, seed=self.cfg.random_state)

        # Edge sets + edge jaccard (thresholded)
        e_full = self._edge_set_from_mat(F, labels, self.cfg.tau)
        e_agg = self._edge_set_from_mat(A, labels, self.cfg.tau)
        edge_jacc = (len(e_full & e_agg) / len(e_full | e_agg)) if (e_full or e_agg) else 0.0

        # Communities (spectral on similarity matrices)
        self.comm_full = self._spectral_partition(F)
        self.comm_aggregated = self._spectral_partition(A)
        ari = float(adjusted_rand_score(self.comm_full, self.comm_aggregated))
        nmi = float(normalized_mutual_info_score(self.comm_full, self.comm_aggregated))

        # Modularity of those partitions on graphs
        q_full = self._modularity_of_partition(self.G_full, self.comm_full)
        q_agg = self._modularity_of_partition(self.G_aggregated, self.comm_aggregated)

        # Edge-weight agreement
        uF = self._upper_triangle_values(F)
        uA = self._upper_triangle_values(A)
        pear_r, pear_p = pearsonr(uF, uA)
        spear_r, spear_p = spearmanr(uF, uA)

        # Global metrics
        gf = self._global_graph_metrics(self.G_full)
        ga = self._global_graph_metrics(self.G_aggregated)

        # Tables
        top_edges = self._top_edge_changes(F, A, labels)
        node_changes = self._node_change_table(self.G_full, self.G_aggregated)

        summary = pd.DataFrame(
            [
                ["Edge Jaccard (thresholded @tau)", edge_jacc],
                ["ARI (communities)", ari],
                ["NMI (communities)", nmi],
                ["Modularity full (partition)", q_full],
                ["Modularity aggregated (partition)", q_agg],
                ["Δ Modularity", abs(q_full - q_agg)],
                ["Pearson r (upper triangle)", float(pear_r)],
                ["Pearson p", float(pear_p)],
                ["Spearman r (upper triangle)", float(spear_r)],
                ["Spearman p", float(spear_p)],
                ["Nodes", len(labels)],
                ["Edges full", self.G_full.number_of_edges()],
                ["Edges aggregated", self.G_aggregated.number_of_edges()],
                ["Common edges", len(e_full & e_agg)],
            ],
            columns=["metric", "value"],
        )

        globals_table = pd.DataFrame(
            {
                "metric": list(gf.keys()),
                "full": list(gf.values()),
                "aggregated": [ga[k] for k in gf.keys()],
                "delta": [ga[k] - gf[k] for k in gf.keys()],
            }
        )

        self.tables = {
            "summary": summary,
            "global_metrics": globals_table,
            "top_edge_changes": top_edges,
            "node_changes": node_changes,
            "community_contingency": self.community_contingency_table(),
        }

        self.results = {
            "config": asdict(self.cfg),
            "edge_jaccard": float(edge_jacc),
            "ari": ari,
            "nmi": nmi,
            "q_full": float(q_full),
            "q_aggregated": float(q_agg),
            "delta_modularity": float(abs(q_full - q_agg)),
            "pearson_r": float(pear_r),
            "pearson_p": float(pear_p),
            "spearman_r": float(spear_r),
            "spearman_p": float(spear_p),
            "n_nodes": int(len(labels)),
            "n_edges_full": int(self.G_full.number_of_edges()),
            "n_edges_aggregated": int(self.G_aggregated.number_of_edges()),
            "common_edges": int(len(e_full & e_agg)),
        }
        return self.results

    # -------------------------
    # Community comparison helper
    # -------------------------
    def community_contingency_table(self) -> pd.DataFrame:
        if self.comm_full is None or self.comm_aggregated is None or self.labels is None:
            return pd.DataFrame()

        df = pd.DataFrame(
            {"node": self.labels, "c_full": self.comm_full.astype(int), "c_aggregated": self.comm_aggregated.astype(int)}
        )
        return pd.crosstab(df["c_full"], df["c_aggregated"])

    # -------------------------
    # Publication figures
    # -------------------------
    def _savefig(self, fig: plt.Figure, name: str):
        png = self.out / f"{name}.png"
        fig.savefig(png, dpi=self.cfg.dpi, bbox_inches="tight")
        if self.cfg.save_pdf:
            pdf = self.out / f"{name}.pdf"
            fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    def fig_matrices(self):
        """Full, Aggregated, and (Aggregated - Full) heatmaps."""
        F, A = self.full_mat, self.aggregated_mat
        assert F is not None and A is not None

        diff = A.values - F.values

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(F.values, ax=axes[0], square=True, cbar=True)
        axes[0].set_title("Full dataset similarity")

        sns.heatmap(A.values, ax=axes[1], square=True, cbar=True)
        axes[1].set_title("Aggregated dataset similarity")

        sns.heatmap(diff, ax=axes[2], square=True, center=0, cmap="RdBu_r", cbar=True)
        axes[2].set_title("Aggregated - Full")

        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")

        plt.tight_layout()
        self._savefig(fig, "fig_matrices")

    def fig_edge_agreement(self):
        """Binary edge agreement matrix at tau."""
        F, A = self.full_mat, self.aggregated_mat
        assert F is not None and A is not None

        Bf = (F.values > self.cfg.tau).astype(int)
        Ba = (A.values > self.cfg.tau).astype(int)
        agree = (Bf == Ba).astype(int)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(agree, ax=ax, square=True, cbar=True)
        ax.set_title(f"Edge agreement (tau={self.cfg.tau})")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        self._savefig(fig, "fig_edge_agreement")

    def fig_edge_weight_scatter(self):
        """Scatter of edge weights Full vs Aggregated (upper triangle) + correlation."""
        F, A = self.full_mat, self.aggregated_mat
        assert F is not None and A is not None

        x = self._upper_triangle_values(F)
        y = self._upper_triangle_values(A)

        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(x, y, s=10, alpha=0.35)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        ax.set_xlabel("Full edge weight")
        ax.set_ylabel("Aggregated edge weight")
        ax.set_title(
            f"Edge-weight agreement\nPearson r={pr:.3f} (p={pp:.1e}), Spearman r={sr:.3f} (p={sp:.1e})"
        )
        plt.tight_layout()
        self._savefig(fig, "fig_edge_weight_scatter")

    def fig_edge_weight_ecdf(self):
        """ECDF of edge weights (distribution shift)."""
        F, A = self.full_mat, self.aggregated_mat
        assert F is not None and A is not None

        x = np.sort(self._upper_triangle_values(F))
        y = np.sort(self._upper_triangle_values(A))
        px = np.linspace(0, 1, len(x), endpoint=True)
        py = np.linspace(0, 1, len(y), endpoint=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, px, label="Full")
        ax.plot(y, py, label="Aggregated")
        ax.set_xlabel("Edge weight (upper triangle)")
        ax.set_ylabel("ECDF")
        ax.set_title("Edge-weight distributions (ECDF)")
        ax.legend()
        plt.tight_layout()
        self._savefig(fig, "fig_edge_weight_ecdf")

    def fig_networks_side_by_side(self):
        """Side-by-side networks with shared layout."""
        Gf, Ga, pos = self.G_full, self.G_aggregated, self.pos
        assert Gf is not None and Ga is not None and pos is not None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        nx.draw_networkx_nodes(Gf, pos, node_size=45, alpha=0.85, ax=axes[0])
        nx.draw_networkx_edges(Gf, pos, alpha=0.25, ax=axes[0])
        axes[0].set_title("Full dataset network")
        axes[0].axis("off")

        nx.draw_networkx_nodes(Ga, pos, node_size=45, alpha=0.85, ax=axes[1])
        nx.draw_networkx_edges(Ga, pos, alpha=0.25, ax=axes[1])
        axes[1].set_title("Aggregated dataset network")
        axes[1].axis("off")

        plt.tight_layout()
        self._savefig(fig, "fig_networks_side_by_side")

    def fig_node_change_bar(self):
        """Bar plot: top nodes with biggest degree/strength change (aggregated - full)."""
        df = self.tables.get("node_changes")
        if df is None or df.empty:
            return

        top = df.head(self.cfg.top_k_nodes).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top))))

        ax.barh(top["node"], top["deg_delta"])
        ax.set_xlabel("Degree change (aggregated - full)")
        ax.set_title(f"Top {self.cfg.top_k_nodes} node degree changes")
        plt.tight_layout()
        self._savefig(fig, "fig_node_degree_change")

        if self.cfg.weighted_graph and "strength_delta" in df.columns:
            topw = df.sort_values("strength_abs_delta", ascending=False).head(self.cfg.top_k_nodes).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(topw))))
            ax.barh(topw["node"], topw["strength_delta"])
            ax.set_xlabel("Strength change (aggregated - full)")
            ax.set_title(f"Top {self.cfg.top_k_nodes} node strength changes")
            plt.tight_layout()
            self._savefig(fig, "fig_node_strength_change")

    def fig_community_contingency(self):
        """Heatmap of how communities map full -> aggregated."""
        tab = self.tables.get("community_contingency")
        if tab is None or tab.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(tab, annot=True, fmt="d", ax=ax, cbar=False)
        ax.set_title("Community mapping (Full vs Aggregated)")
        ax.set_xlabel("Aggregated community")
        ax.set_ylabel("Full community")
        plt.tight_layout()
        self._savefig(fig, "fig_community_contingency")

    def make_all_figures(self):
        self.fig_matrices()
        self.fig_edge_agreement()
        self.fig_edge_weight_scatter()
        self.fig_edge_weight_ecdf()
        self.fig_networks_side_by_side()
        self.fig_node_change_bar()
        self.fig_community_contingency()

    # -------------------------
    # Save outputs (tables + JSON)
    # -------------------------
    def save_all(self):
        if self.results is None:
            raise RuntimeError("Call fit() first")

        for name, df in self.tables.items():
            if df is None or df.empty:
                continue
            df.to_csv(self.out / f"{name}.csv", index=True)

        with open(self.out / "report.json", "w") as f:
            json.dump(self.results, f, indent=2)

        self.make_all_figures()

        print("\n=== Summary ===")
        print(self.tables["summary"].to_string(index=False))
        print(f"\nSaved outputs to: {self.out.resolve()}")

    def show_key_figures(self):
        """If you're in a notebook, regenerate and display key figures without saving."""
        self.fig_matrices()
        self.fig_edge_weight_scatter()
        self.fig_networks_side_by_side()


# -------------------------
# Example usage
# -------------------------
# cfg = CompareConfig(tau=0.3, n_clusters=5, weighted_graph=False, out_dir="./network_compare_pub")
# comp = JaccardNetworkComparator(cfg)
# report = comp.fit(sim_full, sim_aggregated)
# comp.save_all()
# report
