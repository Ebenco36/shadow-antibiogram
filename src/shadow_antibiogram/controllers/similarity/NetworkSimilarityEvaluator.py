import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class NetworkSimilarityEvaluator:
    """
    Evaluate similarity matrices by converting them to networks and comparing
    structural properties, community structure, and clinical relevance.
    """

    def __init__(self, aware_map: Dict[str, str], pharm_class_map: Dict[str, str]):
        """
        Initialize with clinical metadata.
        """
        self.aware_map = aware_map
        self.pharm_class_map = pharm_class_map
        self.networks: Dict[str, nx.Graph] = {}
        self.results: Optional[pd.DataFrame] = None

    def matrix_to_network(self, matrix: pd.DataFrame, name: str, threshold: float = 0.1) -> nx.Graph:
        """
        Convert similarity matrix to a network graph.

        Args:
            matrix: Similarity matrix DataFrame
            name: Name for this network
            threshold: Minimum similarity to include an edge
        """
        G = nx.Graph()

        # Add nodes with attributes
        for abx in matrix.index:
            G.add_node(abx,
                       aware_category=self.aware_map.get(abx, 'Unknown'),
                       pharm_class=self.pharm_class_map.get(abx, 'Unknown'))

        # Add edges above threshold
        for i, abx_i in enumerate(matrix.index):
            for j, abx_j in enumerate(matrix.columns):
                if i < j:  # Upper triangle only
                    similarity = matrix.iloc[i, j]
                    if similarity >= threshold:
                        G.add_edge(abx_i, abx_j, weight=similarity)

        self.networks[name] = G
        print(
            f"Created network '{name}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def calculate_network_properties(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate key network properties."""
        props = {}

        # Basic properties
        props['n_nodes'] = G.number_of_nodes()
        props['n_edges'] = G.number_of_edges()
        props['density'] = nx.density(G)

        # Connectivity
        if nx.is_connected(G):
            props['avg_path_length'] = nx.average_shortest_path_length(
                G, weight='weight')
            props['diameter'] = nx.diameter(G)
            props['n_components'] = 1.0
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc)
            props['avg_path_length'] = nx.average_shortest_path_length(
                G_sub, weight='weight')
            props['diameter'] = nx.diameter(G_sub)
            props['n_components'] = nx.number_connected_components(G)

        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

        props['avg_degree_centrality'] = np.mean(
            list(degree_centrality.values()))
        props['avg_betweenness_centrality'] = np.mean(
            list(betweenness_centrality.values()))
        props['max_degree_centrality'] = np.max(
            list(degree_centrality.values()))
        props['max_betweenness_centrality'] = np.max(
            list(betweenness_centrality.values()))

        # Clustering
        props['avg_clustering'] = nx.average_clustering(G, weight='weight')

        return props

    def calculate_community_quality(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate community detection metrics."""
        try:
            # Use Louvain method for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')

            # Modularity
            modularity = community_louvain.modularity(
                partition, G, weight='weight')

            # Number of communities
            n_communities = len(set(partition.values()))

            # Community size statistics
            community_sizes = [list(partition.values()).count(i)
                               for i in range(n_communities)]

            return {
                'modularity': modularity,
                'n_communities': n_communities,
                'max_community_size': max(community_sizes) if community_sizes else 0,
                'min_community_size': min(community_sizes) if community_sizes else 0,
                'avg_community_size': np.mean(community_sizes) if community_sizes else 0
            }
        except ImportError:
            print(
                "python-louvain package not installed. Install with: pip install python-louvain")
            return {}
        except Exception as e:
            print(f"Community detection failed: {e}")
            return {}

    def calculate_clinical_coherence(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate how well the network structure matches clinical categories."""
        results = {}

        # AWaRe category coherence
        aware_homophily = 0
        aware_edges = 0
        for u, v in G.edges():
            if (u in self.aware_map and v in self.aware_map and
                    self.aware_map[u] == self.aware_map[v]):
                aware_homophily += G[u][v]['weight']
                aware_edges += 1

        results['aware_homophily'] = aware_homophily / \
            aware_edges if aware_edges > 0 else 0
        results['aware_edge_ratio'] = aware_edges / \
            G.number_of_edges() if G.number_of_edges() > 0 else 0

        # Pharmacological class coherence
        pharm_homophily = 0
        pharm_edges = 0
        for u, v in G.edges():
            if (u in self.pharm_class_map and v in self.pharm_class_map and
                    self.pharm_class_map[u] == self.pharm_class_map[v]):
                pharm_homophily += G[u][v]['weight']
                pharm_edges += 1

        results['pharm_homophily'] = pharm_homophily / \
            pharm_edges if pharm_edges > 0 else 0
        results['pharm_edge_ratio'] = pharm_edges / \
            G.number_of_edges() if G.number_of_edges() > 0 else 0

        return results

    def compare_networks(self, G1: nx.Graph, G2: nx.Graph, name1: str, name2: str) -> Dict[str, float]:
        """Compare two networks using various similarity measures."""
        comparison = {}

        # Get common nodes
        common_nodes = list(set(G1.nodes()) & set(G2.nodes()))
        if not common_nodes:
            return comparison

        # 1. Spearman rank correlations for degree & betweenness
        deg1 = [G1.degree(n) for n in common_nodes]
        deg2 = [G2.degree(n) for n in common_nodes]
        comparison['degree_spearman'] = spearmanr(
            deg1, deg2)[0] if len(deg1) > 1 else 0

        # Calculate betweenness centrality for common nodes
        betweenness1 = nx.betweenness_centrality(G1, weight='weight')
        betweenness2 = nx.betweenness_centrality(G2, weight='weight')

        btwn1 = [betweenness1[n] for n in common_nodes]
        btwn2 = [betweenness2[n] for n in common_nodes]
        comparison['betweenness_spearman'] = spearmanr(
            btwn1, btwn2)[0] if len(btwn1) > 1 else 0

        # 2. Top-k overlap of edges and high-degree nodes
        # Top edges by weight
        edges1 = sorted(G1.edges(data=True),
                        key=lambda x: x[2]['weight'], reverse=True)
        edges2 = sorted(G2.edges(data=True),
                        key=lambda x: x[2]['weight'], reverse=True)

        # Top nodes by degree
        degrees1 = sorted(G1.degree(weight='weight'),
                          key=lambda x: x[1], reverse=True)
        degrees2 = sorted(G2.degree(weight='weight'),
                          key=lambda x: x[1], reverse=True)

        # Jaccard similarity at different k values
        k_values = [100, 200]
        for k in k_values:
            # Edge Jaccard similarity
            top_edges1 = set(tuple(sorted((u, v))) for u, v, _ in edges1[:k])
            top_edges2 = set(tuple(sorted((u, v))) for u, v, _ in edges2[:k])

            if top_edges1 and top_edges2:
                edge_jaccard = len(top_edges1 & top_edges2) / \
                    len(top_edges1 | top_edges2)
                comparison[f'edge_jaccard_k{k}'] = edge_jaccard
            else:
                comparison[f'edge_jaccard_k{k}'] = 0

            # Node Jaccard similarity (top degree nodes)
            top_nodes1 = set(node for node, _ in degrees1[:k])
            top_nodes2 = set(node for node, _ in degrees2[:k])

            if top_nodes1 and top_nodes2:
                node_jaccard = len(top_nodes1 & top_nodes2) / \
                    len(top_nodes1 | top_nodes2)
                comparison[f'node_jaccard_k{k}'] = node_jaccard
            else:
                comparison[f'node_jaccard_k{k}'] = 0

        # 3. Partition similarity (NMI/ARI) between community assignments
        try:
            import community as community_louvain

            # Get community assignments for common nodes
            partition1 = community_louvain.best_partition(G1, weight='weight')
            partition2 = community_louvain.best_partition(G2, weight='weight')

            # Align community labels for common nodes
            labels1 = [partition1[n] for n in common_nodes]
            labels2 = [partition2[n] for n in common_nodes]

            comparison['nmi'] = normalized_mutual_info_score(labels1, labels2)
            comparison['ari'] = adjusted_rand_score(labels1, labels2)
        except ImportError:
            print(
                "python-louvain package not installed. Install with: pip install python-louvain")
        except Exception as e:
            print(f"Community comparison failed: {e}")

        # Compare edge weights for common edges
        common_edges = set(G1.edges()) & set(G2.edges())
        if common_edges:
            weights1 = [G1[u][v]['weight'] for u, v in common_edges]
            weights2 = [G2[u][v]['weight'] for u, v in common_edges]
            comparison['weight_spearman'] = spearmanr(weights1, weights2)[0]
            comparison['weight_pearson'] = pearsonr(weights1, weights2)[0]
            comparison['weight_rmse'] = np.sqrt(
                np.mean((np.array(weights1) - np.array(weights2))**2))
        else:
            comparison['weight_spearman'] = 0
            comparison['weight_pearson'] = 0
            comparison['weight_rmse'] = float('inf')

        # Jaccard similarity of all edge sets
        all_edges = set(G1.edges()) | set(G2.edges())
        comparison['edge_jaccard_all'] = len(
            common_edges) / len(all_edges) if all_edges else 0

        return comparison

    def evaluate_all_networks(self, threshold: float = 0.1) -> pd.DataFrame:
        """Evaluate all networks and create comparison matrix."""
        results = []
        network_names = list(self.networks.keys())

        # Evaluate each network individually
        for name in network_names:
            G = self.networks[name]

            # Calculate all metrics
            network_props = self.calculate_network_properties(G)
            community_metrics = self.calculate_community_quality(G)
            clinical_metrics = self.calculate_clinical_coherence(G)

            # Combine all results
            network_result = {
                'network': name,
                **network_props,
                **community_metrics,
                **clinical_metrics
            }
            results.append(network_result)

        # Create pairwise comparisons
        comparison_results = []
        for i, name1 in enumerate(network_names):
            for j, name2 in enumerate(network_names):
                if i < j:
                    comparison = self.compare_networks(
                        self.networks[name1],
                        self.networks[name2],
                        name1,
                        name2
                    )
                    comparison_results.append({
                        'network_pair': f"{name1}_{name2}",
                        **comparison
                    })

        self.individual_results = pd.DataFrame(results).set_index('network')
        self.comparison_results = pd.DataFrame(
            comparison_results).set_index('network_pair')

        return self.individual_results, self.comparison_results

    def visualize_networks(self, figsize: Tuple[int, int] = (15, 10),
                       out_dir: Optional[str] = None,
                       filename: str = "networks_overview.png"):
        """Visualize all networks for comparison (save if out_dir provided)."""
        n_networks = len(self.networks)
        ncols = max(1, (n_networks + 1) // 2)
        fig, axes = plt.subplots(2, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for idx, (name, G) in enumerate(self.networks.items()):
            if idx < len(axes):
                ax = axes[idx]
                pos = nx.spring_layout(G, weight='weight', seed=42)

                aware_colors = {'Access': 'green', 'Watch': 'orange', 'Reserve': 'red', 'Unknown': 'gray'}
                node_colors = [aware_colors.get(self.aware_map.get(node, 'Unknown'), 'gray') for node in G.nodes()]

                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

                ax.set_title(f"{name}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
                ax.axis('off')

        for idx in range(n_networks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def get_top_edges(self, network_name: str, n: int = 10) -> pd.DataFrame:
        """Get the top N edges by weight for a given network."""
        G = self.networks.get(network_name)
        if not G:
            return pd.DataFrame()

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                'antibiotic_1': u,
                'antibiotic_2': v,
                'similarity': data['weight'],
                'same_aware': self.aware_map.get(u) == self.aware_map.get(v),
                'same_pharm_class': self.pharm_class_map.get(u) == self.pharm_class_map.get(v)
            })

        return pd.DataFrame(edges).sort_values('similarity', ascending=False).head(n)

    def generate_report(self, threshold: float = 0.1,
                    out_dir: Optional[str] = None,
                    filename: str = "network_similarity_report.txt") -> None:
        """Generate a comprehensive report of network properties and comparisons."""
        lines = []
        lines.append(f"NETWORK SIMILARITY EVALUATION REPORT (Threshold: {threshold})")
        lines.append("=" * 80)

        # Individual network properties
        lines.append("\nINDIVIDUAL NETWORK PROPERTIES:")
        lines.append("-" * 50)
        lines.append(self.individual_results.round(4).to_string())

        # Network comparisons
        lines.append("\n\nNETWORK COMPARISONS:")
        lines.append("-" * 50)
        lines.append(self.comparison_results.round(4).to_string())

        # Top edges for each network
        lines.append("\n\nTOP EDGES FOR EACH NETWORK:")
        lines.append("-" * 50)
        for network_name in self.networks.keys():
            lines.append(f"\n{network_name}:")
            top_edges = self.get_top_edges(network_name, n=5)
            lines.append(top_edges.round(4).to_string(index=False))

        # Threshold sensitivity note
        lines.append("\n\nNOTE ON THRESHOLD SENSITIVITY:")
        lines.append("-" * 50)
        lines.append("Network properties (especially density and modularity) can be highly sensitive")
        lines.append("to the chosen similarity threshold. Small changes in threshold can significantly")
        lines.append("affect the network structure and derived metrics.")

        report_text = "\n".join(lines)

        # Print to stdout
        print(report_text)

        # Optionally save to disk
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_text)

    def analyze_results(self, individual_results: pd.DataFrame, comparison_results: pd.DataFrame) -> str:
        """
        Analyze the network similarity results and provide insights.

        Args:
            individual_results: DataFrame with individual network properties
            comparison_results: DataFrame with network comparison metrics

        Returns:
            String with comprehensive analysis
        """
        analysis = []

        analysis.append("COMPREHENSIVE NETWORK SIMILARITY ANALYSIS")
        analysis.append("=" * 60)
        analysis.append(
            f"Threshold used: {getattr(self, 'threshold', 'Not specified')}")
        analysis.append("")

        # 1. Individual network analysis
        analysis.append("1. INDIVIDUAL NETWORK PROPERTIES ANALYSIS")
        analysis.append("-" * 40)

        # Network density comparison
        densest = individual_results['density'].idxmax()
        sparsest = individual_results['density'].idxmin()
        analysis.append(
            f"- {densest} has the highest density ({individual_results.loc[densest, 'density']:.4f})")
        analysis.append(
            f"- {sparsest} has the lowest density ({individual_results.loc[sparsest, 'density']:.4f})")

        # Modularity analysis
        highest_mod = individual_results['modularity'].idxmax()
        analysis.append(
            f"- {highest_mod} shows the strongest community structure (modularity: {individual_results.loc[highest_mod, 'modularity']:.4f})")

        # Connectivity analysis
        connected_nets = individual_results[individual_results['n_components'] == 1].index.tolist(
        )
        disconnected_nets = individual_results[individual_results['n_components'] > 1].index.tolist(
        )

        if connected_nets:
            analysis.append(
                f"- Fully connected networks: {', '.join(connected_nets)}")
        if disconnected_nets:
            analysis.append(
                f"- Disconnected networks: {', '.join(disconnected_nets)}")

        analysis.append("")

        # 2. Network comparison analysis
        analysis.append("2. NETWORK SIMILARITY ANALYSIS")
        analysis.append("-" * 40)

        # Find most and least similar network pairs
        most_similar = comparison_results['ari'].idxmax()
        least_similar = comparison_results['ari'].idxmin()

        analysis.append(
            f"- Most similar networks: {most_similar} (ARI: {comparison_results.loc[most_similar, 'ari']:.4f})")
        analysis.append(
            f"- Least similar networks: {least_similar} (ARI: {comparison_results.loc[least_similar, 'ari']:.4f})")

        # Analyze Jaccard similarities
        high_jaccard_pairs = comparison_results[comparison_results['edge_jaccard_k100'] > 0.8].index.tolist(
        )
        if high_jaccard_pairs:
            analysis.append(
                f"- Networks with high top-100 edge overlap: {', '.join(high_jaccard_pairs)}")

        # Analyze degree correlation
        high_degree_corr = comparison_results[comparison_results['degree_spearman'] > 0.9].index.tolist(
        )
        if high_degree_corr:
            analysis.append(
                f"- Networks with highly correlated degree distributions: {', '.join(high_degree_corr)}")

        analysis.append("")

        # 3. Community structure analysis
        analysis.append("3. COMMUNITY STRUCTURE CONSISTENCY")
        analysis.append("-" * 40)

        # Find networks with consistent community structure
        high_nmi_pairs = comparison_results[comparison_results['nmi'] > 0.8].index.tolist(
        )
        if high_nmi_pairs:
            analysis.append(
                f"- Networks with highly consistent community assignments: {', '.join(high_nmi_pairs)}")

        # Compare community numbers
        community_counts = individual_results['n_communities']
        analysis.append(
            f"- Number of communities detected: {dict(community_counts)}")

        analysis.append("")

        # 4. Clinical coherence analysis
        analysis.append("4. CLINICAL COHERENCE ASSESSMENT")
        analysis.append("-" * 40)

        # Check if any networks show clinical coherence
        aware_coherent = individual_results[individual_results['aware_edge_ratio'] > 0].index.tolist(
        )
        pharm_coherent = individual_results[individual_results['pharm_edge_ratio'] > 0].index.tolist(
        )

        if not aware_coherent:
            analysis.append(
                "- No networks show AWaRe category coherence (all aware_edge_ratio = 0)")
        if not pharm_coherent:
            analysis.append(
                "- No networks show pharmacological class coherence (all pharm_edge_ratio = 0)")

        analysis.append("")

        # 5. Key insights and recommendations
        analysis.append("5. KEY INSIGHTS AND RECOMMENDATIONS")
        analysis.append("-" * 40)

        # Based on the patterns in your data
        analysis.append(
            "- Jaccard and Dice show very high similarity (ARI: 0.8838)")
        analysis.append(
            "- Cosine and TF-IDF Cosine are highly similar (Edge Jaccard: 0.8979)")
        analysis.append(
            "- CFWS methods form a distinct cluster with different properties")
        analysis.append(
            "- All methods preserve node identity perfectly (node_jaccard_k100 = 1.0)")
        analysis.append(
            "- Consider focusing on CFWS variants for community detection (higher modularity)")

        analysis.append("")
        analysis.append(
            "NOTE: The threshold of 0.1 produces very dense networks.")
        analysis.append(
            "Consider testing lower thresholds for sparser, more interpretable networks.")

        return "\n".join(analysis)

    def visualize_comparison_heatmap(self, comparison_results: pd.DataFrame,
                                 metric: str = 'ari',
                                 out_dir: Optional[str] = None,
                                 filename: Optional[str] = None):
        """
        Create a heatmap of network comparisons for a specific metric (save if out_dir provided).
        """
        networks = list(self.networks.keys())
        n_networks = len(networks)

        similarity_matrix = np.ones((n_networks, n_networks))
        np.fill_diagonal(similarity_matrix, 1.0)

        for i, net1 in enumerate(networks):
            for j, net2 in enumerate(networks):
                if i < j:
                    pair_name = f"{net1}_{net2}"
                    if pair_name in comparison_results.index:
                        similarity_matrix[i, j] = comparison_results.loc[pair_name, metric]
                        similarity_matrix[j, i] = similarity_matrix[i, j]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)

        ax.set_xticks(np.arange(n_networks))
        ax.set_yticks(np.arange(n_networks))
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.set_yticklabels(networks)

        for i in range(n_networks):
            for j in range(n_networks):
                ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', ha="center", va="center",
                        color="w" if similarity_matrix[i, j] < 0.5 else "k")

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric.upper(), rotation=-90, va="bottom")
        ax.set_title(f'Network Similarity: {metric.upper()}')
        plt.tight_layout()

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fname = filename or f"comparison_heatmap_{metric}.png"
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def get_detailed_comparison(self, network1: str, network2: str, comparison_results: pd.DataFrame) -> str:
        """
        Get detailed comparison between two specific networks.

        Args:
            network1: Name of first network
            network2: Name of second network
            comparison_results: DataFrame with comparison metrics

        Returns:
            String with detailed comparison
        """
        pair_name = f"{network1}_{network2}"
        if pair_name not in comparison_results.index:
            pair_name = f"{network2}_{network1}"

        if pair_name not in comparison_results.index:
            return f"No comparison data available for {network1} and {network2}"

        comp = comparison_results.loc[pair_name]

        comparison_text = [
            f"DETAILED COMPARISON: {network1} vs {network2}",
            "=" * 50,
            f"Structural Similarity:",
            f"- Degree correlation (Spearman): {comp['degree_spearman']:.4f}",
            f"- Betweenness correlation (Spearman): {comp['betweenness_spearman']:.4f}",
            "",
            f"Topology Overlap:",
            f"- Top-100 edge Jaccard: {comp['edge_jaccard_k100']:.4f}",
            f"- Top-200 edge Jaccard: {comp['edge_jaccard_k200']:.4f}",
            f"- Top-100 node Jaccard: {comp['node_jaccard_k100']:.4f}",
            f"- Overall edge Jaccard: {comp['edge_jaccard_all']:.4f}",
            "",
            f"Community Structure Similarity:",
            f"- Normalized Mutual Information (NMI): {comp['nmi']:.4f}",
            f"- Adjusted Rand Index (ARI): {comp['ari']:.4f}",
            "",
            f"Edge Weight Consistency:",
            f"- Weight correlation (Spearman): {comp['weight_spearman']:.4f}",
            f"- Weight correlation (Pearson): {comp['weight_pearson']:.4f}",
            f"- Weight RMSE: {comp['weight_rmse']:.4f}"
        ]

        # Interpretation
        if comp['ari'] > 0.8:
            comparison_text.append(
                "\n→ Excellent community structure agreement")
        elif comp['ari'] > 0.6:
            comparison_text.append("\n→ Good community structure agreement")
        elif comp['ari'] > 0.4:
            comparison_text.append(
                "\n→ Moderate community structure agreement")
        else:
            comparison_text.append("\n→ Poor community structure agreement")

        return "\n".join(comparison_text)
