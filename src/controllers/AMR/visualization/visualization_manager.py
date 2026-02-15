# src/controllers/AMR/visualization/visualization_manager.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

from src.controllers.AMR.config.experiment_config import ExperimentConfig
from src.controllers.AMR.experiments.results import ResultCollection
from src.controllers.DataLoader import DataLoader
from src.controllers.AMR.similarity.engine import SimilarityEngine
from src.controllers.AMR.clustering.graph_builder import GraphBuilder
from src.controllers.AMR.clustering.louvain_clusterer import LouvainClusterer
from src.controllers.AMR.evaluation.label_manager import LabelManager
from src.utils.network import (
    visualize_antibiotic_graph_from_partition,
    visualize_antibiotic_network,
)
from src.utils.helpers import get_label
from src.controllers.AMR.statistics.edge_significance import EdgeSignificancePruner

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"\*scattermapbox\* is deprecated! Use \*scattermap\* instead\..*"
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"setDaemon\(\) is deprecated, set the daemon attribute instead"
)


class VisualizationManager:
    """
    Handles creation of publication-ready plots from the grid search results.

    Uses:
      - results.single_runs (seed-level detail)
      - results.aggregated_results (mean/std across seeds per
        (genus, material, metric, tau, gamma))
    """

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def __init__(self, config: ExperimentConfig, results: ResultCollection):
        self.config = config
        self.results = results
        self._set_publication_style()
        self._init_label_support()

        # Optional user-specified colours per similarity metric, e.g.
        # config.visualization.metric_colors = {"cosine": "#1f77b4", "dice": "#ff7f0e"}
        self.metric_color_map: Optional[Dict[str, str]] = getattr(
            getattr(config, "visualization", None),
            "metric_colors",
            None,
        )

        # FDR / edge pruning options (optional, via config.visualization)
        viz_cfg = getattr(self.config, "visualization", None)
        self.use_fdr_edge_pruning: bool = bool(
            getattr(viz_cfg, "use_fdr_edge_pruning", False)
        )

        self.fdr_alpha: float = float(getattr(viz_cfg, "fdr_alpha", 0.05))
        self.fdr_min_total: int = int(getattr(viz_cfg, "fdr_min_total", 20))
        self.fdr_min_positive: int = int(
            getattr(viz_cfg, "fdr_min_positive", 3))
        self.fdr_alternative: str = str(
            getattr(viz_cfg, "fdr_alternative", "two-sided"))
    # ------------------------------------------------------------------ #
    # Style
    # ------------------------------------------------------------------ #

    def _set_publication_style(self):
        """Set matplotlib parameters (used indirectly by network PNGs)."""
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
            }
        )

    # ------------------------------------------------------------------ #
    # Label support for network visualization
    # ------------------------------------------------------------------ #

    def _init_label_support(self):
        """
        Load antibiotic_class_map from the fine-grained antibiotic
        class JSON so we can generate short / enriched labels
        for antibiotics in network plots.

        Uses:
          - config.evaluation.ground_truth_paths["fine"] if available
          - otherwise falls back to ./datasets/antibiotic_class_grouping.json
        """
        self.abx_class_map = None

        try:
            fine_path = None

            # Prefer the config path if provided
            if getattr(self.config, "evaluation", None) is not None:
                gt_paths = getattr(self.config.evaluation,
                                   "ground_truth_paths", None)
                if isinstance(gt_paths, dict):
                    fine_path = gt_paths.get("fine", None)

            if fine_path is None:
                fine_path = Path("datasets/antibiotic_class_grouping.json")

            fine_path = Path(fine_path)

            if fine_path.is_file():
                with fine_path.open("r", encoding="utf-8") as f:
                    self.abx_class_map = json.load(f)
            else:
                print(
                    f"[VisualizationManager] Warning: fine class JSON not found at {fine_path}. "
                    "Network labels will use raw antibiotic names."
                )
        except Exception as e:
            print(
                f"[VisualizationManager] Could not load antibiotic_class_map: {e}")
            self.abx_class_map = None

    # ------------------------------------------------------------------ #
    # Colour helpers for Plotly
    # ------------------------------------------------------------------ #

    def _get_metric_base_colors(self, metrics) -> Dict[str, str]:
        """
        Return a dict metric -> base hex colour.

        Order of preference:
          - self.metric_color_map if provided
          - otherwise a qualitative Plotly palette.
        """
        metrics = list(sorted(set(metrics)))
        base: Dict[str, str] = dict(self.metric_color_map or {})

        palette = (
            px.colors.qualitative.Set2
            + px.colors.qualitative.Set1
            + px.colors.qualitative.Pastel1
        )

        for i, m in enumerate(metrics):
            if m not in base:
                base[m] = palette[i % len(palette)]
        return base

    @staticmethod
    def _complementary_color(hex_color: str) -> str:
        """
        Return simple RGB complement of a hex colour.
        If the input isn't a #RRGGBB string, return it unchanged.
        """
        if (
            not isinstance(hex_color, str)
            or not hex_color.startswith("#")
            or len(hex_color) != 7
        ):
            return hex_color

        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return "#{:02X}{:02X}{:02X}".format(255 - r, 255 - g, 255 - b)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def create_comprehensive_dashboard(self, output_dir: Path):
        """
        Create a set of key figures that summarize the parameter sensitivity and
        metric comparisons, and generate best-config CSV and networks.

        Produces (all Plotly except networks):
          - heatmaps/parameter_heatmaps_<genus>_<material>.html/.png/.pdf
          - metric_comparison_overall.html/.png/.pdf
          - stability_analysis.html/.png/.pdf
          - tradeoff_analysis.html/.png/.pdf
          - plotly_ics_stability/ics_stability_<genus>_<material>.html/.png/.pdf
          - plotly_external_scores/external_scores_<genus>_<material>.html/.png/.pdf
          - best_configs_per_dataset.csv
          - cluster_composition_best_configs.csv
          - networks/net_<genus>_<material>_<metric>.html/png/gexf
          - networks_original/net_<genus>_<material>_<metric>_original.html/png/gexf
          - networks_fdr/net_<genus>_<material>_<metric>_FDR.html/png/gexf (if FDR enabled)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_best_configs_csv(output_dir)
        self.export_best_cluster_compositions(
            output_dir / "cluster_composition_best_configs.csv"
        )

        # Plotly scalar visualizations
        # self.plot_parameter_heatmaps_per_dataset(output_dir / "heatmaps")
        self.plot_metric_comparison_overall(
            output_dir / "metric_comparison_overall")
        self.plot_stability_analysis(output_dir / "stability_analysis")
        self.plot_optimal_parameter_tradeoffs(output_dir / "tradeoff_analysis")
        self.plot_ics_and_stability_vs_tau_plotly(
            output_dir / "plotly_ics_stability"
        )
        self.plot_external_scores_vs_tau_plotly(
            output_dir / "plotly_external_scores"
        )

        # Network visualizations BEFORE FDR (always)
        self.plot_best_networks(output_dir / "networks")
        self.plot_best_networks_original(output_dir / "networks_original")

        # NEW: UMAP + t-SNE embeddings for the same best configs

        # AFTER FDR: only similarity-based networks (no re-run of original)
        if self.use_fdr_edge_pruning:
            self.plot_best_networks_fdr(output_dir / "networks_fdr")
            self.export_fdr_edge_statistics(
                output_dir / "fdr_edge_statistics_best_configs.csv"
            )

            # --- NEW: retention computation + plot ---
            # --- NEW: retention computation across ALL metrics ---
            retention_dir = output_dir / "retention"
            retention_csv = retention_dir / "retention_summary.csv"

            df_ret = self.compute_retention_summary(
                output_csv=retention_csv,
                metrics_filter=None,   # None = all metrics
            )

            # Multi-metric retention figure
            self.plot_retention_by_metric(
                retention_df=df_ret,
                output_path=retention_dir / "retention_by_metric",
            )

    # ------------------------------------------------------------------ #
    # Helper: choose main score column
    # ------------------------------------------------------------------ #

    def _choose_main_score_col(self, flat_df: pd.DataFrame) -> Optional[str]:
        """
        Main score for ranking parameter configurations.
        Priority:
          1) ics_score_mean
          2) hierarchical_score_mean
          3) external_NMI_broad_mean
          4) first numeric column
        """
        if "ics_score_mean" in flat_df.columns:
            return "ics_score_mean"
        if "hierarchical_score_mean" in flat_df.columns:
            return "hierarchical_score_mean"
        if "external_NMI_broad_mean" in flat_df.columns:
            return "external_NMI_broad_mean"
        num_cols = flat_df.select_dtypes(include=[float, int]).columns
        return num_cols[0] if len(num_cols) > 0 else None

    # ------------------------------------------------------------------ #
    # CSV: best configs per (genus, material)
    # ------------------------------------------------------------------ #

    def save_best_configs_csv(self, output_dir: Path):
        """
        Save a CSV summarizing the best configuration for each
        (genus, material) combination, based on the main score.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        # genus, material, metric, tau, gamma, ...
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_per_dataset = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material"], as_index=False)
            .first()
        )

        best_per_dataset.to_csv(
            output_dir / "best_configs_per_dataset.csv", index=False)

    # ------------------------------------------------------------------ #
    # 1) τ–γ Heatmaps per (genus, material) – Plotly
    # ------------------------------------------------------------------ #

    def plot_parameter_heatmaps_per_dataset(self, output_dir: Path):
        """
        For each (genus, material) subset, create τ–γ heatmaps for each similarity
        metric and selected evaluation scores (Plotly).

        aggregated_results must have index:
        (genus, material, metric, tau, gamma)
        and columns like:
        ('ics_score', 'mean'), ('external_NMI_broad', 'mean'), ...
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # (column_base, pretty title)
        eval_cols: List[Tuple[str, str]] = [
            ("ics_score", "ICS (mean)"),
            ("external_NMI_broad", "NMI broad (mean)"),
            ("external_NMI_fine", "NMI fine (mean)"),
            ("external_NMI_who", "NMI WHO (mean)"),
            ("hierarchical_score", "Hierarchical NMI (mean)"),
            ("silhouette", "Silhouette (mean)"),
        ]

        for (genus, material), df_subset in agg_df.groupby(level=["genus", "material"]):
            df_subset = df_subset.droplevel(["genus", "material"])
            if df_subset.empty:
                continue

            metrics = df_subset.index.get_level_values("metric").unique()
            available_cols = set(df_subset.columns)

            filtered_eval_cols: List[Tuple[str, str]] = []
            for col_base, title in eval_cols:
                if (col_base, "mean") in available_cols:
                    if df_subset[col_base]["mean"].notna().any():
                        filtered_eval_cols.append((col_base, title))

            if not filtered_eval_cols:
                continue

            n_rows = len(metrics)
            n_cols = len(filtered_eval_cols)

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                shared_xaxes=False,
                shared_yaxes=False,
                horizontal_spacing=0.06,
                vertical_spacing=0.08,
                subplot_titles=[title for _, title in filtered_eval_cols],
            )

            # We’ll collect all τ and γ values so tick labels are consistent
            all_tau_vals = set()
            all_gamma_vals = set()

            for i, metric in enumerate(metrics, start=1):
                for j, (col_base, _) in enumerate(filtered_eval_cols, start=1):
                    if (col_base, "mean") not in df_subset.columns:
                        continue
                    try:
                        df_metric = df_subset.loc[metric]
                        series = df_metric[col_base]["mean"]
                    except KeyError:
                        continue

                    df_plot = series.reset_index()
                    df_plot.columns = ["tau", "gamma", "value"]
                    if df_plot.empty:
                        continue

                    pivot = df_plot.pivot(
                        index="gamma", columns="tau", values="value")

                    # Track unique tau/gamma for consistent ticks
                    all_tau_vals.update(pivot.columns.tolist())
                    all_gamma_vals.update(pivot.index.tolist())

                    fig.add_trace(
                        go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns,
                            y=pivot.index,
                            coloraxis="coloraxis",
                            zsmooth="best",
                            showscale=False,  # single shared colorbar defined in layout
                        ),
                        row=i,
                        col=j,
                    )

            # Sort τ/γ so ticks are ordered nicely
            tau_vals = sorted(all_tau_vals)
            # so 1.0 appears at top
            gamma_vals = sorted(all_gamma_vals, reverse=True)

            # Axis formatting:
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    # x-ticks only on bottom row
                    fig.update_xaxes(
                        tickmode="array",
                        tickvals=tau_vals,
                        ticktext=[f"{t:.2f}" for t in tau_vals],
                        tickfont=dict(size=10),
                        showticklabels=(i == n_rows),
                        row=i,
                        col=j,
                    )
                    # y-ticks only on first column
                    fig.update_yaxes(
                        tickmode="array",
                        tickvals=gamma_vals,
                        ticktext=[f"{g:.2f}" for g in gamma_vals],
                        tickfont=dict(size=10),
                        showticklabels=(j == 1),
                        row=i,
                        col=j,
                    )
                    # Put metric names as row labels on the left
                    if j == 1:
                        fig.update_yaxes(
                            title_text=str(metrics[i - 1]),
                            title_font=dict(size=14),
                            row=i,
                            col=j,
                        )

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")

            fig.update_layout(
                title=f"{genus} – {material} : τ–γ heatmaps",
                template="plotly_white",
                title_x=0.5,
                width=1600,
                font=dict(size=14),
                margin=dict(l=90, r=130, t=90, b=70),
                # one shared colorbar on the right
                coloraxis=dict(
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="score", side="right"),
                        thickness=20,
                        len=0.9,
                    ),
                ),
            )

            # Save HTML
            out_html = output_dir / \
                f"parameter_heatmaps_{safe_genus}_{safe_mat}.html"
            pio.write_html(fig, file=str(out_html),
                           include_plotlyjs="cdn", full_html=True)

            # Save PNG (high resolution)
            out_png = out_html.with_suffix(".png")
            fig.write_image(str(out_png), format="png", scale=4)

            # Save PDF (vector)
            out_pdf = out_html.with_suffix(".pdf")
            fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 2) Overall metric comparison & global best params – Plotly
    # ------------------------------------------------------------------ #

    def plot_metric_comparison_overall(self, output_path: Path):
        """
        Compare similarity metrics at their *best* settings across all datasets.
        Saves an interactive Plotly bar chart (HTML + PNG + PDF).
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        # genus, material, metric, tau, gamma, ...
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_rows = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby("metric", as_index=False)
            .first()
        )

        metrics = best_rows["metric"].tolist()
        scores = best_rows[score_col].tolist()

        global_best = flat_df.sort_values(score_col, ascending=False).iloc[0]
        base_colors = self._get_metric_base_colors(metrics)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=scores,
                marker_color=[base_colors[m] for m in metrics],
            )
        )

        fig.update_layout(
            title=(
                "Best configuration per similarity metric<br>"
                f"Global best: {global_best['metric']} "
                f"(τ={global_best['tau']:.2f}, γ={global_best['gamma']:.2f}, "
                f"genus={global_best['genus']}, material={global_best['material']})"
            ),
            title_x=0.5,
            width=1600,
            template="plotly_white",
            font=dict(size=16),
            xaxis=dict(
                title="Similarity metric",
                tickfont=dict(size=14),
                title_font=dict(size=18),
            ),
            yaxis=dict(
                title=score_col.replace("_", " "),
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            margin=dict(l=80, r=40, t=110, b=70),
            showlegend=False,
        )

        out_html = output_path.with_suffix(".html")
        pio.write_html(fig, file=str(out_html),
                       include_plotlyjs="cdn", full_html=True)

        out_png = output_path.with_suffix(".png")
        fig.write_image(str(out_png), format="png", scale=4)

        out_pdf = output_path.with_suffix(".pdf")
        fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 3) Stability analysis plot – Plotly
    # ------------------------------------------------------------------ #

    def plot_stability_analysis(self, output_path: Path):
        """
        Show how stability interacts with the main score in a simpler way.
        Scatter of stability_mean vs main score, coloured by metric.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        if "stability_mean" not in flat_df.columns:
            return

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        metrics = best_df["metric"].unique()
        base_colors = self._get_metric_base_colors(metrics)

        fig = go.Figure()
        for metric in metrics:
            df_m = best_df[best_df["metric"] == metric]
            fig.add_trace(
                go.Scatter(
                    x=df_m["stability_mean"],
                    y=df_m[score_col],
                    mode="markers",
                    name=metric,
                    marker=dict(
                        size=10,
                        color=base_colors[metric],
                        line=dict(width=1, color="black"),
                    ),
                )
            )

        fig.update_layout(
            title="Stability vs label agreement (best per dataset & metric)",
            title_x=0.5,
            width=1600,
            template="plotly_white",
            font=dict(size=16),
            xaxis=dict(
                title="Stability (mean ARI across seeds)",
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=score_col.replace("_", " "),
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
            ),
            margin=dict(l=80, r=40, t=80, b=80),
        )

        out_html = output_path.with_suffix(".html")
        pio.write_html(fig, file=str(out_html),
                       include_plotlyjs="cdn", full_html=True)

        out_png = output_path.with_suffix(".png")
        fig.write_image(str(out_png), format="png", scale=4)

        out_pdf = output_path.with_suffix(".pdf")
        fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 4) Trade-off plot for top configurations – Plotly
    # ------------------------------------------------------------------ #

    def plot_optimal_parameter_tradeoffs(self, output_path: Path, top_k: int = 20):
        """
        Visualize trade-offs between:
          - main score (y)
          - ICS (x)
          - stability (bubble size)
          - metric (colour)
        for the top_k configurations by main score.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        if "ics_score_mean" not in flat_df.columns or "stability_mean" not in flat_df.columns:
            return

        flat_df = flat_df.sort_values(score_col, ascending=False).head(top_k)

        metrics = flat_df["metric"].unique()
        base_colors = self._get_metric_base_colors(metrics)

        fig = go.Figure()
        for metric in metrics:
            df_m = flat_df[flat_df["metric"] == metric]
            sizes = 20 + 120 * df_m["stability_mean"].fillna(0.0)

            fig.add_trace(
                go.Scatter(
                    x=df_m["ics_score_mean"],
                    y=df_m[score_col],
                    mode="markers",
                    name=metric,
                    marker=dict(
                        size=sizes,
                        sizemode="diameter",
                        sizeref=2.0 * max(sizes) / (40.0 ** 2),
                        color=base_colors[metric],
                        opacity=0.8,
                        line=dict(width=1, color="black"),
                    ),
                    text=[
                        f"genus={g}, material={m}, τ={t:.2f}, γ={gma:.2f}"
                        for g, m, t, gma in zip(
                            df_m["genus"], df_m["material"],
                            df_m["tau"], df_m["gamma"]
                        )
                    ],
                    hovertemplate=(
                        "ICS: %{x:.3f}<br>"
                        f"{score_col}: %{{y:.3f}}<br>"
                        "Stability (scaled): %{marker.size:.2f}<br>"
                        "%{text}<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"Trade-offs among top {top_k} configurations (all datasets)",
            title_x=0.5,
            width=1600,
            template="plotly_white",
            font=dict(size=16),
            xaxis=dict(
                title="ICS (mean)",
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=score_col.replace("_", " "),
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
            ),
            margin=dict(l=80, r=40, t=80, b=90),
        )

        out_html = output_path.with_suffix(".html")
        pio.write_html(fig, file=str(out_html),
                       include_plotlyjs="cdn", full_html=True)

        out_png = output_path.with_suffix(".png")
        fig.write_image(str(out_png), format="png", scale=4)

        out_pdf = output_path.with_suffix(".pdf")
        fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 5) ICS & Stability vs τ – Plotly, complementary colours
    # ------------------------------------------------------------------ #

    def plot_ics_and_stability_vs_tau_plotly(self, output_dir: Path):
        """
        For each (genus, material) pair, create an interactive Plotly line chart:

            - x-axis: tau
            - left y-axis: ICS (ics_score_mean)
            - right y-axis: stability (stability_mean)

        For each similarity metric:
            - base colour: ICS curve  (solid)
            - complementary colour: Stability curve (dashed)

        Legend: bottom-centred, large font.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        if "ics_score_mean" not in flat_df.columns or "stability_mean" not in flat_df.columns:
            return

        group_cols = ["genus", "material", "metric", "tau"]
        agg = (
            flat_df
            .groupby(group_cols, as_index=False)
            .agg(
                ics_score_mean=("ics_score_mean", "mean"),
                stability_mean=("stability_mean", "mean"),
            )
        )

        for (genus, material), df_ds in agg.groupby(["genus", "material"]):
            df_ds = df_ds.copy().sort_values(["metric", "tau"])

            metrics = df_ds["metric"].unique()
            base_colors = self._get_metric_base_colors(metrics)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for metric in metrics:
                df_m = df_ds[df_ds["metric"] == metric]
                base = base_colors[metric]
                comp = self._complementary_color(base)

                # ICS (primary y, solid)
                fig.add_trace(
                    go.Scatter(
                        x=df_m["tau"],
                        y=df_m["ics_score_mean"],
                        mode="lines+markers",
                        name=f"{metric} – ICS",
                        line=dict(color=base, width=3),
                        marker=dict(size=9),
                    ),
                    secondary_y=False,
                )

                # Stability (secondary y, dashed, complementary colour)
                fig.add_trace(
                    go.Scatter(
                        x=df_m["tau"],
                        y=df_m["stability_mean"],
                        mode="lines+markers",
                        name=f"{metric} – Stability",
                        line=dict(color=comp, width=3, dash="dash"),
                        marker=dict(size=9, symbol="diamond"),
                    ),
                    secondary_y=True,
                )

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")

            fig.update_layout(
                title=f"ICS & Stability vs τ – {genus} / {material}",
                title_x=0.5,
                width=1600,
                height=900,
                template="plotly_white",
                font=dict(size=16),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=14),
                ),
                margin=dict(l=70, r=70, t=80, b=90),
            )

            fig.update_xaxes(
                title_text="τ (threshold)",
                tickformat=".2f",
                title_font=dict(size=18),
                tickfont=dict(size=14),
            )

            fig.update_yaxes(
                title_text="ICS (mean)",
                secondary_y=False,
                title_font=dict(size=18),
                tickfont=dict(size=14),
            )
            fig.update_yaxes(
                title_text="Stability (mean ARI)",
                secondary_y=True,
                title_font=dict(size=18),
                tickfont=dict(size=14),
            )

            out_html = output_dir / \
                f"ics_stability_{safe_genus}_{safe_mat}.html"
            pio.write_html(fig, file=str(out_html),
                           include_plotlyjs="cdn", full_html=True)

            out_png = output_dir / f"ics_stability_{safe_genus}_{safe_mat}.png"
            fig.write_image(str(out_png), format="png", scale=4)

            out_pdf = output_dir / f"ics_stability_{safe_genus}_{safe_mat}.pdf"
            fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 6) External & hierarchical scores vs τ – Plotly
    # ------------------------------------------------------------------ #

    def plot_external_scores_vs_tau_plotly(self, output_dir: Path):
        """
        For each (genus, material), create interactive Plotly line charts showing:

            - hierarchical_score_mean
            - external_NMI_broad_mean
            - external_NMI_fine_mean
            - external_NMI_who_mean

        as a function of τ for each similarity metric.

        One HTML/PNG/PDF per dataset:
            external_scores_<genus>_<material>.*
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_cols = [
            "hierarchical_score_mean",
            "external_NMI_broad_mean",
            "external_NMI_fine_mean",
            "external_NMI_who_mean",
        ]
        score_cols = [c for c in score_cols if c in flat_df.columns]
        if not score_cols:
            return

        group_cols = ["genus", "material", "metric", "tau"]
        agg = (
            flat_df
            .groupby(group_cols, as_index=False)
            .agg({c: "mean" for c in score_cols})
        )

        for (genus, material), df_ds in agg.groupby(["genus", "material"]):
            df_ds = df_ds.copy().sort_values(["metric", "tau"])

            metrics = df_ds["metric"].unique()
            base_colors = self._get_metric_base_colors(metrics)

            n_rows = len(score_cols)
            fig = make_subplots(
                rows=n_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=[
                    c.replace("_mean", "").replace("_", " ") for c in score_cols
                ],
            )

            for row_idx, col in enumerate(score_cols, start=1):
                for metric in metrics:
                    df_m = df_ds[df_ds["metric"] == metric]
                    fig.add_trace(
                        go.Scatter(
                            x=df_m["tau"],
                            y=df_m[col],
                            mode="lines+markers",
                            name=metric,
                            line=dict(color=base_colors[metric], width=3),
                            marker=dict(size=8),
                            showlegend=(row_idx == 1),
                        ),
                        row=row_idx,
                        col=1,
                    )
                fig.update_yaxes(
                    title_text=col.replace("_mean", "").replace("_", " "),
                    row=row_idx,
                    col=1,
                    title_font=dict(size=16),
                    tickfont=dict(size=13),
                )

            fig.update_xaxes(
                title_text="τ (threshold)",
                tickformat=".2f",
                row=n_rows,
                col=1,
                title_font=dict(size=18),
                tickfont=dict(size=14),
            )

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")

            fig.update_layout(
                title=f"External & Hierarchical Scores vs τ – {genus} / {material}",
                template="plotly_white",
                title_x=0.5,
                width=1600,
                height=350 * n_rows,
                font=dict(size=16),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.14,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=14),
                ),
                margin=dict(l=70, r=40, t=90, b=80),
            )

            out_html = output_dir / \
                f"external_scores_{safe_genus}_{safe_mat}.html"
            pio.write_html(fig, file=str(out_html),
                           include_plotlyjs="cdn", full_html=True)

            out_png = output_dir / \
                f"external_scores_{safe_genus}_{safe_mat}.png"
            fig.write_image(str(out_png), format="png", scale=4)

            out_pdf = output_dir / \
                f"external_scores_{safe_genus}_{safe_mat}.pdf"
            fig.write_image(str(out_pdf), format="pdf")

    # ------------------------------------------------------------------ #
    # 7) Cluster composition CSV for best configs
    # ------------------------------------------------------------------ #

    def export_best_cluster_compositions(self, output_path: Path):
        """
        Export a CSV showing, for each BEST (genus, material, metric, tau, gamma):

        - which antibiotics belong to each Louvain community (cluster_id)
        - their broad / fine / WHO labels
        - cluster size
        - total number of communities

        This clearly explains how ARI/NMI and hierarchical scores are computed.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        # genus, material, metric, tau, gamma, ...
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        # Best per (genus, material, metric)
        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        # Load raw data once
        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_from_loader = loader.abx_tested_cols
        antibiotic_cols_from_config = self.config.data.antibiotic_columns or []

        # Ground truth label maps
        label_manager = LabelManager(self.config)
        label_maps = label_manager.get_label_maps()
        broad_labels = label_maps.get("broad", {})
        fine_labels = label_maps.get("fine", {})
        who_labels = label_maps.get("who", {})

        rows = []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, antibiotic_columns = self._prepare_subset_for_network(
                df_raw,
                genus,
                material,
                antibiotic_cols_from_config,
                tested_cols_from_loader,
            )
            if df_subset is None or not antibiotic_columns:
                continue

            # Build similarity matrix & clusters
            sim_engine = SimilarityEngine(df_subset, antibiotic_columns)
            similarity_matrix = sim_engine.compute(metric)

            G = GraphBuilder(tau).build_graph(similarity_matrix)
            clusterer = LouvainClusterer(gamma)
            partitions = clusterer.run_multiple(G, [self.config.random_seed])

            if not partitions:
                continue

            partition = partitions[0]  # dict {antibiotic: cluster_id}
            total_communities = len(set(partition.values()))

            # Compute cluster sizes
            cluster_sizes = (
                pd.Series(list(partition.values()))
                .value_counts()
                .to_dict()  # {cluster_id: size}
            )

            for abx, cluster_id in partition.items():
                rows.append(
                    {
                        "genus": genus,
                        "material": material,
                        "metric": metric,
                        "tau": tau,
                        "gamma": gamma,
                        "total_communities": total_communities,
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_sizes.get(cluster_id, 1),
                        "antibiotic": abx,
                        "label_broad": broad_labels.get(abx),
                        "label_fine": fine_labels.get(abx),
                        "label_who": who_labels.get(abx),
                    }
                )

        if rows:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(output_path, index=False)

    # ------------------------------------------------------------------ #
    # 8) Helper for subset selection (networks)
    # ------------------------------------------------------------------ #

    def _prepare_subset_for_network(
        self,
        df_raw: pd.DataFrame,
        genus: Optional[str],
        material: Optional[str],
        antibiotic_cols_from_config: List[str],
        tested_cols_from_loader: List[str],
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Reproduce the subset selection used in GridSearchRunner for
        a (genus, material) pair, and determine antibiotic columns.
        """
        mask = np.ones(len(df_raw), dtype=bool)

        if genus is not None and "PathogenGenus" in df_raw.columns:
            mask &= (df_raw["PathogenGenus"] == genus).to_numpy()

        if material is not None and "TextMaterialgroupRkiL0" in df_raw.columns:
            mask &= (df_raw["TextMaterialgroupRkiL0"] == material).to_numpy()

        df_subset = df_raw.loc[mask].copy()
        if df_subset.empty:
            return None, []

        # Decide antibiotic columns (same logic as GridSearchRunner)
        if antibiotic_cols_from_config:
            antibiotic_columns = [
                c for c in antibiotic_cols_from_config if c in df_subset.columns
            ]
        else:
            antibiotic_columns = [
                c for c in tested_cols_from_loader if c in df_subset.columns
            ]

        if not antibiotic_columns:
            return None, []

        return df_subset, antibiotic_columns

    # ------------------------------------------------------------------ #
    # 9) Louvain clustered networks for best params (visualize_antibiotic_network)
    # ------------------------------------------------------------------ #

    def plot_best_networks(self, networks_dir: Path):
        """
        For each (genus, material, metric), take the best (tau, gamma) by main score,
        rebuild the similarity matrix, optionally shorten antibiotic labels,
        and call visualize_antibiotic_network to export HTML + PNG + GEXF.

        Outputs for each best config:
          networks/net_<genus>_<material>_<metric>.html
          networks/net_<genus>_<material>_<metric>.png
          networks/net_<genus>_<material>_<metric>.gexf
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        networks_dir.mkdir(parents=True, exist_ok=True)

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_from_loader = loader.abx_tested_cols
        antibiotic_cols_from_config = self.config.data.antibiotic_columns or []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, antibiotic_columns = self._prepare_subset_for_network(
                df_raw,
                genus,
                material,
                antibiotic_cols_from_config,
                tested_cols_from_loader,
            )
            if df_subset is None or not antibiotic_columns:
                continue

            sim_engine = SimilarityEngine(df_subset, antibiotic_columns)
            similarity_matrix = sim_engine.compute(metric)

            # Shorten antibiotic labels for visualization
            abx_cols = list(similarity_matrix.index)
            if self.abx_class_map is not None:
                try:
                    label_map = get_label(
                        abx_cols,
                        antibiotic_class_map=self.abx_class_map,
                        format_type="abbr",
                        enrich=True,
                        include_class=False,
                    )
                    sim_for_viz = similarity_matrix.rename(
                        index=label_map, columns=label_map
                    )
                except Exception:
                    sim_for_viz = similarity_matrix
            else:
                sim_for_viz = similarity_matrix

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")
            safe_metric = metric.replace(" ", "_")

            base_name = f"net_{safe_genus}_{safe_mat}_{safe_metric}"
            html_name = f"{base_name}.html"
            png_name = f"{base_name}.png"
            pdf_name = f"{base_name}.pdf"
            gexf_name = f"{base_name}.gexf"

            # f"{genus} – {material} – {metric} (tau={tau:.2f}, gamma={gamma:.2f})"
            title = ""

            visualize_antibiotic_network(
                data_input=sim_for_viz,
                threshold=tau,                # use tau as similarity threshold
                community_gamma=gamma,        # use gamma as Louvain resolution
                output_dir=str(networks_dir),
                output_html=html_name,
                output_image=png_name,
                output_pdf=pdf_name,
                gexf_path=gexf_name,
                title=title,
                remove_isolated=False,        # keep singletons visible
            )

    # ------------------------------------------------------------------ #
    # 10) Networks from original evaluation pipeline (GraphBuilder + LouvainClusterer)
    # ------------------------------------------------------------------ #

    def plot_best_networks_original(self, networks_dir: Path):
        """
        For each (genus, material, metric), take the best (tau, gamma) by main score,
        rebuild the similarity matrix, build the graph via GraphBuilder,
        run LouvainClusterer with the SAME settings as the evaluation,
        and visualize that exact partition (no re-clustering).

        This guarantees that:
          - total_communities in the CSV
          - and the visual network clusters
        are generated from the same pipeline.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        networks_dir.mkdir(parents=True, exist_ok=True)

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_from_loader = loader.abx_tested_cols
        antibiotic_cols_from_config = self.config.data.antibiotic_columns or []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, antibiotic_columns = self._prepare_subset_for_network(
                df_raw,
                genus,
                material,
                antibiotic_cols_from_config,
                tested_cols_from_loader,
            )
            if df_subset is None or not antibiotic_columns:
                continue

            # --- Build similarity matrix (same as grid search) ---
            sim_engine = SimilarityEngine(df_subset, antibiotic_columns)
            similarity_matrix = sim_engine.compute(metric)

            # --- Build graph + partition EXACTLY like evaluation ---
            graph_builder = GraphBuilder(tau)
            G = graph_builder.build_graph(similarity_matrix)

            clusterer = LouvainClusterer(gamma)
            partitions = clusterer.run_multiple(G, [self.config.random_seed])
            if not partitions:
                continue
            partition = partitions[0]   # dict node -> cluster_id

            # --- Short labels for visualization (but keep node IDs unchanged) ---
            abx_cols = list(similarity_matrix.index)
            label_map = None
            if self.abx_class_map is not None:
                try:
                    label_map = get_label(
                        abx_cols,
                        antibiotic_class_map=self.abx_class_map,
                        format_type="abbr",
                        enrich=True,
                        include_class=False,
                    )
                except Exception:
                    label_map = None

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")
            safe_metric = metric.replace(" ", "_")

            base_name = f"net_{safe_genus}_{safe_mat}_{safe_metric}"
            html_name = f"{base_name}_original.html"
            png_name = f"{base_name}_original.png"
            pdf_name = f"{base_name}_original.pdf"
            gexf_name = f"{base_name}_original.gexf"

            title = ""  # (
            #    f"{genus} – {material} – {metric} "
            #    f"(tau={tau:.2f}, gamma={gamma:.2f}, seed={self.config.random_seed})"
            # )

            visualize_antibiotic_graph_from_partition(
                G=G,
                partition=partition,
                label_map=label_map,
                output_dir=str(networks_dir),
                output_html=html_name,
                output_image=png_name,
                output_pdf=pdf_name,
                gexf_path=gexf_name,
                title=title,
            )

    # ------------------------------------------------------------------ #
    # 11) FDR-pruned networks (similarity-based, EdgeSignificancePruner)
    # ------------------------------------------------------------------ #

    def plot_best_networks_fdr(self, networks_dir: Path):
        """
        Produce similarity-based networks AFTER FDR pruning.
        Only edges that survive FDR significance are kept.

        Saved to: networks_fdr/
        Filenames:
          net_<genus>_<material>_<metric>_FDR.html/.png/.gexf
          sim_<genus>_<material>_<metric>_FDR.csv     (pruned similarity matrix)
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        networks_dir.mkdir(parents=True, exist_ok=True)

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_loader = loader.abx_tested_cols
        abx_cols_cfg = self.config.data.antibiotic_columns or []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, abx_cols = self._prepare_subset_for_network(
                df_raw, genus, material, abx_cols_cfg, tested_cols_loader
            )
            if df_subset is None or not abx_cols:
                continue

            # --- FDR (on binary table) ---
            pruner = (
                EdgeSignificancePruner(
                    df_binary=df_subset[abx_cols],
                    antibiotic_cols=abx_cols,
                    alpha=self.fdr_alpha,
                    min_total=self.fdr_min_total,
                    min_positive=self.fdr_min_positive,
                    alternative=self.fdr_alternative,
                )
                .fit()
            )

            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")
            safe_metric = metric.replace(" ", "_")

            stats_path = (
                networks_dir
                / f"edge_stats_{safe_genus}_{safe_mat}_{safe_metric}.csv"
            )
            pruner.save_edge_statistics(
                filepath=stats_path,
                upper_only=True,
                extra_columns={
                    "genus": genus,
                    "material": material,
                    "metric": metric,
                    "tau": tau,
                    "gamma": gamma,
                    "alpha": self.fdr_alpha,
                    "min_total": self.fdr_min_total,
                    "min_positive": self.fdr_min_positive,
                    "alternative": self.fdr_alternative,
                },
            )

            allowed_pairs = pruner.get_significant_pairs()  # Set[(u, v)]

            # --- Similarity matrix ---
            sim_engine = SimilarityEngine(df_subset, abx_cols)
            similarity_matrix = sim_engine.compute(metric)

            # --- Mask similarity matrix (remove non-FDR edges) ---
            abx_list = list(similarity_matrix.index)
            mask = pd.DataFrame(False, index=abx_list, columns=abx_list)
            for u, v in allowed_pairs:
                if u in mask.index and v in mask.columns:
                    mask.loc[u, v] = True
                    mask.loc[v, u] = True
            for a in abx_list:
                mask.loc[a, a] = True

            sim_fdr = similarity_matrix.where(mask, 0.0)

            # --- SAVE pruned similarity matrix for this config ---
            safe_genus = (genus or "ALL").replace(" ", "_")
            safe_mat = (material or "ALL").replace(" ", "_")
            safe_metric = metric.replace(" ", "_")
            base = f"{safe_genus}_{safe_mat}_{safe_metric}_FDR"

            sim_csv = networks_dir / f"sim_{base}.csv"
            sim_fdr.to_csv(sim_csv)

            # --- Short labels AFTER pruning ---
            if self.abx_class_map is not None:
                try:
                    label_map = get_label(
                        abx_list,
                        antibiotic_class_map=self.abx_class_map,
                        format_type="abbr",
                        enrich=True,
                        include_class=False,
                    )
                    sim_for_viz = sim_fdr.rename(
                        index=label_map, columns=label_map)
                except Exception:
                    sim_for_viz = sim_fdr
            else:
                sim_for_viz = sim_fdr

            html_name = f"net_{base}.html"
            png_name = f"net_{base}.png"
            pdf_name = f"net_{base}.pdf"
            gexf_name = f"net_{base}.gexf"

            title = ""  # (
            #     f"{genus} – {material} – {metric} "
            #     f"(tau={tau:.2f}, gamma={gamma:.2f}) – [FDR-pruned = {self.fdr_alpha}]"
            # )

            visualize_antibiotic_network(
                data_input=sim_for_viz,
                threshold=tau,
                community_gamma=gamma,
                output_dir=str(networks_dir),
                output_html=html_name,
                output_image=png_name,
                output_pdf=pdf_name,
                gexf_path=gexf_name,
                title=title,
                remove_isolated=False,
            )
            # Create a new subfolder inside the same folder
            networks_dir_default = networks_dir / "default_param"
            networks_dir_default.mkdir(parents=True, exist_ok=True)

            # Use different filenames so you don't overwrite the first set
            html_name_def = f"net_{safe_genus}_{safe_mat}_{safe_metric}_FDR_DEFAULT.html"
            png_name_def = f"net_{safe_genus}_{safe_mat}_{safe_metric}_FDR_DEFAULT.png"
            pdf_name_def = f"net_{safe_genus}_{safe_mat}_{safe_metric}_FDR_DEFAULT.pdf"
            gexf_name_def = f"net_{safe_genus}_{safe_mat}_{safe_metric}_FDR_DEFAULT.gexf"

            visualize_antibiotic_network(
                data_input=sim_for_viz,
                threshold=0.3,
                community_gamma=1,
                output_dir=str(networks_dir_default),
                output_html=html_name_def,
                output_image=png_name_def,
                output_pdf=pdf_name_def,
                gexf_path=gexf_name_def,
                title=title,
                remove_isolated=False,
            )

    def export_fdr_edge_statistics(self, output_path: Path):
        """
        Export a CSV with FDR statistics for each BEST (genus, material, metric):

        Columns:
          genus, material, metric, tau, gamma,
          abx_i, abx_j,
          p_value, q_value, is_significant,
          similarity

        This uses the same FDR settings as plot_best_networks_fdr.
        """
        if not self.use_fdr_edge_pruning:
            return

        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return

        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return

        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_loader = loader.abx_tested_cols
        abx_cols_cfg = self.config.data.antibiotic_columns or []

        all_rows = []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, abx_cols = self._prepare_subset_for_network(
                df_raw, genus, material, abx_cols_cfg, tested_cols_loader
            )
            if df_subset is None or not abx_cols:
                continue

            # --- FDR on binary table ---
            pruner = (
                EdgeSignificancePruner(
                    df_binary=df_subset[abx_cols],
                    antibiotic_cols=abx_cols,
                    alpha=self.fdr_alpha,
                    min_total=self.fdr_min_total,
                    min_positive=self.fdr_min_positive,
                    alternative=self.fdr_alternative,
                )
                .fit()
            )

            # Assume pruner exposes a DataFrame of results, e.g.:
            #   pruner.results_df with columns:
            #   ['abx_i', 'abx_j', 'p_value', 'q_value', 'is_significant']
            if not hasattr(pruner, "results_df") or pruner.results_df is None:
                continue

            stats_df = pruner.results_df.copy()

            # --- Similarity matrix for reference ---
            sim_engine = SimilarityEngine(df_subset, abx_cols)
            similarity_matrix = sim_engine.compute(metric)

            def get_sim(row_edge):
                i = row_edge["abx_i"]
                j = row_edge["abx_j"]
                if i in similarity_matrix.index and j in similarity_matrix.columns:
                    return float(similarity_matrix.loc[i, j])
                return np.nan

            stats_df["similarity"] = stats_df.apply(get_sim, axis=1)

            # Add context columns
            stats_df["genus"] = genus
            stats_df["material"] = material
            stats_df["metric"] = metric
            stats_df["tau"] = tau
            stats_df["gamma"] = gamma

            all_rows.append(stats_df)

        if all_rows:
            out_df = pd.concat(all_rows, axis=0, ignore_index=True)
            # Order columns nicely
            cols = [
                "genus", "material", "metric", "tau", "gamma",
                "abx_i", "abx_j",
                "similarity",
                "p_value", "q_value", "is_significant",
            ]
            cols = [c for c in cols if c in out_df.columns]
            out_df = out_df[cols]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------ #
    # 12) Compute edge-retention summary (per genus × material)
    # ------------------------------------------------------------------ #

    def compute_retention_summary(
        self,
        output_csv: Path,
        metrics_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute FDR-based edge retention for each (genus, material, metric) at the
        best (tau, gamma) configuration for that metric.

        Retention is defined as:
            # edges with (sim >= tau AND q <= alpha AND testable)
            ----------------------------------------------------
            # edges with (sim >= tau AND testable)

        - sim is the similarity value from the given metric.
        - q is the FDR-adjusted p-value from EdgeSignificancePruner.

        Parameters
        ----------
        output_csv : Path
            Where to save the per-(genus, material, metric) retention summary.
        metrics_filter : Optional[List[str]]
            If provided (e.g. ["Jaccard", "cosine"]), restrict to those metrics.
            If None, include all metrics present in the aggregated results.
        """
        agg_df = self.results.aggregated_results
        if agg_df is None or agg_df.empty:
            return pd.DataFrame()

        # Flatten aggregated results
        flat_df = agg_df.copy()
        flat_df.columns = ["_".join(c).rstrip("_")
                           for c in flat_df.columns.to_list()]
        # genus, material, metric, tau, gamma, ...
        flat_df = flat_df.reset_index()

        score_col = self._choose_main_score_col(flat_df)
        if score_col is None:
            return pd.DataFrame()

        # Best per (genus, material, metric)
        best_df = (
            flat_df.sort_values(score_col, ascending=False)
            .groupby(["genus", "material", "metric"], as_index=False)
            .first()
        )

        # Optional: restrict to selected metrics
        if metrics_filter is not None:
            best_df = best_df[best_df["metric"].isin(metrics_filter)]
            if best_df.empty:
                print(
                    f"[Retention] No best configurations found for metrics={metrics_filter}."
                )
                return pd.DataFrame()

        # Load raw data
        loader = DataLoader(str(self.config.data.data_path))
        df_raw = loader.get_combined()
        tested_cols_from_loader = loader.abx_tested_cols
        antibiotic_cols_from_config = self.config.data.antibiotic_columns or []

        rows = []

        for _, row in best_df.iterrows():
            genus = row["genus"]
            material = row["material"]
            metric = row["metric"]
            tau = float(row["tau"])
            gamma = float(row["gamma"])

            df_subset, abx_cols = self._prepare_subset_for_network(
                df_raw,
                genus,
                material,
                antibiotic_cols_from_config,
                tested_cols_from_loader,
            )
            if df_subset is None or not abx_cols:
                continue

            # --- Similarity matrix for this subset + metric ---
            sim_engine = SimilarityEngine(df_subset, abx_cols)
            sim = sim_engine.compute(metric)

            # --- Build binary testing matrix for FDR (0/1) ---
            df_binary = (
                df_subset[abx_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )

            # --- Fit pruner & get q-values ---
            pruner = EdgeSignificancePruner(
                df_binary=df_binary,
                antibiotic_cols=abx_cols,
                alpha=self.fdr_alpha,
                min_total=self.fdr_min_total,
                min_positive=self.fdr_min_positive,
                alternative=self.fdr_alternative,
            ).fit()

            qvals = pruner.qval_df.reindex(
                index=sim.index, columns=sim.columns)
            qvals_np = qvals.to_numpy()

            # --- Define mask: edges with sim >= tau (undirected, off-diagonal only) ---
            sim_np = sim.to_numpy()
            n = sim_np.shape[0]

            sim_mask = (sim_np >= tau)
            np.fill_diagonal(sim_mask, False)

            # "testable" = q-value is finite / not NaN
            testable_mask = np.isfinite(qvals_np) & ~np.isnan(qvals_np)
            np.fill_diagonal(testable_mask, False)

            # Edges considered in retention denominator:
            denom_mask = sim_mask & testable_mask

            # Edges retained (significant) in numerator:
            signif_mask = denom_mask & (qvals_np <= self.fdr_alpha)

            # Convert to undirected counts: use upper triangle only
            triu_idx = np.triu_indices(n, k=1)
            n_edges_jaccard = int(sim_mask[triu_idx].sum())
            n_edges_testable = int(denom_mask[triu_idx].sum())
            n_edges_signif = int(signif_mask[triu_idx].sum())

            retention = (
                float(n_edges_signif) / float(n_edges_testable)
                if n_edges_testable > 0
                else np.nan
            )

            rows.append(
                {
                    "genus": genus,
                    "material": material,
                    "metric": metric,
                    "tau": tau,
                    "gamma": gamma,
                    "n_nodes": len(abx_cols),
                    "n_edges_jaccard": n_edges_jaccard,
                    "n_edges_testable": n_edges_testable,
                    "n_edges_significant": n_edges_signif,
                    "retention_fraction": retention,
                    "retention_percent": retention * 100.0 if not np.isnan(retention) else np.nan,
                    "alpha_fdr": self.fdr_alpha,
                    "min_total": self.fdr_min_total,
                    "min_positive": self.fdr_min_positive,
                }
            )

        if not rows:
            df_ret = pd.DataFrame()
        else:
            df_ret = pd.DataFrame(rows)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_ret.to_csv(output_csv, index=False)
        print(f"[Retention] Saved retention summary → {output_csv}")

        return df_ret

    # ------------------------------------------------------------------ #
    # 13) Plot retention by metric (per genus × material) – Plotly
    # ------------------------------------------------------------------ #

    def plot_retention_by_metric(
        self,
        retention_df: pd.DataFrame,
        output_path: Path,
        genera: Optional[List[str]] = None,
        materials: Optional[List[str]] = None,
    ) -> None:
        """
        Publication-ready: FDR retention (%) by metric × genus × specimen.
        """

        if retention_df.empty:
            print("[Retention] Empty retention_df, nothing to plot.")
            return

        df = retention_df.copy()

        # Ordering defaults
        if genera is None:
            genera = [
                "Escherichia",
                "Staphylococcus",
                "Klebsiella",
                "Pseudomonas",
                "Proteus",
                "Streptococcus",
            ]
        if materials is None:
            materials = ["Blood Culture", "Urine"]

        df = df[df["genus"].isin(genera) & df["material"].isin(materials)]
        if df.empty:
            print("[Retention] No rows after filtering.")
            return

        df["genus"] = pd.Categorical(
            df["genus"], categories=genera, ordered=True)
        df["material"] = pd.Categorical(
            df["material"], categories=materials, ordered=True)
        df = df.sort_values(["material", "genus", "metric"])

        metrics = sorted(df["metric"].unique())
        base_colors = self._get_metric_base_colors(metrics)

        # Larger figure size for readability
        total_height = 550 * len(materials)   # Increase height per row
        total_width = 1500                   # Wider canvas

        fig = make_subplots(
            rows=len(materials),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,  # more spacing between rows
            subplot_titles=materials,
        )

        for row_idx, mat in enumerate(materials, start=1):
            df_m = df[df["material"] == mat]
            if df_m.empty:
                continue

            for metric in metrics:
                df_mm = df_m[df_m["metric"] == metric]
                if df_mm.empty:
                    continue

                fig.add_trace(
                    go.Bar(
                        x=df_mm["genus"],
                        y=df_mm["retention_percent"],
                        name=metric,
                        marker=dict(color=base_colors.get(metric)),
                        text=df_mm["retention_percent"].round(1).astype(str),
                        textposition="outside",
                        textfont=dict(size=16),
                        cliponaxis=False,  # Prevent label clipping
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx, col=1
                )

            fig.update_yaxes(
                title_text="Retention after FDR (%)",
                range=[0, 110],
                title_font=dict(size=20),
                tickfont=dict(size=16),
                row=row_idx, col=1
            )

        fig.update_xaxes(
            title_text="Pathogen genus",
            row=len(materials),
            col=1,
            title_font=dict(size=22),
            tickfont=dict(size=16),
        )

        fig.update_layout(
            title=dict(
                text="FDR-based retention of co-testing edges by metric, genus, and specimen type",
                x=0.5,
                xanchor="center",
                y=0.98,
                font=dict(size=28)
            ),
            template="plotly_white",
            font=dict(size=18),
            barmode="group",
            bargap=0.25,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.13,
                xanchor="center",
                x=0.5,
                font=dict(size=18),
            ),
            margin=dict(l=120, r=120, t=140, b=160),
            height=total_height,
            width=total_width,
        )

        out_html = output_path.with_suffix(".html")
        out_png = output_path.with_suffix(".png")
        out_pdf = output_path.with_suffix(".pdf")
        out_html.parent.mkdir(parents=True, exist_ok=True)

        pio.write_html(fig, file=str(out_html),
                       include_plotlyjs="cdn", full_html=True)
        fig.write_image(str(out_png), format="png", scale=4)
        fig.write_image(str(out_pdf), format="pdf")

        print(f"[Retention] Saved improved retention figure → {out_png}")