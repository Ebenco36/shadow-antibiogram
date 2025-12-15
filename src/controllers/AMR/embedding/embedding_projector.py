# src/controllers/AMR/embedding/embedding_projector.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

try:
    import umap  # pip install umap-learn
except ImportError:
    umap = None

import plotly.express as px
import plotly.io as pio


@dataclass
class EmbeddingProjectorConfig:
    """
    Configuration for dimensionality reduction of antibiotics.

    X can be:
      - feature matrix: antibiotics x features
      - similarity matrix: antibiotics x antibiotics (rows used as features)

    You can use:
      - TSNE: metric on rows of X
      - UMAP: either on features or on a precomputed distance matrix
    """
    n_components: int = 2
    random_state: int = 42

    # t-SNE
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    tsne_metric: str = "euclidean"

    # UMAP
    umap_n_neighbors: int = 10
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"      # set to "precomputed" if you pass a distance matrix
    umap_n_epochs: Optional[int] = None

    # Output
    output_dir: Path = Path("results/embeddings")
    overwrite: bool = True


class EmbeddingProjector:
    """
    Pluggable dimensionality reduction helper for antibiotics.

    Typical usage:
        projector = EmbeddingProjector(EmbeddingProjectorConfig(...))
        coords_tsne = projector.compute_tsne(X)
        fig = projector.plot_embedding(
            coords_tsne,
            labels=cluster_ids,
            label_name="community",
            method="TSNE",
            title="t-SNE of antibiotics"
        )
        projector.save_figure(fig, output_dir / "tsne_example.html")
    """

    def __init__(self, config: Optional[EmbeddingProjectorConfig] = None):
        self.config = config or EmbeddingProjectorConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # t-SNE
    # ------------------------------------------------------------------ #

    def compute_tsne(
        self,
        X: pd.DataFrame,
        metric: Optional[str] = None,
        **tsne_overrides: Any,
    ) -> pd.DataFrame:
        """
        Compute t-SNE embedding on rows of X.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Rows = antibiotics, columns = features (or similarities to other antibiotics).

        metric : str or None
            Passed to sklearn.manifold.TSNE(metric=...).
            If None, uses config.tsne_metric.

        tsne_overrides : dict
            Any sklearn TSNE parameters you want to override (per call).

        Returns
        -------
        coords : DataFrame, index = X.index, columns = ["dim1", "dim2"]
        """
        if X.empty:
            raise ValueError("X is empty in compute_tsne().")

        metric = metric or self.config.tsne_metric

        params = dict(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            perplexity=self.config.tsne_perplexity,
            learning_rate=self.config.tsne_learning_rate,
            max_iter=self.config.tsne_n_iter,
            metric=metric,
            init="random",
        )
        params.update(tsne_overrides)

        tsne = TSNE(**params)
        emb = tsne.fit_transform(X.values)

        coords = pd.DataFrame(
            emb,
            index=X.index,
            columns=[f"dim{i+1}" for i in range(self.config.n_components)],
        )
        return coords

    # ------------------------------------------------------------------ #
    # UMAP
    # ------------------------------------------------------------------ #

    def compute_umap(
        self,
        X: pd.DataFrame,
        metric: Optional[str] = None,
        precomputed_distance: bool = False,
        **umap_overrides: Any,
    ) -> pd.DataFrame:
        """
        Compute UMAP embedding on rows of X.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features) or distance matrix if precomputed_distance=True

        metric : str or None
            UMAP metric. If precomputed_distance=True, metric is forced to "precomputed"
            unless explicitly overridden.

        precomputed_distance : bool
            If True, X is treated as a distance matrix.

        umap_overrides : dict
            Any UMAP parameters to override.

        Returns
        -------
        coords : DataFrame, index = X.index, columns = ["dim1", "dim2"]
        """
        if umap is None:
            raise RuntimeError(
                "UMAP is not installed. Install with `pip install umap-learn`."
            )

        if X.empty:
            raise ValueError("X is empty in compute_umap().")

        if precomputed_distance:
            metric = "precomputed"
        else:
            metric = metric or self.config.umap_metric

        params = dict(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric=metric,
        )
        if self.config.umap_n_epochs is not None:
            params["n_epochs"] = self.config.umap_n_epochs

        params.update(umap_overrides)

        reducer = umap.UMAP(**params)
        emb = reducer.fit_transform(X.values)

        coords = pd.DataFrame(
            emb,
            index=X.index,
            columns=[f"dim{i+1}" for i in range(self.config.n_components)],
        )
        return coords

    # ------------------------------------------------------------------ #
    # Plotting helpers (Plotly)
    # ------------------------------------------------------------------ #

    def plot_embedding(
        self,
        coords: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        label_name: str = "label",
        method: str = "UMAP",
        title: Optional[str] = None,
        color_discrete_map: Optional[Dict[str, str]] = None,
        hover_data: Optional[pd.DataFrame] = None,
    ):
        """
        Generic 2D embedding plot.

        Parameters
        ----------
        coords : DataFrame, index = item IDs, columns = ["dim1", "dim2"]

        labels : Series or None
            Index-aligned with coords.index. Used for colouring.

        label_name : str
            Column name for the label.

        method : str
            "UMAP" or "TSNE" or any string; used in title and axis labels.

        title : str or None
            Plot title. If None, a default based on method is used.

        color_discrete_map : dict or None
            Plotly colour map label -> hex. Optional for custom palette.

        hover_data : DataFrame or None
            Additional columns to show on hover; index must match coords.index.

        Returns
        -------
        fig : plotly.graph_objs.Figure
        """
        if coords.shape[1] < 2:
            raise ValueError("coords must have at least 2 columns for a 2D scatter.")

        df_plot = coords.copy()
        df_plot = df_plot.rename(
            columns={coords.columns[0]: "dim1", coords.columns[1]: "dim2"}
        )

        df_plot["item_id"] = df_plot.index.astype(str)

        if labels is not None:
            labels = labels.reindex(df_plot.index)
            df_plot[label_name] = labels.astype(str)
        else:
            df_plot[label_name] = "all"

        if hover_data is not None:
            hover_aligned = hover_data.reindex(df_plot.index)
            for c in hover_aligned.columns:
                df_plot[c] = hover_aligned[c]

        fig_title = title or f"{method} embedding"

        # Axis labels use the method name (TSNE-1, TSNE-2, UMAP-1, UMAP-2, etc.)
        axis_prefix = str(method).upper()
        x_title = f"{axis_prefix}-1"
        y_title = f"{axis_prefix}-2"

        fig = px.scatter(
            df_plot,
            x="dim1",
            y="dim2",
            color=label_name,
            color_discrete_map=color_discrete_map,
            hover_name="item_id",
            hover_data=[
                c
                for c in df_plot.columns
                if c not in ["dim1", "dim2", "item_id"]
            ],
        )

        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1, color="black"),
            )
        )

        fig.update_layout(
            title=fig_title,
            template="plotly_white",
            font=dict(size=16),
            xaxis=dict(
                title=x_title,
                title_font=dict(size=18),
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=y_title,
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
            margin=dict(l=70, r=40, t=80, b=80),
        )
        return fig

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save_figure(
        self,
        fig,
        output_path: Path,
        save_png: bool = True,
        save_pdf: bool = False,
    ):
        """
        Save Plotly figure to HTML (+ PNG/PDF if requested).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out_html = output_path.with_suffix(".html")
        if out_html.exists() and not self.config.overwrite:
            raise FileExistsError(f"{out_html} already exists and overwrite=False")

        pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)

        if save_png:
            out_png = output_path.with_suffix(".png")
            fig.write_image(str(out_png), format="png", scale=4)

        if save_pdf:
            out_pdf = output_path.with_suffix(".pdf")
            fig.write_image(str(out_pdf), format="pdf")
