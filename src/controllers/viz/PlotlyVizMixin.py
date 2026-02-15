from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import (Optional, Sequence, Union, Mapping)
from src.utils.helpers import _ensure_list
import plotly.graph_objects as go
import plotly.express as px
@dataclass
class PlotlyVizMixin:
    """Add Plotly visualizations to any analyzer class."""

    # ---- parallel coordinates ----
    def parallel_coordinates(
        self,
        df_long: pd.DataFrame,
        group_cols: Union[str, Sequence[str]],
        antibiotic_col: str = "Antibiotic",
        value_col: str = "Pct_tested",
        color_col: Optional[str] = None,
        color_continuous_scale: str = "Blues",
        height: int = 650,
        width: int = 1100,
    ) -> go.Figure:
        group_cols = _ensure_list(group_cols)
        wide = df_long.pivot_table(index=group_cols,
                                   columns=antibiotic_col,
                                   values=value_col,
                                   aggfunc="first").reset_index()

        dims = [c for c in wide.columns if c not in group_cols]  # antibiotics only
        line_color = wide[color_col] if color_col and color_col in wide else None

        fig = go.Figure(
            go.Parcoords(
                line=dict(color=line_color,
                          colorscale=color_continuous_scale,
                          showscale=line_color is not None),
                dimensions=[dict(label=d, values=wide[d].astype(float)) for d in dims]
            )
        )
        fig.update_layout(height=height, width=width)
        return fig

    # ---- sunburst ----
    def sunburst(
        self,
        df_long: pd.DataFrame,
        path: Sequence[str],
        value_col: str,
        color_col: Optional[str] = None,
        height: int = 700,
        width: int = 700,
        **kwargs
    ) -> go.Figure:
        return px.sunburst(df_long, path=path, values=value_col,
                           color=color_col, height=height, width=width, **kwargs)

    # ---- radar / polar ----
    def polar_radar(
        self,
        wide_matrix: pd.DataFrame,
        *,
        row_as_series: bool = True,
        fill: str = "toself",
        mode: str = "lines",
        show_legend: bool = True,
        height: int = 650,
        width: int = 800,
    ) -> go.Figure:
        mat = wide_matrix.copy()
        theta = mat.columns.tolist()
        fig = go.Figure()
        if row_as_series:
            for idx, row in mat.iterrows():
                fig.add_trace(go.Scatterpolar(r=row.values, theta=theta,
                                              name=str(idx), fill=fill, mode=mode))
        else:
            for col in theta:
                fig.add_trace(go.Scatterpolar(r=mat[col].values, theta=mat.index.tolist(),
                                              name=str(col), fill=fill, mode=mode))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=show_legend,
                          height=height, width=width)
        return fig

    # ---- chord/circos-ish with Sankey ----
    def circos_chord(
        self,
        square_matrix: pd.DataFrame,
        *,
        node_order: Optional[Sequence[str]] = None,
        colors: Optional[Mapping[str, str]] = None,
        title: str = "Chord Diagram",
        height: int = 800,
        width: int = 800,
    ) -> go.Figure:
        mat = square_matrix.copy()
        if node_order is None: node_order = mat.columns.tolist()
        mat = mat.loc[node_order, node_order]

        labels = node_order
        src, tgt, val = [], [], []
        for i, r in enumerate(labels):
            for j, c in enumerate(labels):
                w = float(mat.loc[r, c])
                if w <= 0 or i == j:
                    continue
                src.append(i); tgt.append(j); val.append(w)

        if colors is None:
            base = px.colors.qualitative.Plotly
            colors = {lab: base[i % len(base)] for i, lab in enumerate(labels)}
        node_colors = [colors[l] for l in labels]

        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(pad=10, thickness=10, line=dict(color="black", width=0.5),
                      label=labels, color=node_colors),
            link=dict(source=src, target=tgt, value=val,
                      color=[colors[labels[s]] for s in src])
        ))
        fig.update_layout(title_text=title, font_size=10, height=height, width=width)
        return fig