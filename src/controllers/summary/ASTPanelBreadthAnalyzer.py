# src/controllers/summary/ASTPanelBreadthAnalyzer_Plotly.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kruskal, gaussian_kde

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------ helpers ------------------------------

def _is_hex(c: str) -> bool:
    c = str(c).strip()
    if c.startswith("#"):
        c = c[1:]
    return len(c) in (3, 6) and all(ch in "0123456789abcdefABCDEF" for ch in c)

def _pstar(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))

def _ecdf(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF (step) for a 1D array."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(y)
    ps = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ps

def _plotly_pub_layout(fig: go.Figure, *, title: str) -> None:
    # Clean, publication-style base
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="simple_white",
        width=1200, height=700,
        # leave room for a bottom legend (no overlay)
        margin=dict(l=70, r=30, t=70, b=110),
        # true bottom legend, centered, no box overlay
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.18, yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            itemsizing="constant",
            title=None,
            tracegroupgap=10
        ),
        hovermode="x unified",
        font=dict(size=13)
    )

    # Axis polish
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)",
        ticks="outside", tickcolor="rgba(0,0,0,0.25)", mirror=False,
        gridcolor="rgba(0,0,0,0.06)"
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)",
        ticks="outside", tickcolor="rgba(0,0,0,0.25)", mirror=False,
        gridcolor="rgba(0,0,0,0.06)"
    )

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _unique_order_by_median(d: pd.DataFrame, group_col: str, val_col: str) -> List[str]:
    order = (
        d.groupby(group_col, observed=True)[val_col]
         .median()
         .sort_values(ascending=False)
         .index.tolist()
    )
    return [str(o) for o in order]


# ------------------------------ analyzer ------------------------------

class ASTPanelBreadthAnalyzer:
    """
    Plotly-native analyzer for antimicrobial susceptibility testing (AST) panel breadth.

    Returns interactive figures and can export HTML/PNG/SVG/PDF via kaleido.
    """

    def __init__(self, df: pd.DataFrame, test_col_suffix: str = "_Tested"):
        self.test_col_suffix = test_col_suffix
        self.logger = self._setup_logger()
        self.df: pd.DataFrame = pd.DataFrame()
        self.tested_cols: List[str] = []
        self.load_data(df)
        self.process_data()

    # ---- lifecycle ----
    def _setup_logger(self):
        # Avoid forcing global basicConfig; let app handlers manage formatting
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self, df: pd.DataFrame):
        try:
            x = df.copy()
            if "ARS_WardType" in x.columns:
                x["ARS_WardType"] = x["ARS_WardType"].replace({
                    "Early Rehabilitation": "Rehabilitation"
                })
            self.df = x
            self.logger.info(f"Loaded data: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} cols")
        except Exception as e:
            self.logger.error(f"Failed to load dataframe: {e}")
            raise

    def process_data(self):
        self.tested_cols = [c for c in self.df.columns if c.endswith(self.test_col_suffix)]
        if not self.tested_cols:
            raise ValueError(f"No columns end with '{self.test_col_suffix}'.")

        # Prefer explicit total; fallback to indicator sum where missing
        if "TotalAntibioticsTested" in self.df.columns:
            total = pd.to_numeric(self.df["TotalAntibioticsTested"], errors="coerce")
            bin_df = (
                self.df[self.tested_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0).astype(float)
                    .gt(0).astype(int)
            )
            self.df["antibiotics_tested_count"] = total.fillna(bin_df.sum(axis=1)).astype(int)
            used = f"{len(self.tested_cols)} indicators (fallback where TotalAntibioticsTested missing)"
        else:
            bin_df = (
                self.df[self.tested_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0).astype(float)
                    .gt(0).astype(int)
            )
            self.df["antibiotics_tested_count"] = bin_df.sum(axis=1).astype(int)
            used = f"{len(self.tested_cols)} indicators (sum)"
        self.logger.info(f"Panel breadth computed using {used}.")

    # ---- filtering & stats ----
    def filter_data(
        self,
        filter_dict: Optional[Dict] = None,
        pathogen_genus: Optional[str] = None
    ) -> pd.DataFrame:
        df_filtered = self.df.copy()

        if pathogen_genus and "PathogenGenus" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["PathogenGenus"] == pathogen_genus]
            self.logger.info(f"Filtered to pathogen genus: {pathogen_genus}")

        if filter_dict:
            for col, value in filter_dict.items():
                if col not in df_filtered.columns:
                    self.logger.warning(f"Column '{col}' not in dataframe; skipping.")
                    continue

                if callable(value):
                    mask = value(df_filtered[col])
                    df_filtered = df_filtered[mask]
                elif isinstance(value, (list, tuple, set)):
                    df_filtered = df_filtered[df_filtered[col].isin(list(value))]
                elif isinstance(value, dict):
                    m = pd.Series(True, index=df_filtered.index)
                    if "in" in value:
                        m &= df_filtered[col].isin(list(value["in"]))
                    if "notin" in value:
                        m &= ~df_filtered[col].isin(list(value["notin"]))
                    df_filtered = df_filtered[m]
                else:
                    df_filtered = df_filtered[df_filtered[col] == value]

                self.logger.info(f"Filtered {col} to value: {value}")

        return df_filtered

    def compute_summary_stats(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str = "antibiotics_tested_count"
    ) -> pd.DataFrame:
        def pct(x, p): return float(np.percentile(x, p))
        # observed=True => drop empty categorical groups & silence FutureWarning
        summary = (
            df.groupby(group_col, observed=True)[value_col]
              .agg(
                  n="count",
                  median="median",
                  p25=lambda x: pct(x, 25),
                  p75=lambda x: pct(x, 75),
                  p10=lambda x: pct(x, 10),
                  p90=lambda x: pct(x, 90),
                  mean="mean",
                  std="std",
                  min="min",
                  max="max"
              )
              .reset_index()
        )
        summary["IQR"] = summary["p75"] - summary["p25"]
        cols = [group_col, "n", "median", "IQR", "p10", "p90", "mean", "std", "min", "max"]
        return summary[[c for c in cols if c in summary.columns]]

    # ---- inequality helpers ----
    def _lorenz_points(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sort(np.asarray(arr, float))
        x = x[x >= 0]
        if x.size == 0 or x.sum() == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        cum = np.cumsum(x)
        p = np.linspace(0, 1, len(cum) + 1)
        L = np.insert(cum / cum[-1], 0, 0.0)
        return p, L

    def _gini(self, arr: np.ndarray) -> float:
        p, L = self._lorenz_points(arr)
        return float(1 - 2 * np.trapz(L, p))

    # ---- plotly figures ----
    def fig_density_ecdf(
        self, df, group_col, value_col="antibiotics_tested_count",
        title="Testing breadth distribution (KDE + ECDF)",
        color_map: Optional[Dict[str, str]] = None,
        ecdf_dash: str = "dot",
        kde_width: float = 2.3,
        ecdf_width: float = 1.4
    ) -> go.Figure:
        d = df[[group_col, value_col]].dropna()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if d.empty:
            self.logger.warning(f"No data for density/ECDF by {group_col}")
            _plotly_pub_layout(fig, title=title)
            return fig

        order = _unique_order_by_median(d, group_col, value_col)

        # build final color map (custom overrides -> fallback palette)
        palette = px.colors.qualitative.Set2
        cyc = itertools.cycle(palette)
        final_colors: Dict[str, str] = {}
        for g in order:
            col = None
            if color_map and g in color_map:
                user_col = color_map[g]
                if _is_hex(user_col):
                    user_col = "#" + user_col.lstrip("#")
                col = user_col
            if not col:
                col = next(cyc)
            final_colors[g] = col

        x_max = max(5.0, float(d[value_col].max()))
        xs = np.linspace(0, x_max, 512)

        # KDE curves
        for g in order:
            vals = d.loc[d[group_col] == g, value_col].astype(float).values
            n = len(vals)
            if n == 0:
                continue
            col = final_colors[g]

            if n < 2 or np.std(vals) == 0:
                fig.add_trace(
                    go.Scatter(
                        x=vals, y=[0.001]*n, mode="markers",
                        name=f"{g} (KDE, n={n:,})", legendgroup=g, showlegend=True,
                        marker=dict(color=col, size=5, opacity=0.8)
                    ),
                    secondary_y=False
                )
                continue

            kde = gaussian_kde(vals, bw_method="scott")
            ys = kde(xs)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    name=f"{g} (KDE, n={n:,})",
                    legendgroup=g,
                    showlegend=True,
                    line=dict(width=kde_width, color=col)
                ),
                secondary_y=False
            )

        # ECDF
        for g in order:
            vals = d.loc[d[group_col] == g, value_col].astype(float).values
            if vals.size == 0:
                continue
            xs_ecdf, ps = _ecdf(vals)
            if xs_ecdf.size == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs_ecdf, y=ps, mode="lines",
                    name=f"{g} (ECDF)",
                    legendgroup=g,
                    showlegend=True,
                    line=dict(width=ecdf_width, dash=ecdf_dash, color=final_colors[g]),
                    opacity=0.9
                ),
                secondary_y=True
            )

        fig.update_xaxes(
            title_text="Number of antibiotics tested per isolate",
            range=[0, x_max],
            ticks="outside"
        )
        fig.update_yaxes(title_text="Density", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Cumulative proportion", secondary_y=True, range=[0, 1], ticks="outside")

        _plotly_pub_layout(fig, title=title)
        return fig

    def fig_violin_box(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str = "antibiotics_tested_count",
        title: str = "Testing breadth by group",
        color_map: Optional[Dict[str, str]] = None
    ) -> go.Figure:
        """
        Violin per group with inner box & mean line; points shown for small n.
        """
        d = df[[group_col, value_col]].dropna()
        fig = go.Figure()
        if d.empty:
            self.logger.warning(f"No data for violin by {group_col}")
            _plotly_pub_layout(fig, title=title)
            return fig

        order = _unique_order_by_median(d, group_col, value_col)
        counts = d.groupby(group_col, observed=True).size().reindex(order).fillna(0).astype(int)

        # Build color map (reuse custom or fallback palette)
        palette = px.colors.qualitative.Set2
        cyc = itertools.cycle(palette)
        final_colors = {}
        for g in order:
            if color_map and g in color_map:
                c = color_map[g]
                if not str(c).startswith("#") and _is_hex(c):
                    c = "#" + str(c)
            else:
                c = next(cyc)
            final_colors[g] = c

        # Add violins with consistent color
        for g in order:
            vals = d.loc[d[group_col] == g, value_col].tolist()
            if len(vals) == 0:
                continue
            fig.add_trace(go.Violin(
                x=[g]*len(vals),
                y=vals,
                name=f"{g} (n={counts.loc[g]:,})",
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.25,
                scalemode="count",
                side="both",
                line_color=final_colors[g],     # outline
                fillcolor=final_colors[g],      # fill
                opacity=0.9                     # soft fill for overlapping
            ))

        fig.update_xaxes(title_text=group_col, ticks="outside")
        fig.update_yaxes(title_text="Antibiotics tested per isolate", rangemode="tozero", ticks="outside")
        _plotly_pub_layout(fig, title=title)
        return fig

    def fig_lorenz_by_group(
        self, df: pd.DataFrame, group_col: str, value_col: str = "antibiotics_tested_count",
        title: str = "Lorenz curves — Panel breadth by group",
        color_map: Optional[Dict[str,str]] = None
    ) -> go.Figure:
        d = df[[group_col, value_col]].dropna()
        fig = go.Figure()
        if d.empty:
            _plotly_pub_layout(fig, title=title)
            return fig

        order = _unique_order_by_median(d, group_col, value_col)
        palette = px.colors.qualitative.Set2
        cyc = itertools.cycle(palette)
        final_colors = {
            g: (color_map[g] if color_map and g in color_map else next(cyc))
            for g in order
        }

        rows = []
        for g in order:
            vals = d.loc[d[group_col]==g, value_col].values
            if vals.size == 0:
                continue
            p, L = self._lorenz_points(vals)
            fig.add_trace(go.Scatter(
                x=p, y=L, mode="lines", name=f"{g}",
                line=dict(width=2.2, color=final_colors[g])
            ))
            rows.append({
                "group": str(g),
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "gini": self._gini(vals)
            })

        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="Equality", line=dict(width=1, dash="dash"), showlegend=True))
        fig.update_xaxes(title_text="Cumulative share of isolates", range=[0,1])
        fig.update_yaxes(title_text="Cumulative share of total tests", range=[0,1])
        _plotly_pub_layout(fig, title=title)
        # attach quick-summary table
        fig._summary_table = pd.DataFrame(rows, columns=["group","n","median","gini"])
        return fig

    def fig_ks_heatmap(
        self,
        ks_results: pd.DataFrame,
        *,
        title: str = "Pairwise comparisons — KS statistic",
        fdr_alpha: float = 0.05,
        cluster: bool = True
    ) -> go.Figure:
        """
        Heatmap of KS statistic with • markers where FDR-BH q<alpha.
        """
        fig = go.Figure()
        if ks_results.empty:
            self.logger.warning("No KS results to plot.")
            _plotly_pub_layout(fig, title=title)
            return fig

        pairs = ks_results["Comparison"].str.split(" vs ", n=1, expand=True)
        ks_results = ks_results.assign(g1=pairs[0], g2=pairs[1])

        groups = sorted(list(set(ks_results["g1"]) | set(ks_results["g2"])))
        g_idx = {g: i for i, g in enumerate(groups)}
        n = len(groups)

        K = np.zeros((n, n))
        P = np.ones((n, n))

        for _, r in ks_results.iterrows():
            i, j = g_idx[r["g1"]], g_idx[r["g2"]]
            K[i, j] = K[j, i] = float(r["KS_statistic"]) if pd.notna(r["KS_statistic"]) else 0.0
            P[i, j] = P[j, i] = float(r["p_value"]) if pd.notna(r["p_value"]) else 1.0

        order = list(range(n))
        if cluster and n >= 3:
            try:
                import scipy.cluster.hierarchy as sch
                # use 1-K as distance (bounded [0,1])
                D = 1 - K
                # avoid zeros on diagonal issues
                D[np.arange(n), np.arange(n)] = 0
                Z = sch.linkage(D, method="average")
                order = sch.leaves_list(Z).tolist()
                K = K[order][:, order]
                P = P[order][:, order]
                groups = [groups[i] for i in order]
            except Exception:
                pass

        # FDR-BH on lower triangle
        try:
            from statsmodels.stats.multitest import multipletests
            tril = np.tril_indices(n, k=-1)
            rej, qvals, *_ = multipletests(P[tril], alpha=fdr_alpha, method="fdr_bh")
            Q = np.ones_like(P)
            Q[tril] = qvals
            Q = Q + Q.T
        except Exception:
            Q = P.copy()

        # Build text annotations (• for significant)
        text = [["" for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    if Q[i, j] < fdr_alpha:
                        text[i][j] = "•"
                except Exception:
                    pass

        heat = go.Heatmap(
            z=K,
            x=groups,
            y=groups,
            colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="KS statistic")
        )
        fig.add_trace(heat)

        # overlay text
        fig.add_trace(go.Scatter(
            x=[groups[j] for i in range(n) for j in range(n)],
            y=[groups[i] for i in range(n) for j in range(n)],
            mode="text",
            text=[text[i][j] for i in range(n) for j in range(n)],
            textfont=dict(color="white", size=14),
            hoverinfo="skip",
            showlegend=False
        ))

        fig.update_xaxes(tickangle=45)
        _plotly_pub_layout(fig, title=title)
        return fig

    # ---- statistical tests ----
    def run_ks_tests(
        self,
        df: pd.DataFrame,
        group_col: str,
        comparisons: Optional[List[Tuple[str, str]]] = None,
        value_col: str = "antibiotics_tested_count"
    ) -> pd.DataFrame:
        unique_groups = [g for g in df[group_col].dropna().unique().tolist()]
        if comparisons is None:
            comparisons = list(itertools.combinations(unique_groups, 2))

        rows = []
        for g1, g2 in comparisons:
            v1 = df.loc[df[group_col] == g1, value_col].dropna().values
            v2 = df.loc[df[group_col] == g2, value_col].dropna().values
            if len(v1) >= 2 and len(v2) >= 2 and (np.std(v1) > 0 or np.std(v2) > 0):
                stat, p = ks_2samp(v1, v2)
                rows.append(dict(
                    Comparison=f"{g1} vs {g2}",
                    KS_statistic=float(stat),
                    p_value=float(p),
                    n_group1=int(len(v1)),
                    n_group2=int(len(v2)),
                ))
        return pd.DataFrame(rows)

    def run_kruskal_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str = "antibiotics_tested_count"
    ) -> Dict:
        groups, labels = [], []
        for g, sub in df.groupby(group_col, observed=True):
            vals = sub[value_col].dropna().values
            if len(vals) > 1:
                groups.append(vals); labels.append(g)

        if len(groups) < 2:
            return {"H": np.nan, "p_value": np.nan, "n_groups": len(groups), "groups": labels}

        H, p = kruskal(*groups)
        out = {"H": float(H), "p_value": float(p), "p_star": _pstar(p), "n_groups": len(groups), "groups": labels}

        # optional: Dunn post-hoc with FDR if installed
        try:
            import scikit_posthocs as sp
            d = df[[group_col, value_col]].dropna()
            out["dunn_qvalues"] = sp.posthoc_dunn(d, val_col=value_col, group_col=group_col, p_adjust="fdr_bh")
        except Exception:
            pass
        return out

    # ---- orchestrator ----
    def analyze_stratum(self, stratum_config: Dict, create_ks_heatmap: bool = True):
        """
        Runs the full analysis for a stratum and returns Plotly figures.
        """
        filtered = self.filter_data(
            filter_dict=stratum_config.get("filter_dict"),
            pathogen_genus=stratum_config.get("pathogen_genus")
        )
        gcol = stratum_config["group_col"]
        name = stratum_config.get("name", gcol)

        # Optional deterministic order
        if "group_order" in stratum_config and stratum_config["group_order"]:
            filtered = filtered.copy()
            filtered[gcol] = pd.Categorical(
                filtered[gcol],
                categories=list(stratum_config["group_order"]),
                ordered=True
            )
            # Drop unused levels to prevent empty-group aggregations
            filtered[gcol] = filtered[gcol].cat.remove_unused_categories()
            self.logger.info(
                f"{gcol} present levels: {list(filtered[gcol].cat.categories)}"
            )

        summary = self.compute_summary_stats(filtered, gcol)
        self.logger.info(f"Summary statistics for {name}:\n{summary}")

        cm = stratum_config.get("color_map")
        ecdf_dash = stratum_config.get("ecdf_dash", "dot")
        kde_w = stratum_config.get("kde_width", 2.3)
        ecdf_w = stratum_config.get("ecdf_width", 1.4)

        fig_density = self.fig_density_ecdf(
            filtered, gcol,
            value_col="antibiotics_tested_count",
            title=f"Testing Breadth — {name} (KDE + ECDF)",
            color_map=cm, ecdf_dash=ecdf_dash, kde_width=kde_w, ecdf_width=ecdf_w
        )
        fig_violin = self.fig_violin_box(
            filtered, gcol,
            value_col="antibiotics_tested_count",
            title=f"Testing Breadth — {name} (Violin + Box)",
            color_map=cm
        )
        fig_lorenz = self.fig_lorenz_by_group(
            filtered, gcol,
            value_col="antibiotics_tested_count",
            title=f"Lorenz — {name} (Panel breadth inequality)",
            color_map=cm
        )

        ks_df = self.run_ks_tests(filtered, gcol, stratum_config.get("comparisons"))
        self.logger.info(f"KS test results for {name}:\n{ks_df}")

        kr = self.run_kruskal_test(filtered, gcol)
        self.logger.info(f"Kruskal-Wallis result for {name}: {kr}")

        fig_ks = None
        if create_ks_heatmap and not ks_df.empty:
            fig_ks = self.fig_ks_heatmap(
                ks_df,
                title=f"Pairwise {name} — KS statistic (• FDR<0.05)"
            )

        return {
            "summary": summary,
            "density_fig": fig_density,
            "violin_fig": fig_violin,
            "lorenz_fig": fig_lorenz,
            "ks_results": ks_df,
            "ks_heatmap_fig": fig_ks,
            "kruskal_result": kr,
            "filtered_df": filtered,
        }

    def analyze_priority_alignment(self, tier_col: str = "PathogenPriority"):
        """
        WHO Priority-Tier Diagnostic Alignment shortcut for Use Case 1.
        """
        cfg = {
            "name": "WHO Priority Tiers",
            "group_col": tier_col,
            "group_order": ["Critical", "High", "Medium", "Other"],
            "color_map": {
                "Critical": "#D62728",
                "High":     "#1F77B4",
                "Medium":   "#2CA02C",
                "Other":    "#7F7F7F"
            },
            "ecdf_dash": "dot",
            "kde_width": 2.3,
            "ecdf_width": 1.4
        }
        return self.analyze_stratum(cfg)

    # ---- export ----
    def save_plotly(
        self,
        fig: go.Figure,
        basepath: Path | str,
        *,
        width: int = 1400,
        height: int = 800,
        scale: int = 4,
        write_svg: bool = True,
        write_pdf: bool = False
    ) -> Dict[str, str]:
        """
        Save HTML (always) + PNG (+ optional SVG/PDF) via kaleido.
        """
        base = Path(basepath)
        _ensure_dir(base)
        out: Dict[str, str] = {}

        # HTML
        html_p = base.with_suffix(".html")
        html_p.write_text(fig.to_html(full_html=False, include_plotlyjs="cdn"), encoding="utf-8")
        out["html"] = str(html_p)

        # Static
        try:
            fig.write_image(str(base.with_suffix(".png")), width=width, height=height, scale=scale)
            out["png"] = str(base.with_suffix(".png"))
            if write_svg:
                fig.write_image(str(base.with_suffix(".svg")), width=width, height=height, scale=1)
                out["svg"] = str(base.with_suffix(".svg"))
            if write_pdf:
                fig.write_image(str(base.with_suffix(".pdf")), width=width, height=height, scale=1)
                out["pdf"] = str(base.with_suffix(".pdf"))
        except Exception as e:
            self.logger.warning(f"Static export failed for {base}: {e}")

        return out
