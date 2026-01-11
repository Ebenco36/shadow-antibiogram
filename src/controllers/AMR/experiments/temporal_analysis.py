"""
Temporal Trends Analysis Module

A comprehensive toolkit for analyzing temporal patterns in epidemiological data,
featuring trend detection, seasonal decomposition, and publication-quality visualizations.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional analytics (auto-fallback if not installed)
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose as _sm_seasonal_decompose
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

try:
    import pymannkendall as pmk
    _HAS_PMK = True
except ImportError:
    _HAS_PMK = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_STATSMODELS_DIAG = True
except ImportError:
    _HAS_STATSMODELS_DIAG = False


# =============================================================================
# STYLING CONSTANTS
# =============================================================================

_STYLE = {
    "observed": dict(width=3, dash=None, color="#1f77b4"),      # blue
    "ma": dict(width=2, dash="dot", color="#1f77b4"),           # blue, dotted
    "trend": dict(width=3, dash=None, color="#d62728"),         # red
    "seasonal": dict(width=2, dash="dot", color="#2ca02c"),     # green
    "residual": dict(width=2, dash="dashdot", color="#ff7f0e"), # orange
    "shocks": dict(size=10, symbol="diamond-open", color="#111111", line=dict(width=2)),
    "break_band": dict(fillcolor="rgba(0,0,0,0.08)"),
    "break_line": dict(width=1.5, dash="dash", color="#000000"),
}


import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# GLOBAL FIXED COLORS
# -----------------------------
PARTICIPATION_COLORS = {
    "Continuous":   "#0072B2",  # blue
    "Entering":     "#E69F00",  # orange
    "Single Year":  "#009E73",  # bluish green
    "Leaving":      "#D55E00",  # vermilion
    "Intermittent": "#CC79A7",  # reddish purple
}

def has_organisation_column(
    df: pd.DataFrame,
    org_col: str = "NumberOrganisation"
) -> bool:
    return org_col in df.columns and df[org_col].notna().any()


def _style_pub_layout(
    fig: go.Figure,
    title: str,
    y_title: str,
    x_title: str = "Year",
) -> go.Figure:
    """Publication-style layout with centered title and bottom legend."""

    fig.update_layout(
        template="plotly_white",

        # ---- Title ----
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=32, family="Arial"),
            pad=dict(b=40),
        ),

        # ---- Axes ----
        xaxis=dict(
            title=x_title,
            title_font=dict(size=26),
            tickfont=dict(size=22),
            type="category",
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(size=26),
            tickfont=dict(size=22),
        ),

        # ---- Legend ----
        legend=dict(
            title="Participation Type",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            font=dict(size=22),
            bgcolor="rgba(255,255,255,0.6)",
        ),

        # ---- Margins ----
        margin=dict(l=90, r=60, t=100, b=180),
    )

    return fig


def style_time_axes(
    fig: go.Figure,
    *,
    dates: pd.Series,
    x_title: str = "Month",
    y_title: str = "Isolate Count",
    mode: str = "ticks",   # "ticks" (OLD) or "months+years" (NEW)
    month_dtick: str = "M1",
    month_angle: int = -45,
    month_font_size: int = 20,
    title_font_size: int = 22,
    year_font_size: int = 18,
    year_y: float = -0.28,
    bottom_margin_ticks: int = 90,
    bottom_margin_months_years: int = 190,
):
    """
    Style time axes with a switch:
      - mode="ticks": month+year in tick labels (old behavior)
      - mode="months+years": months as ticks (slanted), years as annotations (horizontal)
    """

    # ------------------------------------------------------------------
    # Remove previously-added year annotations (safe re-run)
    # ------------------------------------------------------------------
    if fig.layout.annotations:
        fig.layout.annotations = tuple(
            a for a in fig.layout.annotations
            if not (isinstance(a.text, str) and a.text.isdigit() and len(a.text) == 4)
        )

    # ------------------------------------------------------------------
    # X axis
    # ------------------------------------------------------------------
    if mode == "ticks":
        fig.update_xaxes(
            title_text=x_title,
            dtick=month_dtick,
            tickformat="%b\n%Y",
            tickangle=month_angle,
            showgrid=True,
            ticks="outside",
            automargin=True,
            title_font=dict(size=title_font_size),
            tickfont=dict(size=month_font_size),
        )
        fig.update_layout(margin=dict(b=bottom_margin_ticks))

    elif mode == "months+years":
        fig.update_xaxes(
            title_text=x_title,
            dtick=month_dtick,
            tickformat="%b",
            tickangle=month_angle,
            showgrid=True,
            ticks="outside",
            automargin=True,
            title_font=dict(size=title_font_size),
            tickfont=dict(size=month_font_size),
        )

        # Add years as horizontal annotations
        d = pd.to_datetime(pd.Series(dates).dropna())
        years = pd.date_range(d.min(), d.max(), freq="YS")
        for y in years:
            fig.add_annotation(
                x=y, xref="x",
                y=year_y, yref="paper",
                text=y.strftime("%Y"),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=year_font_size),
            )

        fig.update_layout(margin=dict(b=bottom_margin_months_years))

    else:
        raise ValueError("mode must be 'ticks' or 'months+years'")

    # ------------------------------------------------------------------
    # Y axis (same for both modes)
    # ------------------------------------------------------------------
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        ticks="outside",
        title_font=dict(size=title_font_size),
        tickfont=dict(size=month_font_size),
    )

    return fig


def style_participation_bar(fig: go.Figure, y_title: str) -> go.Figure:
    """Apply consistent, publication-quality styling to participation bar charts."""

    fig.update_traces(
        marker=dict(line=dict(width=0)),
        opacity=0.95,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=24),                   
        title=dict(
            font=dict(size=32),
            x=0.5,               # <-- center title horizontally
            xanchor="center",
            yanchor="top",
            pad=dict(b=40),
        ),
        legend=dict(
            font=dict(size=22),
            title_text="Participation Type",
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=80, r=80, t=120, b=180),
        barmode="stack",
    )

    fig.update_xaxes(
        title_text="Year",
        tickfont=dict(size=22),
        title_font=dict(size=26),
        type="category",
    )

    fig.update_yaxes(
        title_text=y_title,
        tickfont=dict(size=22),
        title_font=dict(size=26),
        tickformat=",",
    )

    return fig


def create_participation_bar_chart(
    df,
    x: str,
    y: str,
    title: str,
    y_title: str,
) -> go.Figure:
    """
    Generic stacked bar chart for participation analysis.
    """
    
    fig = px.bar(
        df,
        x=x,
        y=y,
        color="ParticipationType",
        color_discrete_map=PARTICIPATION_COLORS,   # <-- custom fixed colors
        title=title,
    )

    fig = style_participation_bar(fig, y_title=y_title)

    # Explanation text with large readable fonts
    # participation_explainer = (
    #     "<b>Participation types</b><br>"
    #     "• <b>continuous</b> – present in all study years<br>"
    #     "• <b>entering</b> – first year of reporting<br>"
    #     "• <b>leaving</b> – last year of reporting<br>"
    #     "• <b>single_year</b> – reported only in one year"
    # )

    # fig.add_annotation(
    #     xref="paper", yref="paper",
    #     x=0.01, y=1.12,                       # above plot, aligned left
    #     xanchor="left", yanchor="bottom",
    #     # text=participation_explainer,
    #     showarrow=False,
    #     align="left",
    #     bgcolor="white",
    #     bordercolor="black",
    #     borderwidth=1.2,
    #     opacity=0.95,
    #     font=dict(size=20),                   # large annotation font
    # )

    return fig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_float(value: float, decimals: int = 2, default: str = "NA") -> str:
    """Format floats robustly, returning default for NaNs/non-numerics."""
    try:
        if pd.notna(value) and value == value:  # NaN check
            return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        pass
    return default


def build_summary_text(summary: Dict[str, Any]) -> str:
    """
    Build a tidy HTML snippet for summary annotation.
    
    Args:
        summary: Dictionary containing analysis results
        
    Returns:
        Formatted HTML string for annotation
    """
    def safe_float(x: Any) -> float:
        """Safely convert to float, return NaN on failure."""
        try:
            return float(x) if pd.notna(x) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def extract_amplitude(data: Dict[str, Any]) -> float:
        """Extract seasonal amplitude from multiple possible field names."""
        # Direct amplitude fields
        for key in ("seasonal_amplitude", "seasonality_amp", "seas_amp", "amplitude"):
            if key in data:
                val = safe_float(data[key])
                if not np.isnan(val):
                    return val

        # Derive from seasonal month means
        monthly_means = data.get("seasonal_month_means")
        if isinstance(monthly_means, dict) and monthly_means:
            try:
                values = pd.to_numeric(
                    pd.Series(list(monthly_means.values())),
                    errors="coerce"
                ).dropna()
                if len(values) > 0:
                    return float(values.max() - values.min())
            except (ValueError, TypeError):
                pass

        return np.nan

    # Build summary lines
    lines = [
        f"<b>Period</b>: {summary.get('period_start', '?')}–{summary.get('period_end', '?')}",
        f"<b>Mean</b> = {format_float(summary.get('mean_count'), 0)}/mo, "
        f"<b>SD</b> = {format_float(summary.get('sd_count'), 0)}",
        f"<b>MK</b>: τ={format_float(summary.get('mk_tau'))}, "
        f"p={format_float(summary.get('mk_p'), 3)}, "
        f"trend={summary.get('mk_trend', 'NA')}",
    ]

    # Add seasonal MK if available
    if any(k in summary for k in ["mk_seasonal_tau", "mk_seasonal_p", "mk_seasonal_trend"]):
        lines.append(
            f"<b>Seasonal MK</b>: τ={format_float(summary.get('mk_seasonal_tau'))}, "
            f"p={format_float(summary.get('mk_seasonal_p'), 3)}, "
            f"trend={summary.get('mk_seasonal_trend', 'NA')}"
        )

    # Add amplitude and shocks
    amplitude = extract_amplitude(summary)
    shocks = summary.get('n_shock_months', 0)
    try:
        shocks = int(shocks)
    except (ValueError, TypeError):
        shocks = 0

    lines.append(
        f"<b>Amplitude</b> ≈ {format_float(amplitude, 0)}, "
        f"<b>Shocks</b> = {shocks}"
    )
    
    return "<br>".join(lines)


def build_multi_summary_text(summaries: List[Dict[str, Any]]) -> str:
    """Render a compact multi-group summary block."""
    if not summaries:
        return "No summary available."
    
    parts = []
    first = summaries[0]
    
    # Header with period information
    header_parts = []
    start, end = first.get("period_start"), first.get("period_end")
    if start or end:
        header_parts.append(f"<b>Period</b>: {start or '?'}–{end or '?'}")
    if "months" in first and pd.notna(first["months"]):
        try:
            header_parts.append(f"({int(first['months'])} months)")
        except (TypeError, ValueError):
            pass
    
    if header_parts:
        parts.append("<br>".join(header_parts))

    # One line per group
    for summary in summaries:
        group = summary.get("group", "?")
        line_parts = [
            f"<b>{group}</b>: MK τ={format_float(summary.get('mk_tau'))}, "
            f"p={format_float(summary.get('mk_p'), 3)}, {summary.get('mk_trend', 'NA')}"
        ]
        
        # Optional amplitude
        amplitude = summary.get("seasonal_amplitude")
        if isinstance(amplitude, (int, float)) and not np.isnan(amplitude):
            line_parts.append(f"Amp≈{format_float(amplitude, 0)}")
            
        # Optional model info
        if "model" in summary and "period" in summary:
            line_parts.append(f"(model={summary['model']}, P={summary['period']})")
            
        parts.append("; ".join(line_parts))

    return "<br>".join(parts)


def add_annotation_safely(
    fig: go.Figure, 
    text: str, 
    **kwargs
) -> None:
    """
    Add annotation only if an equivalent one isn't already present.
    
    Args:
        fig: Plotly figure to modify
        text: Annotation text content
        **kwargs: Additional annotation parameters
    """
    annotations = fig.layout.annotations or []

    def annotation_to_dict(ann: Any) -> Dict[str, Any]:
        """Convert annotation to dictionary for comparison."""
        try:
            return ann if isinstance(ann, dict) else ann.to_plotly_json()
        except AttributeError:
            return {
                "text": getattr(ann, "text", None),
                "xref": getattr(ann, "xref", None),
                "yref": getattr(ann, "yref", None),
                "x": getattr(ann, "x", None),
                "y": getattr(ann, "y", None),
            }

    target_xref = kwargs.get("xref")
    target_yref = kwargs.get("yref")

    # Check for existing equivalent annotation
    for annotation in annotations:
        ann_dict = annotation_to_dict(annotation)
        if (
            ann_dict.get("text") == text and
            ann_dict.get("xref") == target_xref and
            ann_dict.get("yref") == target_yref
        ):
            return  # Skip duplicate

    fig.add_annotation(text=text, **kwargs)


def add_summary_box(
    fig: go.Figure, 
    summary: Union[Dict[str, Any], List[Dict[str, Any]]],
    small: bool = False,
    x: float = 0.99,
    y: float = 0.59
) -> None:
    """Add a summary annotation box in paper coordinates."""
    fig.update_layout(autosize=True, width=None)
    
    if isinstance(summary, list):
        text = build_multi_summary_text(summary)
    else:
        text = build_summary_text(summary)
        
    add_annotation_safely(
        fig,
        text=text,
        xref="paper", yref="paper",
        x=x, y=y, xanchor="right", yanchor="top",
        showarrow=False, align="left",
        bordercolor="black", borderwidth=1,
        bgcolor="white", opacity=0.7,
        font=dict(size=18 if small else 18),
    )


def compute_series_summary(
    dates: pd.Series, 
    values: Union[pd.Series, np.ndarray, List], 
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute robust summary statistics for a time series.
    
    Args:
        dates: Series of dates
        values: Series, array, or list of values
        extra: Additional fields to include
        
    Returns:
        Dictionary of summary statistics
    """
    # Convert to datetime safely
    try:
        date_index = pd.to_datetime(dates, errors="coerce")
        valid_dates = date_index[date_index.notna()]
    except (ValueError, TypeError):
        valid_dates = pd.DatetimeIndex([])

    # Period information
    if len(valid_dates) > 0:
        start_date = valid_dates.min().strftime("%Y-%m")
        end_date = valid_dates.max().strftime("%Y-%m")
        try:
            months = len(pd.date_range(
                start=valid_dates.min(), 
                end=valid_dates.max(), 
                freq='ME'
            ))
        except (ValueError, AttributeError):
            months = len({f"{d.year}-{d.month:02d}" for d in valid_dates if pd.notna(d)})
    else:
        start_date, end_date, months = "?", "?", 0

    # Numeric statistics
    if isinstance(values, (np.ndarray, list)):
        values_series = pd.Series(values)
    else:
        values_series = values
        
    numeric_values = pd.to_numeric(values_series, errors="coerce").dropna()
    mean_val = float(numeric_values.mean()) if len(numeric_values) > 0 else np.nan
    std_val = float(numeric_values.std(ddof=1)) if len(numeric_values) > 1 else np.nan

    summary = {
        "period_start": start_date,
        "period_end": end_date,
        "months": months,
        "mean_count": mean_val,
        "sd_count": std_val,
    }
    
    if extra:
        summary.update(extra)
        
    return summary


def run_mann_kendall_test(series: pd.Series) -> Dict[str, Any]:
    """
    Perform robust Mann-Kendall trend test.
    
    Args:
        series: Input time series
        
    Returns:
        Dictionary with test results
    """
    try:
        clean_series = pd.to_numeric(series, errors='coerce').dropna()
    except (ValueError, TypeError):
        return {"Tau": np.nan, "p": np.nan, "trend": "NA"}

    # Check for sufficient data
    if len(clean_series) < 5 or clean_series.nunique() <= 1:
        return {"Tau": np.nan, "p": np.nan, "trend": "no trend"}

    if not _HAS_PMK:
        # Fallback to scipy if pymannkendall not available
        try:
            from scipy.stats import kendalltau
            tau, p_value = kendalltau(
                np.arange(len(clean_series)),
                clean_series.values, 
                nan_policy="omit"
            )
            trend = ("increasing" if (tau or 0) > 0 else 
                     "decreasing" if (tau or 0) < 0 else "no trend")
            return {"Tau": float(tau), "p": float(p_value), "trend": trend}
        except ImportError:
            return {"Tau": np.nan, "p": np.nan, "trend": "NA"}

    try:
        # Prefer modified test if available
        modified_test = getattr(pmk, "hamed_rao_modification_test", None)
        if modified_test:
            result = modified_test(clean_series)
        else:
            result = pmk.original_test(clean_series)
            
        return {
            "Tau": float(result.Tau),
            "p": float(result.p),
            "trend": str(result.trend)
        }
    except (ValueError, TypeError, AttributeError):
        # Fallback to original test
        try:
            result = pmk.original_test(clean_series)
            return {
                "Tau": float(result.Tau),
                "p": float(result.p),
                "trend": str(result.trend)
            }
        except (ValueError, TypeError, AttributeError):
            return {"Tau": np.nan, "p": np.nan, "trend": "NA"}


def apply_publication_layout(
    fig: go.Figure,
    bottom_margin: int = 150,
    title_font_size: int = 26,
    title_bottom_padding: int = 50,
) -> go.Figure:
    """
    Apply publication-quality layout to figure, with centered title
    and extra padding below the title so nothing overlaps it.
    
    Args:
        fig: Plotly figure to format
        bottom_margin: Bottom margin in pixels
        title_font_size: Title font size
        title_bottom_padding: Extra padding below title (px)
        
    Returns:
        Formatted figure
    """
    # Get existing title text, if any
    title_text = ""
    if fig.layout.title and getattr(fig.layout.title, "text", None):
        title_text = fig.layout.title.text

    fig.update_layout(
        autosize=True,
        width=None,  # Responsive
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80 + title_bottom_padding, b=bottom_margin),
        legend=dict(orientation="h", title=None),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=title_font_size),
            pad=dict(b=title_bottom_padding),
        ),
    )
    return fig


def arrange_legend(
    fig: go.Figure,
    columns: int = 5,
    item_width: int = 80,
    outside: bool = True,
    font_size: int = 20,
    base_padding: int = 28,
    bottom_gap: int = 36,
    outside_y: float = -0.25,
    adjust_width: bool = False
) -> go.Figure:
    """
    Arrange legend in columns with proper spacing.
    """
    # Count legend items
    n_items = sum(1 for trace in fig.data if getattr(trace, "showlegend", True))
    columns = max(1, columns)
    rows = max(1, math.ceil(n_items / columns))

    # Position legend
    if outside:
        y_pos, y_anchor = outside_y, "top"
    else:
        y_pos, y_anchor = 0.0, "bottom"

    fig.update_layout(
        autosize=True,
        width=None,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=y_pos, yanchor=y_anchor,
            itemwidth=item_width,
            font=dict(size=font_size),
            title=None
        )
    )

    # Adjust margins
    margin = fig.layout.margin or go.layout.Margin()
    left = margin.l or 50
    right = margin.r or 30
    top = margin.t or 80
    bottom = margin.b or 80

    if outside:
        padding = base_padding + rows * (font_size + 8) + bottom_gap
        bottom = max(bottom, padding)

    fig.update_layout(margin=dict(l=left, r=right, t=top, b=bottom))

    # Optional width adjustment
    if adjust_width:
        current_width = fig.layout.width or 0
        target_width = columns * item_width + 220
        if not current_width or current_width < target_width:
            fig.update_layout(width=target_width)

    return fig


def add_breakpoint_annotations(
    fig: go.Figure,
    breakpoints: List[pd.Timestamp],
    label_prefix: str = "Break",
    # put labels near top of plot area instead of above it
    y_levels: Tuple[float, ...] = (0.58, 0.54, 0.50),
) -> None:
    """
    Add breakpoint annotations with vertical bands and labels.

    Args:
        fig: Plotly figure to modify
        breakpoints: List of breakpoint timestamps
        label_prefix: Prefix for breakpoint labels
        y_levels: Y positions for labels (paper coordinates, 0–1 inside plot)
    """
    if not breakpoints:
        return

    unique_breaks = sorted({pd.Timestamp(bp).normalize() for bp in breakpoints})

    def next_month_start(dt: pd.Timestamp) -> pd.Timestamp:
        """Get start of next month."""
        return dt + pd.offsets.MonthBegin(1)

    for i, breakpoint in enumerate(unique_breaks, 1):
        break_start = breakpoint
        break_end = next_month_start(breakpoint)
        
        # vertical band stays as is
        fig.add_vrect(
            x0=break_start, x1=break_end,
            fillcolor=_STYLE["break_band"]["fillcolor"],
            opacity=1.0,
            line_width=0,
            layer="below",
        )

        break_style = _STYLE["break_line"]
        fig.add_vline(
            x=break_start,
            line_width=break_style.get("width", 1.5),
            line_dash=break_style.get("dash", "dash"),
            line_color=break_style.get("color", "#000000"),
        )

        # label now within plot (below title)
        y_label = y_levels[(i - 1) % len(y_levels)]
        add_annotation_safely(
            fig,
            x=break_start, xref="x",
            y=y_label, yref="paper",
            text=f"{label_prefix} {i}: {breakpoint.strftime('%Y-%m')}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black", borderwidth=1,
            opacity=0.9, font=dict(size=14)
        )


def _infer_year_range(
    df: pd.DataFrame, 
    date_col: str, 
    min_year: Optional[int], 
    max_year: Optional[int]
) -> Tuple[int, int]:
    """Infer [min_year, max_year] from data if not provided."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    years = dates.dt.year.dropna().astype(int)
    if len(years) == 0:
        raise ValueError("No valid dates found to infer year range.")

    if min_year is None:
        min_year = int(years.min())
    if max_year is None:
        max_year = int(years.max())

    return min_year, max_year



# =============================================================================
# PARTICIPATION / ORGANISATION HELPERS
# =============================================================================

def get_continuous_organisations(
    df: pd.DataFrame,
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: int = 2019,
    max_year: int = 2023,
) -> List[str]:
    """
    Identify organisations that participate in *all* years between min_year and max_year.
    One row = one isolate.
    """
    if date_col not in df.columns or org_col not in df.columns:
        raise ValueError(f"Columns '{date_col}' and/or '{org_col}' not found in dataframe.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.dropna(subset=[org_col])
    
    min_year, max_year = _infer_year_range(df, date_col, min_year, max_year)

    df["Year"] = df[date_col].dt.year
    mask = (df["Year"] >= min_year) & (df["Year"] <= max_year)
    df = df.loc[mask]

    # For each organisation, which years does it appear in?
    org_years = df.groupby(org_col)["Year"].agg(lambda x: set(x.unique()))

    full_span = set(range(min_year, max_year + 1))
    continuous_orgs = org_years[org_years.apply(lambda s: full_span.issubset(s))].index.tolist()

    return continuous_orgs


def label_participation_types(
    df: pd.DataFrame,
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: int = 2019,
    max_year: int = 2023,
) -> pd.DataFrame:
    """
    Add a column 'ParticipationType' to df based on organisation's behaviour:
      - 'continuous'   : present in all years [min_year..max_year]
      - 'entering'     : first appear in this year, continue afterwards
      - 'leaving'      : last appear in this year, appeared before
      - 'single_year'  : only appear in that year
      - 'intermittent' : appear in multiple years but not all, and not first/last year of participation
    """
    if date_col not in df.columns or org_col not in df.columns:
        raise ValueError(f"Columns '{date_col}' and/or '{org_col}' not found in dataframe.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["Year"] = df[date_col].dt.year
    mask = (df["Year"] >= min_year) & (df["Year"] <= max_year)
    df = df.loc[mask]

    df = df.dropna(subset=[org_col])

    # Per-organisation participation interval
    org_year_stats = df.groupby(org_col)["Year"].agg(["min", "max", "nunique"]).rename(
        columns={"min": "first_year", "max": "last_year", "nunique": "n_years"}
    )

    full_span = set(range(min_year, max_year + 1))

    # Precompute continuous orgs
    org_year_sets = df.groupby(org_col)["Year"].agg(lambda x: set(x.unique()))
    continuous_orgs = org_year_sets[org_year_sets.apply(lambda s: full_span.issubset(s))].index

    # Map org -> type for each year it contributes
    def classify_row(row):
        org = row[org_col]
        year = row["Year"]
        stats = org_year_stats.loc[org]

        if org in continuous_orgs:
            return "Continuous"

        first_y = stats["first_year"]
        last_y = stats["last_year"]
        n_years = stats["n_years"]

        if n_years == 1:
            return "Single Year"
        if year == first_y and year < last_y:
            return "Entering"
        if year == last_y and year > first_y:
            return "Leaving"
        return "Intermittent"

    df["ParticipationType"] = df.apply(classify_row, axis=1)
    return df


def compute_participation_decomposition(
    df: pd.DataFrame,
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: int = 2019,
    max_year: int = 2023,
) -> pd.DataFrame:
    """
    Return a table with isolate counts by Year × ParticipationType.
    """
    df_labeled = label_participation_types(
        df, date_col=date_col, org_col=org_col,
        min_year=min_year, max_year=max_year
    )

    decomp = (
        df_labeled
        .groupby(["Year", "ParticipationType"])
        .size()
        .reset_index(name="IsolateCount")
        .sort_values(["Year", "ParticipationType"])
    )

    return decomp


def compute_participation_org_counts(
    df: pd.DataFrame,
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: int = 2019,
    max_year: int = 2023,
) -> pd.DataFrame:
    """
    Number of distinct organisations by Year × ParticipationType.
    """
    df_labeled = label_participation_types(
        df, date_col=date_col, org_col=org_col,
        min_year=min_year, max_year=max_year
    )

    org_counts = (
        df_labeled
        .groupby(["Year", "ParticipationType"])[org_col]
        .nunique()
        .reset_index(name="OrganisationCount")
        .sort_values(["Year", "ParticipationType"])
    )

    return org_counts


def run_temporal_with_participation_layers(
    df: pd.DataFrame,
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: int = 2019,
    max_year: int = 2023,
):
    """
    Run analyses with a safe fallback when org_col is missing.

    If org_col exists and has non-null values:
      1) Full national series (all organisations)
      2) Continuous participants only
      3) Participation decomposition (counts by type)
    Else:
      1) Full national series only
    """

    # --- 0) Check org availability ---
    has_orgs = (org_col in df.columns) and df[org_col].notna().any()

    # --- 1) Full series (always) ---
    analyzer_full = TemporalTrendAnalyzer(
        df=df,
        date_column=date_col,
        title="Monthly Isolate Counts – All Organisations"
    )
    monthly_full, shocks_full, breaks_full, amp_full = analyzer_full.run_complete_analysis()

    if not has_orgs:
        # Fallback result: only full series is available
        return {
            "has_orgs": False,
            "full": {
                "analyzer": analyzer_full,
                "monthly": monthly_full,
                "shocks": shocks_full,
                "breaks": breaks_full,
                "amplitude": amp_full,
            },
            "continuous": None,
            "participation": None,
        }

    # --- 2) Continuous participants only (requires org_col) ---
    continuous_orgs = get_continuous_organisations(
        df, date_col=date_col, org_col=org_col,
        min_year=min_year, max_year=max_year
    )

    df_cont = df[df[org_col].isin(continuous_orgs)].copy()

    analyzer_cont = TemporalTrendAnalyzer(
        df=df_cont,
        date_column=date_col,
        title="Monthly Isolate Counts – Continuous Organisations Only"
    )
    monthly_cont, shocks_cont, breaks_cont, amp_cont = analyzer_cont.run_complete_analysis()

    # --- 3) Participation decomposition (requires org_col) ---
    decomp_counts = compute_participation_decomposition(
        df, date_col=date_col, org_col=org_col,
        min_year=min_year, max_year=max_year
    )
    decomp_orgs = compute_participation_org_counts(
        df, date_col=date_col, org_col=org_col,
        min_year=min_year, max_year=max_year
    )

    return {
        "has_orgs": True,
        "full": {
            "analyzer": analyzer_full,
            "monthly": monthly_full,
            "shocks": shocks_full,
            "breaks": breaks_full,
            "amplitude": amp_full,
        },
        "continuous": {
            "analyzer": analyzer_cont,
            "monthly": monthly_cont,
            "shocks": shocks_cont,
            "breaks": breaks_cont,
            "amplitude": amp_cont,
        },
        "participation": {
            "isolate_counts": decomp_counts,
            "organisation_counts": decomp_orgs,
        },
    }




def fig_ribbon_isolate_counts_by_type(decomp_counts: pd.DataFrame) -> go.Figure:
    """
    Stacked area ('ribbon') plot for total isolate counts by participation type per year.
    decomp_counts: output of compute_participation_decomposition
                   cols: ['Year', 'ParticipationType', 'IsolateCount']
    """
    df = decomp_counts.copy()
    df["Year"] = df["Year"].astype(int)

    fig = px.area(
        df,
        x="Year",
        y="IsolateCount",
        color="ParticipationType",
        color_discrete_map=PARTICIPATION_COLORS,
    )
    fig = _style_pub_layout(
        fig,
        title="Isolate counts by participation type – ribbon view",
        y_title="Isolate count",
    )
    return fig


def fig_ribbon_org_counts_by_type(decomp_orgs: pd.DataFrame) -> go.Figure:
    """
    Stacked area plot of number of distinct organisations by participation type per year.
    decomp_orgs: output of compute_participation_org_counts
                 cols: ['Year', 'ParticipationType', 'OrganisationCount']
    """
    df = decomp_orgs.copy()
    df["Year"] = df["Year"].astype(int)

    fig = px.area(
        df,
        x="Year",
        y="OrganisationCount",
        color="ParticipationType",
        color_discrete_map=PARTICIPATION_COLORS,
    )
    fig = _style_pub_layout(
        fig,
        title="Organisation counts by participation type – ribbon view",
        y_title="Organisation count",
    )
    return fig



def create_participation_line_chart(
    df,
    x="Year",
    y="OrganisationCount",
    title="Organisation counts by participation type (line chart)",
    y_title="Count",
):
    df = df.copy()
    df[x] = df[x].astype(str)

    fig = px.line(
        df,
        x=x,
        y=y,
        color="ParticipationType",
        color_discrete_map=PARTICIPATION_COLORS,
        markers=True,
        title=title
    )

    fig.update_traces(marker=dict(size=10), line=dict(width=4))

    fig.update_layout(
        template="plotly_white",
        font=dict(size=24),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=32),
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            font=dict(size=20),
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=80, r=80, t=120, b=180)
    )

    fig.update_xaxes(
        title_text="Year",
        tickfont=dict(size=22),
        title_font=dict(size=26),
        type="category"
    )

    fig.update_yaxes(
        title_text=y_title,
        tickfont=dict(size=22),
        title_font=dict(size=26),
        tickformat=",",
    )

    return fig



# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================

@dataclass
class TemporalTrendAnalyzer:
    """
    Analyzes temporal trends in epidemiological data.
    """
    
    # Configuration
    df: pd.DataFrame
    date_column: str = "Date"
    frequency: str = "MS"  # Month start
    rolling_window: int = 3
    shock_window: int = 12
    shock_threshold: float = 2.5
    title: str = "Monthly Isolate Volume"
    
    # Internal state
    _is_prepared: bool = field(init=False, default=False)
    _monthly_data: Optional[pd.DataFrame] = field(init=False, default=None)
    _shock_data: Optional[pd.DataFrame] = field(init=False, default=None)
    _breakpoints: Optional[List[pd.Timestamp]] = field(init=False, default=None)
    _seasonal_amplitude: Optional[float] = field(init=False, default=None)

    def prepare_data(self) -> "TemporalTrendAnalyzer":
        """
        Prepare and resample data to monthly frequency.
        """
        if self.date_column not in self.df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in dataframe")

        df = self.df.copy()
        
        # Parse and sort dates
        df[self.date_column] = pd.to_datetime(
            df[self.date_column], errors="coerce", utc=False
        )
        df = (df.dropna(subset=[self.date_column])
              .sort_values(self.date_column)
              .set_index(self.date_column))
        
        monthly = (df.resample(self.frequency)
                   .size()
                   .rename("Count")
                   .to_frame())
        
        # Ensure numeric and compute moving average
        monthly["Count"] = pd.to_numeric(monthly["Count"], errors='coerce').fillna(0)
        monthly["MA"] = monthly["Count"].rolling(
            self.rolling_window, min_periods=1
        ).mean()
        
        # Reset index for consistent handling
        monthly = monthly.reset_index().rename(columns={self.date_column: "Date"})
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)
        
        self._monthly_data = monthly
        self._is_prepared = True
        
        return self

    def detect_shocks(self) -> "TemporalTrendAnalyzer":
        """
        Detect unusual months using rolling z-scores.
        """
        self._check_prepared()
        
        monthly = self._monthly_data.copy()
        monthly["Count"] = pd.to_numeric(monthly["Count"], errors='coerce').fillna(0)
        
        # Compute rolling statistics
        min_periods = max(3, self.shock_window // 3)
        monthly["roll_mean"] = monthly["Count"].rolling(
            self.shock_window, min_periods=min_periods
        ).mean()
        monthly["roll_std"] = monthly["Count"].rolling(
            self.shock_window, min_periods=min_periods
        ).std()
        
        # Calculate z-scores
        with np.errstate(divide="ignore", invalid="ignore"):
            monthly["zscore"] = (
                (monthly["Count"] - monthly["roll_mean"]) / monthly["roll_std"]
            )
        
        shocks = monthly.loc[
            monthly["zscore"].abs() >= self.shock_threshold,
            ["Date", "Count", "zscore"]
        ].copy()
        shocks["Date"] = pd.to_datetime(shocks["Date"], utc=False)
        
        self._shock_data = shocks.reset_index(drop=True)
        
        return self

    def detect_structural_breaks(
        self, 
        max_breaks: int = 2, 
        min_segment_length: int = 3
    ) -> "TemporalTrendAnalyzer":
        """
        Detect structural breaks using PELT algorithm.
        """
        self._check_prepared()
        
        series = (self._monthly_data.set_index("Date")["Count"]
                  .astype(float)
                  .dropna())
        n_points = len(series)
        
        self._breakpoints = []
        
        if n_points < 2 * min_segment_length:
            return self  # Insufficient data

        if not _HAS_RUPTURES:
            warnings.warn("ruptures package not available for breakpoint detection")
            return self

        try:
            algorithm = rpt.Pelt(model="l2").fit(series.values.reshape(-1, 1))
            penalty = 3 * np.log(max(n_points, 2))
            raw_breaks = algorithm.predict(pen=penalty)
            
            # Filter breaks by segment length
            filtered_breaks = []
            last_break = 0
            
            for breakpoint in raw_breaks:
                if (breakpoint < n_points and 
                    breakpoint - last_break >= min_segment_length):
                    filtered_breaks.append(breakpoint)
                    last_break = breakpoint
            
            break_dates = [
                pd.Timestamp(series.index[bp - 1]) 
                for bp in filtered_breaks[:max_breaks]
            ]
            self._breakpoints = break_dates
            
        except Exception as e:
            warnings.warn(f"Breakpoint detection failed: {e}")
            self._breakpoints = []
            
        return self

    def estimate_seasonal_amplitude(self, period: int = 12) -> "TemporalTrendAnalyzer":
        """
        Estimate seasonal amplitude from decomposition.
        """
        self._check_prepared()
        
        series = (self._monthly_data.assign(
            Date=pd.to_datetime(self._monthly_data["Date"], utc=False)
        ).set_index("Date")["Count"].asfreq(self.frequency).astype(float))
        
        if len(series.dropna()) < max(6, period):
            self._seasonal_amplitude = np.nan
            return self

        try:
            seasonal_component = None
            
            if _HAS_SM:
                try:
                    decomposition = _sm_seasonal_decompose(
                        series, model="additive", period=period, 
                        extrapolate_trend="freq"
                    )
                    seasonal_component = decomposition.seasonal
                except (ValueError, TypeError):
                    seasonal_component = None
            
            # Fallback decomposition
            if seasonal_component is None:
                trend = series.rolling(
                    period, center=True, min_periods=max(2, period // 2)
                ).mean()
                seasonal_component = (series - trend).rolling(
                    period, center=True, min_periods=max(2, period // 2)
                ).mean()
            
            seasonal_values = pd.to_numeric(seasonal_component, errors="coerce").dropna()
            if len(seasonal_values) > 0 and seasonal_values.nunique() > 1:
                amplitude = float(seasonal_values.max() - seasonal_values.min())
            else:
                amplitude = np.nan
                
            self._seasonal_amplitude = amplitude
            
        except Exception as e:
            warnings.warn(f"Seasonal amplitude estimation failed: {e}")
            self._seasonal_amplitude = np.nan
            
        return self

    def _get_seasonal_amplitude(self, period: int = 12) -> float:
        """Get seasonal amplitude, computing if necessary."""
        amplitude = getattr(self, "_seasonal_amplitude", None)
        if amplitude is None or np.isnan(amplitude):
            self.estimate_seasonal_amplitude(period=period)
            amplitude = self._seasonal_amplitude
        return float(amplitude) if not np.isnan(amplitude) else np.nan

    def create_seasonal_decomposition_plot(
        self, 
        model: str = "additive", 
        period: int = 12, 
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create 4-panel seasonal decomposition plot.
        """
        self._check_prepared()
        
        # Prepare series
        series = (self._monthly_data.assign(
            Date=pd.to_datetime(self._monthly_data["Date"], utc=False)
        ).set_index("Date")["Count"].asfreq(self.frequency).astype(float))
        
        # Perform decomposition
        trend, seasonal, residual = self._decompose_series(series, model, period)
        
        plot_title = title or (
            f"Seasonal decomposition (period={period}, {model}) — {self.title}"
        )
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.06
        )
        
        # Observed + MA
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series, mode="lines", name="Observed",
                line=dict(
                    width=_STYLE["observed"]["width"], 
                    color=_STYLE["observed"]["color"]
                )
            ),
            row=1, col=1
        )
        
        moving_avg = series.rolling(self.rolling_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=moving_avg.index, y=moving_avg, mode="lines", 
                name=f"{self.rolling_window}M MA",
                line=dict(
                    width=_STYLE["ma"]["width"], 
                    dash=_STYLE["ma"]["dash"], 
                    color=_STYLE["ma"]["color"]
                )
            ),
            row=1, col=1
        )
        
        # Trend component
        fig.add_trace(
            go.Scatter(
                x=trend.index, y=trend, mode="lines", name="Trend",
                line=dict(
                    width=_STYLE["trend"]["width"], 
                    color=_STYLE["trend"]["color"]
                )
            ),
            row=2, col=1
        )
        
        # Seasonal component
        fig.add_trace(
            go.Scatter(
                x=seasonal.index, y=seasonal, mode="lines", name="Seasonal",
                line=dict(
                    width=_STYLE["seasonal"]["width"], 
                    dash=_STYLE["seasonal"]["dash"],
                    color=_STYLE["seasonal"]["color"]
                )
            ),
            row=3, col=1
        )
        
        # Residual component
        fig.add_trace(
            go.Scatter(
                x=residual.index, y=residual, mode="lines", name="Residual",
                line=dict(
                    width=_STYLE["residual"]["width"], 
                    dash=_STYLE["residual"]["dash"],
                    color=_STYLE["residual"]["color"]
                )
            ),
            row=4, col=1
        )
        
        # Basic layout, then apply publication layout for title/margins
        fig.update_layout(
            title=plot_title, 
            autosize=True, 
            width=None, 
            template="plotly_white",
            hovermode="x unified", 
            height=800,
            margin=dict(l=60, r=30, t=80, b=90),
            legend=dict(orientation="h", title=None)
        )
        
        # Axes
        fig.update_xaxes(
            dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside", tickangle=-45,
            title_font=dict(size=22),
            tickfont=dict(size=20),
        )
        fig.update_yaxes(
            showgrid=True, ticks="outside",
            title_font=dict(size=22),
            tickfont=dict(size=20),
        )

        # Center/spacing for title
        apply_publication_layout(fig, bottom_margin=90, title_font_size=26, title_bottom_padding=20)
        # style_time_axes(
        #     fig,
        #     dates=pd.Series(series.index),   # <-- use the index for year labels
        #     x_title="Month",
        #     y_title="",                      # decomposition panels usually don’t need a shared y title
        #     mode="months+years",             # NEW option; use "ticks" for old
        #     month_dtick="M1",
        #     month_angle=-45,
        #     year_font_size=18,
        #     year_y=-0.1,                    # a bit less low for subplots; tweak if needed
        #     bottom_margin_months_years=220,
        #     bottom_margin_ticks=90,
        # )
        arrange_legend(fig, columns=4, outside=True, adjust_width=False)
        
        # Add summary
        mk_results = run_mann_kendall_test(series)
        amplitude = self._get_seasonal_amplitude(period=period)
        
        summary = compute_series_summary(
            series.index, series.values,
            extra={
                "model": model,
                "period": period,
                "mk_tau": mk_results.get("Tau"),
                "mk_p": mk_results.get("p"),
                "mk_trend": mk_results.get("trend"),
                "seasonal_amplitude": amplitude,
            }
        )
        
        add_summary_box(fig, summary, small=True)
        
        return fig

    def create_main_plot(
        self,
        show_shocks: bool = True,
        show_moving_average: bool = True,
        show_breakpoints: bool = True
    ) -> go.Figure:
        """
        Create main temporal trends plot.
        """
        self._check_prepared()
        
        monthly = self._monthly_data.copy()
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)
        
        # Base plot
        fig = px.line(
            monthly, x="Date", y="Count",
            title=self.title,
            labels={"Date": "Month", "Count": "Isolates"}
        )
        
        # Style main trace
        fig.update_traces(
            mode="lines+markers",
            marker=dict(size=6, color=_STYLE["observed"]["color"]),
            line=dict(
                width=_STYLE["observed"]["width"], 
                color=_STYLE["observed"]["color"]
            ),
            name="Observed"
        )
        
        # Moving average
        if show_moving_average and "MA" in monthly.columns:
            fig.add_trace(go.Scatter(
                x=monthly["Date"], y=monthly["MA"], mode="lines",
                name=f"{self.rolling_window}M Moving Average",
                line=dict(
                    width=_STYLE["ma"]["width"], 
                    dash=_STYLE["ma"]["dash"], 
                    color=_STYLE["ma"]["color"]
                )
            ))
        
        # Shock points
        if (show_shocks and self._shock_data is not None and 
            len(self._shock_data) > 0):
            shocks = self._shock_data.copy()
            shocks["Date"] = pd.to_datetime(shocks["Date"], utc=False)
            
            fig.add_trace(go.Scatter(
                x=shocks["Date"], y=shocks["Count"],
                mode="markers", 
                name=f"Shocks (|z| ≥ {self.shock_threshold})",
                marker=_STYLE["shocks"]
            ))
        
        # Breakpoint annotations
        if show_breakpoints and self._breakpoints:
            add_breakpoint_annotations(
                fig, self._breakpoints, label_prefix="Break"
            )
        
        # Layout, axis fonts, and title spacing
        apply_publication_layout(fig, bottom_margin=90, title_font_size=26, title_bottom_padding=20)
        # fig.update_xaxes(
        #     dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside", tickangle=-45,
        #     title_font=dict(size=22),
        #     tickfont=dict(size=20),
        # )
        # fig.update_yaxes(
        #     showgrid=True, ticks="outside",
        #     title_font=dict(size=22),
        #     tickfont=dict(size=20),
        # )
        
        style_time_axes(
            fig,
            dates=monthly["Date"],
            x_title="Month",
            y_title="Isolates",
            mode="months+years",    # <-- NEW option (months slanted, years horizontal)
            month_dtick="M1",
            month_angle=-45,
            year_font_size=18,
            year_y=-0.1,
            bottom_margin_ticks=90,
            bottom_margin_months_years=220,
        )
        
        arrange_legend(fig, columns=4, outside=True, adjust_width=False)
        
        # Summary box
        series = pd.Series(
            monthly["Count"].values, 
            index=pd.to_datetime(monthly["Date"])
        )
        
        mk_results = run_mann_kendall_test(series)
        amplitude = self._get_seasonal_amplitude(period=12)
        shock_count = 0 if self._shock_data is None else len(self._shock_data)
        
        summary = compute_series_summary(
            monthly["Date"], monthly["Count"],
            extra={
                "mk_tau": mk_results.get("Tau"),
                "mk_p": mk_results.get("p"),
                "mk_trend": mk_results.get("trend"),
                "n_shock_months": shock_count,
                "seasonal_amplitude": amplitude,
            }
        )
        
        add_summary_box(fig, summary, small=False)
        
        return fig

    def create_comprehensive_plot(
        self,
        show_shocks: bool = True,
        show_moving_average: bool = True,
        show_breakpoints: bool = True,
        model: str = "additive",
        period: int = 12,
        offset_components: bool = False
    ) -> go.Figure:
        """
        Create comprehensive plot with all components.
        """
        self._check_prepared()
        
        monthly = self._monthly_data.copy()
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)
        
        # Prepare series for decomposition
        series = (monthly.set_index("Date")["Count"]
                  .asfreq(self.frequency)
                  .astype(float))
        
        # Decompose series
        trend, seasonal, residual = self._decompose_series(series, model, period)
        
        # Calculate offsets if requested
        if offset_components and not trend.isna().all():
            seasonal_offset = float(trend.mean())
            residual_offset = float(trend.mean() * 2.0)
        else:
            seasonal_offset = 0.0
            residual_offset = 0.0
        
        fig = go.Figure()
        
        # 1. Observed counts
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Count"], 
            mode="lines+markers", 
            name="Observed Count",
            line=dict(
                width=_STYLE["observed"]["width"], 
                color=_STYLE["observed"]["color"]
            ),
            marker=dict(size=5, color=_STYLE["observed"]["color"]), 
            opacity=0.95
        ))
        
        # 2. Moving average
        if show_moving_average and "MA" in monthly.columns:
            fig.add_trace(go.Scatter(
                x=monthly["Date"], y=monthly["MA"], 
                mode="lines", 
                name=f"{self.rolling_window}M Moving Average",
                line=dict(
                    width=_STYLE["ma"]["width"], 
                    dash=_STYLE["ma"]["dash"], 
                    color=_STYLE["ma"]["color"]
                ),
                opacity=0.9
            ))
        
        # 3. Trend component
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend, 
            mode="lines", 
            name="Trend Component",
            line=dict(
                width=_STYLE["trend"]["width"], 
                color=_STYLE["trend"]["color"]
            ), 
            opacity=0.9
        ))
        
        # 4. Seasonal component (with optional offset)
        seasonal_label = "Seasonal Component" + (" (offset)" if offset_components else "")
        fig.add_trace(go.Scatter(
            x=seasonal.index,
            y=seasonal + seasonal_offset,
            mode="lines",
            name=seasonal_label,
            line=dict(
                width=_STYLE["seasonal"]["width"],
                dash=_STYLE["seasonal"]["dash"],
                color=_STYLE["seasonal"]["color"],
            ),
            opacity=0.9
        ))
        
        # 5. Residual component (with optional offset)
        residual_label = "Residual Component" + (" (offset)" if offset_components else "")
        fig.add_trace(go.Scatter(
            x=residual.index,
            y=residual + residual_offset,
            mode="lines",
            name=residual_label,
            line=dict(
                width=_STYLE["residual"]["width"],
                dash=_STYLE["residual"]["dash"],
                color=_STYLE["residual"]["color"],
            ),
            opacity=0.9
        ))
        
        # Add zero reference line if not using offsets
        if not offset_components:
            fig.add_hline(
                y=0, 
                line_width=1, 
                line_dash="dot", 
                line_color="#999999"
            )
        
        # 6. Shock points
        if show_shocks and self._shock_data is not None and len(self._shock_data):
            shocks = self._shock_data.copy()
            shocks["Date"] = pd.to_datetime(shocks["Date"], utc=False)
            fig.add_trace(go.Scatter(
                x=shocks["Date"], y=shocks["Count"], 
                mode="markers", 
                name=f"Shocks (|z| ≥ {self.shock_threshold})",
                marker=_STYLE["shocks"], 
                opacity=0.9
            ))
        
        # 7. Breakpoint annotations
        if show_breakpoints and self._breakpoints:
            add_breakpoint_annotations(
                fig, self._breakpoints, label_prefix="Break"
            )
        
        # Layout + title + spacing
        fig.update_layout(
            title=f"{self.title} - Comprehensive View"
        )
        apply_publication_layout(fig, bottom_margin=90, title_font_size=26, title_bottom_padding=20)
        # fig.update_xaxes(
        #     title_text="Month", 
        #     dtick="M1", 
        #     tickformat="%b\n%Y",
        #     showgrid=True, 
        #     ticks="outside", 
        #     tickangle=-45,
        #     automargin=True,
        #     title_font=dict(size=22),
        #     tickfont=dict(size=20),
        # )
        # fig.update_yaxes(
        #     title_text="Isolate Count",
        #     showgrid=True, 
        #     ticks="outside",
        #     title_font=dict(size=22),
        #     tickfont=dict(size=20),
        # )
        
        style_time_axes(
            fig,
            dates=monthly["Date"],
            x_title="Month",
            y_title="Isolate Count",
            mode="months+years",   # <-- NEW option; set to "ticks" for old
            month_dtick="M1",
            month_angle=-45,
            year_font_size=18,
            year_y=-0.1,
            bottom_margin_ticks=90,
            bottom_margin_months_years=220,
        )
        
        # Legend
        arrange_legend(fig, columns=4, outside=True, adjust_width=False)
        
        # Summary box
        mk_results = run_mann_kendall_test(series)
        amplitude = self._get_seasonal_amplitude(period=period)
        shock_count = 0 if self._shock_data is None else len(self._shock_data)
        
        summary = compute_series_summary(
            series.index, series.values,
            extra={
                "model": model,
                "period": period,
                "mk_tau": mk_results.get("Tau"),
                "mk_p": mk_results.get("p"),
                "mk_trend": mk_results.get("trend"),
                "seasonal_amplitude": amplitude,
                "n_shock_months": shock_count,
            }
        )
        
        add_summary_box(fig, summary, small=True)
        
        return fig

    def run_decomposition_diagnostics(
        self, 
        model: str = "additive", 
        period: int = 12,
        lags: tuple = (6, 12), 
        alpha: float = 0.05,
        export_path: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Run decomposition diagnostics with statistical tests.
        """
        self._check_prepared()
        
        monthly = self._monthly_data.copy()
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)
        
        series = monthly.set_index("Date")["Count"].astype(float)
        
        # Decomposition
        trend, seasonal, residual = self._decompose_series(series, model, period)
        
        # Variance analysis
        clean_series = series.dropna()
        clean_residual = residual.dropna()
        
        total_variance = float(np.var(clean_series, ddof=1)) if len(clean_series) > 1 else np.nan
        residual_variance = float(np.var(clean_residual, ddof=1)) if len(clean_residual) > 1 else np.nan
        
        if total_variance and total_variance != 0:
            residual_variance_ratio = residual_variance / total_variance
            explained_share = 1.0 - residual_variance_ratio
        else:
            residual_variance_ratio = np.nan
            explained_share = np.nan
        
        # Seasonal analysis
        try:
            seasonal_by_month = seasonal.groupby(seasonal.index.month).mean()
            seasonal_centered = bool(np.max(np.abs(seasonal_by_month.fillna(0.0))) < 1e-6)
            seasonal_month_means = seasonal_by_month.round(2)
        except Exception:
            seasonal_month_means = pd.Series(dtype=float)
            seasonal_centered = False
        
        # Ljung-Box test for residual autocorrelation
        ljung_box_results = None
        ljung_box_summary = {}
        
        if _HAS_STATSMODELS_DIAG:
            try:
                clean_residual = residual.dropna()
                if len(clean_residual) >= max(lags) + 2:
                    ljung_box_results = acorr_ljungbox(
                        clean_residual, lags=list(lags), return_df=True
                    )
                    
                    for lag in lags:
                        if lag in ljung_box_results.index:
                            row = ljung_box_results.loc[lag]
                            ljung_box_summary[str(lag)] = {
                                "lb_stat": float(row["lb_stat"]), 
                                "lb_pvalue": float(row["lb_pvalue"])
                            }
                    
                    ljung_box_ok = all(
                        float(row["lb_pvalue"]) > alpha 
                        for _, row in ljung_box_results.iterrows()
                    )
                else:
                    ljung_box_ok = False
                    for lag in lags:
                        ljung_box_summary[str(lag)] = {"lb_stat": None, "lb_pvalue": None}
                    
            except Exception:
                ljung_box_ok = False
                for lag in lags:
                    ljung_box_summary[str(lag)] = {"lb_stat": None, "lb_pvalue": None}
        else:
            ljung_box_ok = False
            for lag in lags:
                ljung_box_summary[str(lag)] = {"lb_stat": None, "lb_pvalue": None}
        
        # Residual trend test
        try:
            clean_residual = residual.dropna()
            if len(clean_residual) >= 5 and clean_residual.nunique() > 1:
                residual_mk = run_mann_kendall_test(clean_residual)
                residual_tau = residual_mk.get("Tau", np.nan)
                residual_p = residual_mk.get("p", np.nan)
                residual_trend = residual_mk.get("trend", "no trend")
                residual_ok = bool(residual_p > alpha)
            else:
                residual_tau, residual_p, residual_trend, residual_ok = np.nan, np.nan, "no trend", True
        except Exception:
            residual_tau, residual_p, residual_trend, residual_ok = np.nan, np.nan, "no trend", False
        
        components_df = pd.DataFrame({
            "Date": series.index,
            "Observed": series.values,
            "Trend": trend.reindex(series.index).values,
            "Seasonal": seasonal.reindex(series.index).values,
            "Residual": residual.reindex(series.index).values
        })
        
        summary = {
            "model": model,
            "period": int(period),
            "n_observations": int(len(clean_series)),
            "residual_variance_ratio": float(residual_variance_ratio) if not np.isnan(residual_variance_ratio) else np.nan,
            "explained_share": float(explained_share) if not np.isnan(explained_share) else np.nan,
            "ljung_box_results": ljung_box_summary,
            "ljung_box_ok": bool(ljung_box_ok),
            "residual_tau": residual_tau,
            "residual_p": residual_p,
            "residual_trend": residual_trend,
            "residual_ok": bool(residual_ok),
            "seasonal_month_means": seasonal_month_means.to_dict() if len(seasonal_month_means) else {},
            "seasonal_centered": bool(seasonal_centered)
        }
        
        # Export if requested
        if export_path:
            try:
                components_df.to_csv(f"{export_path}_components.csv", index=False)
                if ljung_box_results is not None:
                    ljung_box_results.reset_index().rename(columns={"index": "lag"}).to_csv(
                        f"{export_path}_ljungbox.csv", index=False
                    )
                pd.DataFrame([summary]).to_csv(f"{export_path}_summary.csv", index=False)
            except Exception as e:
                warnings.warn(f"Export failed in diagnostics: {e}")
        
        return summary, ljung_box_results, components_df

    def export_results(
        self,
        base_path: str,
        figure: Optional[go.Figure] = None,
        export_csv: bool = True,
        image_formats: Optional[List[str]] = None,
        scale: int = 3,
        width: int = 1400,
        height: int = 900
    ) -> Dict[str, List[str]]:
        """
        Export analysis results to files.
        """
        outputs = {"tables": [], "figures": []}
        
        # Export CSV data
        if export_csv and self._monthly_data is not None:
            csv_path = f"{base_path}.csv"
            self._monthly_data.to_csv(csv_path, index=False)
            outputs["tables"].append(csv_path)
        
        # Export figures
        if figure is not None:
            if image_formats is None:
                image_formats = ["png", "svg"]
                
            for fmt in image_formats:
                try:
                    image_path = f"{base_path}.{fmt}"
                    figure.write_image(
                        image_path, scale=scale, width=width, height=height
                    )
                    outputs["figures"].append(image_path)
                except Exception as e:
                    warnings.warn(
                        f"Failed to export {fmt} image: {e}. "
                        "Install kaleido for static image export."
                    )
                    
        return outputs

    def run_complete_analysis(
        self,
        detect_breaks: bool = True,
        compute_seasonality: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[List[pd.Timestamp]], Optional[float]]:
        """
        Run complete analysis pipeline.
        """
        self.prepare_data().detect_shocks()
        
        if detect_breaks:
            self.detect_structural_breaks()
            
        if compute_seasonality:
            self.estimate_seasonal_amplitude()
            
        return (
            self._monthly_data, 
            self._shock_data, 
            self._breakpoints, 
            self._seasonal_amplitude
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_prepared(self) -> None:
        """Ensure data is prepared before analysis."""
        if not self._is_prepared or self._monthly_data is None:
            raise RuntimeError("Call prepare_data() before analysis")

    def _decompose_series(
        self, 
        series: pd.Series, 
        model: str, 
        period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose series into trend, seasonal, and residual components.
        """
        if _HAS_SM:
            try:
                decomposition = _sm_seasonal_decompose(
                    series, model=model, period=period, extrapolate_trend="freq"
                )
                return decomposition.trend, decomposition.seasonal, decomposition.resid
            except (ValueError, TypeError, Exception):
                pass  # Fall through to simple decomposition
        
        # Simple rolling decomposition
        trend = series.rolling(
            period, center=True, min_periods=max(2, period // 2)
        ).mean()
        seasonal = (series - trend).rolling(
            period, center=True, min_periods=max(2, period // 2)
        ).mean()
        residual = series - trend - seasonal
        
        return trend, seasonal, residual

    # =========================================================================
    # ALIASES FOR BACKWARD COMPATIBILITY
    # =========================================================================
    
    def fig_comprehensive(self, **kwargs) -> go.Figure:
        """Alias for create_comprehensive_plot for backward compatibility."""
        return self.create_comprehensive_plot(**kwargs)
    
    def decompose_diagnostics(self, **kwargs):
        """Alias for run_decomposition_diagnostics for backward compatibility."""
        return self.run_decomposition_diagnostics(**kwargs)
    
    def export_high_resolution_figures(
        self,
        base_path: str,
        figures: Dict[str, go.Figure],
        formats: List[str] = None,
        scale: int = 4,
        width: int = 1600,
        height: int = 1200,
        dpi: int = 300
    ) -> Dict[str, List[str]]:
        """
        Export figures in high-resolution PDF, PNG, and SVG formats.
        """
        if formats is None:
            formats = ['pdf', 'png', 'svg']
        
        outputs = {"figures": []}
        
        for fig_name, figure in figures.items():
            clean_name = fig_name.lower().replace(' ', '_').replace('/', '_')
            
            for fmt in formats:
                try:
                    file_path = f"{base_path}_{clean_name}.{fmt}"
                    
                    figure.write_image(
                        file_path,
                        scale=scale,
                        width=width,
                        height=height,
                        engine="kaleido"
                    )
                    
                    outputs["figures"].append(file_path)
                    print(f"Exported {fmt.upper()}: {file_path}")
                    
                except Exception as e:
                    warnings.warn(f"Failed to export {fmt.upper()} for {fig_name}: {e}")
                    print(f"Failed to export {fmt.upper()}: {e}")
        
        return outputs
    
    
    def export_publication_ready(
        self,
        base_path: str,
        include_all_plots: bool = True,
        export_data: bool = True
    ) -> Dict[str, List[str]]:
        """
        Export publication-ready figures and data in high resolution.
        """
        outputs = {"figures": [], "tables": []}
        
        # Ensure analysis is complete
        if not self._is_prepared:
            self.run_complete_analysis()
        
        figures = {}
        
        if include_all_plots:
            print("Creating publication-ready figures...")
            
            figures["main_trends"] = self.create_main_plot(
                show_shocks=True,
                show_moving_average=True,
                show_breakpoints=True
            )
            
            figures["comprehensive_components"] = self.create_comprehensive_plot(
                show_shocks=True,
                show_moving_average=True,
                show_breakpoints=True,
                offset_components=True
            )
            
            figures["seasonal_decomposition"] = self.create_seasonal_decomposition_plot(
                model="additive",
                period=12
            )
        
        figure_outputs = self.export_high_resolution_figures(
            base_path=base_path,
            figures=figures,
            formats=['pdf', 'png', 'svg'],
            scale=4,
            width=2000,
            height=1500,
            dpi=300
        )
        outputs["figures"].extend(figure_outputs["figures"])
        
        if export_data and self._monthly_data is not None:
            data_files = self.export_data(base_path)
            outputs["tables"].extend(data_files["tables"])
        
        print(f"🎉 Publication-ready export complete!")
        print(f"   Figures: {len(outputs['figures'])} files")
        print(f"   Tables: {len(outputs['tables'])} files")
        
        return outputs

    def export_data(
        self,
        base_path: str
    ) -> Dict[str, List[str]]:
        """
        Export all analysis data to CSV files.
        """
        outputs = {"tables": []}
        
        try:
            # Main monthly data
            if self._monthly_data is not None:
                monthly_path = f"{base_path}_monthly_data.csv"
                self._monthly_data.to_csv(monthly_path, index=False)
                outputs["tables"].append(monthly_path)
                print(f"Monthly data: {monthly_path}")
            
            # Shock data
            if self._shock_data is not None and len(self._shock_data) > 0:
                shocks_path = f"{base_path}_shocks.csv"
                self._shock_data.to_csv(shocks_path, index=False)
                outputs["tables"].append(shocks_path)
                print(f"Shock data: {shocks_path}")
            
            # Breakpoints data
            if self._breakpoints is not None and len(self._breakpoints) > 0:
                breaks_path = f"{base_path}_breakpoints.csv"
                breakpoints_df = pd.DataFrame({
                    'breakpoint': self._breakpoints,
                    'breakpoint_str': [bp.strftime('%Y-%m') for bp in self._breakpoints]
                })
                breakpoints_df.to_csv(breaks_path, index=False)
                outputs["tables"].append(breaks_path)
                print(f"Breakpoints: {breaks_path}")
            
            # Summary statistics
            summary_path = f"{base_path}_summary.json"
            summary_data = {
                'seasonal_amplitude': float(self._seasonal_amplitude) if self._seasonal_amplitude is not None else None,
                'analysis_period': {
                    'start': self._monthly_data['Date'].min().strftime('%Y-%m') if self._monthly_data is not None else None,
                    'end': self._monthly_data['Date'].max().strftime('%Y-%m') if self._monthly_data is not None else None,
                    'months': len(self._monthly_data) if self._monthly_data is not None else 0
                },
                'shocks_detected': len(self._shock_data) if self._shock_data is not None else 0,
                'breakpoints_detected': len(self._breakpoints) if self._breakpoints is not None else 0
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            outputs["tables"].append(summary_path)
            print(f"Summary: {summary_path}")
            
        except Exception as e:
            warnings.warn(f"Data export failed: {e}")
            print(f"Data export error: {e}")
        
        return outputs
    


def run_main_temporal(
    df: pd.DataFrame = pd.DataFrame(),
    date_col: str = "Date",
    org_col: str = "NumberOrganisation",
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    base_dir: str = "./publication_outputs/manuscript",
):
    import os

    # -------------------------------------------------------------------------
    # 0. Basic checks & date parsing
    # -------------------------------------------------------------------------
    if df is None or df.empty:
        print("run_main_temporal: received empty dataframe; no analysis performed.")
        return None

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        print("run_main_temporal: no valid dates after parsing; no analysis performed.")
        return None

    # -------------------------------------------------------------------------
    # 1. Infer year range
    # -------------------------------------------------------------------------
    min_year, max_year = _infer_year_range(
        df, date_col=date_col, min_year=min_year, max_year=max_year
    )

    os.makedirs(base_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Run analysis (auto-fallback if org_col missing)
    # -------------------------------------------------------------------------
    results = run_temporal_with_participation_layers(
        df=df,
        date_col=date_col,
        org_col=org_col,
        min_year=min_year,
        max_year=max_year,
    )

    has_orgs = bool(results.get("has_orgs", False))

    # Unpack full (always present)
    an_full = results["full"]["analyzer"]
    monthly_full = results["full"]["monthly"]
    shocks_full = results["full"]["shocks"]
    breaks_full = results["full"]["breaks"]
    amplitude_full = results["full"]["amplitude"]

    # -------------------------------------------------------------------------
    # 3. Create figures (always full; optionally others)
    # -------------------------------------------------------------------------
    fig_full_main = an_full.create_main_plot(
        show_shocks=True,
        show_moving_average=True,
        show_breakpoints=True,
    )

    fig_full_comp = an_full.create_comprehensive_plot(
        show_shocks=True,
        show_moving_average=True,
        show_breakpoints=True,
        offset_components=False,
    )

    fig_full_decomp = an_full.create_seasonal_decomposition_plot(
        model="additive",
        period=12,
    )

    figures = {
        "figure_1_full_main_trends": fig_full_main,
        "figure_2_full_comprehensive": fig_full_comp,
        "figure_3_full_decomposition": fig_full_decomp,
    }

    # If orgs exist, add continuous + participation figures
    if has_orgs:
        an_cont = results["continuous"]["analyzer"]
        monthly_cont = results["continuous"]["monthly"]
        shocks_cont = results["continuous"]["shocks"]
        breaks_cont = results["continuous"]["breaks"]
        amplitude_cont = results["continuous"]["amplitude"]

        decomp_counts = results["participation"]["isolate_counts"].copy()
        decomp_orgs = results["participation"]["organisation_counts"].copy()

        # Continuous series figs
        fig_cont_main = an_cont.create_main_plot(
            show_shocks=True,
            show_moving_average=True,
            show_breakpoints=True,
        )

        fig_cont_comp = an_cont.create_comprehensive_plot(
            show_shocks=True,
            show_moving_average=True,
            show_breakpoints=True,
            offset_components=False,
        )

        fig_cont_decomp = an_cont.create_seasonal_decomposition_plot(
            model="additive",
            period=12,
        )

        # Participation figs
        fig_part_iso = create_participation_bar_chart(
            df=decomp_counts,
            x="Year",
            y="IsolateCount",
            title="Isolate counts by participation type",
            y_title="Isolate count",
        )

        fig_part_org = create_participation_bar_chart(
            df=decomp_orgs,
            x="Year",
            y="OrganisationCount",
            title="Organisation counts by participation type",
            y_title="Organisation count",
        )

        fig_org_line = create_participation_line_chart(
            df=decomp_orgs,
            x="Year",
            y="OrganisationCount",
            title="Trend in organisation participation types",
            y_title="Number of organisations"
        )

        fig_iso_line = create_participation_line_chart(
            df=decomp_counts,
            x="Year",
            y="IsolateCount",
            title="Trend in isolate contributions by participation type",
            y_title="Isolate count"
        )

        fig_ribbon_iso = fig_ribbon_isolate_counts_by_type(decomp_counts)
        fig_ribbon_org = fig_ribbon_org_counts_by_type(decomp_orgs)

        figures.update({
            "figure_4_continuous_main_trends": fig_cont_main,
            "figure_5_continuous_comprehensive": fig_cont_comp,
            "figure_6_continuous_decomposition": fig_cont_decomp,
            "figure_7_participation_isolate_counts": fig_part_iso,
            "figure_8_participation_organisation_counts": fig_part_org,
            "figure_9_participation_organisation_trends": fig_org_line,
            "figure_10_participation_isolate_trends": fig_iso_line,
            "temporal_participation_ribbon_isolates": fig_ribbon_iso,
            "temporal_participation_ribbon_orgs": fig_ribbon_org,
        })
    else:
        # Placeholders so return tuple stays compatible
        an_cont = None
        monthly_cont = shocks_cont = breaks_cont = amplitude_cont = None

    # -------------------------------------------------------------------------
    # 4. Export figures (always export what's in figures dict)
    # -------------------------------------------------------------------------
    an_full.export_high_resolution_figures(
        base_path=f"{base_dir}/temporal",
        figures=figures,
        formats=["pdf", "png", "svg"],
        scale=6,
        width=1800,
        height=900,
        dpi=300,
    )

    # -------------------------------------------------------------------------
    # 5. Export data tables
    # -------------------------------------------------------------------------
    an_full.export_data(f"{base_dir}/full")

    if has_orgs and an_cont is not None:
        an_cont.export_data(f"{base_dir}/continuous")

    # -------------------------------------------------------------------------
    # 6. Decomposition diagnostics (full series only)
    # -------------------------------------------------------------------------
    summary_full, ljung_full, components_full = an_full.run_decomposition_diagnostics()
    components_full.to_csv(f"{base_dir}/full_components.csv", index=False)

    # -------------------------------------------------------------------------
    # 7. Return (backward compatible)
    # -------------------------------------------------------------------------
    return (
        monthly_full,
        shocks_full,
        breaks_full,
        amplitude_full,
        figures,
        components_full,
    )
