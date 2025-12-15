# temporal_trends.py
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import os
# Plotly (interactive) + optional static export via kaleido
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pymannkendall as pmk

# Optional analytics (auto-fallback if not installed)
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except Exception:
    _HAS_RUPTURES = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose as _sm_seasonal_decompose
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# ------------------------------
# Utilities
# ------------------------------

_STYLE = {
    "observed":   dict(width=3,  dash=None,      color="#1f77b4"),   # blue
    "ma":         dict(width=2,  dash="dot",     color="#1f77b4"),
    "trend":      dict(width=3,  dash=None,      color="#d62728"),   # red
    "seasonal":   dict(width=2,  dash="dot",     color="#2ca02c"),   # green
    "residual":   dict(width=2,  dash="dashdot", color="#ff7f0e"),   # orange
    "shocks":     dict(size=10, symbol="diamond-open", color="#111111", line=dict(width=2)),
    "break_band": dict(fillcolor="rgba(0,0,0,0.08)"),
    "break_line": dict(width=1.5, dash="dash", color="#000000"),
}

def _fmt_float(x, nd: int = 2, default: str = "NA") -> str:
    """Format floats robustly, returning default for NaNs/non-numerics."""
    try:
        if x == x:  # NaN check without numpy dependency
            return f"{float(x):.{nd}f}"
    except Exception:
        pass
    return default

def _summary_annotation_text(summary: Dict[str, Any]) -> str:
    """
    Build a tidy, robust HTML snippet for the summary annotation.
    Works even if some fields are missing or NaN. Amplitude is pulled from several
    possible fields and, if absent, derived from seasonal month means when available.
    """

    def _coerce_float(x) -> float:
        try:
            if isinstance(x, (int, float, np.floating, np.integer)):
                return float(x)
            if isinstance(x, str):
                x = x.strip()
                if x == "":
                    return np.nan
                return float(x)
        except Exception:
            return np.nan
        return np.nan

    def _extract_amplitude(d: Dict[str, Any]) -> float:
        # 1) Preferred direct keys
        for k in ("seasonal_amplitude", "seasonality_amp", "seas_amp", "amplitude"):
            if k in d:
                val = _coerce_float(d.get(k))
                if val == val:  # not NaN
                    return val

        # 2) Derive from seasonal month means, if provided (max - min)
        smm = d.get("seasonal_month_means")
        if isinstance(smm, dict) and len(smm) > 0:
            try:
                vals = pd.to_numeric(pd.Series(list(smm.values())), errors="coerce").dropna()
                if len(vals) > 0:
                    return float(vals.max() - vals.min())
            except Exception:
                pass

        # 3) Nothing found
        return np.nan

    lines = [
        f"<b>Period</b>: {summary.get('period_start','?')}–{summary.get('period_end','?')}",
        f"<b>Mean</b> = {_fmt_float(summary.get('mean_count'),0)}/mo, "
        f"<b>SD</b> = {_fmt_float(summary.get('sd_count'),0)}",
        f"<b>MK</b>: τ={_fmt_float(summary.get('mk_tau'))}, "
        f"p={_fmt_float(summary.get('mk_p'),3)}, "
        f"trend={summary.get('mk_trend','NA')}",
    ]

    if "mk_seasonal_tau" in summary or "mk_seasonal_p" in summary or "mk_seasonal_trend" in summary:
        lines.append(
            f"<b>Seasonal MK</b>: τ={_fmt_float(summary.get('mk_seasonal_tau'))}, "
            f"p={_fmt_float(summary.get('mk_seasonal_p'),3)}, "
            f"trend={summary.get('mk_seasonal_trend','NA')}"
        )

    amp_val = _extract_amplitude(summary)
    amp_txt = _fmt_float(amp_val, 0) if amp_val == amp_val else "NA"
    shocks = summary.get('n_shock_months', 0)
    try:
        shocks = int(shocks)
    except Exception:
        pass

    lines.append(f"<b>Amplitude</b> ≈ {amp_txt}, <b>Shocks</b>={shocks}")
    return "<br>".join(lines)

def _summaries_to_html(summaries: List[Dict[str, Any]]) -> str:
    """Render a compact multi-group block for Fig2/Fig3 overview plots."""
    if not summaries:
        return "No summary available."
    # Optional header from the first summary (period / months if present)
    first = summaries[0]
    head_bits = []
    ps, pe = first.get("period_start"), first.get("period_end")
    if ps or pe:
        head_bits.append(f"<b>Period</b>: {ps or '?'}–{pe or '?'}")
    if "months" in first:
        head_bits.append(f"({int(first['months'])} months)")
    parts = ["<br>".join(head_bits)] if head_bits else []

    # One line per group
    for s in summaries:
        g = s.get("group", "?")
        line = (
            f"<b>{g}</b>: "
            f"MK τ={_fmt_float(s.get('mk_tau'))}, "
            f"p={_fmt_float(s.get('mk_p'), 3)}, "
            f"{s.get('mk_trend','NA')}"
        )
        # Optional extras
        amp = s.get("seasonal_amplitude")
        if isinstance(amp, (int, float)) and amp == amp:
            line += f"; Amp≈{_fmt_float(amp, 0)}"
        if "model" in s and "period" in s:
            line += f" (model={s['model']}, P={s['period']})"
        parts.append(line)
    return "<br>".join(parts)


def _add_annotation_once(fig: go.Figure, *, text: str, **kwargs) -> None:
    """
    Add a paper-anchored annotation only if an equivalent one isn't already present.
    We scope the dedupe to paper-anchored annotations so axis-anchored items
    (e.g., breakpoint labels with xref='x') are unaffected.
    """
    anns = fig.layout.annotations

    # Nothing there yet — just add it.
    if anns is None or len(anns) == 0:
        fig.add_annotation(text=text, **kwargs)
        return

    # Normalize annotations to dicts for robust comparison.
    def _ann_as_dict(a):
        try:
            return a if isinstance(a, dict) else a.to_plotly_json()
        except Exception:
            # Best-effort fallback: use attribute access
            return {
                "text": getattr(a, "text", None),
                "xref": getattr(a, "xref", None),
                "yref": getattr(a, "yref", None),
                "x": getattr(a, "x", None),
                "y": getattr(a, "y", None),
            }

    target_xref = kwargs.get("xref")
    target_yref = kwargs.get("yref")

    for a in anns:
        ad = _ann_as_dict(a)
        # Only dedupe when both are paper-anchored — avoids touching breakpoint labels.
        if ad.get("xref") == "paper" and ad.get("yref") == "paper":
            if ad.get("text") == text and ad.get("xref") == target_xref and ad.get("yref") == target_yref:
                return  # already present — skip

    fig.add_annotation(text=text, **kwargs)




def _add_summary_box(fig: go.Figure, summary: Dict[str, Any] | List[Dict[str, Any]],
                     *, small: bool = False, x: float = 0.99, y: float = 0.99) -> None:
    """Add a top-right summary annotation box (paper coordinates), out of the way."""
    fig.update_layout(autosize=True, width=None)
    if isinstance(summary, list):
        text = _summaries_to_html(summary)
    else:
        text = _summary_annotation_text(summary)
    _add_annotation_once(fig,
        text=text,
        xref="paper", yref="paper",
        x=x, y=y, xanchor="right", yanchor="top",
        showarrow=False, align="left",
        bordercolor="black", borderwidth=1,
        bgcolor="white", opacity=0.7,
        font=dict(size=10 if small else 12),
    )
    

def _add_summary_box_intelligent(
    fig: go.Figure,
    summary: dict | list[dict],
    *,
    small: bool = False,
    box_w: float = 0.30,
    box_h: float = 0.22,
    inset: float = 0.01,
    outside_pad_y: float = 0.06,
    crowd_frac_threshold: float = 0.08,
    crowd_abs_threshold: int = 120,
    **_
) -> None:
    """Place a summary box where it blocks the fewest points, using box_w/box_h."""

    # Build text
    text = _summaries_to_html(summary) if isinstance(summary, list) else _summary_annotation_text(summary)

    # Axis ranges (fallback to data extents if not set yet)
    xaxis = fig.layout.xaxis
    yaxis = fig.layout.yaxis

    def _to_num_x(v):
        # support datetimes
        try:
            return float(v)
        except Exception:
            try:
                return pd.Timestamp(v).value  # ns since epoch
            except Exception:
                return np.nan

    # Collect all points in primary axes (safe for numpy/pandas)
    xs, ys = [], []
    for tr in fig.data:
        if not isinstance(tr, (go.Scatter, go.Scattergl)):
            continue
        if getattr(tr, "xaxis", "x") != "x" or getattr(tr, "yaxis", "y") != "y":
            continue
        X = getattr(tr, "x", None)
        Y = getattr(tr, "y", None)
        if X is None or Y is None:
            continue
        X = np.asarray(X)
        Y = np.asarray(Y)
        # skip mismatched lengths
        n = min(len(X), len(Y))
        if n == 0:
            continue
        X = X[:n]
        Y = Y[:n]
        # convert to numeric
        Xn = np.array([_to_num_x(xi) for xi in X], dtype="float64")
        Yn = pd.to_numeric(Y, errors="coerce").astype("float64")
        mask = np.isfinite(Xn) & np.isfinite(Yn)
        if mask.any():
            xs.extend(Xn[mask].tolist())
            ys.extend(Yn[mask].tolist())

    if not xs:
        # nothing to probe — just place top-right
        _add_summary_box(fig, summary, small=small, x=0.99, y=0.99)
        return

    # Axis ranges (fallbacks to data extents)
    if xaxis is None or xaxis.range is None:
        x0n, x1n = float(np.nanmin(xs)), float(np.nanmax(xs))
    else:
        x0n, x1n = _to_num_x(xaxis.range[0]), _to_num_x(xaxis.range[1])
    if yaxis is None or yaxis.range is None:
        y0n, y1n = float(np.nanmin(ys)), float(np.nanmax(ys))
    else:
        y0n, y1n = float(yaxis.range[0]), float(yaxis.range[1])

    ok_ranges = (
        np.isfinite(x0n) and np.isfinite(x1n) and (x1n > x0n) and
        np.isfinite(y0n) and np.isfinite(y1n) and (y1n > y0n)
    )
    if not ok_ranges:
        _add_summary_box(fig, summary, small=small, x=0.99, y=0.99)
        return

    # Normalize to paper coords
    Px = (np.asarray(xs) - x0n) / (x1n - x0n)
    Py = (np.asarray(ys) - y0n) / (y1n - y0n)
    in_frame = (Px >= 0.0) & (Px <= 1.0) & (Py >= 0.0) & (Py <= 1.0)
    Px, Py = Px[in_frame], Py[in_frame]
    if Px.size == 0:
        _add_summary_box(fig, summary, small=small, x=0.99, y=0.99)
        return

    # Corner rectangles sized by box_w/box_h
    bw = float(np.clip(box_w, 0.05, 0.9))
    bh = float(np.clip(box_h, 0.05, 0.9))
    corners = {
        "TR": (1.0 - bw, 1.0 - bh, 1.0, 1.0),
        "TL": (0.0,       1.0 - bh, bw,  1.0),
        "BR": (1.0 - bw,  0.0,      1.0, bh),
        "BL": (0.0,       0.0,      bw,  bh),
    }

    def crowd(rect):
        x0p, y0p, x1p, y1p = rect
        return int(((Px >= x0p) & (Px <= x1p) & (Py >= y0p) & (Py <= y1p)).sum())

    scores = {k: crowd(r) for k, r in corners.items()}
    total_pts = int(Px.size)
    min_corner, min_score = min(scores.items(), key=lambda kv: kv[1])

    # “too crowded” if either condition is met
    too_crowded = (
        min_score > crowd_abs_threshold or
        (total_pts > 0 and (min_score / total_pts) > crowd_frac_threshold)
    )

    if too_crowded:
        # put above the plot, anchored top-right
        _add_annotation_once(
            fig,
            text=text, xref="paper", yref="paper",
            x=1.0 - inset, y=1.0 + outside_pad_y,
            xanchor="right", yanchor="top",
            showarrow=False, align="left",
            bordercolor="black", borderwidth=1,
            bgcolor="white", opacity=0.75,
            font=dict(size=10 if small else 12),
        )
        return

    pos = {
        "TR": (1.0 - inset, 1.0 - inset, "right", "top"),
        "TL": (0.0 + inset, 1.0 - inset, "left",  "top"),
        "BR": (1.0 - inset, 0.0 + inset, "right", "bottom"),
        "BL": (0.0 + inset, 0.0 + inset, "left",  "bottom"),
    }[min_corner]

    _add_annotation_once(
        fig,
        text=text, xref="paper", yref="paper",
        x=pos[0], y=pos[1],
        xanchor=pos[2], yanchor=pos[3],
        showarrow=False, align="left",
        bordercolor="black", borderwidth=1,
        bgcolor="white", opacity=0.75,
        font=dict(size=10 if small else 12),
    )


def _series_summary(date_idx, values, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Quick, robust summary dict for annotation boxes.
    Accepts date_idx as a Series/Index/array of dates and handles all dtypes safely.
    """
    # Coerce to DatetimeIndex safely
    try:
        idx = pd.Index(date_idx)
    except Exception:
        idx = pd.Index(pd.Series(date_idx))

    try:
        dt_idx = pd.to_datetime(idx, errors="coerce")
        dt_idx = pd.DatetimeIndex(dt_idx)  # ensure DatetimeIndex for .to_period()
    except Exception:
        dt_idx = pd.DatetimeIndex([])

    # Period bounds
    if len(dt_idx):
        dmin = dt_idx.min()
        dmax = dt_idx.max()
        start = dmin.strftime("%Y-%m") if pd.notna(dmin) else "?"
        end   = dmax.strftime("%Y-%m") if pd.notna(dmax) else "?"
        try:
            months = int(dt_idx.to_period("M").nunique())
        except Exception:
            months = int(pd.PeriodIndex(dt_idx, freq="M").nunique())
    else:
        start, end, months = "?", "?", 0

    # Numeric stats on values
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    mean_v = float(v.mean()) if len(v) else np.nan
    sd_v   = float(v.std(ddof=1)) if len(v) > 1 else np.nan

    summ = {
        "period_start": start,
        "period_end": end,
        "months": months,
        "mean_count": mean_v,
        "sd_count": sd_v,
    }
    if extra:
        summ.update(extra)
    return summ


def _add_break_bands(
    fig: go.Figure,
    breaks: List[pd.Timestamp],
    *,
    label_prefix: str = "Break",
    top_y_levels: Tuple[float, ...] = (1.04, 1.08, 1.12)
) -> None:
    """
    Add semi-transparent vertical bands for breakpoints with small top-of-plot labels.
    - Bands span one calendar month (from bp to next MonthBegin).
    - Labels cycle across top_y_levels to reduce overlap.
    """
    if not breaks:
        return

    # De-duplicate by month, normalize to month start
    bps = sorted(pd.to_datetime(list({pd.Timestamp(b).normalize() for b in breaks})))

    def _next_month(dt: pd.Timestamp) -> pd.Timestamp:
        return (dt + pd.offsets.MonthBegin(1))

    for i, bp in enumerate(bps, start=1):
        bp_start = bp
        bp_end = _next_month(bp)
        # Band
        fig.add_vrect(
            x0=bp_start, x1=bp_end,
            fillcolor=_STYLE["break_band"]["fillcolor"],
            opacity=1.0,
            line_width=0,
            layer="below",
        )
        # Crisp band start line (avoid invalid 'width' on shape root)
        bl = _STYLE["break_line"]
        fig.add_vline(
            x=bp_start,
            line_width=bl.get("width", 1.5),
            line_dash=bl.get("dash", "dash"),
            line_color=bl.get("color", "#000000"),
        )
        # Top paper label (cycle vertically)
        ylab = top_y_levels[(i - 1) % len(top_y_levels)]
        _add_annotation_once(
            fig,
            x=bp_start, xref="x",
            y=ylab,   yref="paper",
            text=f"{label_prefix} {i}: {bp.strftime('%Y-%m')}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black", borderwidth=1,
            opacity=0.9, font=dict(size=10)
        )

def apply_fullwidth_layout(fig: go.Figure, *, bottom_margin: int = 90) -> go.Figure:
    """Make figures responsive & full-width with sensible defaults for papers/dashboards."""
    fig.update_layout(
        autosize=True, width=None,  # full-width, responsive
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=int(bottom_margin)),
        legend=dict(orientation="h", title=None)
    )
    return fig

def legend_columns(fig, *,
                   cols: int = 5,
                   itemwidth: int = 80,
                   outside: bool = True,
                   font_size: int = 12,
                   base_pad: int = 28,
                   bottom_gap_for_xticks: int = 36,
                   outside_y: float = -0.25,
                   adjust_width: bool = False) -> go.Figure:
    """
    Arrange legend in `cols` columns (N rows as needed) and keep it away from x-axis labels.
    - Legend is placed BELOW the plot by default (outside_y < 0).
    - We DO NOT force width; default is full-width responsiveness (autosize).
    """
    n_items = sum(1 for t in fig.data if getattr(t, "showlegend", True))
    cols = max(1, int(cols))
    rows = max(1, math.ceil(max(1, n_items) / cols))

    # Legend block config
    if outside:
        y, yanchor = float(outside_y), "top"
    else:
        y, yanchor = 0.0, "bottom"

    fig.update_layout(
        autosize=True, width=None,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=y,   yanchor=yanchor,
            itemwidth=int(itemwidth),
            font=dict(size=font_size),
            title=None
        )
    )

    # Safe margin handling
    m = fig.layout.margin or go.layout.Margin()
    l = 50 if m.l is None else m.l
    r = 30 if m.r is None else m.r
    t = 70 if m.t is None else m.t
    b = 80 if m.b is None else m.b

    if outside:
        pad_b = int(base_pad + rows * (font_size + 8) + bottom_gap_for_xticks)
        b = max(b, pad_b)

    fig.update_layout(margin=dict(l=l, r=r, t=t, b=b))

    # Optional static export width tuning (off by default)
    if adjust_width:
        cur_w = fig.layout.width or 0
        target_w = cols * int(itemwidth) + 220
        if not cur_w or cur_w < target_w:
            fig.update_layout(width=target_w)

    return fig

def _mk_safe(s: pd.Series) -> Dict[str, Any]:
    """Robust MK: handle short/constant series; prefer modified MK; safe fallbacks."""
    try:
        s = pd.Series(s).dropna()
        s = pd.to_numeric(s, errors='coerce').dropna()
    except Exception:
        return {"Tau": np.nan, "p": np.nan, "trend": "NA"}

    if (len(s) < 5) or (s.nunique(dropna=True) <= 1):
        return {"Tau": np.nan, "p": np.nan, "trend": "no trend"}

    try:
        import pymannkendall as pmk  # local import to be robust
    except Exception:
        try:
            from scipy.stats import kendalltau
            tau, p = kendalltau(np.arange(len(s)), s.values, nan_policy="omit")
            trend = "increasing" if (tau or 0) > 0 else ("decreasing" if (tau or 0) < 0 else "no trend")
            return {"Tau": float(tau), "p": float(p), "trend": trend}
        except Exception:
            return {"Tau": np.nan, "p": np.nan, "trend": "NA"}

    try:
        mk_mod = getattr(pmk, "hamed_rao_modification_test", None)
        res = mk_mod(s) if mk_mod else pmk.original_test(s)
        return {"Tau": float(res.Tau), "p": float(res.p), "trend": str(res.trend)}
    except Exception:
        try:
            res = pmk.original_test(s)
            return {"Tau": float(res.Tau), "p": float(res.p), "trend": str(res.trend)}
        except Exception:
            return {"Tau": np.nan, "p": np.nan, "trend": "NA"}


# ------------------------------
# Fig 1 — Volume Context
# ------------------------------

@dataclass
class Fig1VolumeContext:
    """
    Fig. 1 — Monthly isolate volume. Establishes denominators and detects potential data shocks.
    Designed for Date-indexed data where each row is an isolate.
    """
    df: pd.DataFrame
    date_col: str = "Date"
    freq: str = "MS"              # month-start
    rolling_ma: int = 3           # moving average window for smoother overlay
    shock_window: int = 12        # rolling window to compute z-scores for shock detection
    shock_z: float = 2.5          # |z| >= threshold => shock
    title: str = "Monthly Isolate Volume"

    # internals
    _prepared: bool = field(init=False, default=False)
    _monthly: Optional[pd.DataFrame] = field(init=False, default=None)
    _shocks: Optional[pd.DataFrame] = field(init=False, default=None)
    _breakpoints: Optional[List[pd.Timestamp]] = field(init=False, default=None)
    _seasonality_amp: Optional[float] = field(init=False, default=None)

    # ---------- Public API ----------

    def prepare(self) -> "Fig1VolumeContext":
        """Ensure Date is parsed, sorted, indexed, and resampled to complete monthly series."""
        df = self.df.copy()
        if self.date_col not in df.columns:
            raise ValueError(f"`{self.date_col}` not in dataframe.")

        # Parse -> sort -> index
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[self.date_col]).sort_values(self.date_col).set_index(self.date_col)

        # Complete monthly timeline (include empty months as zero counts)
        monthly = df.resample(self.freq).size().rename("Count").to_frame()

        # Ensure Count column is numeric; compute moving average
        monthly["Count"] = pd.to_numeric(monthly["Count"], errors='coerce').fillna(0)
        monthly["MA"] = monthly["Count"].rolling(self.rolling_ma, min_periods=1).mean()

        monthly = monthly.reset_index().rename(columns={self.date_col: "Date"})
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)

        self._monthly = monthly
        self._prepared = True
        return self

    def detect_shocks(self) -> "Fig1VolumeContext":
        """Flag months with unusually high/low counts via rolling z-scores."""
        self._require_prepared()
        m = self._monthly.copy()

        m["Count"] = pd.to_numeric(m["Count"], errors='coerce').fillna(0)

        # Rolling mean/std for z-score (use past window including current)
        m["roll_mean"] = m["Count"].rolling(self.shock_window,
                                            min_periods=max(3, self.shock_window // 3)).mean()
        m["roll_std"] = m["Count"].rolling(self.shock_window,
                                           min_periods=max(3, self.shock_window // 3)).std()

        with np.errstate(divide="ignore", invalid="ignore"):
            m["zscore"] = (m["Count"] - m["roll_mean"]) / m["roll_std"]

        shocks = m.loc[m["zscore"].abs() >= self.shock_z, ["Date", "Count", "zscore"]].copy()
        shocks["Date"] = pd.to_datetime(shocks["Date"], utc=False)
        self._shocks = shocks.reset_index(drop=True)
        return self

    def segmented_breaks(self, max_breaks: int = 2, min_segment_len: int = 3) -> "Fig1VolumeContext":
        """
        Optional: detect structural breaks in volume with PELT L2 (ruptures).
        Useful to annotate onboarding/policy/supply shocks in the volume series itself.
        """
        self._require_prepared()
        y = self._monthly.set_index("Date")["Count"].astype(float).dropna()
        n = len(y)
        self._breakpoints = []
        if n < 2 * min_segment_len:
            return self

        if _HAS_RUPTURES:
            try:
                algo = rpt.Pelt(model="l2").fit(y.values.reshape(-1, 1))
                pen = 3 * np.log(max(n, 2))
                raw = algo.predict(pen=pen)
                keep, last = [], 0
                for bp in raw:
                    if bp < n and bp - last >= min_segment_len:
                        keep.append(bp)
                        last = bp
                # dates = [y.index[bp - 1] for bp in keep[:max_breaks]]
                dates = [pd.Timestamp(y.index[bp - 1]) for bp in keep[:max_breaks]]
                self._breakpoints = dates
            except Exception:
                self._breakpoints = []
        else:
            self._breakpoints = []
        return self

    def decompose_plot(
        self, *, model: str = "additive", period: int = 12, title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot a 4-panel seasonal decomposition (Observed + MA, Trend, Seasonal, Residual).
        Adds a summary box that includes MK on the observed series and amplitude.
        """
        self._require_prepared()

        # Build numeric monthly series
        s = (
            self._monthly.assign(Date=pd.to_datetime(self._monthly["Date"], utc=False))
            .set_index("Date")["Count"].asfreq(self.freq).astype(float)
        )

        # --- compute components (robust) ---
        use_sm = False
        try:
            if _HAS_SM:
                res = _sm_seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")
                trend, seasonal, resid = res.trend, res.seasonal, res.resid
                use_sm = True
        except Exception:
            use_sm = False

        if not use_sm:
            trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (s - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            resid = s - trend - seasonal

        ttl = title or f"Seasonal decomposition (period={period}, {model}) — {self.title}"

        # --- figure scaffold ---
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.06
        )

        # Top: Observed + MA
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s, mode="lines", name="Observed",
                line=dict(width=_STYLE["observed"]["width"], color=_STYLE["observed"]["color"])
            ),
            row=1, col=1
        )
        ma = s.rolling(self.rolling_ma, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index, y=ma, mode="lines", name=f"{self.rolling_ma}M MA",
                line=dict(width=_STYLE["ma"]["width"], dash=_STYLE["ma"]["dash"], color=_STYLE["ma"]["color"])
            ),
            row=1, col=1
        )

        # Trend / Seasonal / Residual
        fig.add_trace(
            go.Scatter(x=trend.index, y=trend, mode="lines", name="Trend",
                    line=dict(width=_STYLE["trend"]["width"], color=_STYLE["trend"]["color"])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=seasonal.index, y=seasonal, mode="lines", name="Seasonal",
                    line=dict(width=_STYLE["seasonal"]["width"], dash=_STYLE["seasonal"]["dash"],
                                color=_STYLE["seasonal"]["color"])),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=resid.index, y=resid, mode="lines", name="Residual",
                    line=dict(width=_STYLE["residual"]["width"], dash=_STYLE["residual"]["dash"],
                                color=_STYLE["residual"]["color"])),
            row=4, col=1
        )

        # Layout (full-width + legend below)
        fig.update_layout(
            title=ttl, autosize=True, width=None, template="plotly_white",
            hovermode="x unified", height=800,
            margin=dict(l=60, r=30, t=70, b=90),
            legend=dict(orientation="h", title=None)
        )
        fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(showgrid=True, ticks="outside")

        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # ---- Summary box — MK (observed) + amplitude ----
        mk = _mk_safe(s)
        amp = self._seasonal_amp_value(period=period)
        summary = _series_summary(
            s.index, s.values,
            extra={
                "model": model,
                "period": period,
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=True, box_w=0.99, box_h=0.99)

        return fig

    def seasonal_amplitude(self, period: int = 12) -> "Fig1VolumeContext":
        """
        Estimate seasonal amplitude (max-min of seasonal component).
        Stores result in `self._seasonality_amp`.
        """
        self._require_prepared()

        # Build monthly series
        s = (
            self._monthly.assign(Date=pd.to_datetime(self._monthly["Date"], utc=False))
            .set_index("Date")["Count"].asfreq(self.freq).astype(float)
        )

        amp = np.nan
        if len(s.dropna()) < max(6, period):  # too short, not meaningful
            self._seasonality_amp = np.nan
            return self

        try:
            seasonal = None
            if _HAS_SM:
                try:
                    res = _sm_seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")
                    seasonal = res.seasonal
                except Exception:
                    seasonal = None

            # Fallback if SM failed or not available
            if seasonal is None:
                trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
                seasonal = (s - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()

            vals = pd.to_numeric(seasonal, errors="coerce").dropna()
            amp = float(vals.max() - vals.min()) if len(vals) > 0 and vals.nunique() > 1 else np.nan
        except Exception:
            amp = np.nan

        self._seasonality_amp = amp
        return self

    def _seasonal_amp_value(self, period: int = 12) -> float:
        """Return a robust seasonal amplitude value, computing it on demand if missing/NaN."""
        amp = getattr(self, "_seasonality_amp", None)
        if amp is None or (isinstance(amp, float) and np.isnan(amp)):
            try:
                self.seasonal_amplitude(period=period)
                amp = self._seasonality_amp
            except Exception:
                amp = np.nan
        return float(amp) if amp == amp else np.nan

    def fig_plotly(self,
                   show_shocks: bool = True,
                   show_ma: bool = True,
                   annotate_breaks: bool = True) -> go.Figure:
        """
        Interactive single-panel chart of monthly counts with optional MA, shocks, and break bands.
        """
        self._require_prepared()
        m = self._monthly.copy()
        m["Date"] = pd.to_datetime(m["Date"], utc=False)

        fig = px.line(
            m, x="Date", y="Count",
            title=self.title,
            labels={"Date": "Month", "Count": "Isolates"}
        )
        # Make the first trace the "Observed"
        fig.update_traces(
            mode="lines+markers",
            marker=dict(size=6, color=_STYLE["observed"]["color"]),
            line=dict(width=_STYLE["observed"]["width"], color=_STYLE["observed"]["color"]),
            name="Observed"
        )

        # MA overlay
        if show_ma and "MA" in m.columns:
            fig.add_trace(go.Scatter(
                x=m["Date"], y=m["MA"], mode="lines",
                name=f"{self.rolling_ma}M Moving Average",
                line=dict(width=_STYLE["ma"]["width"], dash=_STYLE["ma"]["dash"], color=_STYLE["ma"]["color"])
            ))

        # Shocks
        if show_shocks and self._shocks is not None and len(self._shocks):
            sh = self._shocks.copy()
            sh["Date"] = pd.to_datetime(sh["Date"], utc=False)
            fig.add_trace(go.Scatter(
                x=sh["Date"], y=sh["Count"],
                mode="markers", name=f"Shocks (|z| ≥ {self.shock_z})",
                marker=_STYLE["shocks"]
            ))

        # Breaks (from this class' breakpoints)
        if annotate_breaks and self._breakpoints:
            _add_break_bands(fig, self._breakpoints, label_prefix="Break")

        # Layout
        fig.update_layout(
            autosize=True, width=None,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=60, r=30, t=70, b=90),
            legend=dict(orientation="h", title=None)
        )
        fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(showgrid=True, ticks="outside")

        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # Summary w/ MK + amplitude
        y = m.set_index("Date")["Count"].asfreq("MS")
        mk = _mk_safe(y)
        amp = self._seasonal_amp_value(period=12)
        summary = _series_summary(
            m["Date"], m["Count"],
            extra={
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "n_shock_months": 0 if self._shocks is None else int(self._shocks.shape[0]),
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=False, box_w=0.99, box_h=0.99)

        return fig

    def export(self,
               basepath: str,
               fig: Optional[go.Figure] = None,
               csv: bool = True,
               static_formats: Optional[List[str]] = None,
               scale: int = 3,
               width: int = 1400,
               height: int = 900) -> Dict[str, List[str]]:
        """
        Export CSV and static images. For PNG/SVG/PDF from Plotly, install kaleido:  pip install -U kaleido
        """
        out: Dict[str, List[str]] = {"tables": [], "figures": []}
        if csv and self._monthly is not None:
            path_csv = f"{basepath}.csv"
            self._monthly.to_csv(path_csv, index=False)
            out["tables"].append(path_csv)

        if fig is not None:
            if static_formats is None:
                static_formats = ["png", "svg"]  # add "pdf" if you prefer
            for fmt in static_formats:
                try:
                    fig.write_image(f"{basepath}.{fmt}",
                                    scale=scale, width=width, height=height)
                    out["figures"].append(f"{basepath}.{fmt}")
                except Exception as e:
                    warnings.warn(
                        f"Static export failed for .{fmt}. Did you install kaleido? Error: {e}")
        return out

    # ---------- Internals ----------

    def _require_prepared(self):
        if not self._prepared or self._monthly is None:
            raise RuntimeError("Call .prepare() first.")

    # ---------- Convenience one-liner ----------

    def run(self,
            detect_breaks: bool = True,
            compute_seasonality: bool = True
            ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[List[pd.Timestamp]], Optional[float]]:
        """
        Full step for Fig. 1:
        - prepare monthly counts
        - detect shocks via rolling z-scores
        - (optional) detect structural breaks in volume
        - (optional) estimate seasonal amplitude
        Returns (monthly_counts, shocks, breakpoints, seasonal_amp)
        """
        self.prepare().detect_shocks()
        if detect_breaks:
            self.segmented_breaks()
        if compute_seasonality:
            self.seasonal_amplitude()
        return self._monthly, self._shocks, self._breakpoints, self._seasonality_amp

    def fig_with_decomposition(
        self,
        show_shocks: bool = True,
        show_ma: bool = True,
        annotate_breaks: bool = True,
        model: str = "additive",
        period: int = 12
    ) -> go.Figure:
        """
        Composite: main series (left) + 4-panel decomposition (right).
        Adds a unified summary box (MK + amplitude).
        """
        # Build child figures
        main_fig = self.fig_plotly(
            show_shocks=show_shocks, show_ma=show_ma, annotate_breaks=annotate_breaks
        )
        decomp_fig = self.decompose_plot(model=model, period=period)

        # Compose
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Monthly Isolate Volume", "Seasonal Decomposition"),
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.1
        )
        for tr in main_fig.data:
            fig.add_trace(tr, row=1, col=1)
        for tr in decomp_fig.data:
            fig.add_trace(tr, row=1, col=2)

        fig.update_layout(
            title_text=f"{self.title} with Seasonal Decomposition",
            template="plotly_white",
            autosize=True, width=None,
            margin=dict(l=60, r=30, t=80, b=90),
            legend=dict(orientation="h", title=None)
        )
        fig.update_xaxes(title_text="Month", row=1, col=1, dtick="M1", tickformat="%b\n%Y")
        fig.update_yaxes(title_text="Isolate Count", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2, tickformat="%b\n%Y")
        fig.update_yaxes(title_text="Value", row=1, col=2)

        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # ---- Summary box — recompute MK + amplitude here ----
        m = self._monthly.copy()
        m["Date"] = pd.to_datetime(m["Date"], utc=False)
        y = m.set_index("Date")["Count"].asfreq("MS")
        mk = _mk_safe(y)
        amp = self._seasonal_amp_value(period=period)

        summary = _series_summary(
            m["Date"], m["Count"],
            extra={
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "n_shock_months": 0 if self._shocks is None else int(self._shocks.shape[0]),
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=True, box_w=0.99, box_h=0.99)

        return fig



    def fig_comprehensive(self,
                          show_shocks: bool = True,
                          show_ma: bool = True,
                          annotate_breaks: bool = True,
                          model: str = "additive",
                          period: int = 12,
                          offset_components: bool = False) -> go.Figure:
        """
        Comprehensive plot: Observed, (optional) MA, Trend, Seasonal, Residual.
        Set `offset_components=True` to vertically offset Seasonal & Residual
        for visual separation; otherwise they are plotted on true scales.
        """
        self._require_prepared()

        m = self._monthly.copy()
        m["Date"] = pd.to_datetime(m["Date"], utc=False)

        s = (
            self._monthly.assign(Date=pd.to_datetime(self._monthly["Date"], utc=False))
            .set_index("Date")["Count"].asfreq(self.freq).astype(float)
        )

        # Decompose (robust)
        use_sm = False
        if _HAS_SM:
            try:
                res = _sm_seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")
                trend, seasonal, resid = res.trend, res.seasonal, res.resid
                use_sm = True
            except Exception:
                pass
        if not use_sm:
            trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (s - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            resid = s - trend - seasonal

        # --- seasonal & residual offsets ---
        if offset_components and not trend.isna().all():
            seasonal_off = float(trend.mean())
            residual_off = float(trend.mean() * 2.0)
        else:
            seasonal_off = 0.0
            residual_off = 0.0

        seasonal_label = "Seasonal Component" + (" (offset)" if offset_components else "")
        residual_label = "Residual Component" + (" (offset)" if offset_components else "")
        
        fig = go.Figure()

        # Observed
        fig.add_trace(go.Scatter(
            x=m["Date"], y=m["Count"], mode="lines+markers", name="Observed Count",
            line=dict(width=_STYLE["observed"]["width"], color=_STYLE["observed"]["color"]),
            marker=dict(size=5, color=_STYLE["observed"]["color"]), opacity=0.95
        ))

        # MA
        if show_ma and "MA" in m.columns:
            fig.add_trace(go.Scatter(
                x=m["Date"], y=m["MA"], mode="lines", name=f"{self.rolling_ma}M Moving Average",
                line=dict(width=_STYLE["ma"]["width"], dash=_STYLE["ma"]["dash"], color=_STYLE["ma"]["color"]),
                opacity=0.9
            ))

        # Trend (true values)
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend, mode="lines", name="Trend Component",
            line=dict(width=_STYLE["trend"]["width"], color=_STYLE["trend"]["color"]), opacity=0.9
        ))

        # Seasonal
        fig.add_trace(go.Scatter(
            x=seasonal.index,
            y=seasonal + seasonal_off if offset_components else seasonal,
            mode="lines",
            name=seasonal_label,
            line=dict(
                width=_STYLE["seasonal"]["width"],
                dash=_STYLE["seasonal"]["dash"],
                color=_STYLE["seasonal"]["color"],
            ),
            opacity=0.9
        ))

        # Residual
        fig.add_trace(go.Scatter(
            x=resid.index,
            y=resid + residual_off if offset_components else resid,
            mode="lines",
            name=residual_label,
            line=dict(
                width=_STYLE["residual"]["width"],
                dash=_STYLE["residual"]["dash"],
                color=_STYLE["residual"]["color"],
            ),
            opacity=0.9
        ))

        # Optional: baseline at 0 when using true (non-offset) scale
        if not offset_components:
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#999999")

        # Shocks
        if show_shocks and self._shocks is not None and len(self._shocks):
            sh = self._shocks.copy()
            sh["Date"] = pd.to_datetime(sh["Date"], utc=False)
            fig.add_trace(go.Scatter(
                x=sh["Date"], y=sh["Count"], mode="markers", name=f"Shocks (|z| ≥ {self.shock_z})",
                marker=_STYLE["shocks"], opacity=0.9
            ))

        # Breaks
        if annotate_breaks and self._breakpoints:
            _add_break_bands(fig, self._breakpoints, label_prefix="Break")

        # Layout
        apply_fullwidth_layout(fig, bottom_margin=90)
        fig.update_layout(title=f"{self.title} - Comprehensive View")
        fig.update_xaxes(title_text="Month", dtick="M1", tickformat="%b\n%Y",
                         showgrid=True, ticks="outside", automargin=True)
        fig.update_yaxes(title_text="Isolate Count / Component Value",
                         showgrid=True, ticks="outside")

        # Legend below, in columns
        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # ---- Summary box — MK (observed) + amplitude ----
        mk = _mk_safe(s)
        amp = self._seasonal_amp_value(period=period)
        summary = _series_summary(
            s.index, s.values,
            extra={
                "model": model,
                "period": period,
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=True, box_w=0.99, box_h=0.99)
        
        return fig

    def decompose_diagnostics(self, *, model: str = "additive", period: int = 12,
                              lags: tuple = (6, 12), alpha: float = 0.05,
                              export_path: Optional[str] = None
                              ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Run seasonal decomposition and report diagnostics useful for publication.
        Returns (summary_dict, lb_df, components_df).
        """
        self._require_prepared()

        m = self._monthly.copy()
        m["Date"] = pd.to_datetime(m["Date"], utc=False)
        s = m.set_index("Date")["Count"].asfreq(self.freq).astype(float)

        # Decomposition
        use_sm = False
        if _HAS_SM:
            try:
                res = _sm_seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")
                trend, seasonal, resid = res.trend, res.seasonal, res.resid
                use_sm = True
            except Exception:
                pass
        if not use_sm:
            trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (s - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            resid = s - trend - seasonal

        # Variance share
        y = s.dropna(); r = resid.dropna()
        total_var = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
        resid_var = float(np.var(r, ddof=1)) if len(r) > 1 else np.nan
        resid_var_ratio = (resid_var / total_var) if (total_var == total_var and total_var not in (0.0,)) else np.nan
        explained_share = (1.0 - resid_var_ratio) if resid_var_ratio == resid_var_ratio else np.nan

        # Seasonal centering
        try:
            seasonal_by_mo = seasonal.groupby(seasonal.index.month).mean()
            seasonal_centered = bool(np.max(np.abs(seasonal_by_mo.fillna(0.0))) < 1e-6)
            seasonal_by_mo_rounded = seasonal_by_mo.round(2)
        except Exception:
            seasonal_by_mo_rounded = pd.Series(dtype=float)
            seasonal_centered = False

        # Ljung–Box on residuals
        lb_df = None
        lb_table: Dict[str, Any] = {}
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            if len(r.dropna()) >= max(lags) + 2:
                lb_df = acorr_ljungbox(r.dropna(), lags=list(lags), return_df=True)
                for L in lags:
                    row = lb_df.loc[lb_df.index == L].iloc[0]
                    lb_table[str(L)] = {"lb_stat": float(row["lb_stat"]), "lb_pvalue": float(row["lb_pvalue"])}
                lb_ok = all(float(row["lb_pvalue"]) > alpha for _, row in lb_df.iterrows())
            else:
                lb_ok = False
                for L in lags:
                    lb_table[str(L)] = {"lb_stat": None, "lb_pvalue": None}
        except Exception:
            lb_ok = False
            for L in lags:
                lb_table[str(L)] = {"lb_stat": None, "lb_pvalue": None}

        # Residual MK
        try:
            if len(r.dropna()) >= 5 and r.dropna().nunique() > 1:
                mk_res = pmk.original_test(r.dropna())
                resid_mk_tau = float(mk_res.Tau)
                resid_mk_p = float(mk_res.p)
                resid_mk_trend = str(mk_res.trend)
                resid_mk_ok = bool(resid_mk_p > alpha)
            else:
                resid_mk_tau = np.nan; resid_mk_p = np.nan; resid_mk_trend = "no trend"; resid_mk_ok = True
        except Exception:
            resid_mk_tau = np.nan; resid_mk_p = np.nan; resid_mk_trend = "no trend"; resid_mk_ok = False

        components_df = pd.DataFrame({
            "Date": s.index,
            "Observed": s.values,
            "Trend": trend.reindex_like(s).values,
            "Seasonal": seasonal.reindex_like(s).values,
            "Residual": resid.reindex_like(s).values
        })

        summary: Dict[str, Any] = {
            "model": model, "period": int(period), "n_obs": int(len(s.dropna())),
            "resid_var_ratio": float(resid_var_ratio) if resid_var_ratio == resid_var_ratio else np.nan,
            "explained_share": float(explained_share) if explained_share == explained_share else np.nan,
            "lb_table": lb_table, "lb_ok": bool(lb_ok),
            "resid_mk_tau": resid_mk_tau, "resid_mk_p": resid_mk_p,
            "resid_mk_trend": resid_mk_trend, "resid_mk_ok": bool(resid_mk_ok),
            "seasonal_month_means": seasonal_by_mo_rounded.to_dict() if len(seasonal_by_mo_rounded) else {},
            "seasonal_centered": bool(seasonal_centered)
        }

        if export_path:
            try:
                components_df.to_csv(f"{export_path}_decomp_components.csv", index=False)
                if lb_df is not None:
                    lb_df.reset_index().rename(columns={"index": "lag"}).to_csv(
                        f"{export_path}_decomp_ljungbox.csv", index=False)
                pd.DataFrame([summary]).to_csv(f"{export_path}_decomp_summary.csv", index=False)
            except Exception as e:
                warnings.warn(f"Export failed in decompose_diagnostics: {e}")

        return summary, lb_df, components_df

    def analyze_with_mk(self, detect_breaks: bool = True, compute_seasonality: bool = True,
                        basepath: str = "pub_outputs/fig1_volume_context",
                        export: bool = True, show: bool = True,
                        comprehensive_view: bool = False, offset_components: bool = False):
        """
        Run full Fig.1 analysis including MK and seasonal MK; add summary annotation; export artefacts.
        Returns (summary_dict, plotly.Figure, caption_str).
        """
        # 1) Pipeline
        monthly, shocks, breaks, seas_amp = self.run(
            detect_breaks=detect_breaks, compute_seasonality=compute_seasonality
        )

        # 2) MK tests
        y = monthly.set_index("Date")["Count"].asfreq("MS").dropna()
        y_numeric = pd.to_numeric(y, errors='coerce').dropna()
        mk_main = _mk_safe(y_numeric)

        mk_seasonal = None
        try:
            y_s = y_numeric.dropna()
            if (len(y_s) >= 12) and (y_s.nunique() > 1):
                mk_seasonal = pmk.seasonal_test(y_s, period=12)
        except Exception:
            mk_seasonal = None

        # 3) Summary
        summary = {
            "period_start": monthly["Date"].min().strftime("%Y-%m"),
            "period_end": monthly["Date"].max().strftime("%Y-%m"),
            "months": int(monthly.shape[0]),
            "mean_count": float(monthly["Count"].mean()),
            "sd_count": float(monthly["Count"].std(ddof=1)),
            "mk_tau": float(mk_main["Tau"]) if mk_main["Tau"] == mk_main["Tau"] else np.nan,
            "mk_p": float(mk_main["p"]) if mk_main["p"] == mk_main["p"] else np.nan,
            "mk_trend": str(mk_main["trend"]),
            "n_shock_months": 0 if shocks is None else int(shocks.shape[0]),
            "breakpoints": [] if not breaks else [pd.to_datetime(b).strftime("%Y-%m") for b in breaks],
            "seasonal_amplitude": seas_amp,
        }
        if mk_seasonal:
            summary.update({
                "mk_seasonal_tau": float(mk_seasonal.Tau),
                "mk_seasonal_p": float(mk_seasonal.p),
                "mk_seasonal_trend": str(mk_seasonal.trend)
            })

        # 4) Figure
        fig = self.fig_comprehensive(show_shocks=True, show_ma=True, annotate_breaks=True, offset_components=offset_components) \
              if comprehensive_view else \
              self.fig_plotly(show_shocks=True, show_ma=True, annotate_breaks=True)

        # Summary box (again) to ensure amplitude + MK visible on whichever fig we chose
        # _add_summary_box_intelligent(fig, summary, small=comprehensive_view, box_w=0.99, box_h=0.99)

        if show:
            try:
                fig.show()
            except Exception:
                pass

        # 5) Export
        if export:
            suffix = "_comprehensive" if comprehensive_view else ""
            self.export(basepath=f"{basepath}{suffix}", fig=fig, csv=True, static_formats=["png", "svg"])
            if shocks is not None and not shocks.empty:
                shocks.to_csv(f"{basepath}{suffix}_shocks.csv", index=False)
            pd.DataFrame([summary]).to_csv(f"{basepath}{suffix}_summary.csv", index=False)
            # Caption file too
            caption = (
                f"Monthly isolate volume ({summary['period_start']}–{summary['period_end']}). "
                f"Counts: mean {summary['mean_count']:.0f}/mo (SD {summary['sd_count']:.0f}). "
                f"Mann–Kendall: τ={summary['mk_tau']:.2f}, p={summary['mk_p']:.3f}, trend={summary['mk_trend']}. "
                + (f"Seasonal MK: τ={summary['mk_seasonal_tau']:.2f}, "
                   f"p={summary['mk_seasonal_p']:.3f}, trend={summary['mk_seasonal_trend']}. "
                   if "mk_seasonal_tau" in summary else "")
                + (f"Breakpoints: {', '.join(summary['breakpoints'])}. " if summary['breakpoints'] else "")
                + (f"Seasonal amplitude ≈ {summary['seasonal_amplitude']:.0f}. "
                   if summary['seasonal_amplitude'] == summary['seasonal_amplitude'] else "")
                + f"Shock months: {summary['n_shock_months']}."
            )
            with open(f"{basepath}{suffix}_caption.txt", "w", encoding="utf-8") as fcap:
                fcap.write(caption)

        return summary, fig, caption if export else ""



# ===================================
# Fig.2 — Pathogen-group Dynamics
# ===================================

class Fig2PathogenDynamics:
    """
    Fig. 2 — Pathogen-group temporal dynamics (e.g., GramType or other categorical factor).

    Pipeline
    --------
    • prepare():     monthly isolate counts by group (complete Month×Group panel if desired)
    • analyze():     per-group MK (robust), optional Seasonal MK, PELT breakpoints
    • fig_plotly():  multi-group monthly counts (+ optional 3M MA, shaded break bands, full-width)
    • decompose_group_plot():  4-panel seasonal decomposition for one group (Observed+MA, Trend, Seasonal, Residual)
    • create_group_comprehensive_plot(): single-figure overlay (Observed, 3M MA, Trend, Seasonal, Residual)
    • export():      write CSVs and static figures (requires kaleido for images)
    """

    def __init__(self, df: pd.DataFrame, *,
                 date_col: str = "Date",
                 group_col: str = "GramType",
                 title: str = "Pathogen Group Dynamics",
                 fill_missing_months: bool = True):
        self.df = df.copy()
        self.date_col = date_col
        self.group_col = group_col
        self.title = title
        self.fill_missing_months = fill_missing_months
        self.results: Dict[str, pd.DataFrame] = {}

        # Canonicalize time index (strict dtype)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce", utc=False)
        self.df = self.df.dropna(subset=[self.date_col]).sort_values(self.date_col)
        self.df.set_index(self.date_col, inplace=True)

    # ---------------- core pipeline -----------------

    def prepare(self) -> pd.DataFrame:
        """
        Aggregate monthly isolate counts by group (Month Start frequency).
        If fill_missing_months=True, returns a complete Month×Group panel
        with zero-filled counts for missing months, without touching non-numeric
        columns (avoids 'string' dtype fill errors).
        """
        if self.group_col not in self.df.columns:
            raise KeyError(f"'{self.group_col}' not found in dataframe columns.")

        df = self.df.copy()

        # ---- Ensure a proper datetime index named self.date_col ----
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce", utc=False)
            df = df.dropna(subset=[self.date_col]).sort_values(self.date_col).set_index(self.date_col)
            df.index.name = self.date_col
        else:
            # assume date is the index; coerce and name it
            df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
            df = df[~df.index.isna()].sort_index()
            if df.index.name != self.date_col:
                df.index.name = self.date_col

        # ---- Drop rows with missing group values ----
        df = df[df[self.group_col].notna()]

        # ---- Monthly counts by group ----
        monthly = (
            df.groupby([pd.Grouper(freq="MS"), self.group_col], observed=True)
              .size()
              .rename("Count")
              .reset_index()
              .rename(columns={self.date_col: "Date"})
        )
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)

        # ---- Complete panel (only fill numeric 'Count') ----
        if self.fill_missing_months and not monthly.empty:
            groups = monthly[self.group_col].unique()
            all_months = pd.date_range(monthly["Date"].min(), monthly["Date"].max(), freq="MS")
            idx = pd.MultiIndex.from_product([all_months, groups], names=["Date", self.group_col])
            monthly = (
                monthly.set_index(["Date", self.group_col])[["Count"]]
                       .reindex(idx, fill_value=0)
                       .reset_index()
            )

        monthly["Count"] = pd.to_numeric(monthly["Count"], errors="coerce").fillna(0).astype("Int64")
        monthly = monthly.sort_values([self.group_col, "Date"], kind="mergesort").reset_index(drop=True)

        self.results["monthly"] = monthly
        return monthly

    # ---------------- analysis -----------------

    def _seasonal_amp_from_series(self, y: pd.Series, *, period: int = 12) -> float:
        """
        Robust seasonal amplitude for a single group's monthly series y (MS frequency).
        Amplitude := max(seasonal) - min(seasonal) for an additive model.
        Falls back to rolling approx when statsmodels isn't available.
        Returns np.nan when not meaningful.
        """
        try:
            y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
            if len(y) < max(6, period) or y.nunique() <= 1:
                return float("nan")

            seasonal = None
            if _HAS_SM:
                try:
                    res = _sm_seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
                    seasonal = res.seasonal
                except Exception:
                    seasonal = None

            if seasonal is None:
                trend = y.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
                seasonal = (y - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()

            vals = pd.to_numeric(seasonal, errors="coerce").dropna()
            if len(vals) == 0 or vals.nunique() <= 1:
                return float("nan")
            return float(vals.max() - vals.min())
        except Exception:
            return float("nan")


    def _segmented_breaks(self, y: pd.Series, *, penalty: Optional[float] = None,
                          max_breaks: int = 3, min_seg: int = 3) -> List[pd.Timestamp]:
        """
        Use ruptures PELT (model='l2') to detect structural breaks.
        Returns breakpoints as pandas Timestamps (excluding series endpoint).
        """
        s = y.dropna()
        n = len(s)
        if n < (2 * min_seg + 1):
            return []
        try:
            import ruptures as rpt
        except Exception:
            return []
        arr = s.values.reshape(-1, 1)
        algo = rpt.Pelt(model="l2").fit(arr)
        if penalty is None:
            penalty = 3 * np.log(n)
        bk = algo.predict(pen=penalty)
        # remove final endpoint and enforce min segment length
        bk = [b for b in bk if 0 < b < n]
        bk_valid, last = [], 0
        for b in bk:
            if (b - last) >= min_seg:
                bk_valid.append(b)
                last = b
        if len(bk_valid) > max_breaks:
            bk_valid = bk_valid[:max_breaks]
        # return timestamps at the end of each segment
        return [s.index[b - 1] for b in bk_valid]

    def analyze(self, *, include_seasonal: bool = True) -> pd.DataFrame:
        """
        For each group, compute:
          - mean, sd of monthly counts
          - MK (tau, p, trend) via _mk_safe
          - SMK (tau, p, trend, period=12) if feasible and requested
          - breakpoints (YYYY-MM list) using ruptures PELT if available
        """
        if "monthly" not in self.results:
            self.prepare()

        monthly = self.results["monthly"]
        out = []

        # seasonal_test availability
        try:
            import pymannkendall as pmk  # noqa: F401
            _has_pmk = True
        except Exception:
            _has_pmk = False

        for g, sub in monthly.groupby(self.group_col, sort=False):
            y = sub.set_index("Date")["Count"].asfreq("MS").astype(float)

            # MK robust
            mk_d = _mk_safe(y)

            # Seasonal MK only when feasible
            smk_tau = np.nan
            smk_p = np.nan
            smk_trend = "NA"
            if include_seasonal and _has_pmk:
                try:
                    y_s = y.dropna()
                    if (len(y_s) >= 12) and (y_s.nunique() > 1):
                        import pymannkendall as pmk
                        smk = pmk.seasonal_test(y_s, period=12)
                        smk_tau = float(getattr(smk, "Tau", np.nan))
                        smk_p = float(getattr(smk, "p", np.nan))
                        smk_trend = str(getattr(smk, "trend", "NA"))
                except Exception:
                    pass

            breaks = []
            try:
                breaks_ts = self._segmented_breaks(y)
                breaks = [pd.to_datetime(b).strftime("%Y-%m") for b in breaks_ts]
            except Exception:
                breaks = []

            out.append({
                "group": g,
                "mean_count": float(y.mean()) if len(y) else np.nan,
                "sd_count": float(y.std(ddof=1)) if len(y) else np.nan,
                "mk_tau": float(mk_d["Tau"]) if mk_d["Tau"] == mk_d["Tau"] else np.nan,
                "mk_p": float(mk_d["p"]) if mk_d["p"] == mk_d["p"] else np.nan,
                "mk_trend": str(mk_d["trend"]),
                "mk_seasonal_tau": smk_tau,
                "mk_seasonal_p": smk_p,
                "mk_seasonal_trend": smk_trend,
                "breakpoints": breaks,
            })

        trends = pd.DataFrame(out)
        self.results["trends"] = trends
        return trends

    # ---------------- figures -----------------

    def fig_plotly(self, *, show_ma: bool = True, annotate_breaks: bool = True,
                   legend_cols: int = 5) -> go.Figure:
        """
        Interactive line chart of monthly counts by group,
        optional 3M moving average and vertical shaded bands at detected breakpoints.
        • Full-width responsive
        • Legend below in columns
        """
        if "monthly" not in self.results:
            self.prepare()
        monthly = self.results["monthly"].copy().sort_values([self.group_col, "Date"])
        monthly["Date"] = pd.to_datetime(monthly["Date"], utc=False)

        # 3M moving average per group
        if show_ma:
            monthly["MA3"] = (
                monthly
                .groupby(self.group_col, group_keys=False)["Count"]
                .apply(lambda s: s.rolling(3, min_periods=1).mean())
            )

        fig = px.line(
            monthly, x="Date", y="Count", color=self.group_col,
            title=self.title,
            labels={"Date": "Month", "Count": "Isolate Count", self.group_col: "Group"}
        )

        # stylize observed series (keep plotly color cycle per group)
        for tr in fig.data:
            tr.update(line=dict(width=_STYLE["observed"]["width"],
                                dash=_STYLE["observed"]["dash"],
                                color=None))
            tr.update(hovertemplate="%{x|%b %Y}<br>%{y:.0f}<extra>%{fullData.name}</extra>")

        # overlay 3M MA
        if show_ma:
            for g, sub in monthly.groupby(self.group_col, sort=False):
                fig.add_scatter(
                    x=sub["Date"], y=sub["MA3"],
                    mode="lines", name=f"{g} (3M MA)",
                    line=dict(dash=_STYLE["ma"]["dash"], width=_STYLE["ma"]["width"]),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra>" + f"{g} (3M MA)</extra>"
                )

        # annotate break bands if we have trends
        if annotate_breaks and "trends" in self.results and not self.results["trends"].empty:
            all_bps = []
            for _, row in self.results["trends"].iterrows():
                if isinstance(row["breakpoints"], list):
                    all_bps.extend(pd.to_datetime(row["breakpoints"], errors="coerce").to_list())
            all_bps = [b for b in all_bps if pd.notna(b)]
            _add_break_bands(fig, all_bps, label_prefix="Break")

        # full width layout + axes polish
        apply_fullwidth_layout(fig, bottom_margin=90)
        fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(showgrid=True, ticks="outside")

        # Legend layout
        legend_columns(fig, cols=int(legend_cols), outside=True, adjust_width=False)

        # Build per-group summaries and add a single multi-group box
        summaries = []
        for group, sub in monthly.groupby(self.group_col, sort=False):
            y = sub.set_index("Date")["Count"].asfreq("MS").astype(float)
            mk = _mk_safe(y)
            amp = self._seasonal_amp_from_series(y, period=12)
            summaries.append(_series_summary(
                sub["Date"], sub["Count"],
                extra={
                    "group": group,
                    "mk_tau": mk.get("Tau"),
                    "mk_p": mk.get("p"),
                    "mk_trend": mk.get("trend"),
                    "model": "additive",
                    "period": 12,
                    "seasonal_amplitude": amp,
                }
            ))

        _add_summary_box_intelligent(fig, summaries, small=False, box_w=0.99, box_h=0.99)
        
        return fig

    def create_group_comprehensive_plot(self, group: str, *,
                                    show_ma: bool = True,
                                    model: str = "additive",
                                    period: int = 12,
                                    offset_components: bool = False) -> go.Figure:
        if "monthly" not in self.results:
            self.prepare()

        sub = self.results["monthly"][self.results["monthly"][self.group_col] == group].copy()
        if sub.empty:
            raise ValueError(f"No data found for group: {group}")
        sub["Date"] = pd.to_datetime(sub["Date"], utc=False)

        y = sub.set_index("Date")["Count"].asfreq("MS").astype(float)

        # Decompose (robust)
        use_sm = False
        if _HAS_SM:
            try:
                res = _sm_seasonal_decompose(y, model=model, period=period, extrapolate_trend="freq")
                trend, seasonal, resid = res.trend, res.seasonal, res.resid
                use_sm = True
            except Exception:
                pass
        if not use_sm:
            trend = y.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (y - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            resid = y - trend - seasonal

        # Safe offsets
        try:
            trend_mean = float(pd.to_numeric(trend, errors="coerce").dropna().mean())
            if not np.isfinite(trend_mean):
                trend_mean = 0.0
        except Exception:
            trend_mean = 0.0

        seasonal_off = trend_mean if offset_components else 0.0
        residual_off = 2.0 * trend_mean if offset_components else 0.0

        seasonal_label = "Seasonal Component" + (" (offset)" if offset_components else "")
        residual_label = "Residual Component" + (" (offset)" if offset_components else "")

        fig = go.Figure()

        # Observed
        fig.add_trace(go.Scatter(
            x=sub["Date"], y=sub["Count"],
            mode="lines+markers", name="Observed Count",
            line=dict(width=_STYLE["observed"]["width"], color=_STYLE["observed"]["color"]),
            marker=dict(size=5, color=_STYLE["observed"]["color"]), opacity=0.9
        ))

        # 3M MA
        if show_ma:
            sub["MA3"] = sub["Count"].rolling(3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=sub["Date"], y=sub["MA3"],
                mode="lines", name="3M Moving Average",
                line=dict(width=_STYLE["ma"]["width"], dash=_STYLE["ma"]["dash"], color=_STYLE["ma"]["color"]),
                opacity=0.85
            ))

        # Components
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend,
            mode="lines", name="Trend Component",
            line=dict(width=_STYLE["trend"]["width"], color=_STYLE["trend"]["color"]), opacity=0.9
        ))
        fig.add_trace(go.Scatter(
            x=seasonal.index, y=seasonal + seasonal_off,
            mode="lines", name=seasonal_label,
            line=dict(width=_STYLE["seasonal"]["width"], dash=_STYLE["seasonal"]["dash"], color=_STYLE["seasonal"]["color"]),
            opacity=0.9
        ))
        fig.add_trace(go.Scatter(
            x=resid.index, y=resid + residual_off,
            mode="lines", name=residual_label,
            line=dict(width=_STYLE["residual"]["width"], dash=_STYLE["residual"]["dash"], color=_STYLE["residual"]["color"]),
            opacity=0.9
        ))

        if not offset_components:
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#999999")

        apply_fullwidth_layout(fig, bottom_margin=90)
        fig.update_layout(title=f"{group} — Comprehensive Decomposition")
        fig.update_xaxes(title_text="Month", dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(title_text="Count / Component Value", showgrid=True, ticks="outside")
        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # Summary box (MK + amplitude)
        mk = _mk_safe(y)
        amp = self._seasonal_amp_from_series(y, period=period)
        summary = _series_summary(
            sub["Date"], sub["Count"],
            extra={
                "group": group,
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "model": model,
                "period": period,
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=False, box_w=0.99, box_h=0.99)

        return fig


    def decompose_group_plot(
        self,
        group: str,
        *,
        model: str = "additive",
        period: int = 12,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        4-panel seasonal decomposition for a single group.
        Top panel includes Observed and 3M Moving Average for clarity.
        Adds summary box with MK stats + seasonal amplitude.
        """
        if "monthly" not in self.results:
            self.prepare()

        sub = self.results["monthly"][self.results["monthly"][self.group_col] == group].copy()
        if sub.empty:
            raise ValueError(f"No data for group='{group}'")

        # Build monthly series
        y = (
            sub.assign(Date=pd.to_datetime(sub["Date"], utc=False))
            .set_index("Date")["Count"]
            .asfreq("MS")
            .astype(float)
        )

        # --- compute components robustly ---
        trend = seasonal = resid = None
        if _HAS_SM:
            try:
                res = _sm_seasonal_decompose(
                    y, model=model, period=period, extrapolate_trend="freq"
                )
                trend, seasonal, resid = res.trend, res.seasonal, res.resid
            except Exception:
                pass
        if trend is None or seasonal is None or resid is None:
            trend = y.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (y - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            resid = y - trend - seasonal

        ttl = title or f"Seasonal decomposition (period={period}, {model}) — {self.title}: {group}"

        # --- figure scaffold ---
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.06
        )

        # Observed
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, mode="lines", name="Observed",
            line=dict(width=_STYLE["observed"]["width"], color=_STYLE["observed"]["color"])
        ), row=1, col=1)

        # 3M Moving Average
        ma3 = y.rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=ma3.index, y=ma3.values, mode="lines", name="3M MA",
            line=dict(width=_STYLE["ma"]["width"], dash=_STYLE["ma"]["dash"], color=_STYLE["ma"]["color"])
        ), row=1, col=1)

        # Trend
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend.values, mode="lines", name="Trend",
            line=dict(width=_STYLE["trend"]["width"], color=_STYLE["trend"]["color"])
        ), row=2, col=1)

        # Seasonal
        fig.add_trace(go.Scatter(
            x=seasonal.index, y=seasonal.values, mode="lines", name="Seasonal",
            line=dict(width=_STYLE["seasonal"]["width"], dash=_STYLE["seasonal"]["dash"], color=_STYLE["seasonal"]["color"])
        ), row=3, col=1)

        # Residual
        fig.add_trace(go.Scatter(
            x=resid.index, y=resid.values, mode="lines", name="Residual",
            line=dict(width=_STYLE["residual"]["width"], dash=_STYLE["residual"]["dash"], color=_STYLE["residual"]["color"])
        ), row=4, col=1)

        # Layout polish (full-width + bottom legend)
        fig.update_layout(title=ttl)
        apply_fullwidth_layout(fig, bottom_margin=90)
        fig.update_xaxes(tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(showgrid=True, ticks="outside")
        legend_columns(fig, cols=4, outside=True, adjust_width=False)

        # --- summary box ---
        mk = _mk_safe(y)
        amp = self._seasonal_amp_from_series(y, period=period)
        summary = _series_summary(
            y.index, y.values,
            extra={
                "group": group,
                "model": model,
                "period": period,
                "mk_tau": mk.get("Tau"),
                "mk_p": mk.get("p"),
                "mk_trend": mk.get("trend"),
                "seasonal_amplitude": amp,
            }
        )
        _add_summary_box_intelligent(fig, summary, small=True, box_w=0.99, box_h=0.99)

        return fig


    # ---------------- caption & export -----------------

    def caption(self) -> str:
        """
        Build a concise caption summarizing per-group MK/SMK, seasonal amplitude,
        and structural breakpoints. Robust to missing/NaN fields.
        """
        trends = self.results.get("trends")
        if trends is None or trends.empty:
            return "Monthly isolate counts by group."

        parts = []
        for _, r in trends.iterrows():
            group = str(r.get("group", "?"))

            mean_txt = _fmt_float(r.get("mean_count"), 0)
            sd_txt   = _fmt_float(r.get("sd_count"), 0)

            mk_tau   = _fmt_float(r.get("mk_tau"))
            mk_p     = _fmt_float(r.get("mk_p"), 3)
            mk_trend = r.get("mk_trend", "NA")

            smk_txt = ""
            if pd.notna(r.get("mk_seasonal_tau")):
                smk_txt = (
                    f"; Seasonal MK τ={_fmt_float(r.get('mk_seasonal_tau'))}, "
                    f"p={_fmt_float(r.get('mk_seasonal_p'), 3)}, {r.get('mk_seasonal_trend','NA')}"
                )

            amp_txt = ""
            if "seasonal_amplitude" in r and pd.notna(r["seasonal_amplitude"]):
                amp_txt = f"; Amplitude≈{_fmt_float(r['seasonal_amplitude'], 0)}"

            bp_txt = ""
            if isinstance(r.get("breakpoints"), (list, tuple)) and len(r["breakpoints"]):
                bp_txt = ", breaks: " + ", ".join(map(str, r["breakpoints"]))

            parts.append(
                f"{group}: mean {mean_txt}/mo (SD {sd_txt}); "
                f"MK τ={mk_tau}, p={mk_p}, {mk_trend}{smk_txt}{amp_txt}{bp_txt}"
            )

        return "Monthly isolate volume by group (MS). " + " | ".join(parts)


    
    def export(self, *, basepath: str, fig: Optional[go.Figure] = None,
           static_formats: List[str] = ["png", "svg"]) -> Dict[str, List[str]]:
        """
        Save monthly counts and trends tables; export static images via kaleido.
        Returns dict with lists of exported file paths.
        """
        outs: Dict[str, List[str]] = {"csv": [], "fig": []}

        # Ensure base directory exists
        os.makedirs(os.path.dirname(basepath) or ".", exist_ok=True)

        # Export monthly + trends CSVs
        if "monthly" in self.results:
            mp = f"{basepath}_monthly.csv"
            try:
                self.results["monthly"].to_csv(mp, index=False)
                outs["csv"].append(mp)
            except Exception as e:
                warnings.warn(f"Failed to export monthly CSV {mp}: {e}")

        if "trends" in self.results:
            tp = f"{basepath}_trends.csv"
            try:
                self.results["trends"].to_csv(tp, index=False)
                outs["csv"].append(tp)
            except Exception as e:
                warnings.warn(f"Failed to export trends CSV {tp}: {e}")

        # Export static figures
        if fig is not None:
            for fmt in static_formats:
                p = f"{basepath}.{fmt}"
                try:
                    fig.write_image(p)
                    outs["fig"].append(p)
                except Exception as e:
                    warnings.warn(f"Export failed for {p}: {e}")

        return outs


class Fig3TestingPracticeTrends:
    """
    Fig. 3 — Trends and disparities in AST testing practices.
    Core metric: Testing Rate (%) for a given antibiotic, by group (e.g., GramType/Ward/Region).

    Pipeline
    --------
    • prepare():         monthly × group panel with % tested for each antibiotic in `abx_cols`
    • analyze_trends():  MK + optional Seasonal MK + (optional) PELT breaks per (group, antibiotic)
    • fig_plotly():      multi-group rate chart for one antibiotic (+ 3M MA, optional break bands)
    • export():          CSVs and static figure (needs kaleido for images)

    Design & Robustness
    -------------------
    - Strict datetime parsing and sorting
    - Complete Month×Group panel when requested (fills only numeric counts)
    - Coerces *_Tested columns to {0,1} safely
    - Rates in [0,100] with NaN when denominator==0
    - MK/SMK guarded for short/constant series
    - Seasonal amplitude via statsmodels when available, MA fallback otherwise
    - Ruptures is optional; min segment length + penalty guard
    - Full-width Plotly layout + bottom legend columns
    """

    def __init__(self, df: pd.DataFrame, *,
                 date_col: str = "Date",
                 group_col: str = "GramType",
                 antibiotic_cols: List[str],
                 title: str = "AST Testing Practices",
                 fill_missing_months: bool = True):
        self.df = df.copy()
        self.date_col = date_col
        self.group_col = group_col
        self.abx_cols = list(antibiotic_cols)
        self.title = title
        self.fill_missing_months = fill_missing_months
        self.results: Dict[str, pd.DataFrame] = {}

        # Strict datetime + sorting (do not set index; prepare() manages that)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce", utc=False)
        self.df = self.df.dropna(subset=[self.date_col]).sort_values(self.date_col).reset_index(drop=True)

    # --------------------------- Prep ---------------------------

    def prepare(self) -> pd.DataFrame:
        """
        Build monthly × group panel with:
          - total_isolates
          - per-antibiotic tested counts
          - per-antibiotic testing rate (%) in [0,100]
        - Only numeric fields are filled; non-numerics are left untouched.
        """
        x = self.df.copy()

        if self.group_col not in x.columns:
            raise KeyError(f"'{self.group_col}' not found in dataframe columns.")

        missing = [c for c in self.abx_cols if c not in x.columns]
        if missing:
            raise KeyError(f"These antibiotic columns are missing: {missing}")

        # Coerce *_Tested columns to {0,1} (nullable Int64 to keep NA if any pre-aggregation)
        for c in self.abx_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
            x[c] = (x[c] > 0).astype("Int64")

        # Drop rows without group assignment
        x = x[x[self.group_col].notna()].copy()

        # Monthly aggregation per group
        grp = x.groupby([pd.Grouper(key=self.date_col, freq="MS"), self.group_col], observed=True)
        monthly = grp.size().rename("total_isolates").reset_index()

        # Tested counts per antibiotic
        for abx in self.abx_cols:
            tested = grp[abx].sum().rename(f"{abx}_tested").reset_index()
            monthly = monthly.merge(tested, on=[self.date_col, self.group_col], how="left")

        # Complete Month×Group panel (fill only numeric counts with 0)
        if self.fill_missing_months and not monthly.empty:
            groups = monthly[self.group_col].dropna().unique()
            all_months = pd.date_range(monthly[self.date_col].min(), monthly[self.date_col].max(), freq="MS")
            idx = pd.MultiIndex.from_product([all_months, groups], names=[self.date_col, self.group_col])
            numeric_cols = ["total_isolates"] + [f"{abx}_tested" for abx in self.abx_cols]
            monthly = (
                monthly.set_index([self.date_col, self.group_col])[numeric_cols]
                       .reindex(idx, fill_value=0)
                       .reset_index()
            )

        # Compute % rates (NaN if denominator 0), clamp to [0, 100]
        for abx in self.abx_cols:
            tcol = f"{abx}_tested"
            rcol = f"{abx}_rate"
            monthly[rcol] = np.where(
                monthly["total_isolates"] > 0,
                (pd.to_numeric(monthly[tcol], errors="coerce") /
                 pd.to_numeric(monthly["total_isolates"], errors="coerce")) * 100.0,
                np.nan
            )
            monthly[rcol] = monthly[rcol].clip(lower=0, upper=100)

        monthly[self.date_col] = pd.to_datetime(monthly[self.date_col], utc=False)
        monthly = monthly.sort_values([self.group_col, self.date_col], kind="mergesort").reset_index(drop=True)

        self.results["monthly"] = monthly
        return monthly

    # ------------------------- Analysis -------------------------

    def analyze_trends(self, *,
                       include_seasonal: bool = True,
                       detect_breaks: bool = True) -> pd.DataFrame:
        """
        For each (group, antibiotic), compute:
        • MK τ/p/trend on % tested (robust via _mk_safe)
        • optional Seasonal MK (period=12) when feasible
        • optional structural breakpoints (ruptures PELT) on % series
        • seasonal amplitude (max–min of seasonal component, period=12)
        """
        if "monthly" not in self.results:
            self.prepare()
        m = self.results["monthly"]

        out = []
        # availability flags
        try:
            import pymannkendall as pmk  # noqa: F401
            _has_pmk = True
        except Exception:
            _has_pmk = False
        try:
            import ruptures as rpt  # noqa: F401
            _has_rup = True
        except Exception:
            _has_rup = False

        for g in m[self.group_col].dropna().unique():
            sub = (m[m[self.group_col] == g]
                   .set_index(self.date_col)
                   .sort_index())

            for abx in self.abx_cols:
                rcol = f"{abx}_rate"
                if rcol not in sub.columns:
                    continue

                # numeric + monthly frequency
                y = pd.to_numeric(sub[rcol], errors="coerce").dropna()
                if y.empty:
                    continue
                y.index = pd.to_datetime(y.index, errors="coerce")
                y = y.sort_index().asfreq("MS")

                # MK robust
                mk = _mk_safe(y)

                # Seasonal MK
                smk_tau = np.nan
                smk_p = np.nan
                smk_trend = "NA"
                if include_seasonal and _has_pmk and len(y.dropna()) >= 12 and y.nunique() > 1:
                    try:
                        import pymannkendall as pmk
                        smk_res = pmk.seasonal_test(y.dropna(), period=12)
                        smk_tau, smk_p, smk_trend = float(smk_res.Tau), float(smk_res.p), str(smk_res.trend)
                    except Exception:
                        pass

                # Seasonal amplitude (also returns seasonal-by-month means for QA)
                amp, month_means = self._seasonal_amp_from_series(y, period=12)

                # Breakpoints
                breaks_fmt = []
                if detect_breaks and _has_rup and len(y.dropna()) >= 7:
                    try:
                        import ruptures as rpt
                        arr = y.values.reshape(-1, 1)
                        n = len(y)
                        pen = 3 * np.log(max(n, 2))
                        algo = rpt.Pelt(model="l2").fit(arr)
                        idxs = algo.predict(pen=pen)
                        idxs = [b for b in idxs if 0 < b < n]
                        keep, last = [], 0
                        for b in idxs:
                            if (b - last) >= 3:
                                keep.append(b); last = b
                        keep = keep[:3]
                        bps = [y.index[b - 1] for b in keep]
                        breaks_fmt = [pd.to_datetime(b).strftime("%Y-%m") for b in bps]
                    except Exception:
                        breaks_fmt = []

                out.append({
                    "group": g,
                    "antibiotic": abx,
                    "mean_testing_rate": float(y.mean()),
                    "mk_tau": float(mk["Tau"]) if mk["Tau"] == mk["Tau"] else np.nan,
                    "mk_p": float(mk["p"]) if mk["p"] == mk["p"] else np.nan,
                    "mk_trend": str(mk["trend"]),
                    "mk_seasonal_tau": smk_tau,
                    "mk_seasonal_p": smk_p,
                    "mk_seasonal_trend": smk_trend,
                    "seasonal_amplitude": float(amp) if amp == amp else np.nan,
                    "seasonal_month_means": month_means,  # dict for downstream QA/use
                    "breakpoints": breaks_fmt,
                })

        trends = pd.DataFrame(out)
        self.results["trends"] = trends
        return trends

    # --------------------- Seasonality helper -------------------

    def _seasonal_amp_from_series(self, y: pd.Series, period: int = 12) -> Tuple[float, Dict[int, float]]:
        """
        Robust seasonal amplitude: max(seasonal) - min(seasonal) for a monthly series.
        Returns (amplitude, seasonal_month_means_dict).
        Works if statsmodels is available; otherwise uses a moving-average fallback.
        """
        y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
        if y.empty or y.nunique() <= 1 or len(y) < max(6, period):
            return np.nan, {}

        # ensure monthly freq + datetime index
        try:
            y.index = pd.to_datetime(y.index, errors="coerce")
        except Exception:
            return np.nan, {}
        y = y.sort_index().asfreq("MS")

        seasonal = None
        if _HAS_SM:
            try:
                res = _sm_seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
                seasonal = res.seasonal
            except Exception:
                seasonal = None

        if seasonal is None:
            # fallback: rough seasonal via double-centering
            trend = y.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
            seasonal = (y - trend).rolling(period, center=True, min_periods=max(2, period // 2)).mean()

        s = pd.to_numeric(seasonal, errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            return np.nan, {}

        amp = float(s.max() - s.min())

        # month means of seasonal (1..12), useful for QA and summaries
        try:
            month_means = s.groupby(s.index.month).mean().round(3).to_dict()
        except Exception:
            month_means = {}

        return amp, month_means

    # ----------------------- Visualization ----------------------

    def fig_plotly(self, antibiotic: str, *,
                   show_ma: bool = True,
                   annotate_breaks: bool = True,
                   legend_cols: int = 5) -> go.Figure:
        """
        Interactive line chart of testing rate (%) for a single antibiotic, split by group.
        - Optional 3M moving average per group
        - Optional break bands collected from analyze_trends()
        - Full width + bottom legend in columns
        """
        if "monthly" not in self.results:
            self.prepare()
        m = self.results["monthly"].copy()

        rate_col = f"{antibiotic}_rate"
        if rate_col not in m.columns:
            raise KeyError(f"Rate column '{rate_col}' not found. Did you run prepare() with this antibiotic?")

        clean_name = antibiotic.replace("_Tested", "").replace("_", " ")

        # Rolling MA per group (3 months)
        if show_ma:
            m = m.sort_values([self.group_col, self.date_col])
            m["MA3"] = (
                m.groupby(self.group_col, group_keys=False)[rate_col]
                 .apply(lambda s: s.rolling(3, min_periods=1).mean())
            )

        # Base chart
        fig = px.line(
            m, x=self.date_col, y=rate_col, color=self.group_col,
            title=f"Testing Rate for {clean_name}",
            labels={self.date_col: "Month", rate_col: "Testing Rate (%)", self.group_col: "Group"}
        )

        # Style observed
        for tr in fig.data:
            tr.update(line=dict(width=_STYLE["observed"]["width"],
                                dash=_STYLE["observed"]["dash"],
                                color=None))
            tr.update(hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>%{fullData.name}</extra>")

        # Overlay MA per group
        if show_ma:
            for g, sub in m.groupby(self.group_col, sort=False):
                fig.add_scatter(
                    x=sub[self.date_col], y=sub["MA3"],
                    mode="lines", name=f"{g} (3M MA)",
                    line=dict(dash=_STYLE["ma"]["dash"], width=_STYLE["ma"]["width"]),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>" + f"{g} (3M MA)</extra>"
                )

        # Break bands for this antibiotic
        if annotate_breaks and "trends" in self.results and not self.results["trends"].empty:
            all_breaks: List[pd.Timestamp] = []
            for _, row in self.results["trends"].query("antibiotic == @antibiotic").iterrows():
                if isinstance(row.get("breakpoints"), list):
                    all_breaks.extend(pd.to_datetime(row["breakpoints"], errors="coerce").to_list())
            all_breaks = [b for b in all_breaks if pd.notna(b)]
            _add_break_bands(fig, all_breaks, label_prefix="Break")

        # Layout polish
        apply_fullwidth_layout(fig, bottom_margin=90)
        fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(title_text="Testing Rate (%)", range=[0, 100], showgrid=True, ticks="outside")

        legend_columns(fig, cols=int(legend_cols), outside=True, adjust_width=False)

        # --- pooled summary series (handle duplicate months across groups) ---
        monthly = self.results["monthly"]
        # weighted average across groups, using total isolates as weight (preferred)
        if "total_isolates" in monthly.columns:
            pool = (
                monthly[[self.date_col, "total_isolates", rate_col]]
                .dropna(subset=[rate_col, "total_isolates"])
                .groupby(self.date_col, as_index=False)
                .apply(lambda d: np.average(d[rate_col], weights=d["total_isolates"]))
                .rename(columns={None: rate_col})
            )
        else:
            # fallback: simple mean across groups per month
            pool = (
                monthly[[self.date_col, rate_col]]
                .dropna()
                .groupby(self.date_col, as_index=False)
                .mean()
            )
        # Now 1 row per month → safe to asfreq
        y_pool = pool.set_index(self.date_col)[rate_col].sort_index().asfreq("MS")

        mk = _mk_safe(y_pool)
        amp, _smm = self._seasonal_amp_from_series(y_pool, period=12)

        summary = _series_summary(pool[self.date_col], pool[rate_col], extra={
            "metric": "Testing rate (%)",
            "mk_tau": mk.get("Tau"), "mk_p": mk.get("p"), "mk_trend": mk.get("trend"),
            "seasonal_amplitude": float(amp) if amp == amp else np.nan,
        })
        _add_summary_box_intelligent(fig, summary, small=False, box_w=0.99, box_h=0.99)

        return fig

    # -------------------------- Export --------------------------

    def export(self, *, basepath: str, fig: Optional[go.Figure] = None,
               static_formats: Tuple[str, ...] = ("png", "svg")) -> Dict[str, List[str]]:
        """
        Save monthly & trends CSV, and optionally a static figure (requires kaleido).
        Ensures output directory exists and warns (doesn't crash) on failure.
        """
        outs: Dict[str, List[str]] = {"csv": [], "fig": []}

        # Ensure directory
        try:
            os.makedirs(os.path.dirname(basepath) or ".", exist_ok=True)
        except Exception:
            pass

        if "monthly" in self.results:
            p = f"{basepath}_monthly.csv"
            try:
                self.results["monthly"].to_csv(p, index=False)
                outs["csv"].append(p)
            except Exception as e:
                warnings.warn(f"Failed to export monthly CSV {p}: {e}")

        if "trends" in self.results:
            p = f"{basepath}_trends.csv"
            try:
                self.results["trends"].to_csv(p, index=False)
                outs["csv"].append(p)
            except Exception as e:
                warnings.warn(f"Failed to export trends CSV {p}: {e}")

        if fig is not None:
            for fmt in static_formats:
                p = f"{basepath}.{fmt}"
                try:
                    fig.write_image(p)
                    outs["fig"].append(p)
                except Exception as e:
                    warnings.warn(f"Static export failed for {p}: {e}")

        return outs


@dataclass
class ExportSpec:
    """Where to export artefacts (CSV + figures)."""
    out_dir: str = "exports"
    basename: str = "fig4"
    image_scale: int = 2  # for write_image
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    export_svg: bool = True
    export_pdf: bool = False  # requires kaleido or orca installed


class Fig4CoTestingCoverage:
    """
    Publication-ready pipeline for co-testing coverage:
      1) Per-class coverage over time (share of isolates tested for ≥1 drug in class).
      2) Per-antibiotic coverage within each Class (facet by Class).
      3) Per-antibiotic coverage by WHO AWaRe (Access/Watch/Reserve), independent of Class.

    Inputs
    ------
    df : DataFrame with:
        - a date column (default "Date")
        - binary *_Tested columns per antibiotic (0/1, or coercible to int)
    class_map : dict[str, list[str]]
        Class -> list of *_Tested columns
    who_map : dict[str, str]
        *_Tested column -> WHO category ("Access" | "Watch" | "Reserve"), used by prepare_who_level

    Results stored in self.results:
        - 'monthly'       : per-class coverage time series
        - 'monthly_abx'   : per-antibiotic coverage within class
        - 'monthly_who'   : per-antibiotic coverage grouped by WHO
        - 'trends'        : per-class trend tests + breakpoints
    """

    # ----------------- init & validation -----------------
    def __init__(self, df: pd.DataFrame, *,
                 date_col: str = "Date",
                 class_map: Optional[Dict[str, List[str]]] = None,
                 focus_classes: Optional[List[str]] = None,
                 title: str = "Coverage of Key Antibiotic Classes"):
        self.df = df.copy()
        self.date_col = date_col
        self.class_map = class_map or {}
        self.focus_classes = focus_classes
        self.title = title
        self.results: Dict[str, pd.DataFrame] = {}

        # Canonicalize date column
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce", utc=False)
        self.df = self.df.dropna(subset=[self.date_col]).sort_values(self.date_col)

    def _validated_class_map(self) -> Dict[str, List[str]]:
        if not self.class_map:
            raise ValueError("class_map is required: {class_name: [list of *_Tested cols]}")
        present = set(self.df.columns)
        cmap = {k: [c for c in v if c in present] for k, v in self.class_map.items()}
        # drop empty classes
        cmap = {k: v for k, v in cmap.items() if v}
        # optional focus subset
        if self.focus_classes:
            focus = set(self.focus_classes)
            cmap = {k: v for k, v in cmap.items() if k in focus}
        if not cmap:
            raise ValueError("No valid classes after filtering. Check class_map and focus_classes.")
        return cmap

    # ----------------- core prep -----------------
    def prepare(self) -> pd.DataFrame:
        """
        Per-class coverage (monthly): proportion of isolates tested for ≥1 antibiotic in the class.
        Stores self.results['monthly'] with columns: Date, n_isolates, <Class>_coverage ...
        """
        x = self.df.copy()
        cmap = self._validated_class_map()

        monthly = (
            x.groupby(pd.Grouper(key=self.date_col, freq="MS"))
             .size().rename("n_isolates").reset_index()
             .rename(columns={self.date_col: "Date"})
        )

        for cls, cols in cmap.items():
            x[f"{cls}__any"] = x[cols].astype("Int64").fillna(0).astype(int).max(axis=1)
            cov = (
                x.groupby(pd.Grouper(key=self.date_col, freq="MS"))[f"{cls}__any"]
                 .mean().rename(f"{cls}_coverage")
            )
            monthly = monthly.merge(cov.reset_index(), on="Date", how="left")

        self.results["monthly"] = monthly
        return monthly

    def prepare_antibiotic_level(self, *, who_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Per-antibiotic monthly coverage within each Class (for Class-faceted grid).
        Adds: AbxCode, AbxName, WHO (if who_map provided). Stores in self.results['monthly_abx'].
        """
        x = self.df.copy()
        cmap = self._validated_class_map()

        pieces = []
        for cls, cols in cmap.items():
            sub = x[[self.date_col] + cols].melt(
                id_vars=[self.date_col], value_vars=cols,
                var_name="Antibiotic", value_name="Tested"
            )
            sub["Class"] = cls
            sub["WHO"] = sub["Antibiotic"].map(who_map).fillna("Unknown") if who_map else "Unknown"

            toks = sub["Antibiotic"].str.split(" - ", n=1, expand=True)
            sub["AbxCode"] = toks[0].str.replace("_Tested", "", regex=False)
            sub["AbxName"] = toks[1].str.replace("_Tested", "", regex=False).fillna(sub["Antibiotic"])

            m = (
                sub.assign(Tested=pd.to_numeric(sub["Tested"], errors="coerce").fillna(0).astype(int))
                   .groupby([pd.Grouper(key=self.date_col, freq="MS"),
                             "Class", "Antibiotic", "AbxCode", "AbxName", "WHO"])["Tested"]
                   .mean().rename("Coverage").reset_index()
            ).rename(columns={self.date_col: "Date"})
            pieces.append(m)

        m_abx = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
            columns=["Date","Class","Antibiotic","AbxCode","AbxName","WHO","Coverage"]
        )
        self.results["monthly_abx"] = m_abx
        return m_abx

    def prepare_who_level(self, who_map: Dict[str, str]) -> pd.DataFrame:
        """
        Per-antibiotic monthly coverage, grouped by WHO AWaRe (independent of Class).
        Stores in self.results['monthly_who'] with: Date, Antibiotic, AbxCode, AbxName, WHO, Coverage
        """
        x = self.df.copy()
        cols = [c for c in who_map if c in x.columns]
        if not cols:
            raise ValueError("None of the who_map columns exist in df; check names.")

        sub = x[[self.date_col] + cols].melt(
            id_vars=[self.date_col], value_vars=cols,
            var_name="Antibiotic", value_name="Tested"
        )
        sub["WHO"] = sub["Antibiotic"].map(who_map).fillna("Unknown")

        toks = sub["Antibiotic"].str.split(" - ", n=1, expand=True)
        sub["AbxCode"] = toks[0].str.replace("_Tested", "", regex=False)
        sub["AbxName"] = toks[1].str.replace("_Tested", "", regex=False).fillna(sub["Antibiotic"])

        m = (
            sub.assign(Tested=pd.to_numeric(sub["Tested"], errors="coerce").fillna(0).astype(int))
               .groupby([pd.Grouper(key=self.date_col, freq="MS"),
                         "Antibiotic", "AbxCode", "AbxName", "WHO"])["Tested"]
               .mean().rename("Coverage").reset_index()
        ).rename(columns={self.date_col: "Date"})

        # fixed WHO order for consistent facets
        m["WHO"] = pd.Categorical(m["WHO"], categories=["Access", "Watch", "Reserve", "Unknown"], ordered=True)

        self.results["monthly_who"] = m
        return m

    # ----------------- scoring for top-N -----------------
    @staticmethod
    def _score_series(y: pd.Series, method: str, min_points: int) -> float:
        y = y.dropna()
        if len(y) < min_points:
            return -1.0
        if method == "mean":
            return float(y.mean())
        if method == "peak":
            return float(y.max())
        # latest non-NA
        return float(y.iloc[-1])

    def _top_n_per_panel(self, m: pd.DataFrame, panel: str, n: int,
                         method: str = "latest", min_points: int = 3) -> pd.DataFrame:
        """
        Keep only top-N AbxName per `panel` (e.g., 'Class' or 'WHO') using a score
        computed from Coverage over time. Avoids GroupBy.apply deprecation.
        """
        tmp = m[[panel, "AbxName", "Date", "Coverage"]].copy()
        tmp = tmp.sort_values(["Date"], kind="mergesort")  # stable for latest

        g = tmp.groupby([panel, "AbxName"], observed=True, sort=False)
        counts = g["Coverage"].count().rename("non_na")

        if method == "mean":
            scores = g["Coverage"].mean()
        elif method == "peak":
            scores = g["Coverage"].max()
        else:
            def last_non_na(s: pd.Series) -> float:
                s = s.dropna()
                return float(s.iloc[-1]) if len(s) else np.nan
            scores = g["Coverage"].agg(last_non_na)

        scores = scores.rename("score").to_frame().join(counts, how="left")
        scores.loc[scores["non_na"] < int(min_points), "score"] = -1.0
        scores = scores.reset_index()

        keep = (
            scores.sort_values([panel, "score"], ascending=[True, False])
                  .groupby(panel, observed=True, sort=False)
                  .head(max(1, int(n)))[[panel, "AbxName"]]
        )

        return m.merge(keep, on=[panel, "AbxName"], how="inner")

    # ----------------- analysis -----------------
    def _segmented_breaks(self, y: pd.Series, *, penalty: Optional[float] = None,
                          max_breaks: int = 2, min_seg: int = 3) -> List[pd.Timestamp]:
        """PELT (l2) breakpoints with minimal segment length guard."""
        s = y.dropna()
        n = len(s)
        if n < (2 * min_seg + 1):
            return []
        try:
            import ruptures as rpt
        except Exception:
            return []
        arr = s.values.reshape(-1, 1)
        algo = rpt.Pelt(model="l2").fit(arr)
        if penalty is None:
            penalty = 3 * np.log(n)
        bk = algo.predict(pen=penalty)
        bk = [b for b in bk if 0 < b < n]
        bk_valid, last = [], 0
        for b in bk:
            if (b - last) >= min_seg:
                bk_valid.append(b)
                last = b
        if len(bk_valid) > max_breaks:
            bk_valid = bk_valid[:max_breaks]
        return [s.index[b] for b in bk_valid]

    def analyze(self) -> pd.DataFrame:
        """
        Trend tests & breakpoints per class coverage.
        Stores self.results['trends'] with MK + Seasonal MK and breakpoint months.
        """
        if "monthly" not in self.results:
            self.prepare()
        monthly = self.results["monthly"]
        metrics = [c for c in monthly.columns if c.endswith("_coverage")]

        out = []
        for col in metrics:
            y = monthly.set_index("Date")[col].asfreq("MS")
            if y.isna().all():
                continue

            mk = _mk_safe(y)

            # Seasonal MK (robust try)
            smk_tau = np.nan
            smk_p = np.nan
            smk_trend = "NA"
            try:
                import pymannkendall as pmk
                y_s = y.dropna()
                if (len(y_s) >= 12) and (y_s.nunique() > 1):
                    smk_res = pmk.seasonal_test(y_s, period=12)
                    smk_tau = float(getattr(smk_res, "Tau", np.nan))
                    smk_p = float(getattr(smk_res, "p", np.nan))
                    smk_trend = str(getattr(smk_res, "trend", "NA"))
            except Exception:
                pass

            breaks = []
            try:
                bks = self._segmented_breaks(y)
                breaks = [pd.to_datetime(b).strftime("%Y-%m") for b in bks]
            except Exception:
                breaks = []

            out.append({
                "metric": col,
                "mean": float(y.mean()),
                "sd": float(y.std(ddof=1)),
                "mk_tau": float(mk["Tau"]) if mk["Tau"] == mk["Tau"] else np.nan,
                "mk_p": float(mk["p"]) if mk["p"] == mk["p"] else np.nan,
                "mk_trend": str(mk["trend"]),
                "mk_seasonal_tau": smk_tau,
                "mk_seasonal_p": smk_p,
                "mk_seasonal_trend": smk_trend,
                "breakpoints": breaks,
            })
        trends = pd.DataFrame(out)
        self.results["trends"] = trends
        return trends

    # ----------------- plotting -----------------
    def fig_plotly(self, *, annotate_breaks: bool = True,
                   height: int = 700, tick_every: int = 3,
                   legend_cols: int = 5) -> go.Figure:
        """
        Multi-class coverage (one curve per class). Full-width responsive with
        bottom legend arranged in columns. Adds shaded break bands if analyze() ran.
        """
        if "monthly" not in self.results:
            self.prepare()
        monthly = self.results["monthly"].copy()

        metrics = [c for c in monthly.columns if c.endswith("_coverage")]
        if not metrics:
            raise ValueError("No *_coverage columns found; run prepare() with a valid class_map.")

        mlong = monthly.melt(id_vars=["Date"], value_vars=metrics,
                             var_name="Class", value_name="Coverage")

        fig = px.line(
            mlong, x="Date", y="Coverage", color="Class",
            title=self.title,
            labels={"Coverage": "Coverage (proportion of isolates tested)", "Date": "Month"}
        )

        # styling: thicker observed lines (coverage)
        for tr in fig.data:
            tr.update(mode="lines+markers")
            tr.update(line=dict(width=_STYLE["observed"]["width"]))

        # x/y axes
        apply_fullwidth_layout(fig, bottom_margin=100)
        fig.update_xaxes(dtick=f"M{max(1, int(tick_every))}", tickformat="%b\n%Y", showgrid=True, ticks="outside")
        fig.update_yaxes(range=[0, 1], showgrid=True, ticks="outside")

        # Break bands from trends
        if annotate_breaks and "trends" in self.results and not self.results["trends"].empty:
            all_bps = []
            for _, row in self.results["trends"].iterrows():
                if isinstance(row["breakpoints"], list):
                    all_bps.extend(pd.to_datetime(row["breakpoints"], errors="coerce").to_list())
            all_bps = [b for b in all_bps if pd.notna(b)]
            _add_break_bands(fig, all_bps, label_prefix="Break")

        # Legend below in columns
        legend_columns(fig, cols=int(legend_cols), outside=True, adjust_width=False)
        return fig

    def fig_antibiotic_grid(self, *,
                            facet_cols: int = 2,
                            tick_every: int = 3,
                            height: int = 1700,
                            top_n: Optional[int] = None,
                            rank_by: str = "latest",
                            color_map: Optional[Dict[str, str]] = None,
                            legend_cols: int = 5) -> go.Figure:
        """
        Facet grid by Class; lines = antibiotics in that class.
        Honors: facet_cols, tick_every, height, color_map, top_n.
        """
        if "monthly_abx" not in self.results:
            self.prepare_antibiotic_level()

        m_abx = self.results["monthly_abx"].copy()
        if m_abx.empty:
            raise ValueError("monthly_abx is empty; ensure class_map matches dataframe columns.")

        if top_n is not None:
            m_abx = self._top_n_per_panel(m_abx, panel="Class", n=top_n, method=rank_by, min_points=3)

        fig = px.line(
            m_abx, x="Date", y="Coverage",
            color="AbxName",
            facet_col="Class", facet_col_wrap=int(max(1, facet_cols)),
            title=f"Antibiotic-level coverage within class (faceted {facet_cols} cols)",
            labels={"Coverage": "Coverage (proportion)", "Date": "Month", "AbxName": "Antibiotic"},
            color_discrete_map=color_map or {}
        )

        # styling
        for tr in fig.data:
            tr.update(mode="lines+markers", line=dict(width=_STYLE["observed"]["width"]))

        # axes/layout
        apply_fullwidth_layout(fig, bottom_margin=140)
        fig.update_layout(height=int(height))
        fig.update_xaxes(dtick=f"M{max(1, int(tick_every))}", tickformat="%b\n%Y", tickangle=-45, showgrid=True, ticks="outside")
        fig.update_yaxes(range=[0, 1], showgrid=True, ticks="outside")

        # Legend grouping by Class (legend group titles)
        class_by_curve = {abx: cls for (cls, abx), _ in m_abx.groupby(["Class", "AbxName"], sort=False)}
        seen = set()
        for tr in fig.data:
            abx = tr.name
            cls = class_by_curve.get(abx)
            if not cls:
                continue
            tr.legendgroup = cls
            tr.showlegend = True
            if cls not in seen:
                tr.legendgrouptitle = {"text": cls}
                seen.add(cls)

        legend_columns(fig, cols=int(legend_cols), outside=True, adjust_width=False)
        
        
        return fig

    def fig_who_grid(self, *,
                     facet_cols: int = 3,
                     tick_every: int = 3,
                     height: int = 1200,
                     top_n: Optional[int] = 8,
                     rank_by: str = "latest",
                     color_map: Optional[Dict[str, str]] = None,
                     legend_cols: int = 5) -> go.Figure:
        """
        Facet grid by WHO (Access/Watch/Reserve); lines = antibiotics in that WHO category.
        Honors: facet_cols, tick_every, height, color_map, top_n.
        """
        if "monthly_who" not in self.results:
            raise ValueError("Run prepare_who_level(who_map=...) first.")
        m = self.results["monthly_who"].copy()
        if m.empty:
            raise ValueError("monthly_who is empty; verify who_map and dataframe columns.")

        if top_n is not None:
            m = self._top_n_per_panel(m, panel="WHO", n=top_n, method=rank_by, min_points=3)

        fig = px.line(
            m, x="Date", y="Coverage",
            color="AbxName",
            facet_col="WHO", facet_col_wrap=int(max(1, facet_cols)),
            title="Antibiotic-level coverage by WHO AWaRe group",
            labels={"Coverage": "Coverage (proportion)", "Date": "Month", "AbxName": "Antibiotic"},
            color_discrete_map=color_map or {}
        )

        # styling
        for tr in fig.data:
            tr.update(mode="lines+markers", line=dict(width=_STYLE["observed"]["width"]))

        apply_fullwidth_layout(fig, bottom_margin=140)
        fig.update_layout(height=int(height))
        fig.update_xaxes(dtick=f"M{max(1, int(tick_every))}", tickformat="%b\n%Y", tickangle=-45, showgrid=True, ticks="outside")
        fig.update_yaxes(range=[0, 1], showgrid=True, ticks="outside")

        # Legend grouping by WHO
        who_by_curve = {abx: who for (who, abx), _ in m.groupby(["WHO", "AbxName"], sort=False)}
        seen = set()
        for tr in fig.data:
            abx = tr.name
            who = who_by_curve.get(abx)
            if not who:
                continue
            tr.legendgroup = who
            tr.showlegend = True
            if who not in seen:
                tr.legendgrouptitle = {"text": who}
                seen.add(who)

        legend_columns(fig, cols=int(legend_cols), outside=True, adjust_width=False)
        return fig

    # ----------------- caption & export -----------------
    def caption(self) -> str:
        """
        Concise, methods-style caption fragment (include in figure caption or supplement).
        """
        trends = self.results.get("trends")
        if trends is None or trends.empty:
            return "Monthly coverage dynamics for key antibiotic classes (MS aggregation)."
        parts = []
        for _, r in trends.iterrows():
            bp_txt = f", breaks: {', '.join(r['breakpoints'])}" if r["breakpoints"] else ""
            parts.append(
                f"{r['metric']}: mean {_fmt_float(r['mean'],3)} (SD {_fmt_float(r['sd'],3)}); "
                f"MK τ={_fmt_float(r['mk_tau'])}, p={_fmt_float(r['mk_p'],3)}, {r['mk_trend']}; "
                f"Seasonal MK τ={_fmt_float(r['mk_seasonal_tau'])}, p={_fmt_float(r['mk_seasonal_p'],3)}, "
                f"{r['mk_seasonal_trend']}{bp_txt}"
            )
        return "Monthly coverage dynamics for key classes (MS). " + " | ".join(parts)

    def export_all(self, *,
                   who_map_used: Optional[Dict[str, str]] = None,
                   export: ExportSpec = ExportSpec(),
                   save_json_maps: bool = True,
                   fig_class_kwargs: Optional[dict] = None,
                   fig_who_kwargs: Optional[dict] = None) -> Dict[str, str]:
        """
        Write CSVs (monthly, monthly_abx, monthly_who, trends) and figures (SVG/PDF).
        Returns dict of paths written.
        """
        os.makedirs(export.out_dir, exist_ok=True)
        written: Dict[str, str] = {}

        # CSVs
        for key in ["monthly", "monthly_abx", "monthly_who", "trends"]:
            dfk = self.results.get(key)
            if dfk is not None and not dfk.empty:
                p = os.path.join(export.out_dir, f"{export.basename}_{key}.csv")
                dfk.to_csv(p, index=False)
                written[key] = p

        # Save maps (for reproducibility)
        if save_json_maps:
            if self.class_map:
                p = os.path.join(export.out_dir, f"{export.basename}_class_map.json")
                with open(p, "w") as f:
                    json.dump(self.class_map, f, indent=2)
                written["class_map"] = p
            if who_map_used:
                p = os.path.join(export.out_dir, f"{export.basename}_who_map.json")
                with open(p, "w") as f:
                    json.dump(who_map_used, f, indent=2)
                written["who_map"] = p

        # Figures — Class grid
        try:
            fig_cls = self.fig_antibiotic_grid(**(fig_class_kwargs or {}))
            if export.export_svg:
                p = os.path.join(export.out_dir, f"{export.basename}_class_grid.svg")
                fig_cls.write_image(p, scale=export.image_scale,
                                    width=export.image_width, height=export.image_height)
                written["class_grid_svg"] = p
            if export.export_pdf:
                p = os.path.join(export.out_dir, f"{export.basename}_class_grid.pdf")
                fig_cls.write_image(p, scale=export.image_scale,
                                    width=export.image_width, height=export.image_height)
                written["class_grid_pdf"] = p
        except Exception as e:
            written["class_grid_error"] = str(e)

        # Figures — WHO grid
        try:
            fig_who = self.fig_who_grid(**(fig_who_kwargs or {}))
            if export.export_svg:
                p = os.path.join(export.out_dir, f"{export.basename}_who_grid.svg")
                fig_who.write_image(p, scale=export.image_scale,
                                    width=export.image_width, height=export.image_height)
                written["who_grid_svg"] = p
            if export.export_pdf:
                p = os.path.join(export.out_dir, f"{export.basename}_who_grid.pdf")
                fig_who.write_image(p, scale=export.image_scale,
                                    width=export.image_width, height=export.image_height)
                written["who_grid_pdf"] = p
        except Exception as e:
            written["who_grid_error"] = str(e)

        # Figures — Class multi-line plot
        try:
            fig_multi = self.fig_plotly()
            if export.export_svg:
                p = os.path.join(export.out_dir, f"{export.basename}_class_multiline.svg")
                fig_multi.write_image(p, scale=export.image_scale,
                                      width=export.image_width, height=export.image_height)
                written["class_multiline_svg"] = p
            if export.export_pdf:
                p = os.path.join(export.out_dir, f"{export.basename}_class_multiline.pdf")
                fig_multi.write_image(p, scale=export.image_scale,
                                      width=export.image_width, height=export.image_height)
                written["class_multiline_pdf"] = p
        except Exception as e:
            written["class_multiline_error"] = str(e)

        return written

    
    