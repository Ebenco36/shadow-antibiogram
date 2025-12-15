# publication_stats.py  — FULL IMPLEMENTATION

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


import scipy.stats as sps
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import DomainWarning

from src.utils.LoadClasses import LoadClasses

# style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
warnings.filterwarnings("ignore", category=DomainWarning)

# ---------- constants ----------
CLASS_ORDER = ["Access", "Watch", "Reserve", "Unclassified"]
DEFAULT_CLASS_COLORS = {
    "Access":  "#56B4E9",
    "Watch":   "#E69F00",
    "Reserve": "#D55E00",
    "Unclassified": "#999999",
}
WHO_CLASSES = ("Access", "Watch", "Reserve")
REQUIRED_COLS = [
    "ARS_HospitalLevelManual", "CareType", "HighLevelAgeRange", "ARS_WardType",
    "TextMaterialgroupRkiL0", "PathogengroupL1", "GramType", "Year", "SeasonName", "PathogenGenus"
]

# ---------------------- helpers ----------------------


def gini(arr: Union[pd.Series, np.ndarray]) -> float:
    """Gini coefficient on nonnegative data (NaN if empty)."""
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return np.nan
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


def clean_antibiotic_label(raw: str) -> str:
    """Drop '_Tested' and tidy labels like 'AMX - Amoxicillin_Tested' -> 'Amoxicillin'."""
    if not isinstance(raw, str):
        return str(raw)
    name = raw.replace("_Tested", "")
    # keep the human name if format is 'ABBR - Name'
    parts = [p.strip() for p in name.split(" - ", 1)]
    if len(parts) == 2:
        return parts[1]
    return name


# ====================== MAIN PIPELINE CLASS =======================

@dataclass
class PublicationStats:
    """
    Publication-grade variation & disparity stats for antibiotic test coverage,
    with WHO class awareness, pagination, and export helpers.
    """
    long_df: pd.DataFrame
    antibiotic_col: str = "Antibiotic"
    stratum_col: str = "Stratum"
    compare_col: Optional[str] = "Compare"
    count_col: str = "Count"
    total_col: str = "N"

    # internal class map & palette (optional)
    # raw column name (e.g., *_Tested) -> WHO class
    _class_map: Optional[Dict[str, str]] = None
    _class_palette: Optional[Dict[str, str]] = None

    # ---------- Constructors ----------
    @classmethod
    def from_wide(
        cls,
        df: pd.DataFrame,
        stratum_col: str,
        compare_col: Optional[str] = None,
        antibiotic_suffix: str = "_Tested",
        antibiotic_cols: Optional[List[str]] = None,
    ) -> "PublicationStats":
        """
        Build long counts from row-level wide data with *_Tested columns.
        For each (stratum, [compare], antibiotic) compute Count and N.
        """
        
        if antibiotic_cols is None:
            antibiotic_cols = [
                c for c in df.columns if c.endswith(antibiotic_suffix)]
        if not antibiotic_cols:
            raise ValueError("No antibiotic *_Tested columns found.")

        id_cols = [stratum_col] + ([compare_col] if compare_col else [])
        work = df[id_cols + antibiotic_cols].copy()

        # Robust coercion to 0/1
        work[antibiotic_cols] = (
            work[antibiotic_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .astype(int)
        )
        long = work.melt(
            id_vars=id_cols, value_vars=antibiotic_cols,
            var_name="Antibiotic", value_name="Tested"
        )
        grp_cols = [stratum_col, "Antibiotic"] + \
            ([compare_col] if compare_col else [])
        agg = (
            long.groupby(grp_cols, observed=False)["Tested"]
            .agg(Count="sum", N="size")
            .reset_index()
        )
        agg = agg.rename(columns={stratum_col: "Stratum"})
        if compare_col:
            agg = agg.rename(columns={compare_col: "Compare"})
        else:
            agg["Compare"] = "All"
        agg["Compare"] = agg["Compare"].astype(str).fillna("Unknown")
        
        return cls(
            long_df=agg,
            antibiotic_col="Antibiotic",
            stratum_col="Stratum",
            compare_col="Compare",
            count_col="Count",
            total_col="N",
        )

    # ---------- Public config ----------
    def with_class_map(
        self,
        class_map: Dict[str, str],
        class_palette: Optional[Dict[str, str]] = None
    ) -> "PublicationStats":
        """Attach WHO class map and optional palette (returns self for chaining)."""
        self._class_map = {str(k): str(v) for k, v in class_map.items()}
        self._class_palette = class_palette or DEFAULT_CLASS_COLORS.copy()
        return self

    # ---------- Internal: labels & class attachment ----------
    def _attach_labels_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add AntibioticLabel + WHO_Class to a DF with Antibiotic column."""
        out = df.copy()
        out["AntibioticLabel"] = out[self.antibiotic_col].map(
            clean_antibiotic_label)
        if self._class_map:
            out["WHO_Class"] = out[self.antibiotic_col].map(
                self._class_map).fillna("Unclassified")
        else:
            out["WHO_Class"] = "Unclassified"
        return out

    def _subset_by_labels(
        self,
        df: pd.DataFrame,
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter by WHO class and optionally paginate by AntibioticLabel."""
        d = self._attach_labels_classes(df)
        if include_classes:
            include_classes = [c for c in include_classes if c in CLASS_ORDER]
            d = d[d["WHO_Class"].isin(include_classes)]
        if page and page_size:
            labels = sorted(d["AntibioticLabel"].dropna().unique().tolist())
            start = (page - 1) * page_size
            end = start + page_size
            keep = set(labels[start:end])
            d = d[d["AntibioticLabel"].isin(keep)]
        return d

    def _ensure_palette(self) -> Dict[str, str]:
        pal = self._class_palette or DEFAULT_CLASS_COLORS
        # make sure we cover all keys we might use
        for k in CLASS_ORDER:
            pal.setdefault(k, DEFAULT_CLASS_COLORS.get(k, "#999999"))
        return pal

    # ---------- Core tables ----------
    def proportions_table(self, ci: str = "wilson", alpha: float = 0.05) -> pd.DataFrame:
        """Row-level Percent + CI for each antibiotic/stratum/compare."""
        df = self.long_df.copy()
        df[self.count_col] = pd.to_numeric(
            df[self.count_col], errors="coerce").fillna(0).clip(lower=0).astype(int)
        df[self.total_col] = pd.to_numeric(
            df[self.total_col], errors="coerce").fillna(0).clip(lower=0).astype(int)

        method = "wilson" if ci.lower() == "wilson" else "beta"
        ci_lo, ci_hi = [], []
        for k, n in df[[self.count_col, self.total_col]].itertuples(index=False):
            if n <= 0:
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
            else:
                lo, hi = proportion_confint(
                    count=int(k), nobs=int(n), alpha=alpha, method=method)
                ci_lo.append(100.0 * lo)
                ci_hi.append(100.0 * hi)

        denom = df[self.total_col].replace(0, np.nan)
        df["Percent"] = 100.0 * df[self.count_col] / denom
        df["CI_L"] = ci_lo
        df["CI_U"] = ci_hi
        return df

    def variation_summary(
        self,
        threshold: float = 50.0,
        by_compare: bool = True,
        include_classes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Variability across strata for each antibiotic (by compare if requested)."""
        d = self.long_df.copy()
        d[self.count_col] = pd.to_numeric(
            d[self.count_col], errors="coerce").fillna(0)
        d[self.total_col] = pd.to_numeric(
            d[self.total_col], errors="coerce").fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d["Percent"] = 100.0 * d[self.count_col] / \
                d[self.total_col].replace(0, np.nan)

        # attach label/class and filter
        d = self._subset_by_labels(d, include_classes=include_classes)

        grp = [self.antibiotic_col]
        if by_compare and self.compare_col:
            grp.append(self.compare_col)

        rows = []
        for keys, sub in d.groupby(grp, observed=False):
            s = pd.to_numeric(sub["Percent"], errors="coerce")
            s = s[np.isfinite(s)]
            if s.empty:
                continue
            q1, med, q3 = np.quantile(s, [0.25, 0.5, 0.75])
            iqr = float(sps.iqr(s, nan_policy="omit", rng=(25, 75)))
            mean = float(np.nanmean(s))
            sd = float(np.nanstd(s, ddof=1))
            mad = float(sps.median_abs_deviation(
                s, scale=1.0, nan_policy="omit"))
            rows.append({
                "Antibiotic": keys[0] if isinstance(keys, tuple) else keys,
                "Compare": (keys[1] if isinstance(keys, tuple) and len(keys) > 1
                            else (sub[self.compare_col].iloc[0] if self.compare_col else "All")),
                "AntibioticLabel": clean_antibiotic_label(keys[0] if isinstance(keys, tuple) else keys),
                "WHO_Class": sub["WHO_Class"].iloc[0] if "WHO_Class" in sub else "Unclassified",
                "n_strata": int(sub[self.stratum_col].nunique()),
                "median_pct": float(med),
                "iqr": float(iqr),
                "rel_iqr": float(iqr/med) if med else np.nan,
                "mean_pct": mean,
                "sd": sd,
                "cv": float(sd/mean) if mean else np.nan,
                "mad": mad,
                "gini": float(gini((s.values/100.0))),
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                f"pct_below_{threshold}": float((s < threshold).mean() * 100.0)
            })
        out = pd.DataFrame(rows)
        return out.sort_values(["WHO_Class", "AntibioticLabel", "Compare"]).reset_index(drop=True)

    def disparity_summary(
        self,
        # 'logit' (OR), 'log' (RR via modified Poisson), 'rd' (risk diff, pp)
        model: str = "logit",
        reference: Optional[str] = None,
        fdr_alpha: float = 0.05,
        min_strata: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Disparity across Compare with cluster-robust SE by stratum.
        Stable against boundary proportions (0%/100%), separation, and rank issues.
        """
        from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning
        import warnings as _warnings

        if not self.compare_col:
            raise ValueError(
                "compare_col is None; initialize with a compare column.")

        d = self.long_df.copy()
        d[self.count_col] = pd.to_numeric(
            d[self.count_col], errors="coerce").fillna(0).clip(lower=0)
        d[self.total_col] = pd.to_numeric(
            d[self.total_col], errors="coerce").fillna(0).clip(lower=0)
        d = d[d[self.total_col] > 0].copy()
        if d.empty:
            return pd.DataFrame(), pd.DataFrame()
        d[self.count_col] = np.minimum(d[self.count_col], d[self.total_col])

        # keep antibiotics with at least min_strata strata
        keep = d.groupby(self.antibiotic_col, observed=False)[
            self.stratum_col].nunique()
        d = d[d[self.antibiotic_col].isin(
            keep[keep >= min_strata].index)].copy()
        if d.empty:
            return pd.DataFrame(), pd.DataFrame()

        d[self.compare_col] = d[self.compare_col].astype(str).fillna("Unknown")
        levels = sorted(d[self.compare_col].dropna().unique().tolist())
        if reference is None:
            reference = levels[0]
        if reference not in levels:
            raise ValueError(
                f"reference '{reference}' not in compare levels: {levels}")

        # choose model label
        if model == "logit":
            metric_name = "OR"
        elif model == "log":
            metric_name = "RR"
        elif model == "rd":
            metric_name = "RD_pp"
        else:
            raise ValueError("model must be 'logit', 'log', or 'rd'")

        pairwise_rows, global_rows = [], []

        for abx, sub in d.groupby(self.antibiotic_col, observed=False):
            # design matrix (reference is the first category because drop_first=True)
            cmp_cat = pd.Categorical(
                sub[self.compare_col], categories=levels, ordered=True)
            X = pd.get_dummies(cmp_cat, drop_first=True, dtype=float)
            X = sm.add_constant(X, has_constant="add").astype(float)

            # Drop any all-zero (or constant) columns beyond the intercept to avoid rank defects
            nz = [c for c in X.columns if c != "const" and X[c].var() == 0.0]
            if nz:
                X = X.drop(columns=nz)
            if X.shape[1] <= 1:
                # no contrasts left
                continue

            col = self.stratum_col
            # ensure it’s a single column Series, even if a name collision happens
            if isinstance(sub[col], pd.DataFrame):
                clusters = pd.Categorical(sub[col].iloc[:, 0]).codes
            else:
                clusters = sub[col].astype("category").cat.codes

            # ---- Fit according to model ----
            if model == "logit":
                # Grouped binomial: [successes, failures]
                successes = sub[self.count_col].astype(float).values
                failures = (sub[self.total_col] - sub[self.count_col]
                            ).clip(lower=0).astype(float).values

                # Detect boundary rows (0% or 100%) and apply tiny continuity correction
                boundary = (successes == 0) | (failures == 0)
                if np.any(boundary):
                    successes = successes.copy()
                    failures = failures.copy()
                    successes[boundary] += 0.5
                    failures[boundary] += 0.5

                endog = np.column_stack([successes, failures]).astype(float)
                fam = sm.families.Binomial(link=sm.families.links.Logit())

                # Try standard GLM; if separation triggers, use gentle ridge then sandwich around it
                try:
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings(
                            "ignore", category=PerfectSeparationWarning)
                        res = sm.GLM(endog, X, family=fam).fit(
                            cov_type="cluster", cov_kwds={"groups": clusters}
                        )
                except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                    pen = sm.GLM(endog, X, family=fam).fit_regularized(
                        alpha=1.0, L1_wt=0.0)
                    start = pen.params
                    res = sm.GLM(endog, X, family=fam).fit(
                        start_params=start, maxiter=0, method="newton",
                        cov_type="cluster", cov_kwds={"groups": clusters}
                    )

            elif model == "log":
                # Modified Poisson with exposure (Zou 2004) for stable RR
                y = sub[self.count_col].astype(float).values
                exposure = sub[self.total_col].astype(float).values
                exposure = np.clip(exposure, 1e-12, np.inf)

                # --- NEW: continuity on rates for boundary rows (rate==0 or 1) ---
                rate = y / exposure
                b0 = (rate == 0.0)
                b1 = (rate == 1.0)
                if np.any(b0 | b1):
                    y = y.copy()
                    exposure = exposure.copy()
                    # add 0.5 success and +1.0 exposure to zero-rate rows
                    y[b0] += 0.5
                    exposure[b0] += 1.0
                    # subtract 0.5 success and +1.0 exposure to one-rate rows (keep strictly <1)
                    y[b1] = np.maximum(y[b1] - 0.5, 1e-9)
                    exposure[b1] += 1.0
                    # final clip to (0, exposure)
                    y = np.clip(y, 1e-9, exposure - 1e-9)

                # Fit with cluster-robust SE
                res = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exposure).fit(
                    cov_type="cluster", cov_kwds={"groups": clusters}
                )

            else:  # 'rd' — risk difference via linear probability with weights
                y = (sub[self.count_col] /
                     sub[self.total_col]).astype(float).values
                # clip to (0,1) slightly to avoid degenerate HC sandwich at exact 0/1
                y = np.clip(y, 1e-9, 1 - 1e-9)
                w = sub[self.total_col].astype(float).clip(lower=1.0).values
                res = sm.WLS(y, X, weights=w).fit(
                    cov_type="cluster", cov_kwds={"groups": clusters})

            # Global Wald (skip if no contrasts)
            if len(res.params) > 1:
                L = np.eye(len(res.params))[1:, :]
                try:
                    wald = res.wald_test(L, scalar=True)
                    global_rows.append({
                        "Antibiotic": abx,
                        "metric": metric_name,
                        "global_wald_stat": float(np.squeeze(wald.statistic)),
                        "global_df": int(L.shape[0]),
                        "global_p": float(np.squeeze(wald.pvalue)),
                    })
                except Exception:
                    pass

            # Pairwise contrasts vs reference
            try:
                conf = res.conf_int()
            except Exception:
                conf = None

            for cname in X.columns:
                if cname == "const":
                    continue
                beta = float(res.params.get(cname, np.nan))
                se = float(res.bse.get(cname, np.nan)) if hasattr(
                    res, "bse") else np.nan
                if conf is not None and cname in conf.index:
                    ci_l, ci_u = [float(v) for v in conf.loc[cname].tolist()]
                else:
                    ci_l, ci_u = beta - 1.96 * se, beta + 1.96 * se
                pval = float(res.pvalues.get(cname, np.nan)) if hasattr(
                    res, "pvalues") else np.nan

                if model in {"logit", "log"}:
                    est, lo, hi = np.exp(beta), np.exp(ci_l), np.exp(ci_u)
                else:
                    est, lo, hi = 100.0 * beta, 100.0 * ci_l, 100.0 * ci_u

                pairwise_rows.append({
                    "Antibiotic": abx,
                    "AntibioticLabel": clean_antibiotic_label(abx),
                    "WHO_Class": self._class_map.get(abx, "Unclassified") if self._class_map else "Unclassified",
                    "contrast": f"{reference} vs {cname}",
                    "metric": metric_name,
                    "estimate": float(est),
                    "ci_l": float(lo),
                    "ci_u": float(hi),
                    "beta": beta,
                    "se": se,
                    "p": pval,
                    "n_strata": int(sub[self.stratum_col].nunique()),
                    "n_obs": int(len(sub))
                })

        pairwise = pd.DataFrame(pairwise_rows)
        globals_ = pd.DataFrame(global_rows)

        # FDR within metric
        if not pairwise.empty:
            pairwise["p_fdr"] = np.nan
            for metric, subm in pairwise.groupby("metric", observed=False):
                try:
                    _, p_adj, _, _ = multipletests(
                        subm["p"].values, alpha=fdr_alpha, method="fdr_bh")
                    pairwise.loc[subm.index, "p_fdr"] = p_adj
                except Exception:
                    pass

        if not globals_.empty:
            globals_["global_p_fdr"] = np.nan
            for metric, subm in globals_.groupby("metric", observed=False):
                try:
                    _, p_adj, _, _ = multipletests(
                        subm["global_p"].values, alpha=fdr_alpha, method="fdr_bh")
                    globals_.loc[subm.index, "global_p_fdr"] = p_adj
                except Exception:
                    pass

        # tidy sort
        if not pairwise.empty:
            pairwise = pairwise.sort_values(
                ["metric", "WHO_Class", "AntibioticLabel", "contrast"]).reset_index(drop=True)
        if not globals_.empty:
            globals_ = globals_.sort_values(
                ["metric", "Antibiotic"]).reset_index(drop=True)

        return pairwise, globals_

    # ---------- Visualization (class-aware + pagination) ----------

    def plot_coverage_heatmap(
        self,
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        figsize: Tuple[int, int] = (16, 12),
        cmap: str = "RdYlGn",
        center: float = 50.0,
        annot: bool = True,
        fmt: str = ".1f"
    ) -> plt.Figure:
        """Heatmap of testing percentages across antibiotics and strata (class-aware & paginated)."""
        props = self.proportions_table()
        props = self._subset_by_labels(
            props, include_classes=include_classes, page=page, page_size=page_size)

        pivot = props.pivot_table(
            values="Percent", index=self.stratum_col, columns="AntibioticLabel", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap, center=center,
                    cbar_kws={'label': 'Testing Percentage (%)'}, ax=ax)
        title_bits = ["Antibiotic Testing Coverage"]
        if include_classes:
            title_bits.append(f"({', '.join(include_classes)})")
        if page and page_size:
            title_bits.append(f"— Page {page}")
        ax.set_title(" ".join(title_bits), fontsize=16, pad=18)
        ax.set_xlabel("Antibiotic")
        ax.set_ylabel(self.stratum_col)
        plt.tight_layout()
        return fig

    def plot_antibiotic_comparison(
        self,
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        figsize: Tuple[int, int] = (14, 8),
        color_by_class: bool = True
    ) -> plt.Figure:
        """Box plot comparing testing rates across antibiotics (class-colored & paginated)."""
        props = self.proportions_table()
        props = self._subset_by_labels(
            props, include_classes=include_classes, page=page, page_size=page_size)

        fig, ax = plt.subplots(figsize=figsize)
        if color_by_class:
            pal = self._ensure_palette()
            sns.boxplot(data=props, x="AntibioticLabel", y="Percent",
                        hue="WHO_Class", palette=pal, ax=ax)
            ax.legend(title="WHO class")
        else:
            sns.boxplot(data=props, x="AntibioticLabel", y="Percent", ax=ax)

        ax.set_title("Distribution of Testing Rates by Antibiotic",
                     fontsize=16, pad=18)
        ax.set_ylabel("Testing Percentage (%)")
        ax.set_xlabel("Antibiotic")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    def plot_inequality_metrics(
        self,
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        figsize: Tuple[int, int] = (16, 8)
    ) -> plt.Figure:
        """Side-by-side: Gini and Relative IQR."""
        varsum = self.variation_summary(include_classes=include_classes)
        if page and page_size:
            labels = sorted(varsum["AntibioticLabel"].unique().tolist())
            start = (page - 1) * page_size
            end = start + page_size
            keep = set(labels[start:end])
            varsum = varsum[varsum["AntibioticLabel"].isin(keep)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        pal = self._ensure_palette()

        sns.barplot(data=varsum, x="AntibioticLabel", y="gini",
                    hue="WHO_Class", palette=pal, ax=ax1)
        ax1.set_title("Gini Coefficient of Testing Coverage", fontsize=14)
        ax1.set_ylabel("Gini (0–1)")
        ax1.set_xlabel("Antibiotic")
        ax1.tick_params(axis='x', rotation=45)

        sns.barplot(data=varsum, x="AntibioticLabel", y="rel_iqr",
                    hue="WHO_Class", palette=pal, ax=ax2)
        ax2.set_title("Relative IQR of Testing Coverage", fontsize=14)
        ax2.set_ylabel("IQR / Median")
        ax2.set_xlabel("Antibiotic")
        ax2.tick_params(axis='x', rotation=45)

        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="WHO class",
                       loc="lower center", ncol=len(pal))
        ax1.get_legend().remove() if ax1.get_legend() else None
        ax2.get_legend().remove() if ax2.get_legend() else None

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        return fig

    def plot_disparity_forest(
        self,
        pairwise_df: pd.DataFrame,
        metric: str = "OR",
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 10),
        color_by_class: bool = True,
    ) -> plt.Figure:
        """Forest plot of effect estimates with confidence intervals (class-aware & paginated)."""
        subset = pairwise_df[pairwise_df["metric"] == metric].copy()
        if include_classes:
            subset = subset[subset["WHO_Class"].isin(include_classes)]
        if subset.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data for selected filter", ha="center")
            ax.axis("off")
            return fig

        # deterministic sort: class, antibiotic, contrast
        class_order = [
            c for c in CLASS_ORDER if c in subset["WHO_Class"].unique()]
        subset["WHO_Class"] = pd.Categorical(
            subset["WHO_Class"], categories=class_order, ordered=True)
        subset = subset.sort_values(
            ["WHO_Class", "AntibioticLabel", "contrast"]).reset_index(drop=True)

        # paginate by AntibioticLabel if requested
        if page and page_size:
            labels = subset["AntibioticLabel"].unique().tolist()
            start, end = (page - 1) * page_size, (page - 1) * \
                page_size + page_size
            keep = set(labels[start:end])
            subset = subset[subset["AntibioticLabel"].isin(
                keep)].reset_index(drop=True)
            if subset.empty:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No data on this page", ha="center")
                ax.axis("off")
                return fig

        subset["label"] = subset["AntibioticLabel"] + \
            " — " + subset["contrast"]

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(subset))
        x = subset["estimate"].values
        lo = subset["ci_l"].values
        hi = subset["ci_u"].values

        if color_by_class:
            pal = self._ensure_palette()
            # draw per-class (preserve the same y positions)
            for cls in class_order:
                idx = np.where(subset["WHO_Class"].values == cls)[0]
                if idx.size == 0:
                    continue
                ax.errorbar(
                    x[idx], y[idx],
                    xerr=[x[idx] - lo[idx], hi[idx] - x[idx]],
                    fmt="o", capsize=4, markersize=6,
                    color=pal.get(cls, "#666666"), label=cls,
                )
            ax.legend(title="WHO class")
        else:
            ax.errorbar(x, y, xerr=[x - lo, hi - x],
                        fmt="o", capsize=4, markersize=6)

        # reference line
        ref = 1 if metric in {"OR", "RR"} else 0
        ax.axvline(ref, color="red", ls="--", lw=1.8, alpha=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(subset["label"])
        ax.set_xlabel(f"{metric} with 95% CI")
        ax.set_title(f"Disparity in Testing Coverage ({metric})", pad=16)
        if metric in {"OR", "RR"}:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig

    def plot_significant_disparities(
        self,
        pairwise_df: pd.DataFrame,
        alpha: float = 0.05,
        include_classes: Optional[List[str]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8),
        color_by_class: bool = True,
        *,
        show_names: bool = True,
        name_col: str = "AntibioticLabel",
        max_name_len: int = 22,
        # ⬅️ spacing from point (in screen points)
        name_offset_pts: float = 6.0,
        name_fontsize: int = 8,
        x_margin: float = 0.08,            # ⬅️ extra margin so labels aren’t clipped
        outline_text: bool = True,         # ⬅️ white outline for readability
    ) -> plt.Figure:
        """Scatter of significant disparities (FDR < alpha), class-filtered & paginated.
        Adds small spacing between points and labels so text is clear.
        """
        import matplotlib.patheffects as pe

        sig = pairwise_df[pairwise_df["p_fdr"] < alpha].copy()
        if include_classes:
            sig = sig[sig["WHO_Class"].isin(include_classes)]
        if sig.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No significant disparities", ha="center")
            ax.axis("off")
            return fig

        # paginate by label if requested
        if page and page_size:
            labels = sig[name_col].unique().tolist()
            start, end = (page - 1) * page_size, (page - 1) * \
                page_size + page_size
            keep = set(labels[start:end])
            sig = sig[sig[name_col].isin(keep)]
            if sig.empty:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No data on this page", ha="center")
                ax.axis("off")
                return fig

        def _shorten(s: str) -> str:
            s = str(s)
            return s if len(s) <= max_name_len else s[:max_name_len-1] + "…"

        fig, ax = plt.subplots(figsize=figsize)
        metrics = sig["metric"].unique().tolist()
        pal = self._ensure_palette()

        for m in metrics:
            subm = sig[sig["metric"] == m].copy()
            subm = subm.sort_values(
                ["WHO_Class", name_col, "contrast"]).reset_index(drop=True)
            subm["pos"] = np.arange(len(subm))

            if color_by_class:
                classes = [c for c in ("Access", "Watch", "Reserve", "Unclassified")
                           if c in subm["WHO_Class"].unique()]
                if not classes:
                    classes = sorted(subm["WHO_Class"].unique().tolist())

                for cls in classes:
                    scls = subm[subm["WHO_Class"] == cls]
                    pos = scls["pos"].to_numpy()
                    col = pal.get(cls, "#666666")

                    ax.scatter(pos, scls["estimate"], s=90, alpha=0.85,
                               color=col, label=f"{m} — {cls}")
                    for x, (ci_l, ci_u) in zip(pos, scls[["ci_l", "ci_u"]].to_numpy()):
                        ax.plot([x, x], [ci_l, ci_u], color=col, alpha=0.85)

                    if show_names:
                        for x, y, label in zip(pos, scls["estimate"], scls[name_col]):
                            txt = ax.annotate(_shorten(label),
                                              xy=(x, y),
                                              # ⬅️ offset in *points*
                                              xytext=(name_offset_pts, 0),
                                              textcoords="offset points",
                                              ha="left", va="center",
                                              fontsize=name_fontsize,
                                              color=col,
                                              clip_on=False)
                            if outline_text:
                                txt.set_path_effects(
                                    [pe.withStroke(linewidth=2, foreground="white")])
            else:
                pos = subm["pos"].to_numpy()
                ax.scatter(pos, subm["estimate"], s=90, alpha=0.85, label=m)
                for x, (ci_l, ci_u) in zip(pos, subm[["ci_l", "ci_u"]].to_numpy()):
                    ax.plot([x, x], [ci_l, ci_u], alpha=0.85)
                if show_names:
                    for x, y, label in zip(pos, subm["estimate"], subm[name_col]):
                        txt = ax.annotate(_shorten(label),
                                          xy=(x, y), xytext=(
                                              name_offset_pts, 0),
                                          textcoords="offset points",
                                          ha="left", va="center",
                                          fontsize=name_fontsize)
                        if outline_text:
                            txt.set_path_effects(
                                [pe.withStroke(linewidth=2, foreground="white")])

        # reference lines
        if {"OR", "RR"} & set(metrics):
            ax.axhline(1, color="red", ls="--", alpha=0.7)
        if "RD_pp" in metrics:
            ax.axhline(0, color="red", ls="--", alpha=0.7)

        ax.set_xlabel("Contrast index")
        ax.set_ylabel("Effect size")
        ax.set_title(
            f"Statistically Significant Disparities (FDR < {alpha})", pad=14)
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.25)

        # give some horizontal breathing room for labels
        ax.margins(x=x_margin)

        plt.tight_layout()
        return fig

    def plot_stratum_performance(
        self,
        figsize: Tuple[int, int] = (14, 8),
        top_n: int = 20
    ) -> plt.Figure:
        """Top N strata by average testing percentage."""
        props = self.proportions_table()
        stratum_avg = props.groupby(self.stratum_col, observed=False)[
            "Percent"].mean().sort_values(ascending=False)
        top = stratum_avg.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top)))
        bars = ax.bar(range(len(top)), top.values, color=colors)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top.index, rotation=45, ha="right")
        ax.set_ylabel("Average Testing Percentage (%)")
        ax.set_title(
            f"Top {top_n} Performing {self.stratum_col}s by Testing Coverage", pad=14)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        return fig

    # ---------- Exporters ----------

    def export_all(self,
                   out_dir: str,
                   ci: str = "wilson",
                   disparity_models: Iterable[str] = ("logit", "log", "rd"),
                   fdr_alpha: float = 0.05,
                   threshold: float = 50.0
                   ) -> Dict[str, str]:
        """CSV tables: proportions, variation, disparity pairwise/global."""
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        props = self.proportions_table(ci=ci)
        p_props = f"{out_dir}/proportions_with_ci.csv"
        props.to_csv(p_props, index=False, encoding="utf-8-sig")
        paths["proportions"] = p_props

        varsum = self.variation_summary(threshold=threshold, by_compare=True)
        p_var = f"{out_dir}/variation_summary.csv"
        varsum.to_csv(p_var, index=False, encoding="utf-8-sig")
        paths["variation"] = p_var

        for m in disparity_models:
            pairwise, globals_ = self.disparity_summary(
                model=m, fdr_alpha=fdr_alpha)
            p_pw = f"{out_dir}/disparity_{m}_pairwise.csv"
            p_gl = f"{out_dir}/disparity_{m}_global.csv"
            pairwise.to_csv(p_pw, index=False, encoding="utf-8-sig")
            globals_.to_csv(p_gl, index=False, encoding="utf-8-sig")
            paths[f"disparity_{m}_pairwise"] = p_pw
            paths[f"disparity_{m}_global"] = p_gl

        return paths

    def export_all_visualizations(
        self,
        out_dir: str,
        disparity_models: Iterable[str] = ("logit", "log", "rd"),
        fdr_alpha: float = 0.05,
        threshold: float = 50.0,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Overall figures. If `classes` is provided, also save per-class variants.
        Dynamically scales figure height for readability."""
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # ---- sizing helpers ----
        props = self._attach_labels_classes(self.proportions_table())
        n_strata = int(props[self.stratum_col].nunique())
        n_abx = int(props["AntibioticLabel"].nunique())

        def h_heatmap(n_rows): return max(
            6, 0.50 * n_rows)          # 0.50" per stratum

        # 0.38" per abx (cap at 36")
        def h_box(n_items): return max(8, min(36, 0.38 * n_items))
        def h_bar(n_items): return max(8, 0.45 * n_items)
        def h_forest(n_rows): return max(
            10, min(48, 0.28 * n_rows))  # 0.28" per row, cap 48"

        def h_sig(n_points): return max(8, min(30, 0.22 * n_points))

        # ---- unfiltered overall figures ----
        # fig = self.plot_coverage_heatmap(figsize=(16, h_heatmap(n_strata)))
        # p = os.path.join(out_dir, "coverage_heatmap.png")
        # fig.savefig(p, dpi=300, bbox_inches="tight")
        # plt.close(fig)
        # paths["coverage_heatmap"] = p

        fig = self.plot_inequality_metrics(figsize=(16, h_box(n_abx)))
        p = os.path.join(out_dir, "inequality_metrics.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["inequality_metrics"] = p

        fig = self.plot_stratum_performance(
            figsize=(14, h_bar(min(20, n_strata))))
        p = os.path.join(out_dir, "stratum_performance.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["stratum_performance"] = p

        # fig = self.plot_antibiotic_comparison(
        #     color_by_class=True, figsize=(14, h_box(n_abx)))
        # p = os.path.join(out_dir, "antibiotic_comparison.png")
        # fig.savefig(p, dpi=300, bbox_inches="tight")
        # plt.close(fig)
        # paths["antibiotic_comparison"] = p

        for m in disparity_models:
            pw, _ = self.disparity_summary(model=m, fdr_alpha=fdr_alpha)
            if pw.empty:
                continue
            metric = "OR" if m == "logit" else "RR" if m == "log" else "RD_pp"

            # rows in the forest = number of lines plotted on y-axis
            rows_forest = len(pw[pw["metric"] == metric])
            fig = self.plot_disparity_forest(pw, metric=metric, color_by_class=True,
                                             figsize=(12, h_forest(rows_forest)))
            p = os.path.join(out_dir, f"disparity_forest_{m}.png")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths[f"disparity_forest_{m}"] = p

            sig = pw[(pw["metric"] == metric) & (pw["p_fdr"] < fdr_alpha)]
            fig = self.plot_significant_disparities(pw, alpha=fdr_alpha, color_by_class=True,
                                                    figsize=(12, h_sig(len(sig))))
            p = os.path.join(out_dir, f"significant_disparities_{m}.png")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths[f"significant_disparities_{m}"] = p

        # ---- optional per-class overall variants ----
        if classes:
            classes = [c for c in classes if c in CLASS_ORDER]
            for cls in classes:
                cls_dir = os.path.join(out_dir, f"by_class_{cls}")
                os.makedirs(cls_dir, exist_ok=True)

                n_abx_cls = int(
                    props.loc[props["WHO_Class"] == cls, "AntibioticLabel"].nunique())

                # fig = self.plot_coverage_heatmap(include_classes=[cls],
                #                                  figsize=(16, h_heatmap(n_strata)))
                # p = os.path.join(cls_dir, f"coverage_heatmap_{cls}.png")
                # fig.savefig(p, dpi=300, bbox_inches="tight")
                # plt.close(fig)
                # paths[f"coverage_heatmap_{cls}"] = p

                fig = self.plot_inequality_metrics(include_classes=[cls],
                                                   figsize=(16, h_box(n_abx_cls)))
                p = os.path.join(cls_dir, f"inequality_metrics_{cls}.png")
                fig.savefig(p, dpi=300, bbox_inches="tight")
                plt.close(fig)
                paths[f"inequality_metrics_{cls}"] = p

                # fig = self.plot_antibiotic_comparison(include_classes=[cls],
                #                                       color_by_class=True,
                #                                       figsize=(14, h_box(n_abx_cls)))
                # p = os.path.join(cls_dir, f"antibiotic_comparison_{cls}.png")
                # fig.savefig(p, dpi=300, bbox_inches="tight")
                # plt.close(fig)
                # paths[f"antibiotic_comparison_{cls}"] = p

                for dm in disparity_models:
                    pw_dm, _ = self.disparity_summary(
                        model=dm, fdr_alpha=fdr_alpha)
                    if pw_dm.empty:
                        continue
                    metric = "OR" if dm == "logit" else "RR" if dm == "log" else "RD_pp"

                    sub = pw_dm[(pw_dm["metric"] == metric) &
                                (pw_dm["WHO_Class"] == cls)]
                    if sub.empty:
                        continue

                    fig = self.plot_disparity_forest(pw_dm, metric=metric, include_classes=[cls],
                                                     color_by_class=True,
                                                     figsize=(12, h_forest(len(sub))))
                    p = os.path.join(
                        cls_dir, f"disparity_forest_{dm}_{cls}.png")
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    paths[f"disparity_forest_{dm}_{cls}"] = p

                    sig = sub[sub["p_fdr"] < fdr_alpha]
                    fig = self.plot_significant_disparities(pw_dm, alpha=fdr_alpha,
                                                            include_classes=[
                                                                cls], color_by_class=True,
                                                            figsize=(12, h_sig(len(sig))))
                    p = os.path.join(
                        cls_dir, f"significant_disparities_{dm}_{cls}.png")
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    paths[f"significant_disparities_{dm}_{cls}"] = p

        return paths

    def export_all_paginated_figures(
        self,
        out_dir: str,
        classes: Optional[List[str]] = None,
        page_size: int = 30,
        disparity_models: Iterable[str] = ("logit", "log", "rd"),
        fdr_alpha: float = 0.05,
    ) -> Dict[str, List[str]]:
        """Save class-filtered, paginated figures with dynamic heights."""
        os.makedirs(out_dir, exist_ok=True)
        saved: Dict[str, List[str]] = {}

        props = self._attach_labels_classes(self.proportions_table())
        avail = sorted(props["WHO_Class"].dropna().unique().tolist())
        if classes is None:
            classes = [c for c in CLASS_ORDER if c in avail]
        else:
            classes = [c for c in classes if c in avail]

        # helpers
        def h_heatmap(n_rows): return max(6, 0.50 * n_rows)
        def h_box(n): return max(8, min(36, 0.38 * n))
        def h_ineq(n): return max(8, min(36, 0.38 * n))
        def h_forest(n): return max(10, min(48, 0.28 * n))
        def h_sig(n): return max(8, min(30, 0.22 * n))

        # precompute disparity
        pw_by = {m: self.disparity_summary(model=m, fdr_alpha=fdr_alpha)[0]
                 for m in disparity_models}

        def _save(fig, path):
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.setdefault("figures", []).append(path)

        for cls in classes:
            labels_all = sorted(
                props.loc[props["WHO_Class"] == cls, "AntibioticLabel"].unique().tolist())
            pages = max(1, int(np.ceil(len(labels_all) / page_size))
                        ) if page_size else 1

            base_dir = os.path.join(out_dir, f"class_{cls}")
            os.makedirs(base_dir, exist_ok=True)

            for page in range(1, pages + 1):
                page_labels = labels_all[(
                    page - 1) * page_size: page * page_size]
                n_page = len(page_labels)
                page_dir = os.path.join(base_dir, f"page_{page:02d}")
                os.makedirs(page_dir, exist_ok=True)

                # fig = self.plot_coverage_heatmap(include_classes=[cls], page=page, page_size=page_size,
                #                                  figsize=(16, h_heatmap(props[self.stratum_col].nunique())))
                # _save(fig, os.path.join(
                #     page_dir, f"heatmap_{cls}_p{page:02d}.png"))

                # fig = self.plot_antibiotic_comparison(include_classes=[cls], page=page, page_size=page_size,
                #                                       color_by_class=True,
                #                                       figsize=(14, h_box(n_page)))
                # _save(fig, os.path.join(
                #     page_dir, f"boxplot_{cls}_p{page:02d}.png"))

                fig = self.plot_inequality_metrics(include_classes=[cls], page=page, page_size=page_size,
                                                   figsize=(16, h_ineq(n_page)))
                _save(fig, os.path.join(
                    page_dir, f"inequality_{cls}_p{page:02d}.png"))

                for m, pw in pw_by.items():
                    if pw.empty:
                        continue
                    metric = "OR" if m == "logit" else "RR" if m == "log" else "RD_pp"
                    sub = pw[(pw["metric"] == metric) & (pw["WHO_Class"] == cls) &
                             (pw["AntibioticLabel"].isin(page_labels))]

                    if sub.empty:
                        continue

                    fig = self.plot_disparity_forest(pw, metric=metric, include_classes=[cls],
                                                     page=page, page_size=page_size,
                                                     figsize=(12, h_forest(len(sub))))
                    _save(fig, os.path.join(
                        page_dir, f"forest_{metric}_{cls}_p{page:02d}.png"))

                    sig = sub[sub["p_fdr"] < fdr_alpha]
                    fig = self.plot_significant_disparities(pw, include_classes=[cls], page=page,
                                                            page_size=page_size, alpha=fdr_alpha,
                                                            figsize=(12, h_sig(len(sig))))
                    _save(fig, os.path.join(
                        page_dir, f"significant_{metric}_{cls}_p{page:02d}.png"))

        return saved

    def save_all_everything(
        self,
        out_dir: str,
        ci: str = "wilson",
        disparity_models: Iterable[str] = ("logit", "log", "rd"),
        fdr_alpha: float = 0.05,
        threshold: float = 50.0,
        page_size: int = 30,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Save CSV tables, overall figures (optionally per-class),
        and paginated WHO-class figure sets.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(out_dir, f"analysis_{timestamp}")
        tables_dir = os.path.join(root, "tables")
        figures_dir = os.path.join(root, "figures_overall")
        class_dir = os.path.join(root, "figures_by_class")
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(class_dir, exist_ok=True)

        tables = self.export_all(
            tables_dir, ci=ci, disparity_models=disparity_models,
            fdr_alpha=fdr_alpha, threshold=threshold
        )
        figs = self.export_all_visualizations(
            figures_dir, disparity_models=disparity_models,
            fdr_alpha=fdr_alpha, threshold=threshold,
            classes=classes,                                   # <— pass classes here
        )
        classf = self.export_all_paginated_figures(
            class_dir, classes=classes, page_size=page_size,
            disparity_models=disparity_models, fdr_alpha=fdr_alpha
        )

        meta_path = os.path.join(root, "README.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("Antibiotic Testing Coverage — Publication Bundle\n")
            f.write(f"Generated: {timestamp}\n\n")
            f.write("Tables:\n")
            for k, v in tables.items():
                f.write(f"  {k}: {os.path.relpath(v, root)}\n")
            f.write("\nOverall figures:\n")
            for k, v in figs.items():
                f.write(f"  {k}: {os.path.relpath(v, root)}\n")
            f.write("\nPaginated class figures:\n")
            for p in classf.get("figures", []):
                f.write(f"  {os.path.relpath(p, root)}\n")

        return {
            "tables": list(tables.values()),
            "overall_figures": list(figs.values()),
            "class_figures": classf.get("figures", []),
            "readme": meta_path,
        }


# ---------------------- one-pass scenario runner ----------------------


def prepare_ward_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add WardGroup and WardPhase once. Uses ARS_WardType or UniqueARS_WardType if present."""
    df = df.copy()
    ward_src = "ARS_WardType"
    if ward_src not in df.columns:
        ward_src = "UniqueARS_WardType" if "UniqueARS_WardType" in df.columns else None
    if ward_src is None:
        df["WardGroup"] = "Other/Unknown"
    else:
        ward_map = {
            "Outpatient": "Outpatient",
            "Day Clinic": "Day Clinic",
            "Normal Ward": "Normal Ward",
            "Intermediate Care/Awake Station": "Intermediate Care",
            "Intensive Care Unit": "ICU",
            "Operating Room": "Operating Room",
            "Early Rehabilitation": "Rehabilitation",
            "Rehabilitation": "Rehabilitation",
            "Other Treatment Type": "Other/Unknown",
            "Unknown": "Other/Unknown",
        }
        df["WardGroup"] = df[ward_src].map(ward_map).fillna("Other/Unknown")

    phase_map = {
        "ICU": "Acute",
        "Intermediate Care": "Acute",
        "Normal Ward": "Acute",
        "Rehabilitation": "Post-acute",
    }
    df["WardPhase"] = df["WardGroup"].map(
        phase_map).fillna("Other/Not-included")
    df["WardPhase"] = pd.Categorical(
        df["WardPhase"], ["Acute", "Post-acute", "Other/Not-included"], ordered=True
    )
    # print(df.columns)
    return df


def preflight(df: pd.DataFrame, test_suffix: str = "_Tested"):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    abx_cols = [c for c in df.columns if c.endswith(test_suffix)]
    if not abx_cols:
        raise ValueError(
            f"No antibiotic indicator columns ending with '{test_suffix}'.")
    nonbin = []
    for c in abx_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not s.isin([0, 1]).all():
            nonbin.append(c)
    if nonbin:
        raise ValueError(
            f"Non-binary values found in: {nonbin[:12]}{' …' if len(nonbin) > 12 else ''}")
    print(
        f"Preflight OK — {len(df):,} rows, {len(df.columns)} cols, {len(abx_cols)} antibiotics.")



def build_class_map() -> Dict[str, str]:
    load = LoadClasses()
    class_map: Dict[str, str] = {}
    for cls in WHO_CLASSES:
        for col in load.convert_to_tested_columns(load.get_antibiotics_by_category([cls])):
            class_map[str(col)] = str(cls)
    return class_map


def filter_ab_dict(df, ab_dict):
    filtered = {k: v for k, v in ab_dict.items() if k in df.columns}
    missing = [k for k in ab_dict if k not in df.columns]
    return filtered, missing


def run_publication_stats_over_time(
    df: pd.DataFrame,
    *,
    out_root: Union[str, Path],
    stratum_col: str,
    compare_col: str,
    antibiotic_suffix: str,
    class_map: Dict[str, str],
    who_classes: Iterable[str],
    disparity_models: Iterable[str] = ("logit", "log", "rd"),
    fdr_alpha: float = 0.05,
    var_threshold: float = 50.0,
    page_size: int = 30,
    ci: str = "wilson",
) -> Dict[str, Any]:
    """
    Run PublicationStats separately for each year (or season),
    save outputs, and combine summary tables for trend analysis.
    """
    out_root = Path(out_root)
    os.makedirs(out_root, exist_ok=True)

    results_by_year: Dict[str, Any] = {}
    years = sorted(df["Year"].dropna().unique().tolist())

    for year in years:
        dfi = df[df["Year"] == year].copy()
        if dfi.empty:
            continue

        pub = (
            PublicationStats.from_wide(
                df=dfi,
                stratum_col=stratum_col,
                compare_col=compare_col,
                antibiotic_suffix=antibiotic_suffix,
            )
            .with_class_map(class_map)
        )

        bundle = pub.save_all_everything(
            out_dir=str(out_root / f"year_{year}"),
            ci=ci,
            disparity_models=disparity_models,
            fdr_alpha=fdr_alpha,
            threshold=var_threshold,
            page_size=page_size,
            classes=list(who_classes),
        )
        results_by_year[str(year)] = bundle

    return results_by_year


def collect_variation_trends(out_root: Union[str, Path]) -> pd.DataFrame:
    """
    Intelligently combines yearly variation summaries from dynamic, timestamped analysis folders.

    This function searches for files matching the pattern:
    'year_*/analysis_*/tables/variation_summary.csv'

    Args:
        out_root: The root directory containing the 'year_*' folders.

    Returns:
        A single pandas DataFrame containing the combined data from all years,
        or an empty DataFrame if no files are found.
    """
    out_root = Path(out_root)
    all_dataframes: List[pd.DataFrame] = []

    # Use a single, more powerful glob pattern to find all relevant files at once.
    # The first '*' matches the year number, and the second '*' matches the dynamic timestamp.
    search_pattern = "year_*/analysis_*/tables/variation_summary.csv"

    for file_path in out_root.glob(search_pattern):
        try:
            # Intelligently extract the year from the path.
            # We look for the part of the path that starts with 'year_' and split it.
            year_str = next(part.split(
                '_')[1] for part in file_path.parts if part.startswith('year_'))

            df = pd.read_csv(file_path)
            df["Year"] = int(year_str)
            all_dataframes.append(df)

        except (StopIteration, IndexError, ValueError):
            # This handles cases where a folder might be named incorrectly (e.g., 'year_')
            print(
                f"Warning: Could not extract a valid year from path: {file_path}")
            continue

    if not all_dataframes:
        return pd.DataFrame()

    return pd.concat(all_dataframes, ignore_index=True)


def plot_trends_as_bars(var_df: pd.DataFrame, metric: str = "median_pct"):
    """
    Plots elegant time trends as a grouped bar chart with error bars.

    Each year on the x-axis has a cluster of bars, one for each WHO Class.
    Error bars show the 95% confidence interval, calculated by Seaborn.
    """
    if var_df.empty:
        print("No variation data found.")
        return None

    METRIC_LABELS = {
        "median_pct": "Median Testing Coverage (%)",
        "gini": "Gini Coefficient (Inequality)",
        "rel_iqr": "Relative Interquartile Range (%)",
    }
    y_label = METRIC_LABELS.get(metric, metric)

    sns.set_theme(style="whitegrid", context="talk")
    # Make the figure a bit wider to comfortably fit the grouped bars
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- CORE CHANGE: Use barplot instead of lineplot ---
    sns.barplot(
        data=var_df,
        x="Year",
        y=metric,
        hue="WHO_Class",
        palette="viridis",  # A nice colorblind-friendly palette
        ax=ax
    )

    # --- All the "perfection" from before is retained ---
    # a) Force the X-axis (Year) to only show integers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=0)

    # b) Make the Y-axis truly intelligent by auto-detecting the data's scale
    is_percent_metric = "pct" in metric.lower() or "iqr" in metric.lower()
    if is_percent_metric:
        max_val = var_df[metric].max()
        if max_val <= 1.0:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
            ax.set_ylim(0, 1.0)
        else:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, var_df[metric].max() * 1.1)

    # --- Improve Titles, Labels, and Legend ---
    fig.suptitle("Evolution of Diagnostic Testing Variation by WHO AWaRe Class",
                 fontsize=20, weight='bold')
    ax.set_title(
        f"Metric: {y_label.split('(')[0].strip()}", fontsize=14, style='italic', y=1.02)

    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.legend(title='WHO AWaRe Class', bbox_to_anchor=(
        1.02, 1), loc='upper left', frameon=True)

    # --- Final Touches ---
    sns.despine(left=True, bottom=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def run_all_publication_stats(
    df: pd.DataFrame,
    *,
    out_root: Union[str, Path],
    stratum_col: str,
    antibiotic_suffix: str,
    class_map: Dict[str, str],
    who_classes: Iterable[str],
    disparity_models: Iterable[str] = ("logit", "log", "rd"),
    fdr_alpha: float = 0.05,
    var_threshold: float = 50.0,
    page_size: int = 30,
    ci: str = "wilson",
) -> Dict[str, Any]:
    """
    Execute the full publication bundle once for each scenario.
    Returns {scenario_tag: bundle_dict}.
    """
    df = prepare_ward_columns(df)
    scenarios: List[Tuple[str, str, List[str]]] = [
        ("sex_man_woman",              "Sex",
         ["Man", "Woman"]),
        ("gram_gp_gn",                 "GramType",
         ["Gram-positive", "Gram-negative"]),
        ("materials_urine_blood",      "TextMaterialgroupRkiL0",
         ["Urine", "Blood Culture"]),
        ("caretype_in_vs_out",         "CareType",
         ["In-Patient", "Out-Patient"]),
        ("wardphase_acute_vs_post",    "WardPhase",
         ["Acute", "Post-acute"]),
        ("wardgap_ICU_vs_normal",      "WardGroup",
         ["ICU", "Normal Ward"]),
        ("wardgap_ICU_vs_IMC",         "WardGroup",
         ["ICU", "Intermediate Care"]),
        ("age_pediatric_vs_elderly",   "AgeGroup",
         ["Pediatric", "Elderly"]),
        ("age_elderly_vs_adult",       "AgeGroup",
         ["Elderly", "Adult"]),
    ]

    results: Dict[str, Any] = {}
    out_root = Path(out_root)

    for tag, compare_col, values in scenarios:
        if compare_col not in df.columns:
            # skip gracefully if a column is missing
            results[tag] = {"skipped": f"missing column {compare_col}"}
            continue

        dfi = df[df[compare_col].isin(values)].copy()
        if dfi.empty or dfi[compare_col].nunique() < len(values):
            results[tag] = {
                "skipped": f"insufficient levels for {compare_col}: {values}"}
            continue

        # enforce desired display order
        dfi[compare_col] = pd.Categorical(
            dfi[compare_col], categories=values, ordered=True)

        pub = (
            PublicationStats.from_wide(
                df=dfi,
                stratum_col=stratum_col,
                compare_col=compare_col,
                antibiotic_suffix=antibiotic_suffix,
            )
            .with_class_map(class_map)
        )

        bundle = pub.save_all_everything(
            out_dir=str(out_root / "publication_stats" / tag),
            ci=ci,
            disparity_models=disparity_models,
            fdr_alpha=fdr_alpha,
            threshold=var_threshold,
            page_size=page_size,
            classes=list(who_classes),
        )
        results[tag] = bundle

    return results


class DisparityTrends:
    """Helper class to collect and summarize disparity results across years."""

    @classmethod
    def collect(cls, out_root: str, model: str = "log") -> pd.DataFrame:
        """
        Stack disparity_{model}_pairwise.csv across year_* subfolders under out_root.
        Returns tidy DF with Year, Antibiotic, WHO_Class, contrast, metric, estimate, ci_l, ci_u, p_fdr.
        """
        out_root = Path(out_root)
        metric = {"log": "RR", "logit": "OR", "rd": "RD_pp"}[model]
        rows = []
        for f in out_root.glob(f"year_*/analysis_*/tables/disparity_{model}_pairwise.csv"):
            # print(f)
            try:
                year = int(next(p.split("_")[1]
                           for p in f.parts if p.startswith("year_")))
                df = pd.read_csv(f)
                df["Year"] = year
                rows.append(df[df["metric"] == metric])
            except Exception:
                continue
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
