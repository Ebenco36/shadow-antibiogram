from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# Your existing wide-matrix metrics (only used in WIDE mode)
from shadow_antibiogram.controllers.similarity.Metrics import (
    SimilarityMetric,
    JaccardMetric, DiceMetric, CosineMetric, OverlapMetric,
    PhiMetric,
    # (keep the rest imported if you want, but we'll guard them)
)

# ----------------------------
# Helpers
# ----------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (q-values) for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.zeros_like(num, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


@dataclass(frozen=True)
class PairwiseFDRResult:
    pvals: np.ndarray
    qvals: np.ndarray
    keep: np.ndarray


# ----------------------------
# Pairwise metric formulas from a,b,c,d
# ----------------------------

class PairwiseMetrics:
    @staticmethod
    def jaccard(a, b, c, d=None):
        # a / (a+b+c)
        return _safe_div(a, (a + b + c))

    @staticmethod
    def dice(a, b, c, d=None):
        # 2a / (2a+b+c)
        return _safe_div(2 * a, (2 * a + b + c))

    @staticmethod
    def cosine(a, b, c, d=None):
        # a / sqrt((a+b)(a+c))
        return _safe_div(a, np.sqrt((a + b) * (a + c)))

    @staticmethod
    def overlap(a, b, c, d=None):
        # a / min(a+b, a+c)
        return _safe_div(a, np.minimum(a + b, a + c))

    @staticmethod
    def phi(a, b, c, d):
        # (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        num = (a * d) - (b * c)
        den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        return _safe_div(num, den)


# ----------------------------
# Dual backend CoTestAnalyzer
# ----------------------------

class CoTestAnalyzer:
    """
    Backwards-compatible analyzer supporting TWO dataset layouts:

    1) WIDE isolate-level:
        - antibiotics are columns ending with _Tested
        - uses your existing Metrics classes in src.controllers.similarity.Metrics

    2) PAIRWISE aggregated:
        - required columns: ab_1, ab_2, a, b, c, d
        - computes valid metrics directly from contingency counts
        - supports Fisher + BH-FDR pruning

    IMPORTANT:
      Many of your advanced metrics (TFIDF/CFWS/JS/PMI/NPMI/ACT...) require isolate-level data.
      In PAIRWISE mode we raise a clear error for those.
    """

    PAIRWISE_CORE = {"ab_1", "ab_2", "a", "b", "c", "d"}

    def __init__(self, transactions: pd.DataFrame, antibiotic_cols: Optional[List[str]] = None):
        self.transactions = transactions.copy()
        self.is_pairwise = self.PAIRWISE_CORE.issubset(set(self.transactions.columns))

        if self.is_pairwise:
            # abx "cols" are actually antibiotic labels (values) in ab_1/ab_2
            self.abx_cols = antibiotic_cols or self._infer_pairwise_abx_labels(self.transactions)
        else:
            if antibiotic_cols is None:
                raise ValueError("WIDE mode requires antibiotic_cols (list of *_Tested columns).")
            self.abx_cols = antibiotic_cols

        # output caches (optional)
        self.last_fdr: Optional[PairwiseFDRResult] = None

    # ----------------------------
    # Format detection helpers
    # ----------------------------

    @staticmethod
    def _infer_pairwise_abx_labels(df: pd.DataFrame) -> List[str]:
        s1 = set(df["ab_1"].astype("string").dropna().astype(str).unique())
        s2 = set(df["ab_2"].astype("string").dropna().astype(str).unique())
        return sorted(s1.union(s2))

    def _require_wide(self, what: str) -> None:
        if self.is_pairwise:
            raise TypeError(
                f"{what} requires isolate-level WIDE data (binary *_Tested columns). "
                f"Your input is PAIRWISE aggregated (ab_1/ab_2/a/b/c/d)."
            )

    def _require_pairwise(self, what: str) -> None:
        if not self.is_pairwise:
            raise TypeError(
                f"{what} requires PAIRWISE aggregated data (ab_1/ab_2/a/b/c/d). "
                f"Your input is WIDE isolate-level."
            )

    # ----------------------------
    # Label mapping (works for both)
    # ----------------------------

    def create_label_mapping(self, format_type: str = "combined", remove_suffix: str = "_Tested") -> dict:
        label_map = {}
        for col in self.abx_cols:
            cleaned = str(col).replace(remove_suffix, "")
            parts = cleaned.split(" - ", 1)
            if len(parts) == 2:
                abbr, full = parts
                if format_type == "abbr":
                    label_map[col] = abbr
                elif format_type == "full":
                    label_map[col] = full
                else:
                    label_map[col] = cleaned
            else:
                label_map[col] = cleaned
        return label_map

    # ============================================================
    # WIDE backend (your existing implementation style)
    # ============================================================

    def _maybe_apply_fdr_wide(self, metric_obj: Any, mat: pd.DataFrame, fdr: dict | None):
        if not fdr:
            return mat, None, None
        if not hasattr(metric_obj, "fdr"):
            raise TypeError(f"{type(metric_obj).__name__} does not support FDR (add FDRMixin).")
        res = metric_obj.fdr(**fdr)
        pruned = pd.DataFrame(res.sim_pruned, index=mat.index, columns=mat.columns)
        edges = metric_obj.edges_from_fdr(res)
        return pruned, edges, res

    def compute_metric(
        self,
        metric_cls: Type[SimilarityMetric],
        *,
        right=None,
        fdr: dict | None = None,
        return_edges: bool = False,
        **kwargs,
    ):
        self._require_wide(f"compute_metric({getattr(metric_cls, '__name__', metric_cls)})")

        metric = metric_cls(self.transactions, left_cols=self.abx_cols, right=right, **kwargs)
        mat = metric.compute()
        mat2, edges, _ = self._maybe_apply_fdr_wide(metric, mat, fdr)
        return (mat2, edges) if return_edges else mat2

    def compute_metric_long(
        self,
        metric_cls: Type[SimilarityMetric],
        *,
        right=None,
        drop_self=True,
        triangle="upper",
        sort_by="similarity",
        ascending=False,
        round_to=3,
        topk=None,
        include_pvalues=False,
        fdr: dict | None = None,
        **kwargs,
    ):
        self._require_wide(f"compute_metric_long({getattr(metric_cls, '__name__', metric_cls)})")

        metric = metric_cls(self.transactions, left_cols=self.abx_cols, right=right, **kwargs)
        mat = metric.compute()
        mat2, _, res = self._maybe_apply_fdr_wide(metric, mat, fdr)
        df_long = metric.as_long(
            mat2,
            drop_self=drop_self,
            triangle=triangle,
            sort_by=sort_by,
            ascending=ascending,
            round_to=round_to,
            topk=topk,
        )
        if include_pvalues and res is not None:
            p_df = pd.DataFrame(res.pvals, index=mat.index, columns=mat.columns).stack().rename("p_value").reset_index()
            q_df = pd.DataFrame(res.qvals, index=mat.index, columns=mat.columns).stack().rename("q_value").reset_index()
            p_df.columns = ["left", "right", "p_value"]
            q_df.columns = ["left", "right", "q_value"]
            df_long = df_long.merge(p_df, on=["left", "right"], how="left").merge(q_df, on=["left", "right"], how="left")
        return df_long

    # ============================================================
    # PAIRWISE backend (new aggregated data)
    # ============================================================

    def _pairwise_filter_to_abx(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.abx_cols:
            return df
        s = set(map(str, self.abx_cols))
        out = df[df["ab_1"].astype(str).isin(s) & df["ab_2"].astype(str).isin(s)].copy()
        return out

    def _pairwise_metric_values(self, metric: str, df: pd.DataFrame) -> np.ndarray:
        a = df["a"].to_numpy(dtype=float)
        b = df["b"].to_numpy(dtype=float)
        c = df["c"].to_numpy(dtype=float)
        d = df["d"].to_numpy(dtype=float)

        m = metric.lower()
        if m in ("jaccard", "jac"):
            return PairwiseMetrics.jaccard(a, b, c, d)
        if m in ("dice", "sorensen"):
            return PairwiseMetrics.dice(a, b, c, d)
        if m in ("cosine", "cos"):
            return PairwiseMetrics.cosine(a, b, c, d)
        if m in ("overlap", "ovl"):
            return PairwiseMetrics.overlap(a, b, c, d)
        if m in ("phi",):
            return PairwiseMetrics.phi(a, b, c, d)

        raise ValueError(
            f"Metric '{metric}' not available in PAIRWISE mode. "
            f"Supported: jaccard, dice, cosine, overlap, phi."
        )

    def _pairwise_square_matrix(self, df: pd.DataFrame, vals: np.ndarray, fill_diagonal: float = 1.0) -> pd.DataFrame:
        ab1 = df["ab_1"].astype(str).to_numpy()
        ab2 = df["ab_2"].astype(str).to_numpy()
        nodes = pd.Index(sorted(set(ab1).union(set(ab2))))
        mat = pd.DataFrame(0.0, index=nodes, columns=nodes)

        tmp = pd.DataFrame({"ab1": ab1, "ab2": ab2, "val": vals})
        tmp = tmp.groupby(["ab1", "ab2"], as_index=False)["val"].mean()

        for r in tmp.itertuples(index=False):
            mat.at[r.ab1, r.ab2] = r.val
            mat.at[r.ab2, r.ab1] = r.val

        np.fill_diagonal(mat.values, fill_diagonal)
        return mat

    def compute_pairwise_matrix(
        self,
        metric: str,
        *,
        fill_diagonal: float = 1.0,
        tau: Optional[float] = None,
        fdr: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        PAIRWISE: compute a similarity matrix from (ab_1,ab_2,a,b,c,d).

        Optional:
          - tau: keep only similarities >= tau (sets others to 0)
          - fdr: dict like {"tau":0.3,"alpha":0.05,"alternative":"greater"} to prune by Fisher+BH
        """
        self._require_pairwise(f"compute_pairwise_matrix({metric})")

        df = self.transactions.copy()
        for c in ["a", "b", "c", "d"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        df = self._pairwise_filter_to_abx(df)
        if df.empty:
            return pd.DataFrame()

        vals = self._pairwise_metric_values(metric, df)

        # optional tau threshold
        if tau is not None:
            vals = np.where(vals >= float(tau), vals, 0.0)

        # optional Fisher+BH pruning
        if fdr is not None:
            tau0 = float(fdr.get("tau", 0.0))
            alpha = float(fdr.get("alpha", 0.05))
            alternative = str(fdr.get("alternative", "greater"))

            # apply tau gate for tests
            gate = vals >= tau0
            df_g = df.loc[gate].copy()
            vals_g = vals[gate]

            if df_g.empty:
                return self._pairwise_square_matrix(df, np.zeros(len(df)), fill_diagonal=fill_diagonal)

            pvals = []
            for r in df_g.itertuples(index=False):
                a = int(r.a); b = int(r.b); c = int(r.c); d = int(r.d)
                _, p = fisher_exact([[a, b], [c, d]], alternative=alternative)
                pvals.append(p)

            pvals = np.asarray(pvals, dtype=float)
            qvals = bh_fdr(pvals)
            keep = qvals <= alpha
            self.last_fdr = PairwiseFDRResult(pvals=pvals, qvals=qvals, keep=keep)

            # zero-out non-kept similarities
            vals_g = np.where(keep, vals_g, 0.0)

            # put back into full-length vector aligned to df
            vals_full = np.zeros(len(df), dtype=float)
            vals_full[gate] = vals_g
            vals = vals_full

        return self._pairwise_square_matrix(df, vals, fill_diagonal=fill_diagonal)

    def compute_pairwise_long(
        self,
        metric: str,
        *,
        drop_self: bool = True,
        triangle: str = "upper",
        sort_by: str = "similarity",
        ascending: bool = False,
        round_to: Optional[int] = 6,
        topk: Optional[int] = None,
        tau: Optional[float] = None,
        fdr: Optional[dict] = None,
        include_pq: bool = False,
    ) -> pd.DataFrame:
        """
        PAIRWISE: return long edge list with similarity (+ optional p/q if fdr used).
        """
        self._require_pairwise(f"compute_pairwise_long({metric})")

        df = self.transactions.copy()
        for c in ["a", "b", "c", "d"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        df = self._pairwise_filter_to_abx(df)
        if df.empty:
            return pd.DataFrame(columns=["left", "right", "similarity", "a", "b", "c", "d"])

        sim = self._pairwise_metric_values(metric, df)

        if tau is not None:
            keep_tau = sim >= float(tau)
            df = df.loc[keep_tau].copy()
            sim = sim[keep_tau]

        out = df[["ab_1", "ab_2", "a", "b", "c", "d"]].copy()
        out.rename(columns={"ab_1": "left", "ab_2": "right"}, inplace=True)
        out["similarity"] = sim

        # apply fdr pruning (also attaches p/q if requested)
        if fdr is not None:
            tau0 = float(fdr.get("tau", 0.0))
            alpha = float(fdr.get("alpha", 0.05))
            alternative = str(fdr.get("alternative", "greater"))

            gate = out["similarity"].to_numpy() >= tau0
            pvals = np.full(len(out), np.nan, dtype=float)

            idx = np.where(gate)[0]
            for ii in idx:
                r = out.iloc[ii]
                _, p = fisher_exact([[int(r.a), int(r.b)], [int(r.c), int(r.d)]], alternative=alternative)
                pvals[ii] = p

            qvals = bh_fdr(np.nan_to_num(pvals, nan=1.0))
            keep = qvals <= alpha

            out["p_value"] = pvals
            out["q_value"] = qvals
            out["keep"] = keep
            out.loc[~keep, "similarity"] = 0.0

            if not include_pq:
                out = out.drop(columns=["p_value", "q_value", "keep"])

        if drop_self:
            out = out[out["left"] != out["right"]]

        if triangle.lower() in ("upper", "lower"):
            l = out["left"].astype(str)
            r = out["right"].astype(str)
            if triangle.lower() == "upper":
                out = out[l < r]
            else:
                out = out[l > r]

        if round_to is not None:
            out["similarity"] = out["similarity"].round(round_to)

        if sort_by:
            out = out.sort_values(by=sort_by, ascending=ascending)

        if topk is not None:
            out = out.head(topk)

        return out.reset_index(drop=True)

    # ============================================================
    # Public convenience methods (same names)
    # ============================================================

    # --- metrics that work in BOTH modes ---
    def jaccard(self, *, right=None, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            # In pairwise mode, "right" is not supported (rectangular sims need original matrix)
            return self.compute_pairwise_matrix("jaccard", fdr=fdr, tau=kwargs.get("tau"))
        return self.compute_metric(JaccardMetric, right=right, fdr=fdr, **kwargs)

    def dice(self, *, right=None, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            return self.compute_pairwise_matrix("dice", fdr=fdr, tau=kwargs.get("tau"))
        return self.compute_metric(DiceMetric, right=right, fdr=fdr, **kwargs)

    def cos(self, *, right=None, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            return self.compute_pairwise_matrix("cosine", fdr=fdr, tau=kwargs.get("tau"))
        return self.compute_metric(CosineMetric, right=right, fdr=fdr, **kwargs)

    def overlap(self, *, right=None, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            return self.compute_pairwise_matrix("overlap", fdr=fdr, tau=kwargs.get("tau"))
        return self.compute_metric(OverlapMetric, right=right, fdr=fdr, **kwargs)

    def phi(self, *, right=None, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            return self.compute_pairwise_matrix("phi", fdr=fdr, tau=kwargs.get("tau"))
        return self.compute_metric(PhiMetric, right=right, fdr=fdr, **kwargs)

    # --- long / edge list ---
    def jaccard_pairs(self, *, topk: Optional[int] = 20, fdr=None, **kwargs) -> pd.DataFrame:
        if self.is_pairwise:
            return self.compute_pairwise_long("jaccard", topk=topk, fdr=fdr, tau=kwargs.get("tau"), include_pq=bool(fdr))
        return self.compute_metric_long(JaccardMetric, topk=topk, fdr=fdr, **kwargs)

    # --- metrics that are WIDE-only (guard with clear errors) ---
    def tfidf(self, *args, **kwargs):
        self._require_wide("tfidf")
        raise NotImplementedError  # keep your old implementation here if needed

    def cfws(self, *args, **kwargs):
        self._require_wide("cfws")
        raise NotImplementedError

    # Add guards similarly for ACT/PMI/NPMI/JS/etc if you keep them in your API
