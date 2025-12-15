# =======================
# FDR helpers + Mixin  (production-ready)
# =======================
from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from scipy.stats import fisher_exact, hypergeom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class FDRResult:
    sim_original: np.ndarray
    sim_pruned:   np.ndarray
    overlaps:     np.ndarray
    pvals:        np.ndarray
    qvals:        np.ndarray
    keep_mask:    np.ndarray             # same shape as sim; True where kept
    meta:         Dict[str, Any]


# ---------- FDR corrections ----------
def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values (vectorized)."""
    x = np.asarray(p, dtype=float).ravel()
    m = x.size
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    adj = x * m / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    q = np.empty_like(x)
    q[order] = np.clip(adj_sorted, 0.0, 1.0)
    return q.reshape(p.shape)

def _by(p: np.ndarray) -> np.ndarray:
    """Benjamini–Yekutieli q-values (conservative under dependency)."""
    m = p.size
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return np.clip(_bh(p) * c_m, 0.0, 1.0)


# ---------- Vectorized tests on 2×2 ----------
def _gtest_p(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    tail: Literal["greater", "less", "two-sided"] = "greater",
    eps: float = 0.5
) -> np.ndarray:
    """
    Likelihood-ratio test (G-test) p-values via chi-square(1) survival.
    Stable, fully vectorized; used as default and SciPy fallback.
    """
    a = a.astype(float) + eps
    b = b.astype(float) + eps
    c = c.astype(float) + eps
    d = d.astype(float) + eps

    N  = a + b + c + d
    r1 = a + b; r0 = c + d
    c1 = a + c; c0 = b + d

    E11 = r1 * c1 / N; E10 = r1 * c0 / N
    E01 = r0 * c1 / N; E00 = r0 * c0 / N

    O = np.stack([a, b, c, d], axis=0)
    E = np.stack([E11, E10, E01, E00], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(O > 0, O * (np.log(O) - np.log(E)), 0.0)
    G = 2.0 * term.sum(axis=0)  # ~ chi2(df=1)

    from math import erfc, sqrt
    sf = np.vectorize(lambda x: erfc(sqrt(max(x, 0.0) / 2.0)))  # χ1^2 survival
    p_two = sf(G)

    if tail == "two-sided":
        return p_two
    # one-sided by sign of association (ad - bc)
    sign = a * d - b * c
    if tail == "greater":
        return np.where(sign >= 0, p_two, 1.0)
    else:
        return np.where(sign <= 0, p_two, 1.0)

def _fisher_p(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    tail: Literal["greater", "less", "two-sided"] = "greater"
) -> np.ndarray:
    """Exact Fisher p-values; iterates pairs (robust for small counts)."""
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for Fisher's exact test.")
    alt = {"greater": "greater", "less": "less", "two-sided": "two-sided"}[tail]
    out = np.empty_like(a, dtype=float)
    it = np.ndindex(a.shape)
    for idx in it:
        aa = max(float(a[idx]), 0.0)
        bb = max(float(b[idx]), 0.0)
        cc = max(float(c[idx]), 0.0)
        dd = max(float(d[idx]), 0.0)
        # cast to non-negative integers
        tbl = [[int(round(aa)), int(round(bb))],
               [int(round(cc)), int(round(dd))]]
        _, p = fisher_exact(tbl, alternative=alt)
        out[idx] = p
    return out

def _hypergeom_sf(
    k: np.ndarray, N: int, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """
    Hypergeometric survival p = P[X >= k] with population N,
    K = n1[i], draws = n2[j], observed = k[i,j].
    Works for rectangular matrices (broadcast-safe).
    """
    if _HAS_SCIPY:
        k = np.asarray(k, dtype=float)
        A, B = k.shape
        K = np.asarray(n1, dtype=float).reshape(A)   # |A| per left item
        n = np.asarray(n2, dtype=float).reshape(B)   # |B| per right item
        kv = k.ravel()
        Kv = np.repeat(K, B)
        nv = np.tile(n, A)
        out = np.array(
            [hypergeom.sf(int(round(kk)) - 1, int(N), int(round(Ki)), int(round(nj)))
             for kk, Ki, nj in zip(kv, Kv, nv)],
            dtype=float
        )
        return out.reshape(A, B)
    # fallback: approximate via G-test one-sided tail
    a = k.astype(float)
    n1 = n1.astype(float).reshape(-1, 1)
    n2 = n2.astype(float).reshape(1, -1)
    b = n1 - a
    c = n2 - a
    d = float(N) - (a + b + c)
    return _gtest_p(a, b, c, d, tail="greater", eps=0.0)


class FDRMixin:
    """
    Plug-and-play FDR for any SimilarityMetric subclass.
    Requires: self.co (overlap counts), self.f_left, self.f_right, self.N,
              self.left_labels, self.right_labels, self.is_symmetric,
              and self.compute() returning a dense similarity matrix.
    """

    def fdr(
        self,
        *,
        alpha: float = 0.05,
        fdr: Literal["bh", "by"] = "bh",
        test: Literal["g", "fisher", "hypergeom"] = "g",
        tail: Literal["greater", "less", "two-sided"] = "greater",
        min_weight: float = 0.0,                 # similarity floor to keep
        mode: Literal["hard", "soft"] = "hard",  # prune vs. shrink
        triangle: Optional[Literal["upper", "lower"]] = "upper",
        enforce_diag_one: bool = True,
    ) -> FDRResult:
        """
        Run FDR on the metric’s similarity matrix using the true 2×2 contingency tables.
        Supports square or rectangular; for square, tests upper/lower triangle only.
        """
        # --- 1) similarity & counts ---
        S = np.asarray(self.compute(round_to=None).values, dtype=float)
        co = self.co.toarray() if sparse.issparse(self.co) else np.asarray(self.co, dtype=float)
        if co.shape != S.shape:
            raise ValueError(f"co shape {co.shape} != similarity shape {S.shape}")
        fL = np.asarray(self.f_left, dtype=float).reshape(-1, 1)
        fR = np.asarray(self.f_right, dtype=float).reshape(1, -1)
        N = int(self.N)

        a = co
        b = fL - a
        c = fR - a
        d = float(N) - (a + b + c)
        np.maximum(a, 0.0, out=a); np.maximum(b, 0.0, out=b)
        np.maximum(c, 0.0, out=c); np.maximum(d, 0.0, out=d)

        # --- 2) test mask ---
        A, B = S.shape
        if self.is_symmetric and triangle in {"upper", "lower"} and A == B:
            iu = np.triu_indices(A, k=1) if triangle == "upper" else np.tril_indices(A, k=-1)
            mask = np.zeros_like(S, dtype=bool)
            mask[iu] = True
        else:
            mask = np.ones_like(S, dtype=bool)

        # --- 3) p-values ---
        if test == "g":
            P = _gtest_p(a, b, c, d, tail=tail)
        elif test == "fisher":
            P = _fisher_p(a, b, c, d, tail=tail)
        elif test == "hypergeom":
            P = _hypergeom_sf(a, N=N, n1=fL.squeeze(), n2=fR.squeeze())
        else:
            raise ValueError("test must be 'g', 'fisher', or 'hypergeom'")

        # --- 4) FDR q-values on tested cells only ---
        p_vec = P[mask]
        q_vec = _bh(p_vec) if fdr == "bh" else _by(p_vec)
        keep = (q_vec <= float(alpha)) & (S[mask] >= float(min_weight))

        # --- 5) prune / shrink ---
        Spruned = S.copy()
        if mode == "hard":
            Spruned[mask] = np.where(keep, S[mask], 0.0)
        else:
            # soft shrink proportional to (1 - q/alpha), floor at 0
            w = np.clip(1.0 - q_vec / max(alpha, 1e-12), 0.0, 1.0)
            Spruned[mask] = S[mask] * w

        if self.is_symmetric and A == B and triangle in {"upper", "lower"}:
            Spruned = 0.5 * (Spruned + Spruned.T)
            if enforce_diag_one:
                np.fill_diagonal(Spruned, 1.0)

        q_full = np.ones_like(S, dtype=float)
        q_full[mask] = q_vec
        keep_full = np.zeros_like(S, dtype=bool)
        keep_full[mask] = keep

        meta = dict(
            alpha=float(alpha), fdr=fdr, test=test, tail=tail, mode=mode,
            pairs=int(mask.sum()), kept=int(keep.sum()),
            kept_pct=float(100.0 * keep.sum() / max(1, mask.sum()))
        )

        return FDRResult(
            sim_original=S,
            sim_pruned=Spruned,
            overlaps=a,
            pvals=P,
            qvals=q_full,
            keep_mask=keep_full,
            meta=meta,
        )

    # ---------- Tidy edges ----------
    def edges_from_fdr(self, res: FDRResult) -> pd.DataFrame:
        """
        Build a tidy edge table from FDRResult (for plotting/export).
        For square matrices, returns only i<j (no self/dups).
        """
        A, B = res.sim_original.shape
        lefts  = np.repeat(self.left_labels, B)
        rights = np.tile(self.right_labels, A)
        df = pd.DataFrame({
            "left":       lefts,
            "right":      rights,
            "similarity": res.sim_original.ravel(),
            "similarity_pruned": res.sim_pruned.ravel(),
            "support":    res.overlaps.ravel(),
            "pval":       res.pvals.ravel(),
            "qval":       res.qvals.ravel(),
            "keep":       res.keep_mask.ravel(),
        })
        if getattr(self, "is_symmetric", False) and A == B:
            pos = {lab: i for i, lab in enumerate(self.left_labels)}
            i = df["left"].map(pos); j = df["right"].map(pos)
            df = df[i < j].copy()
        return df.sort_values(["keep", "similarity"], ascending=[False, False]).reset_index(drop=True)
