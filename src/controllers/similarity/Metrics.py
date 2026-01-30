import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm
from scipy import sparse

from typing import List, Optional, Union, Literal, Tuple
# Main TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.controllers.similarity.FDRMixin import FDRMixin
from src.controllers.similarity.DomainWeightsMixin import (
    _DomainWeightsMixin, AlphaBackboneParams, 
    DomainWeightParams, _alpha_backbone_from_counts, _ppmi_sigmoid, _saturation_level
)


def _row_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.sqrt((X * X).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    return X / norms

# ----------------------------- utilities -----------------------------
def _validate_binary_01_sampled(
    df: pd.DataFrame,
    cols: List[str],
    *,
    sample_n: int = 200_000,
    random_state: int = 0,
) -> None:
    """
    Validate that selected columns contain only binary values {0,1} (plus NaN / bool).
    Uses sampling for performance on very large DataFrames.

    Raises
    ------
    ValueError if any non-binary values are found.
    """
    if not cols:
        return

    n = len(df)
    sample = df if n <= sample_n else df.sample(n=sample_n, random_state=random_state)

    allowed = {0, 1, 0.0, 1.0, True, False}
    bad_cols = []

    for c in cols:
        if c not in sample.columns:
            continue

        s = sample[c].dropna()

        # If object-like, attempt numeric coercion (helps with "0"/"1" strings)
        if s.dtype == "object":
            s_num = pd.to_numeric(s, errors="coerce")
            # if coercion succeeds for most values, use numeric
            if s_num.notna().mean() > 0.9:
                s = s_num

        vals = pd.unique(s)
        if len(vals) == 0:
            continue

        if not set(vals).issubset(allowed):
            examples = [v for v in vals if v not in allowed][:8]
            mx = None
            try:
                mx = float(pd.to_numeric(pd.Series(vals), errors="coerce").max())
            except Exception:
                pass
            bad_cols.append((c, examples, mx))

    if bad_cols:
        lines = []
        for c, ex, mx in bad_cols[:12]:
            hint = ""
            if mx is not None and mx > 1:
                hint = f" (max≈{mx:g} → looks like counts/aggregation)"
            lines.append(f"- {c}: examples={ex}{hint}")

        raise ValueError(
            "Non-binary values detected in columns expected to be 0/1.\n"
            "This similarity engine assumes isolate-level binary indicators per row.\n\n"
            "Offending columns (sampled):\n"
            + "\n".join(lines)
            + "\n\nFix:\n"
            "  • Ensure you pass isolate-level rows into CoTestAnalyzer/SimilarityEngine (not aggregated counts), OR\n"
            "  • If you intended month-level networks, compute them from isolate-level subsets by YearMonth (no groupby-sum),\n"
            "    OR export pairwise co-test counts n_ij as an edge list.\n"
        )


def _to_binary(df: pd.DataFrame, cols: List[str], positive: bool = True) -> np.ndarray:
    """
    Convert selected columns to a 0/1 array.

    Dynamic behavior:
    - Validates input is binary 0/1 first (sampled).
    - Prevents silently treating aggregated counts (e.g., 160) as 1.
    """
    _validate_binary_01_sampled(df, cols)

    arr = df[cols].to_numpy()
    if positive:
        arr = (arr > 0).astype(int)
    else:
        arr = arr.astype(int)
    return arr

def _safe_div(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Elementwise safe division with default for zero denominators.
    Works with both dense and sparse inputs, returns dense ndarray."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)

        # If result is sparse, convert to dense
        if sparse.issparse(result):
            result = result.toarray()

        # Always ensure ndarray
        result = np.asarray(result)

        # Replace division by zero with default
        if not np.isscalar(result):
            result[denominator == 0] = default
        else:
            if denominator == 0:
                result = default

    return result

def _check_duplicate_columns(left_cols: List[str], right_cols: List[str]) -> None:
    """Warn about duplicate column names in asymmetric case."""
    duplicates = set(left_cols) & set(right_cols)
    if duplicates:
        warnings.warn(
            f"Duplicate column names found across sides: {duplicates}. "
            "This may cause ambiguity in outputs."
        )
    
def _normalize_cols(M):
    """
    Column-wise normalize to probabilities (each column sums to 1).
    Works for both dense ndarrays and scipy.sparse matrices.
    Returns a *dense* (numpy) array, which simplifies Jensen–Shannon code.
    """
    if sparse.issparse(M):
        # column sums as 1D array
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        # avoid division by zero
        inv = np.reciprocal(np.maximum(col_sums, 1e-15))
        # scale columns via right-multiplication by diagonal inv(col_sums)
        P = M @ sparse.diags(inv)
        return P.toarray()  # dense for downstream vector ops
    else:
        s = M.sum(axis=0, keepdims=True)  # keepdims OK for ndarrays
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.divide(M, s, where=(s > 0))
        # zero out columns that were all-zero
        zero_cols = (s.squeeze() == 0)
        if np.any(zero_cols):
            P[:, zero_cols] = 0.0
        return P


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon distance between two probability vectors p, q (both sum to 1).
    Uses natural log and converts to the standard distance in [0,1]:
      JSDist = sqrt( JS / ln(2) ),  where JS = 0.5*(KL(p||m)+KL(q||m)), m=(p+q)/2
    """
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], 1e-15, None)))).sum())

    js = 0.5 * (_kl(p, m) + _kl(q, m))
    return float(np.sqrt(js / np.log(2))) if js > 0 else 0.0

# ------------------------- dynamic base class -------------------------

class SimilarityMetric:
    """
    Flexible binary co-occurrence similarity base.

    Left axis:  `left_cols`  (e.g., antibiotic *_Tested columns).
    Right axis:
      - None (default): reuse left_cols -> square (A×A)
      - List[str]: binary columns -> rectangular (A×B)
      - str: categorical column -> one-hot -> rectangular (A×C)

    Subclasses must implement `_compute_matrix()` using:
      - self.co        : X.T @ Y  (|∩| counts), shape (A×B)
      - self.f_left    : X.sum(0) (A,)
      - self.f_right   : Y.sum(0) (B,)
      - self.N         : number of rows (samples)

    Helpful extras:
      - `.compute(round_to=None)` -> DataFrame with pretty labels
      - `.as_long(...)`           -> tidy pair list with support stats
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        right_name: Optional[str] = None,
        one_hot: bool = True,
        dropna_right: bool = True,
        binarize_positive: bool = True,
        sparse_threshold: float = 0.30,
    ) -> None:
        if not left_cols:
            raise ValueError("left_cols must be non-empty.")
        missing_left = [c for c in left_cols if c not in df.columns]
        if missing_left:
            raise KeyError(f"Left columns not found: {missing_left}")

        self.df = df.copy()
        self.left_cols = left_cols
        self.sparse_threshold = float(sparse_threshold)

        # --- Left matrix (X) ---
        X = _to_binary(self.df, left_cols, positive=binarize_positive)
        self.left_labels = list(left_cols)

        # Sparse toggle
        x_density = np.mean(X > 0)
        self.use_sparse = bool(x_density < self.sparse_threshold)
        self.X = sparse.csr_matrix(X) if self.use_sparse else X

        # --- Right matrix (Y) ---
        self.is_symmetric = right is None
        if right is None:
            self.Y = self.X
            self.right_labels = self.left_labels
            self.right_kind = "columns"
            # No duplicate warning in symmetric mode (expected)
        elif isinstance(right, list):
            missing_right = [c for c in right if c not in df.columns]
            if missing_right:
                raise KeyError(f"Right columns not found: {missing_right}")
            Y = _to_binary(self.df, right, positive=binarize_positive)
            self.Y = sparse.csr_matrix(Y) if self.use_sparse else Y
            self.right_labels = list(right)
            self.right_kind = "columns"
            _check_duplicate_columns(self.left_labels, self.right_labels)
        elif isinstance(right, str):
            if right not in df.columns:
                raise KeyError(f"Right column '{right}' not in df.")
            if one_hot:
                dummies = pd.get_dummies(self.df[right], dummy_na=not dropna_right)
                Y = dummies.to_numpy(dtype=int)
                self.right_labels = list(dummies.columns)
                self.right_kind = "categorical"
            else:
                Y = _to_binary(self.df, [right], positive=binarize_positive)
                self.right_labels = [right]
                self.right_kind = "binary"
            self.Y = sparse.csr_matrix(Y) if self.use_sparse else Y
        else:
            raise TypeError("right must be None, List[str], or str (categorical)")

        # --- core counts ---
        self.N = int(self.X.shape[0])
        if self.use_sparse:
            self.f_left = np.asarray(self.X.sum(axis=0)).ravel()
            self.f_right = np.asarray(self.Y.sum(axis=0)).ravel()
            self.co = self.X.T.dot(self.Y)  # (A×B), may be sparse
            # Densify if not too sparse to simplify downstream ops
            if sparse.issparse(self.co):
                area = self.co.shape[0] * self.co.shape[1]
                if area > 0 and (self.co.nnz / area) > 0.30:
                    self.co = self.co.toarray()
        else:
            self.f_left = self.X.sum(axis=0)
            self.f_right = self.Y.sum(axis=0)
            self.co = self.X.T.dot(self.Y)

        # axis names for nice outputs
        self.index_name = "Left"
        self.columns_name = right_name or ("Right" if not self.is_symmetric else "Right (same as Left)")

        # shaded internals exposed by as_long()
        self._last_union: Optional[np.ndarray] = None
        self._last_sim: Optional[np.ndarray] = None

    # --------------------------- public API ---------------------------
    @property
    def is_square(self) -> bool:
        """Historical alias used by older code."""
        return bool(self.is_symmetric)

    @property
    def right_cols(self) -> List[str]:
        """Historical alias used by some callers."""
        return list(self.right_labels)

    def compute(self, round_to: Optional[int] = None) -> pd.DataFrame:
        """
        Compute similarity matrix. Returns DataFrame indexed by left labels and columned by right labels.
        """
        mat = self._compute_matrix()
        if sparse.issparse(mat):
            mat = mat.toarray()
        out = pd.DataFrame(mat, index=self.left_labels, columns=self.right_labels)
        out.columns.name = self.columns_name
        if round_to is not None:
            out = out.round(round_to)
        return out

    def as_long(
        self,
        mat: Optional[pd.DataFrame] = None,
        *,
        drop_self: bool = True,
        triangle: Optional[Literal["upper", "lower"]] = "upper",
        sort_by: Literal["similarity", "intersection", "union"] = "similarity",
        ascending: bool = False,
        round_to: Optional[int] = 3,
        topk: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Tidy long form: left, right, similarity, intersection, union, f_left, f_right.
        For square matrices, you can keep only the upper/lower triangle.
        """
        if mat is None:
            mat = self.compute()

        sim = mat.values
        co_dense = self.co.toarray() if sparse.issparse(self.co) else self.co

        # If caller metric stored a custom "union" (e.g., weighted), use it; else raw set union.
        union = self._last_union
        if union is None:
            union = self.f_left[:, None] + self.f_right[None, :] - co_dense

        df_long = pd.DataFrame({
            "left": np.repeat(mat.index.values, mat.shape[1]),
            "right": np.tile(mat.columns.values, mat.shape[0]),
            "similarity": sim.ravel(),
            "intersection": co_dense.ravel(),
            "union": union.ravel(),
            "f_left": np.repeat(self.f_left, mat.shape[1]),
            "f_right": np.tile(self.f_right, mat.shape[0]),
        })

        if self.is_symmetric and triangle in {"upper", "lower"}:
            pos = {lab: i for i, lab in enumerate(mat.index)}
            i = df_long["left"].map(pos)
            j = df_long["right"].map(pos)
            df_long = df_long[(i < j) if triangle == "upper" else (i > j)]

        if drop_self and self.is_symmetric:
            df_long = df_long[df_long["left"] != df_long["right"]]

        if threshold is not None:
            df_long = df_long[df_long["similarity"] >= threshold]

        key = {"similarity": "similarity", "intersection": "intersection", "union": "union"}[sort_by]
        df_long = df_long.sort_values(key, ascending=ascending)
        if round_to is not None:
            df_long["similarity"] = df_long["similarity"].round(round_to)
        if topk:
            df_long = df_long.head(topk)
        return df_long.reset_index(drop=True)

    # -------------------------- to override --------------------------
    def _compute_matrix(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _compute_matrix().")

# ---------------- IDF-Enhanced Base Class ----------------
class BaseIDFMetric(SimilarityMetric):
    """
    Base class for IDF-based similarity metrics with optional NPMI.

    Parameters:
        alpha          : blending parameter in [0,1]
        use_npmi       : use NPMI (else normalized co-occurrence)
        eps            : numerical epsilon
        smoothing      : Laplace smoothing for probabilities
        normalize_idf  : normalize IDF vectors to unit norm (default False)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        alpha: float = 0.5,
        use_npmi: bool = True,
        eps: float = 1e-10,
        smoothing: float = 0.5,
        normalize_idf: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        self.alpha = float(alpha)
        self.use_npmi = bool(use_npmi)
        self.eps = float(eps)
        self.smoothing = float(smoothing)
        self.normalize_idf = bool(normalize_idf)

        self._calculate_idf_weights()
        if self.use_npmi:
            self._calculate_npmi_matrix()

    def _calculate_idf_weights(self) -> None:
        """Compute IDF arrays and frequency weights."""
        # Smoothed IDF (common +1 form)
        self.idf_left_ = np.log((self.N + 1.0) / (self.f_left + 1.0)) + 1.0
        self.idf_right_ = np.log((self.N + 1.0) / (self.f_right + 1.0)) + 1.0

        if self.normalize_idf:
            # unit-norm scaling (opt-in; can change absolute weighting)
            lnorm = np.linalg.norm(self.idf_left_)
            rnorm = np.linalg.norm(self.idf_right_)
            if lnorm > 0:
                self.idf_left_ = self.idf_left_ / lnorm
            if rnorm > 0:
                self.idf_right_ = self.idf_right_ / rnorm

        self.w_f_left = self.f_left * self.idf_left_
        self.w_f_right = self.f_right * self.idf_right_

    def _calculate_npmi_matrix(self) -> None:
        """Compute NPMI similarity ∈ [0,1] with Laplace smoothing."""
        s = self.smoothing
        p_i = (self.f_left + s) / (self.N + 2 * s)
        p_j = (self.f_right + s) / (self.N + 2 * s)

        co_dense = self.co.toarray() if sparse.issparse(self.co) else self.co
        p_ij = (co_dense + s) / (self.N + 2 * s)

        pmi = np.log(np.clip(p_ij, self.eps, 1.0)) - np.log(np.clip(np.outer(p_i, p_j), self.eps, 1.0))
        denom = -np.log(np.clip(p_ij, self.eps, 1.0))
        npmi = pmi / np.maximum(denom, self.eps)
        npmi = np.nan_to_num(npmi, nan=0.0, posinf=1.0, neginf=0.0)
        self.npmi_sim = (np.clip(npmi, -1.0, 1.0) + 1.0) / 2.0

    # def _finalize_matrix(self, mat: np.ndarray, union: Optional[np.ndarray]) -> np.ndarray:
    #     """Finalize similarity: store union, enforce symmetry, clear diagonal, clamp."""
    #     self._last_union = union
    #     self._last_sim = mat
    #     if self.is_symmetric:
    #         mat = 0.5 * (mat + mat.T)
    #         np.fill_diagonal(mat, 0.0)
    #     return np.clip(mat, 0.0, 1.0)
    
    def _finalize_matrix(self, mat: np.ndarray, union: np.ndarray) -> np.ndarray:
        self._last_union = union
        self._last_sim = mat

        # Symmetrize if square
        if self.is_symmetric:
            mat = 0.5 * (mat + mat.T)

        # Clip into [0, 1]
        mat = np.clip(mat, 0.0, 1.0)

        # If you want self-similarity = 1
        np.fill_diagonal(mat, 1.0)

        return mat

    def as_distance(self, round_to: Optional[int] = None) -> pd.DataFrame:
        """
        Convert similarity matrix to distance matrix (1 - similarity).

        Args:
            round_to: Number of decimal places to round results to

        Returns:
            DataFrame of pairwise distances
        """
        sim_mat = self.compute(round_to=None)  # keep raw similarity
        dist_mat = 1 - sim_mat

        if round_to is not None:
            dist_mat = dist_mat.round(round_to)

        dist_mat.index.name = sim_mat.index.name
        dist_mat.columns.name = sim_mat.columns.name
        return dist_mat 
    


# ---------------- IDF-Enhanced Composite FWS ----------------
class NPMIIDFCFWSMetric(BaseIDFMetric):
    """
    IDF-Enhanced Composite Frequency-Weighted Similarity.

    similarity = (Jaccard^α) * (CondFrac^(1-α))

    with:
        weighted_assoc = assoc * avg_idf
        weighted_union = w_f_left + w_f_right - weighted_assoc
        Jaccard = weighted_assoc / weighted_union
        CondFrac = weighted_assoc / max(w_f_left, w_f_right)
        assoc = NPMI∈[0,1]  (or normalized co-occurrence if use_npmi=False)
    """

    def _compute_matrix(self) -> np.ndarray:
        # Association strength
        if self.use_npmi:
            assoc = self.npmi_sim
        else:
            co_dense = self.co.toarray() if sparse.issparse(self.co) else self.co
            max_co = max(np.max(co_dense), self.eps)
            assoc = co_dense / max_co

        avg_idf = (self.idf_left_[:, None] + self.idf_right_[None, :]) / 2.0
        weighted_assoc = assoc * avg_idf

        weighted_union = self.w_f_left[:, None] + self.w_f_right[None, :] - weighted_assoc
        jaccard = _safe_div(weighted_assoc, np.maximum(weighted_union, self.eps))

        max_freq = np.maximum(self.w_f_left[:, None], self.w_f_right[None, :])
        condfrac = _safe_div(weighted_assoc, np.maximum(max_freq, self.eps))

        if self.alpha == 1.0:
            out = jaccard
        elif self.alpha == 0.0:
            out = condfrac
        else:
            out = np.exp(self.alpha * np.log(jaccard + self.eps) +
                         (1.0 - self.alpha) * np.log(condfrac + self.eps))

        # Store weighted_union so as_long() can display it (documented as such)
        return self._finalize_matrix(out, union=weighted_union)

# ---------------- IDF-Enhanced Cosine FWS ----------------
class NPMIIDFCosineFWSMetric(BaseIDFMetric):
    """
    IDF-Enhanced Cosine Frequency-Weighted Similarity.

    similarity = (Cosine^α) * (CondFrac^(1-α))

    with:
        weighted_assoc = assoc * avg_idf
        Cosine = weighted_assoc / sqrt(w_f_left * w_f_right)
        CondFrac = weighted_assoc / max(w_f_left, w_f_right)
        assoc = NPMI∈[0,1] (or normalized co-occurrence if use_npmi=False)
    """

    def _compute_matrix(self) -> np.ndarray:
        if self.use_npmi:
            assoc = self.npmi_sim
        else:
            co_dense = self.co.toarray() if sparse.issparse(self.co) else self.co
            max_co = max(np.max(co_dense), self.eps)
            assoc = co_dense / max_co

        avg_idf = (self.idf_left_[:, None] + self.idf_right_[None, :]) / 2.0
        weighted_assoc = assoc * avg_idf

        denom = np.sqrt(np.outer(self.w_f_left, self.w_f_right))
        cosine = _safe_div(weighted_assoc, np.maximum(denom, self.eps))

        max_freq = np.maximum(self.w_f_left[:, None], self.w_f_right[None, :])
        condfrac = _safe_div(weighted_assoc, np.maximum(max_freq, self.eps))

        if self.alpha == 1.0:
            out = cosine
        elif self.alpha == 0.0:
            out = condfrac
        else:
            out = np.exp(self.alpha * np.log(cosine + self.eps) +
                         (1.0 - self.alpha) * np.log(condfrac + self.eps))

        # DO NOT store cosine denom as "union" (it’s not a union) → let as_long compute raw union
        return self._finalize_matrix(out, union=None)
    
# ----------------------------- metrics ------------------------------

class IDFCFWSMetric(_DomainWeightsMixin, SimilarityMetric):
    """
    IDF-tempered CFWS with optional domain (guideline/novelty) effect + optional alpha-backbone (with BH-FDR).
    alpha: 1 -> Weighted Jaccard, 0 -> CFWS, (0,1) -> geometric blend.
    Output strictly in [0,1].
    """
    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        alpha: float = 0.5,
        domain: Optional[DomainWeightParams] = None,
        alpha_backbone: Optional[AlphaBackboneParams] = None,
        **kwargs
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self.domain = domain or DomainWeightParams(enable=False)
        self.alpha_backbone = alpha_backbone or AlphaBackboneParams(enable=False)
        # caches
        self._alpha_mat = None
        self._alpha_gate = None
        self._alpha_w = None
        self._alpha_p = None
        self._alpha_q = None

    def _ensure_alpha_backbone(self, co_raw: np.ndarray):
        if not self.alpha_backbone.enable:
            return
        if self._alpha_gate is not None:
            return
        A, G, W, P, Q = _alpha_backbone_from_counts(
            co=co_raw,
            fL=self.f_left,
            fR=self.f_right,
            N=self.N,
            params=self.alpha_backbone
        )
        self._alpha_mat = A
        self._alpha_gate = G
        self._alpha_w = W
        self._alpha_p = P
        self._alpha_q = Q

    def _compute_matrix(self) -> np.ndarray:
        N, fL, fR = self.N, self.f_left, self.f_right
        co = self.co.toarray() if sparse.issparse(self.co) else self.co

        # IDF + domain vectors
        w_idf_L, w_idf_R, w_dom_L, w_dom_R = self._init_domain_weights(
            self.left_labels, self.right_labels, N, fL, fR, self.domain
        )

        # *** BOUNDED weighting: intersection uses min(IDF) ***
        min_idf = np.minimum(w_idf_L[:, None], w_idf_R[None, :])
        weighted_co = co * min_idf

        # Side weights use their own IDF (consistent with set properties)
        w_fL = fL * w_idf_L
        w_fR = fR * w_idf_R

        # Weighted Jaccard pieces (bounded)
        w_union = w_fL[:, None] + w_fR[None, :] - weighted_co
        self._last_union = w_union
        jacc = _safe_div(weighted_co, w_union)              # in [0,1]

        # CFWS (bounded)
        max_w_freq = np.maximum(w_fL[:, None], w_fR[None, :])
        cfws = _safe_div(weighted_co, max_w_freq)           # in [0,1]

        # Geometric blend (still in [0,1])
        if self.alpha == 1.0:
            base = jacc
        elif self.alpha == 0.0:
            base = cfws
        else:
            base = np.power(jacc, self.alpha) * np.power(cfws, 1.0 - self.alpha)

        # Domain effect (bounded)
        if self.domain.enable:
            theta = ((w_dom_L[:, None] + w_dom_R[None, :]) / 2.0) ** max(self.domain.pair_exponent, 0.0)
            if self.domain.combine_mode == "stretch":
                base = 1.0 - np.power(1.0 - base, np.clip(theta, 1.0, None))
            else:  # "multiply"
                base = np.minimum(1.0, base * theta)

        # ---- α backbone (gate + optional soft weight + BH-FDR) ----
        if self.alpha_backbone.enable:
            self._ensure_alpha_backbone(co_raw=co)  # uses raw co, not weighted
            gate = self._alpha_gate
            aw = self._alpha_w
            if self.alpha_backbone.hard_gate:
                base = base * gate
            if self.alpha_backbone.multiply_weight:
                base = base * aw

        # Numerical safety
        base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0)
        base = np.clip(base, 0.0, 1.0)
        return base

    # Optional: expose alpha/p/q if you want to inspect or export
    @property
    def alpha_matrix(self) -> Optional[np.ndarray]:
        return self._alpha_mat
    @property
    def alpha_pvalues(self) -> Optional[np.ndarray]:
        return self._alpha_p
    @property
    def alpha_qvalues(self) -> Optional[np.ndarray]:
        return self._alpha_q
    @property
    def alpha_gate(self) -> Optional[np.ndarray]:
        return self._alpha_gate

class IDFCosineFWSMetric(_DomainWeightsMixin, SimilarityMetric):
    """
    Blend of standard binary Cosine and IDF-tempered CFWS, with optional domain effect + optional alpha-backbone (with BH-FDR).
    alpha: 1 -> Cosine only; 0 -> CFWS only; (0,1) -> geometric blend.
    Output strictly in [0,1].
    """
    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        alpha: float = 0.5,
        domain: Optional[DomainWeightParams] = None,
        alpha_backbone: Optional[AlphaBackboneParams] = None,
        **kwargs
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self.domain = domain or DomainWeightParams(enable=False)
        self.alpha_backbone = alpha_backbone or AlphaBackboneParams(enable=False)
        # caches
        self._alpha_mat = None
        self._alpha_gate = None
        self._alpha_w = None
        self._alpha_p = None
        self._alpha_q = None

    def _ensure_alpha_backbone(self, co_raw: np.ndarray):
        if not self.alpha_backbone.enable:
            return
        if self._alpha_gate is not None:
            return
        A, G, W, P, Q = _alpha_backbone_from_counts(
            co=co_raw,
            fL=self.f_left,
            fR=self.f_right,
            N=self.N,
            params=self.alpha_backbone
        )
        self._alpha_mat = A
        self._alpha_gate = G
        self._alpha_w = W
        self._alpha_p = P
        self._alpha_q = Q

    def _compute_matrix(self) -> np.ndarray:
        N, fL, fR = self.N, self.f_left, self.f_right
        co = self.co.toarray() if sparse.issparse(self.co) else self.co

        # IDF + domain (for CFWS & domain combine)
        w_idf_L, w_idf_R, w_dom_L, w_dom_R = self._init_domain_weights(
            self.left_labels, self.right_labels, N, fL, fR, self.domain
        )

        # Standard cosine on binary columns (bounded)
        denom = np.sqrt(np.outer(fL, fR))
        cos = _safe_div(co, denom)                           # in [0,1]

        # IDF-tempered CFWS (bounded via min-idf intersection)
        min_idf = np.minimum(w_idf_L[:, None], w_idf_R[None, :])
        weighted_co = co * min_idf
        w_fL = fL * w_idf_L
        w_fR = fR * w_idf_R
        max_w_freq = np.maximum(w_fL[:, None], w_fR[None, :])
        cfws = _safe_div(weighted_co, max_w_freq)            # in [0,1]

        # Blend (still in [0,1])
        if self.alpha == 1.0:
            base = cos
        elif self.alpha == 0.0:
            base = cfws
        else:
            base = np.power(cos, self.alpha) * np.power(cfws, 1.0 - self.alpha)

        # Domain effect (bounded)
        if self.domain.enable:
            theta = ((w_dom_L[:, None] + w_dom_R[None, :]) / 2.0) ** max(self.domain.pair_exponent, 0.0)
            if self.domain.combine_mode == "stretch":
                base = 1.0 - np.power(1.0 - base, np.clip(theta, 1.0, None))
            else:
                base = np.minimum(1.0, base * theta)

        # ---- α backbone (gate + optional soft weight + BH-FDR) ----
        if self.alpha_backbone.enable:
            self._ensure_alpha_backbone(co_raw=co)
            gate = self._alpha_gate
            aw = self._alpha_w
            if self.alpha_backbone.hard_gate:
                base = base * gate
            if self.alpha_backbone.multiply_weight:
                base = base * aw

        base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0)
        base = np.clip(base, 0.0, 1.0)
        return base

    # Optional getters
    @property
    def alpha_matrix(self) -> Optional[np.ndarray]:
        return self._alpha_mat
    @property
    def alpha_pvalues(self) -> Optional[np.ndarray]:
        return self._alpha_p
    @property
    def alpha_qvalues(self) -> Optional[np.ndarray]:
        return self._alpha_q
    @property
    def alpha_gate(self) -> Optional[np.ndarray]:
        return self._alpha_gate

#######################################################################
################# WITH PPMI / NPMI INSTEAD OF RAW CO ##################
#######################################################################

class IDFCFWSPPMIMetric(_DomainWeightsMixin, SimilarityMetric):
    """IDF-tempered CFWS/Jaccard with optional domain effect, alpha-backbone, and PPMI blend.

    Parameters
    ----------
    alpha : float in [0,1]
        Blend between IDF-Weighted Jaccard (alpha=1) and IDF-CFWS (alpha=0).
    ppmi_eta : float in [0,1]
        Geometric weight on S_ppmi enrichment (0 disables PPMI).
    ppmi_gamma : float
        Logistic slope for mapping PPMI -> [0,1].
    ppmi_cap : Optional[float]
        Clip PPMI before logistic (stability).
    ppmi_min_support : int
        Require n11 >= this for PPMI contribution.
    domain : DomainWeightParams
        Domain/IDF parameters for weighted counts and stretch/multiply combine.
    alpha_backbone : AlphaBackboneParams
        Statistical backbone (Fisher MLE + FDR/FWER) for gating/soft weights.

    Output
    ------
    Matrix in [0,1]. If alpha_backbone.hard_gate is True, zeros outside significant, positive edges.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        alpha: float = 0.5,
        ppmi_eta: float = 0.0,
        ppmi_gamma: float = 2.0,
        ppmi_cap: Optional[float] = 10.0,
        ppmi_min_support: int = 5,
        domain: Optional[DomainWeightParams] = None,
        alpha_backbone: Optional[AlphaBackboneParams] = None,
        **kwargs,
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if not (0.0 <= ppmi_eta <= 1.0):
            raise ValueError("ppmi_eta must be in [0, 1].")
        self.alpha = float(alpha)
        self.ppmi_eta = float(ppmi_eta)
        self.ppmi_gamma = float(ppmi_gamma)
        self.ppmi_cap = ppmi_cap
        self.ppmi_min_support = int(ppmi_min_support)
        self.domain = domain or DomainWeightParams(enable=False)
        self.alpha_backbone = alpha_backbone or AlphaBackboneParams(enable=False)
        # caches
        self._alpha_mat: Optional[np.ndarray] = None
        self._alpha_gate: Optional[np.ndarray] = None
        self._alpha_w: Optional[np.ndarray] = None
        self._alpha_p: Optional[np.ndarray] = None
        self._alpha_q: Optional[np.ndarray] = None
        self._ppmi: Optional[np.ndarray] = None
        self._last_union: Optional[np.ndarray] = None

    def _ensure_alpha_backbone(self, co_raw: np.ndarray):
        if not self.alpha_backbone.enable:
            return
        if self._alpha_gate is not None:
            return
        A, G, W, P, Q = _alpha_backbone_from_counts(
            co=co_raw,
            fL=self.f_left,
            fR=self.f_right,
            N=self.N,
            params=self.alpha_backbone,
        )
        self._alpha_mat = A
        self._alpha_gate = G
        self._alpha_w = W
        self._alpha_p = P
        self._alpha_q = Q

    def _compute_matrix(self) -> np.ndarray:
        N, fL, fR = self.N, self.f_left, self.f_right
        co = self.co.toarray() if sparse.issparse(self.co) else self.co

        # IDF + domain vectors
        w_idf_L, w_idf_R, w_dom_L, w_dom_R = self._init_domain_weights(
            self.left_labels, self.right_labels, N, fL, fR, self.domain
        )

        # *** BOUNDED weighting: intersection uses min(IDF) ***
        min_idf = np.minimum(w_idf_L[:, None], w_idf_R[None, :])
        weighted_co = co * min_idf

        # Side weights use their own IDF (consistent with set properties)
        w_fL = fL * w_idf_L
        w_fR = fR * w_idf_R

        # Weighted Jaccard pieces (bounded)
        w_union = w_fL[:, None] + w_fR[None, :] - weighted_co
        self._last_union = w_union
        jacc = _safe_div(weighted_co, w_union)  # in [0,1]

        # CFWS (bounded)
        max_w_freq = np.maximum(w_fL[:, None], w_fR[None, :])
        cfws = _safe_div(weighted_co, max_w_freq)  # in [0,1]

        # Geometric blend of Jaccard/CFWS
        if self.alpha == 1.0:
            base = jacc
        elif self.alpha == 0.0:
            base = cfws
        else:
            base = np.power(jacc, self.alpha) * np.power(cfws, 1.0 - self.alpha)

        # ----- Optional PPMI blend (geometric, preserves bounds) -----
        if self.ppmi_eta > 0.0:
            S_ppmi = _ppmi_sigmoid(
                co=co, fL=fL, fR=fR, N=N,
                gamma=self.ppmi_gamma, cap=self.ppmi_cap,
                min_support=self.ppmi_min_support,
            )
            base = np.power(base, 1.0 - self.ppmi_eta) * np.power(S_ppmi, self.ppmi_eta)
            self._ppmi = S_ppmi

        # Domain effect (bounded)
        if self.domain.enable:
            theta = ((w_dom_L[:, None] + w_dom_R[None, :]) / 2.0) ** max(self.domain.pair_exponent, 0.0)
            if self.domain.combine_mode == "stretch":
                base = 1.0 - np.power(1.0 - base, np.clip(theta, 1.0, None))
            else:  # "multiply"
                base = np.minimum(1.0, base * theta)

        # ---- α backbone (gate + optional soft weight + corrections) ----
        if self.alpha_backbone.enable:
            self._ensure_alpha_backbone(co_raw=co)  # raw co, not weighted
            gate = self._alpha_gate
            aw = self._alpha_w
            if self.alpha_backbone.hard_gate:
                base = base * gate
            if self.alpha_backbone.multiply_weight:
                base = base * aw

        # Numerical safety
        base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0)
        base = np.clip(base, 0.0, 1.0)
        return base

    # Optional: expose internals
    @property
    def alpha_matrix(self) -> Optional[np.ndarray]:
        return self._alpha_mat

    @property
    def alpha_pvalues(self) -> Optional[np.ndarray]:
        return self._alpha_p

    @property
    def alpha_qvalues(self) -> Optional[np.ndarray]:
        return self._alpha_q

    @property
    def alpha_gate(self) -> Optional[np.ndarray]:
        return self._alpha_gate

    @property
    def ppmi_matrix(self) -> Optional[np.ndarray]:
        return self._ppmi
    
    @property
    def ppmi_saturation(self) -> Optional[float]:
        """Return implied S* if PPMI is enabled and cap is finite; else None/1.0."""
        if self.ppmi_eta <= 0.0:
            return None
        if self.ppmi_cap is None or self.ppmi_cap <= 0.0:
            return 1.0  # effectively no cap
        return _saturation_level(self.ppmi_gamma, float(self.ppmi_cap))

class IDFCosinePPMIFWSMetric(_DomainWeightsMixin, SimilarityMetric):
    """Blend of binary Cosine and IDF-tempered CFWS, plus optional domain, backbone, and PPMI.

    Parameters are the same as IDFCFWSMetric; `alpha` controls Cosine vs CFWS here.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        alpha: float = 0.5,
        ppmi_eta: float = 0.0,
        ppmi_gamma: float = 2.0,
        ppmi_cap: Optional[float] = 10.0,
        ppmi_min_support: int = 5,
        domain: Optional[DomainWeightParams] = None,
        alpha_backbone: Optional[AlphaBackboneParams] = None,
        **kwargs,
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if not (0.0 <= ppmi_eta <= 1.0):
            raise ValueError("ppmi_eta must be in [0, 1].")
        self.alpha = float(alpha)
        self.ppmi_eta = float(ppmi_eta)
        self.ppmi_gamma = float(ppmi_gamma)
        self.ppmi_cap = ppmi_cap
        self.ppmi_min_support = int(ppmi_min_support)
        self.domain = domain or DomainWeightParams(enable=False)
        self.alpha_backbone = alpha_backbone or AlphaBackboneParams(enable=False)
        # caches
        self._alpha_mat: Optional[np.ndarray] = None
        self._alpha_gate: Optional[np.ndarray] = None
        self._alpha_w: Optional[np.ndarray] = None
        self._alpha_p: Optional[np.ndarray] = None
        self._alpha_q: Optional[np.ndarray] = None
        self._ppmi: Optional[np.ndarray] = None

    def _ensure_alpha_backbone(self, co_raw: np.ndarray):
        if not self.alpha_backbone.enable:
            return
        if self._alpha_gate is not None:
            return
        A, G, W, P, Q = _alpha_backbone_from_counts(
            co=co_raw,
            fL=self.f_left,
            fR=self.f_right,
            N=self.N,
            params=self.alpha_backbone,
        )
        self._alpha_mat = A
        self._alpha_gate = G
        self._alpha_w = W
        self._alpha_p = P
        self._alpha_q = Q

    def _compute_matrix(self) -> np.ndarray:
        N, fL, fR = self.N, self.f_left, self.f_right
        co = self.co.toarray() if sparse.issparse(self.co) else self.co

        # IDF + domain (for CFWS & domain combine)
        w_idf_L, w_idf_R, w_dom_L, w_dom_R = self._init_domain_weights(
            self.left_labels, self.right_labels, N, fL, fR, self.domain
        )

        # Standard cosine on binary columns (bounded)
        denom = np.sqrt(np.outer(fL, fR))
        cos = _safe_div(co, denom)  # in [0,1]

        # IDF-tempered CFWS (bounded via min-idf intersection)
        min_idf = np.minimum(w_idf_L[:, None], w_idf_R[None, :])
        weighted_co = co * min_idf
        w_fL = fL * w_idf_L
        w_fR = fR * w_idf_R
        max_w_freq = np.maximum(w_fL[:, None], w_fR[None, :])
        cfws = _safe_div(weighted_co, max_w_freq)  # in [0,1]

        # Blend (still in [0,1])
        if self.alpha == 1.0:
            base = cos
        elif self.alpha == 0.0:
            base = cfws
        else:
            base = np.power(cos, self.alpha) * np.power(cfws, 1.0 - self.alpha)

        # ----- Optional PPMI blend -----
        if self.ppmi_eta > 0.0:
            S_ppmi = _ppmi_sigmoid(
                co=co, fL=fL, fR=fR, N=N,
                gamma=self.ppmi_gamma, cap=self.ppmi_cap,
                min_support=self.ppmi_min_support,
            )
            base = np.power(base, 1.0 - self.ppmi_eta) * np.power(S_ppmi, self.ppmi_eta)
            self._ppmi = S_ppmi

        # Domain effect (bounded)
        if self.domain.enable:
            theta = ((w_dom_L[:, None] + w_dom_R[None, :]) / 2.0) ** max(self.domain.pair_exponent, 0.0)
            if self.domain.combine_mode == "stretch":
                base = 1.0 - np.power(1.0 - base, np.clip(theta, 1.0, None))
            else:
                base = np.minimum(1.0, base * theta)

        # ---- α backbone (gate + optional soft weight + corrections) ----
        if self.alpha_backbone.enable:
            self._ensure_alpha_backbone(co_raw=co)
            gate = self._alpha_gate
            aw = self._alpha_w
            if self.alpha_backbone.hard_gate:
                base = base * gate
            if self.alpha_backbone.multiply_weight:
                base = base * aw

        base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0)
        base = np.clip(base, 0.0, 1.0)
        return base

    # Optional getters
    @property
    def alpha_matrix(self) -> Optional[np.ndarray]:
        return self._alpha_mat

    @property
    def alpha_pvalues(self) -> Optional[np.ndarray]:
        return self._alpha_p

    @property
    def alpha_qvalues(self) -> Optional[np.ndarray]:
        return self._alpha_q

    @property
    def alpha_gate(self) -> Optional[np.ndarray]:
        return self._alpha_gate

    @property
    def ppmi_matrix(self) -> Optional[np.ndarray]:
        return self._ppmi
    
    @property
    def ppmi_saturation(self) -> Optional[float]:
        """Return implied S* if PPMI is enabled and cap is finite; else None/1.0."""
        if self.ppmi_eta <= 0.0:
            return None
        if self.ppmi_cap is None or self.ppmi_cap <= 0.0:
            return 1.0  # effectively no cap
        return _saturation_level(self.ppmi_gamma, float(self.ppmi_cap))
    
# TF-IDF Model
class TFIDFSimilarity(SimilarityMetric):
    r"""
    TF–IDF similarity (item–item), using TF-IDF values directly as similarity scores.
    This assumes that higher TF-IDF values indicate greater similarity between items.

    Assumes `df` has rows = entities (e.g., isolates/patients) and columns = items (e.g., antibiotics).
    `left_cols` are the item names to compare on the rows of the similarity output.
    `right` (optional) are the item names for the columns; if None, uses `left_cols` (square matrix).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        smooth_idf: bool = True,
        fit_scope: str = "both",   # "both" -> fit IDF on [left; right] (recommended), "left" -> fit only on left
        similarity_method: str = "raw",  # "dot_product", "cosine", or "raw"
        **kwargs
    ):
        super().__init__(df, left_cols, right, **kwargs)
        self.smooth_idf = bool(smooth_idf)
        self.fit_scope = fit_scope
        self.similarity_method = similarity_method

        # cached for inspection/debugging
        self.idf_vector = None
        self.tfidf_left_ = None
        self.tfidf_right_ = None
        self._last_sim = None

    def _build_item_blocks(self):
        """
        Build item×entity blocks (rows=items, cols=entities) from df (entities×items).
        """
        left_items = list(self.left_labels)
        X_left = self.df[left_items].T.to_numpy(dtype=float)  # (n_left_items, n_entities)

        if self.right_labels is None or list(self.right_labels) == left_items:
            X_right = X_left
            right_items = left_items
        else:
            right_items = list(self.right_labels)
            X_right = self.df[right_items].T.to_numpy(dtype=float)  # (n_right_items, n_entities)

        if X_left.shape[1] != X_right.shape[1]:
            raise ValueError(
                f"Left and right blocks must have the same number of entities (columns). "
                f"Got {X_left.shape[1]} vs {X_right.shape[1]}."
            )
        return X_left, left_items, X_right, right_items

    def _compute_matrix(self) -> np.ndarray:
        # 1) items×entities blocks
        X_left, left_items, X_right, right_items = self._build_item_blocks()

        # 2) build TF-IDF
        if (X_right is X_left) or (self.fit_scope == "left"):
            X_fit = X_left
        else:
            X_fit = np.vstack([X_left, X_right])

        tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=self.smooth_idf)
        X_fit_tfidf = tfidf.fit_transform(X_fit)
        self.idf_vector = tfidf.idf_.copy()

        # Transform each block
        if X_right is X_left:
            X_left_tfidf = X_fit_tfidf.toarray()
            X_right_tfidf = X_left_tfidf
        else:
            X_left_tfidf = tfidf.transform(X_left).toarray()
            X_right_tfidf = tfidf.transform(X_right).toarray()

        # 3) Compute similarity using different methods
        if self.similarity_method == "raw":
            # Use TF-IDF values directly (assuming they represent similarity)
            S = X_left_tfidf @ X_right_tfidf.T  # Dot product of TF-IDF vectors
            
        elif self.similarity_method == "dot_product":
            # Simple dot product (equivalent to unnormalized cosine similarity)
            S = X_left_tfidf @ X_right_tfidf.T
            
        elif self.similarity_method == "cosine":
            # Traditional cosine similarity (your original approach)
            from sklearn.metrics.pairwise import cosine_similarity
            S = cosine_similarity(X_left_tfidf, X_right_tfidf)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

        # Cache and return
        self.tfidf_left_ = X_left_tfidf
        self.tfidf_right_ = X_right_tfidf
        self._last_sim = S
        return S
    
class JaccardMetric(FDRMixin, SimilarityMetric):
    """Jaccard similarity: |A ∩ B| / |A ∪ B| (works for square or rectangular)."""
    def _compute_matrix(self) -> np.ndarray:
        union = self.f_left[:, None] + self.f_right[None, :] - self.co
        J = _safe_div(self.co, union)
        self._last_union, self._last_sim = union, J
        return J

class DiceMetric(FDRMixin, SimilarityMetric):
    """Sørensen–Dice: 2|A ∩ B| / (|A| + |B|)."""
    def _compute_matrix(self) -> np.ndarray:
        denom = self.f_left[:, None] + self.f_right[None, :]
        return _safe_div(2 * self.co, denom)

class OverlapMetric(FDRMixin, SimilarityMetric):
    """Overlap (Szymkiewicz–Simpson): |A ∩ B| / min(|A|, |B|)."""
    def _compute_matrix(self) -> np.ndarray:
        denom = np.minimum(self.f_left[:, None], self.f_right[None, :])
        return _safe_div(self.co, denom)

class CosineMetric(FDRMixin, SimilarityMetric):
    """Cosine / Ochiai: |A ∩ B| / sqrt(|A|·|B|)."""
    def _compute_matrix(self) -> np.ndarray:
        denom = np.sqrt(np.outer(self.f_left, self.f_right))
        return _safe_div(self.co, denom)

class ConditionalFractionMetric(FDRMixin, SimilarityMetric):
    """Conditional co-testing fraction: |A ∩ B| / max(|A|, |B|)."""
    def _compute_matrix(self) -> np.ndarray:
        denom = np.maximum(self.f_left[:, None], self.f_right[None, :])
        return _safe_div(self.co, denom)

class LiftMetric(FDRMixin, SimilarityMetric):
    """Lift: (N · |A ∩ B|) / (|A| · |B|)."""
    def _compute_matrix(self) -> np.ndarray:
        denom = np.outer(self.f_left, self.f_right)
        return _safe_div(self.N * self.co, denom)

class ScaledLiftMetric(FDRMixin, SimilarityMetric):
    """Log-normalized Lift into [0,1]:  log1p(Lift) / max(log1p(Lift))."""
    def _compute_matrix(self) -> np.ndarray:
        denom = np.outer(self.f_left, self.f_right)
        lift = _safe_div(self.N * self.co, denom)
        loglift = np.log1p(lift)
        mx = float(np.nanmax(loglift)) if loglift.size else 0.0
        return loglift / mx if mx > 0 else np.zeros_like(loglift)

class TverskyMetric(FDRMixin, SimilarityMetric):
    """
        Tversky index (generalizes Dice and Jaccard):
        T = |A∩B| / (|A∩B| + α·|A\B| + β·|B\A|)
        α=β=0.5 -> Dice,  α=β=1 -> Jaccard
    """
    def __init__(self, df: pd.DataFrame, left_cols: List[str], right: Optional[Union[List[str], str]] = None,
                 *, alpha: float = 0.5, beta: float = 0.5, **kwargs) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative.")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _compute_matrix(self) -> np.ndarray:
        a_minus_b = self.f_left[:, None] - self.co
        b_minus_a = self.f_right[None, :] - self.co
        denom = self.co + self.alpha * a_minus_b + self.beta * b_minus_a
        return _safe_div(self.co, denom)

class CFWSMetric(FDRMixin, SimilarityMetric):
    """
    Composite Frequency-Weighted Similarity:
      CFWS = Jaccard^α * ConditionalFraction^(1-α)
    """
    def __init__(self, df: pd.DataFrame, left_cols: List[str], right: Optional[Union[List[str], str]] = None,
                 *, alpha: float = 0.5, **kwargs) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")
        self.alpha = float(alpha)

    def _compute_matrix(self) -> np.ndarray:
        union = self.f_left[:, None] + self.f_right[None, :] - self.co
        J = _safe_div(self.co, union)
        CF = _safe_div(self.co, np.maximum(self.f_left[:, None], self.f_right[None, :]))
        return (J ** self.alpha) * (CF ** (1.0 - self.alpha))

class CFWSCosMetric(FDRMixin, SimilarityMetric):
    """
    CFWS using Cosine:
      CFWS = Jaccard^α * Cosine^(1-α)
    """
    def __init__(self, df: pd.DataFrame, left_cols: List[str], right: Optional[Union[List[str], str]] = None,
                 *, alpha: float = 0.5, **kwargs) -> None:
        super().__init__(df, left_cols, right, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")
        self.alpha = float(alpha)

    def _compute_matrix(self) -> np.ndarray:
        union = self.f_left[:, None] + self.f_right[None, :] - self.co
        J = _safe_div(self.co, union)
        Cos = _safe_div(self.co, np.sqrt(np.outer(self.f_left, self.f_right)))
        return (J ** self.alpha) * (Cos ** (1.0 - self.alpha))

class JensenShannonMetric(FDRMixin, SimilarityMetric):
    """
    Jensen–Shannon similarity (1 - distance) between column-wise probability
    profiles of the two sides (left columns vs right columns) over samples.
    Values in [0,1]. Works for square or rectangular.
    """
    def _compute_matrix(self) -> np.ndarray:
        # Cast to float; _normalize_cols handles sparse/dense and returns dense
        Xf = self.X.astype(float)
        Yf = self.Y.astype(float)

        P = _normalize_cols(Xf)  # shape (n_samples, A), dense
        Q = _normalize_cols(Yf)  # shape (n_samples, B), dense

        A, B = P.shape[1], Q.shape[1]
        sim = np.zeros((A, B), dtype=float)

        for i in range(A):
            pi = P[:, i]
            for j in range(B):
                qj = Q[:, j]
                d = _js_distance(pi, qj)   # distance in [0,1]
                sim[i, j] = 1.0 - d       # similarity
        return sim

# ----------------------------- NEW METRICS -----------------------------
# Information-theoretic & association measures built on 2×2 counts

class PhiMetric(FDRMixin, SimilarityMetric):
    """
    Pearson φ (Matthews correlation) for binary pair (left_i, right_j):
      φ = (ad - bc) / sqrt((a+b)(a+c)(b+d)(c+d)) ∈ [-1,1]
    """
    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float)
        fL = self.f_left.astype(float)[:, None]
        fR = self.f_right.astype(float)[None, :]
        b = fL - a
        c = fR - a
        d = self.N - (a + b + c)
        num = a * d - b * c
        den = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
        return _safe_div(num, den)

class PMIMetric(FDRMixin, SimilarityMetric):
    """
    Pointwise Mutual Information (PMI) between two binary events:
      PMI = log( p11 / (p1 * p2) ), with Laplace smoothing eps.
    Returns nats. Larger => stronger positive association.
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)

    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float) + self.eps
        fL = self.f_left.astype(float)[:, None] + self.eps
        fR = self.f_right.astype(float)[None, :] + self.eps
        N  = float(self.N) + 4.0 * self.eps
        p11 = a / N
        p1  = fL / N
        p2  = fR / N
        with np.errstate(divide="ignore"):
            PMI = np.log(np.clip(p11 / (p1 * p2), 1e-15, None))
        return PMI

class NPMIMetric(SimilarityMetric):
    """
    Normalized PMI in [-1,1]:
      NPMI = PMI / ( -log p11 )
    with Laplace smoothing. 1 = perfect co-occurrence, 0 ≈ independence, -1 = never co-occur.
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)

    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float) + self.eps
        fL = self.f_left.astype(float)[:, None] + self.eps
        fR = self.f_right.astype(float)[None, :] + self.eps
        N  = float(self.N) + 4.0 * self.eps
        p11 = a / N
        p1  = fL / N
        p2  = fR / N
        with np.errstate(divide="ignore"):
            PMI = np.log(np.clip(p11 / (p1 * p2), 1e-15, None))
            denom = -np.log(np.clip(p11, 1e-15, None))
            NPMI = np.where(denom > 0, PMI / denom, 1.0)
        return np.clip(NPMI, -1.0, 1.0)

class NPMI01Metric(SimilarityMetric):
    """
    Shifted NPMI to [0,1] for visualization:
      NPMI01 = (NPMI + 1) / 2
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self._npmi = NPMIMetric(df, left_cols, right, eps=eps, **kwargs)

    def _compute_matrix(self) -> np.ndarray:
        npmi = self._npmi._compute_matrix()
        return (npmi + 1.0) / 2.0

class MutualInformationMetric(SimilarityMetric):
    """
    Mutual Information I(A;B) in nats for binary variables, using full 2×2 table:
      I = sum_{a,b∈{0,1}} Pab log( Pab / (Pa Pb) )  ≥ 0
    Laplace smoothing keeps it finite.
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)

    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float)
        fL = self.f_left.astype(float)[:, None]
        fR = self.f_right.astype(float)[None, :]
        b = fL - a
        c = fR - a
        d = self.N - (a + b + c)

        # Laplace smoothing to avoid zeros
        a += self.eps; b += self.eps; c += self.eps; d += self.eps
        N = float(self.N) + 4.0 * self.eps

        p11 = a / N; p10 = b / N; p01 = c / N; p00 = d / N
        p1 = p11 + p10
        p2 = p11 + p01
        with np.errstate(divide="ignore", invalid="ignore"):
            MI = (p11 * (np.log(p11) - np.log(p1 * p2)) +
                  p10 * (np.log(p10) - np.log(p1 * (1.0 - p2))) +
                  p01 * (np.log(p01) - np.log((1.0 - p1) * p2)) +
                  p00 * (np.log(p00) - np.log((1.0 - p1) * (1.0 - p2))))
        MI = np.maximum(MI, 0.0)
        return MI

class NMIMetric(SimilarityMetric):
    """
    Symmetric Normalized MI in [0,1]:
      NMI = MI / sqrt( H(A) H(B) )
    where H(X) is binary entropy in nats.
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)
        self._mi = MutualInformationMetric(df, left_cols, right, eps=eps, **kwargs)

    def _compute_matrix(self) -> np.ndarray:
        # Entropy per variable
        fL = (self.f_left.astype(float) + self.eps) / (float(self.N) + 2.0 * self.eps)
        fR = (self.f_right.astype(float) + self.eps) / (float(self.N) + 2.0 * self.eps)
        H = lambda p: -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        HL = H(fL)[:, None]
        HR = H(fR)[None, :]
        den = np.sqrt(HL * HR)

        MI = self._mi._compute_matrix()
        return _safe_div(MI, den)

class YulesQMetric(SimilarityMetric):
    """
    Yule's Q association in [-1,1]:
      Q = (ad - bc) / (ad + bc) = (OR - 1) / (OR + 1)
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)

    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float) + self.eps
        fL = self.f_left.astype(float)[:, None] + self.eps
        fR = self.f_right.astype(float)[None, :] + self.eps
        b = fL - a
        c = fR - a
        d = self.N + 4.0 * self.eps - (a + b + c)
        ad = a * d
        bc = b * c
        return _safe_div(ad - bc, ad + bc)

class YulesYMetric(SimilarityMetric):
    """
    Yule's Y (Tschuprow’s coefficient variant) in [-1,1]:
      Y = (sqrt(OR) - 1) / (sqrt(OR) + 1)
    """
    def __init__(self, df, left_cols, right=None, *, eps: float = 0.5, **kwargs):
        super().__init__(df, left_cols, right, **kwargs)
        self.eps = float(eps)

    def _compute_matrix(self) -> np.ndarray:
        a = self.co.astype(float) + self.eps
        fL = self.f_left.astype(float)[:, None] + self.eps
        fR = self.f_right.astype(float)[None, :] + self.eps
        b = fL - a
        c = fR - a
        d = self.N + 4.0 * self.eps - (a + b + c)
        with np.errstate(divide="ignore", invalid="ignore"):
            OR = (a * d) / (b * c)
            sq = np.sqrt(np.clip(OR, 1e-15, None))
            Y = (sq - 1.0) / (sq + 1.0)
        return np.clip(Y, -1.0, 1.0)

class ACTCompositeMetric(SimilarityMetric):
    """
    Antibiotic Co-Testing Composite Metric (ACT)
    --------------------------------------------
    Combines:
      1) IDF-enhanced frequency-weighted similarity (IDF-CFWS)
      2) NPMI (normalized PMI) mapped to [0,1]

    Formula (two blend options):
      additive:  ACT = w1 * IDF-CFWS + w2 * NPMI01
      geometric: ACT = exp( w1*log(IDF-CFWS) + w2*log(NPMI01) )

    Optional support gating (applied to the final score):
      cap:  g(a) = min(1, a / t)
      exp:  g(a) = 1 - exp(-lambda * a / N)

    Attributes after compute():
      - idfcfws_  : np.ndarray (A×B)
      - npmi01_   : np.ndarray (A×B)
      - gate_     : np.ndarray (A×B) or None
      - composite_: np.ndarray (A×B)  (returned matrix)
      - llr_G_    : np.ndarray (A×B) or None
      - pvalues_  : np.ndarray (A×B) or None
    """

    def __init__(
        self,
        df: pd.DataFrame,
        left_cols: List[str],
        right: Optional[Union[List[str], str]] = None,
        *,
        weights: Tuple[float, float] = (0.5, 0.5),
        idf_alpha: float = 0.5,
        npmi_eps: float = 0.5,
        blend: Literal["add", "geo"] = "geo",
        use_support_gate: bool = True,
        gate_mode: Literal["cap", "exp"] = "cap",
        gate_t: int = 5,
        gate_lambda: float = 12.0,
        compute_llr: bool = False,
        llr_eps: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(df, left_cols, right, **kwargs)

        # --- validate weights ---
        w1, w2 = float(weights[0]), float(weights[1])
        if not np.isclose(w1 + w2, 1.0, atol=1e-8):
            raise ValueError("weights must sum to 1.0")
        if not (0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0):
            raise ValueError("each weight must be in [0,1]")
        self.w1, self.w2 = w1, w2

        # --- store params ---
        if not (0.0 <= idf_alpha <= 1.0):
            raise ValueError("idf_alpha must be in [0,1]")
        self.idf_alpha = float(idf_alpha)
        self.npmi_eps = float(npmi_eps)
        if blend not in {"add", "geo"}:
            raise ValueError("blend must be 'add' or 'geo'")
        self.blend = blend

        self.use_support_gate = bool(use_support_gate)
        self.gate_mode = gate_mode
        self.gate_t = int(gate_t)
        self.gate_lambda = float(gate_lambda)

        self.compute_llr = bool(compute_llr)
        self.llr_eps = float(llr_eps)

        # --- precompute IDF weights by position (no label collisions) ---
        # IDF = log((N+1)/(f+1)) + 1
        self.idf_left_  = np.log((self.N + 1.0) / (self.f_left.astype(float)  + 1.0)) + 1.0
        self.idf_right_ = np.log((self.N + 1.0) / (self.f_right.astype(float) + 1.0)) + 1.0

        # placeholders exposed after compute()
        self.idfcfws_ = None
        self.npmi01_ = None
        self.gate_ = None
        self.llr_G_ = None
        self.pvalues_ = None
        self._last_sim = None  # for as_long()
        self._last_union = None

    # ---------------- internal components ----------------

    def _idfcfws(self) -> np.ndarray:
        """IDF-enhanced CFWS component in [0,1]."""
        # weighted intersection
        avg_idf = 0.5 * (self.idf_left_[:, None] + self.idf_right_[None, :])
        weighted_co = self.co.astype(float) * avg_idf

        # weighted sizes
        wL = self.f_left.astype(float)  * self.idf_left_
        wR = self.f_right.astype(float) * self.idf_right_

        # Jaccard with IDF
        weighted_union = wL[:, None] + wR[None, :] - weighted_co
        jacc = _safe_div(weighted_co, weighted_union)

        # Conditional fraction with IDF
        max_w = np.maximum(wL[:, None], wR[None, :])
        cf = _safe_div(weighted_co, max_w)

        if self.idf_alpha == 1.0:
            return jacc
        if self.idf_alpha == 0.0:
            return cf
        return (jacc ** self.idf_alpha) * (cf ** (1.0 - self.idf_alpha))

    def _npmi01(self) -> np.ndarray:
        """NPMI mapped to [0,1] with Laplace smoothing."""
        a  = self.co.astype(float) + self.npmi_eps
        fL = self.f_left.astype(float)[:, None] + self.npmi_eps
        fR = self.f_right.astype(float)[None, :] + self.npmi_eps
        N  = float(self.N) + 4.0 * self.npmi_eps

        p11 = a / N
        p1  = fL / N
        p2  = fR / N

        with np.errstate(divide="ignore", invalid="ignore"):
            PMI   = np.log(np.clip(p11 / (p1 * p2), 1e-15, None))
            denom = -np.log(np.clip(p11, 1e-15, None))
            NPMI  = np.where(denom > 0, PMI / denom, 1.0)

        return (np.clip(NPMI, -1.0, 1.0) + 1.0) / 2.0  # -> [0,1]

    def _support_gate(self) -> Optional[np.ndarray]:
        """Optional gating matrix g(a) in [0,1] based on support a=|∩|."""
        if not self.use_support_gate:
            return None
        a = self.co.astype(float)
        if self.gate_mode == "cap":
            return np.minimum(1.0, a / max(1, self.gate_t))
        # smooth exponential
        return 1.0 - np.exp(-self.gate_lambda * (a / max(1.0, float(self.N))))

    def _llr_gtest(self) -> Tuple[np.ndarray, np.ndarray]:
        """G-test (LLR) for independence on each 2×2. Returns (G, p)."""
        a = self.co.astype(float)
        fL = self.f_left.astype(float)[:, None]
        fR = self.f_right.astype(float)[None, :]
        b = fL - a
        c = fR - a
        d = self.N - (a + b + c)

        # Laplace smoothing to avoid zeros
        eps = self.llr_eps
        a += eps; b += eps; c += eps; d += eps

        N = a + b + c + d
        r1 = a + b; r0 = c + d
        c1 = a + c; c0 = b + d

        E11 = r1 * c1 / N; E10 = r1 * c0 / N
        E01 = r0 * c1 / N; E00 = r0 * c0 / N

        O = np.stack([a, b, c, d], axis=0)
        E = np.stack([E11, E10, E01, E00], axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(O > 0, O * (np.log(O) - np.log(E)), 0.0)
        G = 2.0 * term.sum(axis=0)  # chi-square df=1 approx

        # p-value via chi2(1) survival: sf(x) = erfc(sqrt(x/2))
        from math import erfc, sqrt
        sf = np.vectorize(lambda x: erfc(sqrt(max(x, 0.0) / 2.0)))
        p = sf(G)
        return G, p

    # ---------------- core API ----------------

    def _compute_matrix(self) -> np.ndarray:
        # components
        idfcfws = self._idfcfws()
        npmi01  = self._npmi01()

        # blend
        if self.blend == "add":
            comp = self.w1 * idfcfws + self.w2 * npmi01
        else:  # "geo"
            comp = np.exp(
                self.w1 * np.log(np.clip(idfcfws, 1e-12, 1.0)) +
                self.w2 * np.log(np.clip(npmi01,  1e-12, 1.0))
            )

        # gate by support if requested
        gate = self._support_gate()
        if gate is not None:
            comp = comp * gate

        # optional LLR p-values for downstream filtering
        if self.compute_llr:
            G, p = self._llr_gtest()
            self.llr_G_, self.pvalues_ = G, p

        # expose internals for auditing and as_long()
        self.idfcfws_ = idfcfws
        self.npmi01_  = npmi01
        self.gate_    = gate
        self.composite_ = comp
        self._last_sim = comp  # so as_long() can reuse it
        # (union not uniquely defined here; keep default from base)
        return comp
    
class ACTJensenMetric(SimilarityMetric):
    def __init__(
        self,
        df,
        left_cols,
        right=None,
        *,
        weights=(0.5, 0.5),
        idf_alpha=0.5,
        js_eps=1e-12,
        blend="geo",
        use_support_gate=True,
        gate_mode="cap",
        gate_t=5,
        gate_lambda=12.0,
        compute_llr=False,
        llr_eps=0.5,
        **kwargs,  # keep if you want to accept extra args, but don't pass them up
    ):
        # DO NOT forward **kwargs to the base (it doesn't accept them)
        super().__init__(df, left_cols, right)

        # validate/store params
        w1, w2 = float(weights[0]), float(weights[1])
        if not (0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0) or not np.isclose(w1 + w2, 1.0, atol=1e-8):
            raise ValueError("weights must each be in [0,1] and sum to 1.0")
        if not (0.0 <= idf_alpha <= 1.0):
            raise ValueError("idf_alpha must be in [0,1]")
        if blend not in {"add", "geo"}:
            raise ValueError("blend must be 'add' or 'geo'")

        self.w1, self.w2 = w1, w2
        self.idf_alpha = float(idf_alpha)
        self.js_eps = float(js_eps)
        self.blend = blend
        self.use_support_gate = bool(use_support_gate)
        self.gate_mode = gate_mode
        self.gate_t = int(gate_t)
        self.gate_lambda = float(gate_lambda)
        self.compute_llr = bool(compute_llr)
        self.llr_eps = float(llr_eps)

        # precompute IDF (same pattern as ACT)
        self.idf_left_  = np.log((self.N + 1.0) / (self.f_left.astype(float)  + 1.0)) + 1.0
        self.idf_right_ = np.log((self.N + 1.0) / (self.f_right.astype(float) + 1.0)) + 1.0

        # placeholders
        self.idfcfws_ = None
        self.js01_ = None
        self.gate_ = None
        self.llr_G_ = None
        self.pvalues_ = None
        self._last_sim = None
        self._last_union = None

    # --- components (same IDF-CFWS as ACT) ---
    def _idfcfws(self) -> np.ndarray:
        avg_idf = 0.5 * (self.idf_left_[:, None] + self.idf_right_[None, :])
        weighted_co = self.co.astype(float) * avg_idf
        wL = self.f_left.astype(float)  * self.idf_left_
        wR = self.f_right.astype(float) * self.idf_right_
        weighted_union = wL[:, None] + wR[None, :] - weighted_co
        jacc = _safe_div(weighted_co, weighted_union)
        max_w = np.maximum(wL[:, None], wR[None, :])
        cf = _safe_div(weighted_co, max_w)
        if self.idf_alpha == 1.0: return jacc
        if self.idf_alpha == 0.0: return cf
        return (jacc ** self.idf_alpha) * (cf ** (1.0 - self.idf_alpha))

    def _js01(self) -> np.ndarray:
        # Jensen–Shannon between the joint P and the independence baseline Q (both 2×2),
        # then map to similarity in [0,1] via 1 - JSD/log2.
        eps = self.js_eps
        a  = self.co.astype(float) + eps
        fL = self.f_left.astype(float)[:, None] + eps
        fR = self.f_right.astype(float)[None, :] + eps
        N  = float(self.N) + 4.0*eps

        p11 = a / N
        p10 = (fL - a) / N
        p01 = (fR - a) / N
        p00 = 1.0 - (p11 + p10 + p01)
        P = np.stack([p11, p10, p01, p00], axis=0)

        p1dot = p11 + p10; p0dot = 1.0 - p1dot
        pdot1 = p11 + p01; pdot0 = 1.0 - pdot1
        Q = np.stack([p1dot*pdot1, p1dot*pdot0, p0dot*pdot1, p0dot*pdot0], axis=0)

        M = 0.5 * (P + Q)
        with np.errstate(divide="ignore", invalid="ignore"):
            KL_PM = np.where(P > 0, P * (np.log(P) - np.log(M)), 0.0).sum(axis=0)
            KL_QM = np.where(Q > 0, Q * (np.log(Q) - np.log(M)), 0.0).sum(axis=0)
        JSD = 0.5 * (KL_PM + KL_QM) / np.log(2.0)      # base 2
        js01 = 1.0 - np.clip(JSD, 0.0, 1.0)            # map to [0,1]
        return js01

    def _support_gate(self) -> Optional[np.ndarray]:
        if not self.use_support_gate: return None
        a = self.co.astype(float)
        if self.gate_mode == "cap":
            return np.minimum(1.0, a / max(1, self.gate_t))
        return 1.0 - np.exp(-self.gate_lambda * (a / max(1.0, float(self.N))))

    def _compute_matrix(self) -> np.ndarray:
        idfcfws = self._idfcfws()
        js01    = self._js01()
        if self.blend == "add":
            comp = self.w1 * idfcfws + self.w2 * js01
        else:  # geometric
            comp = np.exp(
                self.w1 * np.log(np.clip(idfcfws, 1e-12, 1.0)) +
                self.w2 * np.log(np.clip(js01,    1e-12, 1.0))
            )
        gate = self._support_gate()
        if gate is not None:
            comp = comp * gate
        if self.compute_llr:
            G, p = self._llr_gtest()
            self.llr_G_, self.pvalues_ = G, p

        self.idfcfws_ = idfcfws
        self.js01_    = js01
        self.gate_    = gate
        self.composite_ = comp
        self._last_sim = comp
        return comp

def _clip01(x, eps=1e-12):
    return np.clip(x, eps, 1.0 - eps)

def _nz_std(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    return s if s > 1e-12 else 1.0
