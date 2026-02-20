from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

# ---------- tiny utilities ----------


def _check_duplicate_columns(left: List[str], right: List[str]) -> None:
    dup = set(left).intersection(right)
    if dup:
        # soft warning; keep behavior compatible with your base class
        pass


def _to_binary(df: pd.DataFrame, cols: List[str], positive: bool = True) -> np.ndarray:
    arr = df[cols].to_numpy()
    return (arr > 0).astype(int) if positive else (arr != 0).astype(int)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _robust_tanh_cap(x: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    z = (x - med) / mad
    return 1.0 + kappa * np.tanh(z)

# ---------- BH–FDR ----------

def bh_fdr(pvals: np.ndarray, q: float = 0.05):
    """Benjamini–Hochberg procedure.
    Returns: discoveries mask (bool) and BH-adjusted p-values (q-values)."""
    pv = np.asarray(pvals, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]

    # Find largest k with p_(k) <= (k/m) q
    crit = (np.arange(1, m+1) / m) * q
    le = ranked <= crit
    k = le.nonzero()[0].max() + 1 if le.any() else 0

    disc = np.zeros(m, dtype=bool)
    if k > 0:
        disc[order[:k]] = True

    # BH-adjusted p-values (monotone step-up)
    qvals = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m-1, -1, -1):
        val = ranked[i] * m / (i+1)
        prev = min(prev, val)
        qvals[order[i]] = prev
    return disc, qvals

def bonferroni(pvals: np.ndarray, alpha: float = 0.05):
    """Bonferroni correction.
    Returns: discoveries mask (bool) and Bonferroni-adjusted p-values.

    - disc[i] = True if hypothesis i is significant at level alpha under Bonferroni.
    - pvals_adj[i] = min(p_i * m, 1.0), the corrected p-value.
    """
    pv = np.asarray(pvals, dtype=float)
    m = pv.size
    pvals_adj = np.minimum(pv * m, 1.0)
    disc = pvals_adj <= alpha
    return disc, pvals_adj

# ---------- alpha backbone params ----------

@dataclass
class AlphaBackboneParams:
    enable: bool = True
    min_support_n11: int = 5       # min co-tests for a pair
    min_marginal: int = 20         # min tests per single antibiotic
    sigmoid_tau: float = 1.0       # softness of mapping alpha-> [0,1]
    # drop edges with alpha <= 0 (and failing filters)
    hard_gate: bool = True
    # additionally multiply by sigmoid(alpha/tau)
    multiply_weight: bool = True
    correction: str = "bh"      # "bh", "bonferroni", or "none"
    fdr_q: float = 0.05            # target FDR
    fwer_alpha: float = 0.05       # used if correction == "bonferroni"

# ---------- domain weight config ----------

@dataclass
class DomainWeightParams:
    enable: bool = True
    guideline_scale: float = 0.20
    novelty_scale: float = 0.30
    pair_exponent: float = 0.35
    out_of_scope: str = "neutral"         # "neutral" | "mask"
    # "stretch" (bounded) | "multiply" (clamped)
    combine_mode: str = "stretch"
    idf_beta: float = 20.0
    idf_cap_kappa: float = 1.0
    lookup_df: Optional[pd.DataFrame] = None
    label_to_code: Optional[Callable[[str], str]] = None
    use_heuristics_if_missing: bool = True

# ---------- simple heuristics (safe defaults) ----------

def _heuristic_weights_for_codes(codes: Iterable[str]) -> pd.DataFrame:
    codes = list(codes)
    reserve = {"CZA", "CZT", "CCO", "CTL", "BPR", "DAL",
               "ORI", "TLV", "TZD", "LEF", "OMA", "DEL", "PLA"}
    last_resort = {"COL", "PMB", "CST"}
    advanced = {"LIZ", "TGC", "DPT", "IPM", "MER",
                "MEM", "ETP", "DOR", "FEP", "CPI", "CPO"}
    urinary = {"NFT", "NIT", "FOS", "PIC", "TRP", "NOR", "CIN", "PIM"}
    mac_lin = {"ERY", "AZM", "CLR", "CLI", "LIN", "ROX", "TEL"}
    flq = {"CIP", "LEV", "MOX", "OFX", "NOR", "ENO",
           "GAT", "GEM", "GRE", "DEL"}  # DEL handled above
    amg = {"GEN", "TOB", "AMK", "NET", "KAN", "SIS", "PLA"}
    folate = {"SXT", "TRP"}
    first_line = {"AMX", "AMP", "AMC", "AMS", "TZP",
                  "PIT", "CEZ", "CXM", "CTX", "CRO", "CAZ", "FEP"}
    topicals_or_niche = {"MUP", "FUS", "NOV", "PRI"}
    antifungals = {"FCA", "ITR", "KET", "MCZ", "NYS",
                   "VOR", "POS", "IVC", "MIC", "ANI", "5FC"}
    tb_only = {"INH", "EMB", "PZA", "ETH", "PAS", "CPR", "CSE", "PTH"}

    rows = []
    for code in codes:
        gw, nw = 1, 1
        note = []
        if code in antifungals or code in tb_only:
            gw, nw = 0, 0
            note.append("out_of_scope")
        if code in topicals_or_niche:
            gw = 0
            nw = 0
            note.append("topical")
        if code in urinary:
            gw = max(gw, 2)
            nw = max(nw, 0)
            note.append("urinary")
        if code in reserve:
            gw = max(gw, 2)
            nw = max(nw, 3)
            note.append("reserve")
        if code in last_resort:
            gw = max(gw, 2)
            nw = max(nw, 3)
            note.append("last_resort")
        if code in advanced:
            gw = max(gw, 2)
            nw = max(nw, 2)
            note.append("advanced")
        if code in first_line:
            gw = max(gw, 2)
        if code in flq:
            gw = max(gw, 2)
            nw = max(nw, 1)
        if code in mac_lin:
            gw = max(gw, 2)
            nw = max(nw, 1)
        if code in amg:
            gw = max(gw, 2)
            nw = max(nw, 1)
        if code in folate:
            gw = max(gw, 2)
        rows.append((code, gw, nw, ";".join(note)))
    return pd.DataFrame(rows, columns=["antibiotic_code", "guideline_weight_0to3", "novelty_weight_0to3", "notes"])

def _default_label_to_code(label: str) -> str:
    s = str(label).strip()
    token = ""
    for ch in s:
        if ch.isalnum():
            token += ch
        else:
            break
    return token or s

# ---------- mixin that computes left/right domain multipliers ----------

class _DomainWeightsMixin:
    def _init_domain_weights(
        self,
        labels_left: List[str],
        labels_right: List[str],
        N_events: int,
        f_left: np.ndarray,
        f_right: np.ndarray,
        params: DomainWeightParams,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # IDF tempering (vectorized)
        idf_left = np.log((N_events + params.idf_beta) /
                          (f_left + params.idf_beta))
        idf_right = np.log((N_events + params.idf_beta) /
                           (f_right + params.idf_beta))
        w_idf_left = _robust_tanh_cap(idf_left, params.idf_cap_kappa)
        w_idf_right = _robust_tanh_cap(idf_right, params.idf_cap_kappa)

        if not params.enable:
            ones_l = np.ones_like(f_left, dtype=float)
            ones_r = np.ones_like(f_right, dtype=float)
            return w_idf_left, w_idf_right, ones_l, ones_r

        label_to_code = params.label_to_code or _default_label_to_code
        codes_left = [label_to_code(x) for x in labels_left]
        codes_right = [label_to_code(x) for x in labels_right]

        need_codes = list(dict.fromkeys(codes_left + codes_right))
        lu = params.lookup_df
        if lu is None and params.use_heuristics_if_missing:
            lu = _heuristic_weights_for_codes(need_codes)
        elif lu is not None:
            lu = lu.copy()
        else:
            lu = pd.DataFrame({"antibiotic_code": need_codes,
                               "guideline_weight_0to3": 1,
                               "novelty_weight_0to3": 1})

        lu = lu.set_index("antibiotic_code").reindex(need_codes)
        gw = lu["guideline_weight_0to3"].fillna(1).astype(float).to_numpy()
        nw = lu["novelty_weight_0to3"].fillna(1).astype(float).to_numpy()

        gw_norm = gw / 3.0
        nw_norm = nw / 3.0
        multi = (1.0 + params.guideline_scale * gw_norm) * \
            (1.0 + params.novelty_scale * nw_norm)

        idx_map = {code: i for i, code in enumerate(need_codes)}
        w_dom_left = np.array([multi[idx_map[c]]
                              for c in codes_left], dtype=float)
        w_dom_right = np.array([multi[idx_map[c]]
                               for c in codes_right], dtype=float)

        if params.out_of_scope == "mask":
            mask = (gw == 0) & (nw == 0)
            for cidx, code in enumerate(need_codes):
                if mask[cidx]:
                    if code in codes_left:
                        w_dom_left[[i for i, c in enumerate(
                            codes_left) if c == code]] = 0.0
                    if code in codes_right:
                        w_dom_right[[j for j, c in enumerate(
                            codes_right) if c == code]] = 0.0

        return w_idf_left, w_idf_right, w_dom_left, w_dom_right

# ---------- combinatorics helpers ----------

def _logC(n, k):
    """
    Vectorized log binomial coefficient log C(n, k).
    Works with scalars or numpy arrays (broadcasted).
    Returns -inf where k < 0 or k > n.
    """
    n = np.asarray(n)
    k = np.asarray(k)

    # Broadcast to common shape
    n_b, k_b = np.broadcast_arrays(n, k)

    out = np.full(n_b.shape, -np.inf, dtype=float)
    valid = (k_b >= 0) & (k_b <= n_b)

    if np.any(valid):
        nb = n_b[valid].astype(float)
        kb = k_b[valid].astype(float)
        out[valid] = (
            gammaln(nb + 1.0)
            - gammaln(kb + 1.0)
            - gammaln(nb - kb + 1.0)
        )
    # If both inputs were pure scalars, return a scalar
    return out.item() if out.shape == () else out

# ---------- Fisher (noncentral) log-likelihood for alpha MLE ----------

def _fisher_loglik(n11, m1, m2, k, omega):
    # support of X
    lo, hi = max(0, k - m2), min(k, m1)
    xs = np.arange(lo, hi+1)
    logw = _logC(m1, xs) + _logC(m2, k - xs) + xs*np.log(omega)
    logZ = logsumexp(logw)
    if not np.isfinite(logZ):
        return -np.inf
    return _logC(m1, n11) + _logC(m2, k - n11) + n11*np.log(omega) - logZ

def _alpha_mle_fisher(n11, n10, n01, n00):
    m1 = n11 + n10
    m2 = n01 + n00
    k = n11 + n01
    # degenerate margins -> neutral alpha
    if (m1 == 0) or (m2 == 0) or (k == 0) or (k == m1 + m2):
        return 0.0

    def nll(logw):
        ll = _fisher_loglik(n11, m1, m2, k, np.exp(logw))
        return -(ll if np.isfinite(ll) else -1e12)
    res = minimize_scalar(nll, bounds=(-20, 20), method="bounded",
                          options={"xatol": 1e-4, "maxiter": 200})
    return float(res.x) if res.success else np.nan  # alpha = log(omega)

# ---------- Fisher exact two-sided p-value (central hypergeometric, omega=1) ----------

def _fisher_two_sided_pval(n11, n10, n01, n00):
    """Two-sided Fisher p-value via enumeration of support, central hypergeometric."""
    m1 = n11 + n10
    m2 = n01 + n00
    k = n11 + n01
    N = m1 + m2
    if (m1 == 0) or (m2 == 0) or (k == 0) or (k == N):
        return 1.0
    lo, hi = max(0, k - m2), min(k, m1)
    xs = np.arange(lo, hi+1)

    # log pmf up to normalization, omega=1 => standard hypergeometric
    logw = _logC(m1, xs) + _logC(m2, k - xs)
    logZ = logsumexp(logw)
    logp = logw - logZ
    # observed probability
    lp_obs = _logC(m1, n11) + _logC(m2, k - n11) - logZ
    # "Extreme" definition: sum probs with p <= p_obs
    mask = logp <= lp_obs + 1e-12
    pval = float(np.clip(np.exp(logsumexp(logp[mask])), 0.0, 1.0))
    return pval

# ---------- Fisher’s exact test (central hypergeometric) ----------
def fisher_exact_table(a: int, b: int, c: int, d: int, *, alternative: str = "two-sided") -> Tuple[float, float]:
    """
    Fisher’s exact test on a 2x2 table [[a,b],[c,d]].
    Returns (odds_ratio, p_value).

    alternative: "two-sided" | "greater" | "less"
        "greater": tests for positive association (odds ratio > 1)
        "less":    tests for negative association (odds ratio < 1)
        "two-sided": sum of probabilities of all tables with probability <= observed
                     under the central hypergeometric with fixed margins.

    Notes:
    - Uses log-domain enumeration over the support to avoid underflow/overflow.
    - Matches the standard definition used by SciPy for two-sided.
    """
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    # Margins
    m1 = a + b
    m2 = c + d
    k = a + c
    N = m1 + m2

    # Odds ratio (handle zeros gracefully)
    if b * c == 0:
        if a * d == 0:
            odds = np.nan  # undefined (all zero or one entire row/col zero)
        else:
            odds = np.inf  # division by zero -> infinite odds
    else:
        odds = (a * d) / (b * c)

    # Degenerate margins -> p = 1.0
    if (m1 == 0) or (m2 == 0) or (k == 0) or (k == N):
        return float(odds), 1.0

    # Support of X ~ Hypergeometric under H0
    lo, hi = max(0, k - m2), min(k, m1)
    xs = np.arange(lo, hi + 1)

    # log pmf up to normalization: log C(m1, x) + log C(m2, k-x)
    logw = _logC(m1, xs) + _logC(m2, k - xs)
    logZ = logsumexp(logw)
    logp = logw - logZ  # normalized log-probabilities
    p = np.exp(logp)

    # Observed table probability
    # Observed a = 'a'
    idx_obs = a - lo  # index into xs
    # Guard if a out of support due to illegal input (shouldn't happen if inputs consistent)
    if idx_obs < 0 or idx_obs >= len(xs):
        return float(odds), 1.0
    p_obs = p[idx_obs]
    logp_obs = logp[idx_obs]

    alt = alternative.lower()
    if alt == "greater":
        # right tail: X >= a
        pval = float(np.clip(p[idx_obs:].sum(), 0.0, 1.0))
    elif alt == "less":
        # left tail: X <= a
        pval = float(np.clip(p[:idx_obs + 1].sum(), 0.0, 1.0))
    elif alt == "two-sided":
        # Sum probabilities of tables with prob <= p_obs (the “extreme” criterion)
        mask = logp <= (logp_obs + 1e-12)
        pval = float(np.clip(p[mask].sum(), 0.0, 1.0))
    else:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'.")

    return float(odds), pval

# ---------- alpha backbone with BH-FDR and Benferroni ----------
def _alpha_backbone_from_counts(co: np.ndarray,
                                fL: np.ndarray,
                                fR: np.ndarray,
                                N: int,
                                params: AlphaBackboneParams):
    """
    Returns:
      alpha:   MxM real log-odds (diag=0)
      gate:    MxM bool mask (alpha>0 & support/margins ok & significant if enabled)
      aweight: MxM in [0,1] = sigmoid(alpha/tau)
      pvals:   MxM p-value matrix (two-sided Fisher), 1 on diag; 1 where not computed
      qvals:   MxM adjusted p-values (BH q-values or Bonferroni adj p); else = pvals
    """
    M = co.shape[0]
    n11 = co.astype(int)
    n10 = (fL[:, None] - n11).astype(int)
    n01 = (fR[None, :] - n11).astype(int)
    n00 = (N - n11 - n10 - n01).astype(int)

    alpha = np.zeros((M, M), dtype=float)
    gate = np.zeros((M, M), dtype=bool)
    aweight = np.ones((M, M), dtype=float)
    pvals = np.ones((M, M), dtype=float)
    qvals = np.ones((M, M), dtype=float)

    # Collect p-values for multiple testing only for candidate pairs we actually test
    idx_i, idx_j, pv_list = [], [], []

    for i in range(M):
        for j in range(i+1, M):
            # basic support/marginals screening
            if n11[i, j] < params.min_support_n11:
                continue
            if (fL[i] < params.min_marginal) or (fR[j] < params.min_marginal):
                continue

            # alpha (log-odds) MLE
            a = _alpha_mle_fisher(n11[i, j], n10[i, j], n01[i, j], n00[i, j])
            if not np.isfinite(a):
                continue
            alpha[i, j] = alpha[j, i] = a

            # two-sided Fisher p-value under independence
            pv = _fisher_two_sided_pval(
                n11[i, j], n10[i, j], n01[i, j], n00[i, j])
            pvals[i, j] = pvals[j, i] = pv

            idx_i.append(i)
            idx_j.append(j)
            pv_list.append(pv)

    # Multiple testing correction
    sig_pairs = set()
    if len(pv_list) > 0 and params.enable:
        pv_arr = np.array(pv_list, dtype=float)

        if params.correction.lower() == "bh":
            disc, adj = bh_fdr(pv_arr, q=params.fdr_q)
        elif params.correction.lower() == "bonferroni":
            disc, adj = bonferroni(pv_arr, alpha=params.fwer_alpha)
        elif params.correction.lower() == "none":
            # no adjustment; simple alpha threshold on raw p
            alpha0 = float(params.alpha_threshold)
            adj = pv_arr.copy()
            disc = pv_arr <= alpha0
        else:
            raise ValueError(f"Unknown correction: {params.correction}")

        # fill adjusted values and significance mask back into matrices
        for (i, j, is_sig, adj_ij) in zip(idx_i, idx_j, disc, adj):
            qvals[i, j] = qvals[j, i] = adj_ij
            if is_sig:
                sig_pairs.add((i, j))
    else:
        # No correction applied: report qvals=pvals for convenience
        qvals[:] = pvals

    # Gate & alpha-based soft weights
    tau = max(params.sigmoid_tau, 1e-6)
    for i in range(M):
        for j in range(i+1, M):
            # skip entries that never passed the basic tests above
            if alpha[i, j] == 0.0 and pvals[i, j] == 1.0:
                continue

            keep = True
            if params.hard_gate:
                # require positive association
                keep = keep and (alpha[i, j] > 0.0)

                # require significance if correction enabled
                if params.enable and params.correction.lower() in {"bh", "bonferroni", "none"}:
                    keep = keep and ((i, j) in sig_pairs)

            gate[i, j] = gate[j, i] = keep

            # soft weight from alpha
            aw = 1.0 / (1.0 + np.exp(-alpha[i, j] / tau))
            aweight[i, j] = aweight[j, i] = aw

    # diagonals
    np.fill_diagonal(alpha, 0.0)
    np.fill_diagonal(gate, True)
    np.fill_diagonal(aweight, 1.0)
    np.fill_diagonal(pvals, 1.0)
    np.fill_diagonal(qvals, 1.0)
    return alpha, gate, aweight, pvals, qvals

def _saturation_level(gamma: float, cap: float) -> float:
    """Implied saturation S* = 1 / (1 + exp(-gamma * cap)) for the PPMI logistic."""
    return 1.0 / (1.0 + np.exp(-gamma * cap))

# ---------- PPMI with optional sigmoid scaling ----------
def _ppmi_sigmoid(
    co: np.ndarray,
    fL: np.ndarray,
    fR: np.ndarray,
    N: int,
    *,
    gamma: float = 2.0,
    cap: Optional[float] = 10.0,
    eps: float = 1e-9,
    min_support: int = 5
) -> np.ndarray:
    """Vectorized PPMI mapped to [0,1] with a logistic.

    Parameters
    ----------
    co : (M,M) ndarray
        Co-occurrence counts n11 (raw, unweighted) for left x right.
    fL, fR : (M,) ndarray
        Marginal counts (#1s) per left/right feature.
    N : int
        Number of rows/samples.
    gamma : float
        Logistic slope controlling how fast S saturates with PPMI.
    cap : Optional[float]
        If provided, clip PPMI to this value for stability (typ. 6–10).
    eps : float
        Numerical safety for logs.
    min_support : int
        Require n11 >= min_support; else return 0.

    Returns
    -------
    S_ppmi : (M,M) ndarray in [0,1]
    """
    fL = fL.astype(float)
    fR = fR.astype(float)
    Nf = float(N)

    px = fL / Nf                      # (M,)
    py = fR / Nf                      # (M,)
    pxy = co.astype(float) / Nf       # (M,M)

    pxpy = np.outer(px, py)           # independence baseline

    pmi = np.log(np.maximum(pxy, eps)) - np.log(np.maximum(pxpy, eps))
    ppmi = np.maximum(0.0, pmi)
    if cap is not None and cap > 0:
        ppmi = np.minimum(ppmi, cap)

    S_ppmi = 1.0 / (1.0 + np.exp(-gamma * ppmi))
    S_ppmi = np.where(co >= min_support, S_ppmi, 0.0)
    return S_ppmi
