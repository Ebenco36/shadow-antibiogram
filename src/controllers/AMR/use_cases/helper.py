# from __future__ import annotations
# import json
# import os
# from pathlib import Path
# from typing import Optional,Tuple, List, Iterable

# import pandas as pd

# from src.controllers.AMR.similarity.engine import SimilarityEngine
# from src.controllers.AMR.clustering.graph_builder import GraphBuilder
# from src.controllers.AMR.clustering.louvain_clusterer import LouvainClusterer
# from src.controllers.AMR.statistics.edge_significance import EdgeSignificancePruner
# from src.utils.helpers import get_label
# from src.utils.network import visualize_antibiotic_network
# from src.controllers.AMR.use_cases.config_similarity import (
#     BEST_SIMILARITY_METRIC,
#     BEST_TAU,
#     BEST_GAMMA,
# )

# from dataclasses import dataclass


# @dataclass(frozen=True)
# class ContinuousFilterResult:
#     df_continuous: pd.DataFrame
#     continuous_orgs: list[str]
#     year_range: Tuple[int, int]
#     org_col: str
#     year_col: str


# def _ensure_year_column(
#     df: pd.DataFrame,
#     *,
#     year_col: str = "Year",
#     date_col: Optional[str] = "Date",
# ) -> pd.DataFrame:
#     """
#     Ensure df has an integer Year column.
#     Priority: existing year_col -> derive from date_col.
#     """
#     out = df.copy()

#     if year_col in out.columns and out[year_col].notna().any():
#         out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
#         return out

#     if date_col is None or date_col not in out.columns:
#         raise ValueError(
#             f"Cannot create '{year_col}'. Provide a valid date_col. "
#             f"Missing '{year_col}' and '{date_col}'."
#         )

#     out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
#     out = out.dropna(subset=[date_col]).copy()
#     out[year_col] = out[date_col].dt.year.astype("Int64")
#     return out


# def filter_continuous_organisations(
#     df: pd.DataFrame,
#     *,
#     org_col: str = "NumberOrganisation",
#     year_col: str = "Year",
#     date_col: Optional[str] = "Date",
#     min_year: int = 2019,
#     max_year: int = 2023,
#     keep_only_year_span: bool = True,
#     verbose: bool = True,
# ) -> ContinuousFilterResult:
#     """
#     Return isolates from organisations that appear in every year in [min_year, max_year].

#     Notes
#     - Uses year_col if present; otherwise derives year_col from date_col.
#     - Organisation IDs are coerced to string for stable grouping.
#     """
#     if df is None or df.empty:
#         raise ValueError("Input dataframe is empty.")

#     if org_col not in df.columns:
#         raise ValueError(f"Organisation column not found: '{org_col}'")

#     work = _ensure_year_column(df, year_col=year_col, date_col=date_col)

#     # Drop rows missing org/year
#     work = work.dropna(subset=[org_col, year_col]).copy()

#     # Restrict to year span for the continuous definition
#     span = set(range(min_year, max_year + 1))
#     work[year_col] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
#     work = work[work[year_col].between(min_year, max_year, inclusive="both")].copy()

#     # Normalise org ids
#     work[org_col] = work[org_col].astype(str)

#     if work.empty:
#         if verbose:
#             print("After dropping NA org/year and restricting to the year span, dataframe is empty.")
#             print(f"Check: org_col='{org_col}', year_col='{year_col}', date_col='{date_col}'")
#         return ContinuousFilterResult(work, [], (min_year, max_year), org_col, year_col)

#     # Compute year coverage per org
#     years_by_org = work.groupby(org_col)[year_col].agg(lambda s: set(int(x) for x in s.dropna().unique()))
#     continuous_orgs = sorted([org for org, ys in years_by_org.items() if span.issubset(ys)])

#     df_cont = work[work[org_col].isin(continuous_orgs)].copy()

#     if keep_only_year_span:
#         df_cont = df_cont[df_cont[year_col].between(min_year, max_year, inclusive="both")].copy()

#     if verbose:
#         observed_years = sorted(work[year_col].dropna().unique().tolist())
#         print(f"Observed years in data (after cleaning): {observed_years[:10]}{'...' if len(observed_years) > 10 else ''}")
#         print(f"Target continuous span: {min_year}-{max_year} ({len(span)} years)")
#         print(f"Total organisations in span: {work[org_col].nunique():,}")
#         print(f"Continuous organisations: {len(continuous_orgs):,}")
#         print(f"Isolates retained (continuous): {len(df_cont):,}")

#         if len(continuous_orgs) == 0:
#             # Helpful diagnostics: how many years do orgs typically cover?
#             coverage = years_by_org.apply(len)
#             print("No continuous orgs found. Coverage distribution (#years per org) in span:")
#             print(coverage.value_counts().sort_index())

#     return ContinuousFilterResult(df_cont, continuous_orgs, (min_year, max_year), org_col, year_col)




# def build_and_save_network_for_cohort(
#     cohort_df: pd.DataFrame,
#     cohort_name: str,
#     output_dir: Path,
#     metric: str = BEST_SIMILARITY_METRIC,
#     tau: float = BEST_TAU,
#     gamma: float = BEST_GAMMA,
#     use_fdr: bool = False,
#     fdr_alpha: float = 0.05,
#     fdr_min_total: int = 20,
#     fdr_min_positive: int = 3,
#     fdr_alternative: str = "two-sided",
#     suffix: Optional[str] = None,
#     n_louvain_iterations: int = 100,   # match ParameterConfig.n_iterations default
#     base_seed: int = 100,              # match ExperimentConfig.random_seed default
#     title_prefix: Optional[str] = None,
# ) -> None:
#     """
#     Build a similarity-based network for a cohort and save HTML/PNG/GEXF.

#     - Uses global best metric/params by default (Jaccard, tau, gamma).
#     - If use_fdr=True, prunes edges using EdgeSignificancePruner before
#       thresholding with tau.
#     - Runs Louvain multiple times (n_louvain_iterations) with different seeds.
#     - Keeps get_label-based antibiotic relabeling for visualization.
#     """

#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Antibiotic columns
#     abx_cols = [c for c in cohort_df.columns if c.endswith("_Tested")]
#     if not abx_cols:
#         raise ValueError(f"No _Tested columns found for cohort {cohort_name}")

#     # 2) Similarity matrix (e.g. Jaccard)
#     sim_engine = SimilarityEngine(cohort_df, abx_cols)
#     similarity_matrix = sim_engine.compute(metric)

#     # 3) Optional FDR edge pruning (work on the original antibiotic names)
#     if use_fdr:
#         df_binary = (
#             cohort_df[abx_cols]
#             .apply(pd.to_numeric, errors="coerce")
#             .fillna(0)
#             .astype(int)
#         )

#         pruner = EdgeSignificancePruner(
#             df_binary=df_binary,
#             antibiotic_cols=abx_cols,
#             alpha=fdr_alpha,
#             min_total=fdr_min_total,
#             min_positive=fdr_min_positive,
#             alternative=fdr_alternative,
#         ).fit()

#         allowed_pairs = pruner.get_significant_pairs()  # Set[(u, v)]

#         # Build mask: only allow significant pairs + diagonal
#         abx_list = list(similarity_matrix.index)
#         mask = pd.DataFrame(False, index=abx_list, columns=abx_list)

#         for u, v in allowed_pairs:
#             if u in mask.index and v in mask.columns:
#                 mask.loc[u, v] = True
#                 mask.loc[v, u] = True

#         for a in abx_list:
#             mask.loc[a, a] = True

#         similarity_matrix = similarity_matrix.where(mask, 0.0)

#     # 4) Build graph and run Louvain (multiple seeds)
#     graph_builder = GraphBuilder(tau)
#     G = graph_builder.build_graph(similarity_matrix)

#     clusterer = LouvainClusterer(gamma)

#     # seeds: base_seed, base_seed+1, ..., base_seed+n_louvain_iterations-1
#     if n_louvain_iterations <= 0:
#         seeds = [base_seed]
#     else:
#         seeds = [base_seed + i for i in range(n_louvain_iterations)]

#     partitions = clusterer.run_multiple(G, seeds)
#     if not partitions:
#         raise RuntimeError(f"Louvain failed for cohort {cohort_name}")

#     # NOTE: run_multiple currently returns partitions only; if you later
#     # extend it to return modularities, you can select the best here.
#     partition = partitions[0]  # currently unused but kept for future stats

#     # 5) Safe file base name
#     safe_name = cohort_name.replace(" ", "_")
#     if suffix:
#         safe_name = f"{safe_name}{suffix}"

#     html_name = f"net_{safe_name}.html"
#     png_name = f"net_{safe_name}.png"
#     pdf_name = f"net_{safe_name}.pdf"
#     gexf_name = f"net_{safe_name}.gexf"

#     base_title = title_prefix or f"Network – {cohort_name}"
#     title = (
#         f"{base_title} – {metric} (τ={tau}, γ={gamma})"
#         + (" – FDR-pruned" if use_fdr else "")
#     )

#     # 6) Optional: load antibiotic class map + get_label
#     abx_class_map = None
#     antibiotic_class_map_path: str = "./datasets/antibiotic_class_grouping.json"
#     if os.path.exists(antibiotic_class_map_path):
#         try:
#             with open(antibiotic_class_map_path, "r") as f:
#                 abx_class_map = json.load(f)
#         except Exception:
#             abx_class_map = None  # fall back to raw names if broken file

#     label_map = None
#     if abx_class_map is not None:
#         try:
#             label_map = get_label(
#                 abx_cols,
#                 antibiotic_class_map=abx_class_map,
#                 format_type="abbr",
#                 enrich=True,
#                 include_class=False,
#             )
#         except Exception:
#             # If label creation fails, just keep original antibiotic names
#             label_map = None

#     AWARE_PALETTE = {
#         "Access":  "#56B4E9",
#         "Watch":   "#E69F00",
#         "Reserve": "#D55E00",
#         "Unknown": "#999999",
#     }
#     from src.runners.Phases.Phase_I2 import build_who_map
#     who_map = build_who_map()
    
#     def _norm_aware(cls: str) -> str:
#         s = str(cls).strip()
#         # robust normalization (handles "access", "ACCESS", "Access ")
#         s_low = s.lower()
#         if s_low == "access":  return "Access"
#         if s_low == "watch":   return "Watch"
#         if s_low == "reserve": return "Reserve"
#         return "Unknown"

#     if label_map:
#         sim_for_viz = similarity_matrix.rename(index=label_map, columns=label_map)
#         # 2) build color map in the *display-label space*
#         aware_color_map = {}
#         for raw_abx, cls in who_map.items():
#             disp = label_map.get(raw_abx)  # e.g. "AMS"
#             if disp is None:
#                 continue
#             aware_color_map[disp] = AWARE_PALETTE.get(_norm_aware(cls), AWARE_PALETTE["Unknown"])
#     else:
#         sim_for_viz = similarity_matrix
#         aware_color_map = {
#             raw_abx: AWARE_PALETTE.get(_norm_aware(cls), AWARE_PALETTE["Unknown"])
#             for raw_abx, cls in who_map.items()
#         }


#     # 7) Visualize & save
#     visualize_antibiotic_network(
#         data_input=sim_for_viz,
#         threshold=tau,
#         community_gamma=gamma,
#         output_dir=str(output_dir),
#         output_html=html_name,
#         output_image=png_name,
#         output_pdf=pdf_name,
#         gexf_path=gexf_name,
#         title=None,
#         remove_isolated=False,
#         semantic_color_map=aware_color_map,
#         node_color_mode="semantic",

#     )


# def save_raw_and_pruned_networks(
#     cohort_df: pd.DataFrame,
#     cohort_name: str,
#     output_dir: Path,
#     metric: str = BEST_SIMILARITY_METRIC,
#     tau: float = BEST_TAU,
#     gamma: float = BEST_GAMMA,
#     fdr_alpha: float = 0.05,
#     fdr_min_total: int = 20,
#     fdr_min_positive: int = 3,
#     fdr_alternative: str = "two-sided",
#     n_louvain_iterations: int = 100,
#     base_seed: int = 100,
#     title_prefix: Optional[str] = None,
# ) -> None:
#     """
#     Convenience wrapper: for a given cohort, save both:
#       - raw network
#       - FDR-pruned network
#     using the same Louvain multi-run setup.
#     """

#     # RAW
#     build_and_save_network_for_cohort(
#         cohort_df=cohort_df,
#         cohort_name=cohort_name,
#         output_dir=output_dir,
#         metric=metric,
#         tau=tau,
#         gamma=gamma,
#         use_fdr=False,
#         suffix="_raw",
#         n_louvain_iterations=n_louvain_iterations,
#         base_seed=base_seed,
#         title_prefix=title_prefix,
#     )

#     # FDR-PRUNED
#     build_and_save_network_for_cohort(
#         cohort_df=cohort_df,
#         cohort_name=cohort_name,
#         output_dir=output_dir,
#         metric=metric,
#         tau=tau,
#         gamma=gamma,
#         use_fdr=True,
#         fdr_alpha=fdr_alpha,
#         fdr_min_total=fdr_min_total,
#         fdr_min_positive=fdr_min_positive,
#         fdr_alternative=fdr_alternative,
#         suffix="_FDR",
#         n_louvain_iterations=n_louvain_iterations,
#         base_seed=base_seed,
#         title_prefix=title_prefix,
#     )




from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Iterable, Dict, Set

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.controllers.AMR.similarity.engine import SimilarityEngine
from src.controllers.AMR.clustering.graph_builder import GraphBuilder
from src.controllers.AMR.clustering.louvain_clusterer import LouvainClusterer
from src.controllers.AMR.statistics.edge_significance import EdgeSignificancePruner
from src.utils.helpers import get_label
from src.utils.network import visualize_antibiotic_network
from src.controllers.AMR.use_cases.config_similarity import (
    BEST_SIMILARITY_METRIC,
    BEST_TAU,
    BEST_GAMMA,
)

# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class ContinuousFilterResult:
    df_continuous: pd.DataFrame
    continuous_orgs: list[str]
    year_range: Tuple[int, int]
    org_col: str
    year_col: str


# ============================================================
# Format detection helpers
# ============================================================

PAIRWISE_CORE = {"ab_1", "ab_2", "a", "b", "c", "d"}

def is_pairwise_df(df: pd.DataFrame) -> bool:
    return PAIRWISE_CORE.issubset(set(df.columns))

def get_wide_abx_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).endswith("_Tested")]

def get_pairwise_abx_labels(df: pd.DataFrame) -> List[str]:
    s1 = set(df["ab_1"].astype("string").dropna().astype(str).unique())
    s2 = set(df["ab_2"].astype("string").dropna().astype(str).unique())
    return sorted(s1.union(s2))


# ============================================================
# Year utilities (unchanged)
# ============================================================

def _ensure_year_column(
    df: pd.DataFrame,
    *,
    year_col: str = "Year",
    date_col: Optional[str] = "Date",
) -> pd.DataFrame:
    out = df.copy()

    if year_col in out.columns and out[year_col].notna().any():
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        return out

    if date_col is None or date_col not in out.columns:
        raise ValueError(
            f"Cannot create '{year_col}'. Provide a valid date_col. "
            f"Missing '{year_col}' and '{date_col}'."
        )

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out[year_col] = out[date_col].dt.year.astype("Int64")
    return out


def filter_continuous_organisations(
    df: pd.DataFrame,
    *,
    org_col: str = "NumberOrganisation",
    year_col: str = "Year",
    date_col: Optional[str] = "Date",
    min_year: int = 2019,
    max_year: int = 2023,
    keep_only_year_span: bool = True,
    verbose: bool = True,
) -> ContinuousFilterResult:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    if org_col not in df.columns:
        raise ValueError(f"Organisation column not found: '{org_col}'")

    work = _ensure_year_column(df, year_col=year_col, date_col=date_col)

    work = work.dropna(subset=[org_col, year_col]).copy()

    span = set(range(min_year, max_year + 1))
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work = work[work[year_col].between(min_year, max_year, inclusive="both")].copy()

    work[org_col] = work[org_col].astype(str)

    if work.empty:
        if verbose:
            print("After dropping NA org/year and restricting to the year span, dataframe is empty.")
            print(f"Check: org_col='{org_col}', year_col='{year_col}', date_col='{date_col}'")
        return ContinuousFilterResult(work, [], (min_year, max_year), org_col, year_col)

    years_by_org = work.groupby(org_col)[year_col].agg(lambda s: set(int(x) for x in s.dropna().unique()))
    continuous_orgs = sorted([org for org, ys in years_by_org.items() if span.issubset(ys)])

    df_cont = work[work[org_col].isin(continuous_orgs)].copy()

    if keep_only_year_span:
        df_cont = df_cont[df_cont[year_col].between(min_year, max_year, inclusive="both")].copy()

    if verbose:
        observed_years = sorted(work[year_col].dropna().unique().tolist())
        print(f"Observed years in data (after cleaning): {observed_years[:10]}{'...' if len(observed_years) > 10 else ''}")
        print(f"Target continuous span: {min_year}-{max_year} ({len(span)} years)")
        print(f"Total organisations in span: {work[org_col].nunique():,}")
        print(f"Continuous organisations: {len(continuous_orgs):,}")
        print(f"Rows retained (continuous): {len(df_cont):,}")

        if len(continuous_orgs) == 0:
            coverage = years_by_org.apply(len)
            print("No continuous orgs found. Coverage distribution (#years per org) in span:")
            print(coverage.value_counts().sort_index())

    return ContinuousFilterResult(df_cont, continuous_orgs, (min_year, max_year), org_col, year_col)


# ============================================================
# Pairwise similarity + FDR (NEW)
# ============================================================

def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.zeros_like(num, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out

def _pairwise_metric_values(df: pd.DataFrame, metric: str) -> np.ndarray:
    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    c = df["c"].to_numpy(dtype=float)
    d = df["d"].to_numpy(dtype=float)

    m = metric.lower()
    if m in ("jaccard", "jac"):
        return _safe_div(a, (a + b + c))
    if m in ("dice", "sorensen"):
        return _safe_div(2 * a, (2 * a + b + c))
    if m in ("cosine", "cos"):
        return _safe_div(a, np.sqrt((a + b) * (a + c)))
    if m in ("overlap", "ovl"):
        return _safe_div(a, np.minimum(a + b, a + c))
    if m == "phi":
        num = (a * d) - (b * c)
        den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        return _safe_div(num, den)

    raise ValueError(
        f"Metric '{metric}' not supported for PAIRWISE data. "
        f"Use: jaccard, dice, cosine, overlap, phi."
    )

def _pairwise_square_matrix(df: pd.DataFrame, vals: np.ndarray, fill_diagonal: float = 1.0) -> pd.DataFrame:
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

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
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

def pairwise_fdr_significant_pairs(
    df_pairwise: pd.DataFrame,
    *,
    tau_gate: float = 0.30,
    alpha: float = 0.05,
    alternative: str = "greater",
    metric_for_gate: str = "jaccard",
) -> Set[Tuple[str, str]]:
    """
    For PAIRWISE data, compute Fisher+BH-FDR over all pairs passing similarity >= tau_gate.
    Returns a set of significant undirected pairs (u,v) with u< v lexicographically.
    """
    work = df_pairwise.copy()
    for c in ["a", "b", "c", "d"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)

    sim = _pairwise_metric_values(work, metric_for_gate)
    gate = sim >= float(tau_gate)
    work = work.loc[gate].copy()
    if work.empty:
        return set()

    pvals = []
    pairs = []
    for r in work.itertuples(index=False):
        a = int(r.a); b = int(r.b); c = int(r.c); d = int(r.d)
        _, p = fisher_exact([[a, b], [c, d]], alternative=alternative)
        pvals.append(p)
        u = str(r.ab_1); v = str(r.ab_2)
        pairs.append((u, v))

    pvals = np.asarray(pvals, dtype=float)
    qvals = _bh_fdr(pvals)
    keep = qvals <= float(alpha)

    out: Set[Tuple[str, str]] = set()
    for (u, v), ok in zip(pairs, keep):
        if not ok:
            continue
        uu, vv = (u, v) if u < v else (v, u)
        out.add((uu, vv))
    return out


# ============================================================
# Main helper: build and save network (updated for BOTH)
# ============================================================

def build_and_save_network_for_cohort(
    cohort_df: pd.DataFrame,
    cohort_name: str,
    output_dir: Path,
    metric: str = BEST_SIMILARITY_METRIC,
    tau: float = BEST_TAU,
    gamma: float = BEST_GAMMA,
    use_fdr: bool = False,
    fdr_alpha: float = 0.05,
    fdr_min_total: int = 20,
    fdr_min_positive: int = 3,
    fdr_alternative: str = "two-sided",
    suffix: Optional[str] = None,
    n_louvain_iterations: int = 100,
    base_seed: int = 100,
    title_prefix: Optional[str] = None,
) -> None:
    """
    Build a similarity-based network for a cohort and save HTML/PNG/PDF/GEXF.

    Works for:
      - WIDE isolate-level cohorts with *_Tested columns
      - PAIRWISE aggregated cohorts with ab_1/ab_2/a/b/c/d

    Notes:
      - In PAIRWISE mode, only metrics that are computable from a,b,c,d are allowed:
        jaccard, dice, cosine, overlap, phi
      - In PAIRWISE mode, FDR uses Fisher+BH directly on a,b,c,d (no binary expansion).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_mode = is_pairwise_df(cohort_df)

    # ----------------------------
    # 1) Similarity matrix
    # ----------------------------
    if not pairwise_mode:
        # WIDE mode: your original approach
        abx_cols = get_wide_abx_cols(cohort_df)
        if not abx_cols:
            raise ValueError(f"No _Tested columns found for cohort {cohort_name} (WIDE mode).")

        sim_engine = SimilarityEngine(cohort_df, abx_cols)
        similarity_matrix = sim_engine.compute(metric)

    else:
        # PAIRWISE mode: compute directly from a,b,c,d
        work = cohort_df.copy()
        for c in ["a", "b", "c", "d"]:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)

        sim_vals = _pairwise_metric_values(work, metric)
        similarity_matrix = _pairwise_square_matrix(work, sim_vals, fill_diagonal=1.0)

        # labels for viz
        abx_cols = list(similarity_matrix.index)

    # ----------------------------
    # 2) Optional FDR pruning
    # ----------------------------
    if use_fdr:
        if not pairwise_mode:
            # original WIDE FDR pruning
            df_binary = (
                cohort_df[abx_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )

            pruner = EdgeSignificancePruner(
                df_binary=df_binary,
                antibiotic_cols=abx_cols,
                alpha=fdr_alpha,
                min_total=fdr_min_total,
                min_positive=fdr_min_positive,
                alternative=fdr_alternative,
            ).fit()

            allowed_pairs = pruner.get_significant_pairs()  # Set[(u, v)]

        else:
            # PAIRWISE FDR pruning directly on a,b,c,d
            # Use tau as gate for tests (same as your paper Stage2)
            allowed_pairs = pairwise_fdr_significant_pairs(
                cohort_df,
                tau_gate=tau,
                alpha=fdr_alpha,
                alternative=("greater" if fdr_alternative == "two-sided" else fdr_alternative),
                metric_for_gate=metric,
            )

        # Apply allowed_pairs mask to similarity_matrix
        nodes = list(similarity_matrix.index)
        mask = pd.DataFrame(False, index=nodes, columns=nodes)

        for u, v in allowed_pairs:
            if u in mask.index and v in mask.columns:
                mask.loc[u, v] = True
                mask.loc[v, u] = True

        for a0 in nodes:
            mask.loc[a0, a0] = True

        similarity_matrix = similarity_matrix.where(mask, 0.0)

    # ----------------------------
    # 3) Build graph and run Louvain
    # ----------------------------
    graph_builder = GraphBuilder(tau)
    G = graph_builder.build_graph(similarity_matrix)

    clusterer = LouvainClusterer(gamma)

    seeds = [base_seed + i for i in range(max(1, n_louvain_iterations))]
    partitions = clusterer.run_multiple(G, seeds)
    if not partitions:
        raise RuntimeError(f"Louvain failed for cohort {cohort_name}")

    # keep for future
    _ = partitions[0]

    # ----------------------------
    # 4) Safe file names
    # ----------------------------
    safe_name = cohort_name.replace(" ", "_")
    if suffix:
        safe_name = f"{safe_name}{suffix}"

    html_name = f"net_{safe_name}.html"
    png_name = f"net_{safe_name}.png"
    pdf_name = f"net_{safe_name}.pdf"
    gexf_name = f"net_{safe_name}.gexf"

    base_title = title_prefix or f"Network – {cohort_name}"
    # title = (
    #     f"{base_title} – {metric} (τ={tau}, γ={gamma})"
    #     + (" – FDR-pruned" if use_fdr else "")
    #     + (" – pairwise" if pairwise_mode else "")
    # )
    title = ""

    # ----------------------------
    # 5) Label + semantic colors (works in both)
    # ----------------------------
    abx_class_map = None
    antibiotic_class_map_path: str = "./datasets/antibiotic_class_grouping.json"
    if os.path.exists(antibiotic_class_map_path):
        try:
            with open(antibiotic_class_map_path, "r") as f:
                abx_class_map = json.load(f)
        except Exception:
            abx_class_map = None

    label_map = None
    if abx_class_map is not None:
        try:
            label_map = get_label(
                abx_cols,
                antibiotic_class_map=abx_class_map,
                format_type="abbr",
                enrich=True,
                include_class=False,
            )
        except Exception:
            label_map = None

    AWARE_PALETTE = {
        "Access":  "#56B4E9",
        "Watch":   "#E69F00",
        "Reserve": "#D55E00",
        "Unknown": "#999999",
    }
    from src.runners.Phases.Phase_I2 import build_who_map
    who_map = build_who_map()

    def _norm_aware(cls: str) -> str:
        s = str(cls).strip().lower()
        if s == "access":  return "Access"
        if s == "watch":   return "Watch"
        if s == "reserve": return "Reserve"
        return "Unknown"

    if label_map:
        sim_for_viz = similarity_matrix.rename(index=label_map, columns=label_map)
        aware_color_map = {}
        for raw_abx, cls in who_map.items():
            disp = label_map.get(raw_abx)
            if disp is None:
                continue
            aware_color_map[disp] = AWARE_PALETTE.get(_norm_aware(cls), AWARE_PALETTE["Unknown"])
    else:
        sim_for_viz = similarity_matrix
        aware_color_map = {
            raw_abx: AWARE_PALETTE.get(_norm_aware(cls), AWARE_PALETTE["Unknown"])
            for raw_abx, cls in who_map.items()
        }
    
    # ----------------------------
    # 6) Visualize & save
    # ----------------------------
    visualize_antibiotic_network(
        data_input=sim_for_viz,
        threshold=tau,
        community_gamma=gamma,
        output_dir=str(output_dir),
        output_html=html_name,
        output_image=png_name,
        output_pdf=pdf_name,
        gexf_path=gexf_name,
        title=title,
        remove_isolated=False,
        semantic_color_map=aware_color_map,
        node_color_mode="semantic",
    )
    
    return sim_for_viz, aware_color_map, title


def save_raw_and_pruned_networks(
    cohort_df: pd.DataFrame,
    cohort_name: str,
    output_dir: Path,
    metric: str = BEST_SIMILARITY_METRIC,
    tau: float = BEST_TAU,
    gamma: float = BEST_GAMMA,
    fdr_alpha: float = 0.05,
    fdr_min_total: int = 20,
    fdr_min_positive: int = 3,
    fdr_alternative: str = "two-sided",
    n_louvain_iterations: int = 100,
    base_seed: int = 100,
    title_prefix: Optional[str] = None,
) -> None:
    # RAW
    build_and_save_network_for_cohort(
        cohort_df=cohort_df,
        cohort_name=cohort_name,
        output_dir=output_dir,
        metric=metric,
        tau=tau,
        gamma=gamma,
        use_fdr=False,
        suffix="_raw",
        n_louvain_iterations=n_louvain_iterations,
        base_seed=base_seed,
        title_prefix=title_prefix,
    )

    # FDR-PRUNED
    build_and_save_network_for_cohort(
        cohort_df=cohort_df,
        cohort_name=cohort_name,
        output_dir=output_dir,
        metric=metric,
        tau=tau,
        gamma=gamma,
        use_fdr=True,
        fdr_alpha=fdr_alpha,
        fdr_min_total=fdr_min_total,       # used only in WIDE mode
        fdr_min_positive=fdr_min_positive, # used only in WIDE mode
        fdr_alternative=fdr_alternative,
        suffix="_FDR",
        n_louvain_iterations=n_louvain_iterations,
        base_seed=base_seed,
        title_prefix=title_prefix,
    )
