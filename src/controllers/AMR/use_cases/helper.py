import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

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
    n_louvain_iterations: int = 100,   # match ParameterConfig.n_iterations default
    base_seed: int = 100,              # match ExperimentConfig.random_seed default
    title_prefix: Optional[str] = None,
) -> None:
    """
    Build a similarity-based network for a cohort and save HTML/PNG/GEXF.

    - Uses global best metric/params by default (Jaccard, tau, gamma).
    - If use_fdr=True, prunes edges using EdgeSignificancePruner before
      thresholding with tau.
    - Runs Louvain multiple times (n_louvain_iterations) with different seeds.
    - Keeps get_label-based antibiotic relabeling for visualization.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Antibiotic columns
    abx_cols = [c for c in cohort_df.columns if c.endswith("_Tested")]
    if not abx_cols:
        raise ValueError(f"No _Tested columns found for cohort {cohort_name}")

    # 2) Similarity matrix (e.g. Jaccard)
    sim_engine = SimilarityEngine(cohort_df, abx_cols)
    similarity_matrix = sim_engine.compute(metric)

    # 3) Optional FDR edge pruning (work on the original antibiotic names)
    if use_fdr:
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

        # Build mask: only allow significant pairs + diagonal
        abx_list = list(similarity_matrix.index)
        mask = pd.DataFrame(False, index=abx_list, columns=abx_list)

        for u, v in allowed_pairs:
            if u in mask.index and v in mask.columns:
                mask.loc[u, v] = True
                mask.loc[v, u] = True

        for a in abx_list:
            mask.loc[a, a] = True

        similarity_matrix = similarity_matrix.where(mask, 0.0)

    # 4) Build graph and run Louvain (multiple seeds)
    graph_builder = GraphBuilder(tau)
    G = graph_builder.build_graph(similarity_matrix)

    clusterer = LouvainClusterer(gamma)

    # seeds: base_seed, base_seed+1, ..., base_seed+n_louvain_iterations-1
    if n_louvain_iterations <= 0:
        seeds = [base_seed]
    else:
        seeds = [base_seed + i for i in range(n_louvain_iterations)]

    partitions = clusterer.run_multiple(G, seeds)
    if not partitions:
        raise RuntimeError(f"Louvain failed for cohort {cohort_name}")

    # NOTE: run_multiple currently returns partitions only; if you later
    # extend it to return modularities, you can select the best here.
    partition = partitions[0]  # currently unused but kept for future stats

    # 5) Safe file base name
    safe_name = cohort_name.replace(" ", "_")
    if suffix:
        safe_name = f"{safe_name}{suffix}"

    html_name = f"net_{safe_name}.html"
    png_name = f"net_{safe_name}.png"
    pdf_name = f"net_{safe_name}.pdf"
    gexf_name = f"net_{safe_name}.gexf"

    base_title = title_prefix or f"Network – {cohort_name}"
    title = (
        f"{base_title} – {metric} (τ={tau}, γ={gamma})"
        + (" – FDR-pruned" if use_fdr else "")
    )

    # 6) Optional: load antibiotic class map + get_label
    abx_class_map = None
    antibiotic_class_map_path: str = "./datasets/antibiotic_class_grouping.json"
    if os.path.exists(antibiotic_class_map_path):
        try:
            with open(antibiotic_class_map_path, "r") as f:
                abx_class_map = json.load(f)
        except Exception:
            abx_class_map = None  # fall back to raw names if broken file

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
            # If label creation fails, just keep original antibiotic names
            label_map = None

    if label_map:
        sim_for_viz = similarity_matrix.rename(index=label_map, columns=label_map)
    else:
        sim_for_viz = similarity_matrix

    # 7) Visualize & save
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
    )


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
    """
    Convenience wrapper: for a given cohort, save both:
      - raw network
      - FDR-pruned network
    using the same Louvain multi-run setup.
    """

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
        fdr_min_total=fdr_min_total,
        fdr_min_positive=fdr_min_positive,
        fdr_alternative=fdr_alternative,
        suffix="_FDR",
        n_louvain_iterations=n_louvain_iterations,
        base_seed=base_seed,
        title_prefix=title_prefix,
    )
