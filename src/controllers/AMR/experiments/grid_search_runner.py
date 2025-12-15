# src/controllers/AMR/experiments/grid_search_runner.py

import logging
from itertools import product
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

from src.controllers.AMR.config.experiment_config import ExperimentConfig
from src.controllers.DataLoader import DataLoader
from src.controllers.AMR.data.preprocessor import DataPreprocessor
from src.controllers.AMR.similarity.engine import SimilarityEngine
from src.controllers.AMR.clustering.graph_builder import GraphBuilder
from src.controllers.AMR.clustering.louvain_clusterer import LouvainClusterer
from src.controllers.AMR.evaluation.label_manager import LabelManager
from src.controllers.AMR.evaluation.external_evaluator import ExternalEvaluator
from src.controllers.AMR.evaluation.internal_evaluator import InternalEvaluator
from src.controllers.AMR.evaluation.stability_analyzer import StabilityAnalyzer
from src.controllers.AMR.evaluation.ics_evaluator import ICSEvaluator
from src.controllers.AMR.experiments.results import ResultCollection, SingleRunResult


class GridSearchRunner:
    """
    Run a comprehensive τ–γ grid search for Louvain clustering across multiple
    similarity metrics, *for each (PathogenGenus, Material) combination* specified
    in ExperimentConfig.data.

    For each pair (genus, material), we:
      - Filter the dataframe to that subset
      - Determine antibiotic columns present in that subset
      - Compute similarity matrices (jaccard, dice, cosine, phi)
      - Run Louvain for all τ–γ combinations and random seeds
      - Evaluate external (ARI/NMI vs ground truths), ICS, and stability

    All runs are collected into a single ResultCollection. Aggregation currently
    groups by (metric, tau, gamma); if you later want per-(genus, material)
    aggregates, we can extend SingleRunResult and the grouping logic.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = ResultCollection()

        # global RNG init (for reproducibility of any numpy-based randomness)
        np.random.seed(self.config.random_seed)

        # data-related
        self.data_loader = DataLoader(str(config.data.data_path))
        self.preprocessor = DataPreprocessor(
            genera=config.data.genera,
            materials=config.data.materials,
            antibiotic_columns=config.data.antibiotic_columns,
        )

        # evaluation-related
        self.label_manager = LabelManager(self.config)
        self.external_eval = ExternalEvaluator(self.label_manager.get_label_maps())
        self.stability_analyzer = StabilityAnalyzer()

    # ------------------------------------------------------------------ #
    # Data preparation: per (genus, material) subset
    # ------------------------------------------------------------------ #

    def _prepare_subset(
        self,
        df_raw,
        genus: Optional[str],
        material: Optional[str],
    ) -> Tuple[Optional[object], List[str]]:
        """
        Given the full combined dataframe from DataLoader, filter it to a single
        (PathogenGenus, TextMaterialgroupRkiL0) combination and determine
        antibiotic columns present in that subset.

        Returns
        -------
        df_subset : pd.DataFrame or None
            Filtered dataframe. None if empty after filtering.
        antibiotic_columns : list of str
            Antibiotic columns to use for similarity computation (non-empty).
        """
        # Start with all rows
        mask = np.ones(len(df_raw), dtype=bool)

        # Filter by genus if available
        if genus is not None and "PathogenGenus" in df_raw.columns:
            mask &= (df_raw["PathogenGenus"] == genus).to_numpy()

        # Filter by material if available
        if material is not None and "TextMaterialgroupRkiL0" in df_raw.columns:
            mask &= (df_raw["TextMaterialgroupRkiL0"] == material).to_numpy()

        df_subset = df_raw.loc[mask].copy()

        if df_subset.empty:
            return None, []

        # Decide antibiotic columns:
        # 1) If config explicitly provides some, intersect with df_subset.
        # 2) Else, use DataLoader.abx_tested_cols intersected with df_subset.
        if self.config.data.antibiotic_columns:
            antibiotic_columns = [
                c for c in self.config.data.antibiotic_columns if c in df_subset.columns
            ]
        else:
            antibiotic_columns = [
                c for c in self.data_loader.abx_tested_cols if c in df_subset.columns
            ]

        if not antibiotic_columns:
            # No antibiotics for this subset → skip
            return None, []

        return df_subset, antibiotic_columns

    # ------------------------------------------------------------------ #
    # Hierarchical label score
    # ------------------------------------------------------------------ #

    def _compute_hierarchical_score(self, external_scores: Dict[str, float]) -> float:
        """
        Weighted average of NMI over label levels using weights from config.
        external_scores keys like: 'ARI_broad', 'NMI_broad', ...
        We define hierarchical_score based on NMI only.
        """
        weights = self.config.evaluation.label_hierarchy_weights
        if not weights:
            return np.nan

        total = 0.0
        norm = 0.0
        for level, w in weights.items():
            key = f"NMI_{level}"
            if key in external_scores:
                total += w * external_scores[key]
                norm += w
        return total / norm if norm > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Main runner
    # ------------------------------------------------------------------ #

    def run(self) -> ResultCollection:
        """
        Run the full grid search across:
          - all specified genera
          - all specified materials
          - all similarity metrics
          - all τ–γ combinations
          - all random seeds (for stability)

        Results (across all subsets) are stored in self.results.
        """
        # Load the full combined dataframe once
        df_raw = self.data_loader.get_combined()

        genera = self.config.data.genera or [None]
        materials = self.config.data.materials or [None]

        n_metrics = len(self.config.parameters.similarity_metrics)
        n_tau = len(self.config.parameters.tau_range)
        n_gamma = len(self.config.parameters.gamma_range)
        n_genera = len(genera)
        n_materials = len(materials)

        total_combinations = n_metrics * n_tau * n_gamma * n_genera * n_materials

        pbar = tqdm(total=total_combinations, desc="Parameter combinations")

        # loop over all (genus, material) subsets
        for genus in genera:
            for material in materials:
                subset_label = f"{genus or 'ALL'} | {material or 'ALL'}"
                self.logger.info("Processing subset: %s", subset_label)

                df_subset, antibiotic_columns = self._prepare_subset(
                    df_raw, genus, material
                )

                # skip empty or invalid subsets
                if df_subset is None or not antibiotic_columns:
                    self.logger.info("Skipping subset %s (no data or no antibiotics).", subset_label)
                    # still advance pbar for this subset over all τ–γ–metric combos
                    # to keep progress roughly consistent
                    pbar.update(n_metrics * n_tau * n_gamma)
                    continue

                # similarity engine for this subset
                sim_engine = SimilarityEngine(df_subset, antibiotic_columns)

                # loop over similarity metrics
                for metric in self.config.parameters.similarity_metrics:
                    self.logger.info(
                        "Processing metric=%s for subset=%s", metric, subset_label
                    )
                    similarity_matrix = sim_engine.compute(metric)
                    nodes = list(similarity_matrix.index)

                    internal_eval = InternalEvaluator(similarity_matrix)
                    ics_evaluator = (
                        ICSEvaluator(similarity_matrix)
                        if self.config.evaluation.compute_ics
                        else None
                    )

                    # tau–gamma loop
                    for tau, gamma in product(
                        self.config.parameters.tau_range,
                        self.config.parameters.gamma_range,
                    ):
                        graph_builder = GraphBuilder(tau)
                        G = graph_builder.build_graph(similarity_matrix)
                        clusterer = LouvainClusterer(gamma)

                        # multiple seeds for stability
                        seed_list = list(
                            range(
                                self.config.random_seed,
                                self.config.random_seed
                                + self.config.parameters.n_iterations,
                            )
                        )
                        partitions = clusterer.run_multiple(G, seed_list)

                        # stability (optional toggle)
                        stability = None
                        if self.config.evaluation.compute_stability:
                            stability = self.stability_analyzer.compute_stability(
                                partitions, nodes
                            )

                        # evaluate each seed
                        for seed, partition in zip(seed_list, partitions):
                            external_scores = self.external_eval.evaluate(
                                partition, nodes
                            )

                            silhouette = None
                            if self.config.evaluation.compute_silhouette:
                                # print("here we are: ", list(partition.values()))
                                silhouette = internal_eval.compute_silhouette(
                                    partition
                                )

                            ics_score = None
                            if ics_evaluator is not None:
                                ics_scores = ics_evaluator.evaluate(partition)
                                ics_score = ics_scores["ICS"]

                            hierarchical_score = self._compute_hierarchical_score(
                                external_scores
                            )

                            result = SingleRunResult(
                                metric=metric,
                                tau=tau,
                                gamma=gamma,
                                seed=seed,
                                n_clusters=len(set(partition.values())),
                                silhouette=silhouette,
                                ics_score=ics_score,
                                stability=stability,
                                external_scores=external_scores,
                                hierarchical_score=hierarchical_score,
                                genus=genus,
                                material=material,
                            )
                            self.results.add_single_run(result)

                        pbar.update(1)

        pbar.close()
        self.results.compute_aggregates()
        return self.results
