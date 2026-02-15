# src/controllers/AMR/experiments/results.py

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class SingleRunResult:
    """
    Results from a single (genus, material, metric, tau, gamma, seed) combination.
    """
    metric: str
    tau: float
    gamma: float
    seed: int
    n_clusters: int
    silhouette: Optional[float] = None
    ics_score: Optional[float] = None
    stability: Optional[float] = None  # same across seeds for (subset, metric, tau, gamma)
    external_scores: Dict[str, float] = field(default_factory=dict)
    hierarchical_score: Optional[float] = None

    # NEW: which dataset this run belongs to
    genus: Optional[str] = None
    material: Optional[str] = None

    def to_dict(self) -> Dict:
        """
        Flatten the dataclass into a dict, expanding external_scores
        into columns named external_<key>.
        """
        base = asdict(self)
        external = base.pop("external_scores", {})
        base.update({f"external_{k}": v for k, v in external.items()})
        return base


class ResultCollection:
    """
    Collects SingleRunResult objects and provides:
      - conversion to a single-run DataFrame
      - aggregated statistics (mean/std) per (genus, material, metric, tau, gamma)
    """

    def __init__(self):
        self.single_runs: List[SingleRunResult] = []
        self.aggregated_results: Optional[pd.DataFrame] = None

    def add_single_run(self, result: SingleRunResult):
        self.single_runs.append(result)

    def to_single_run_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.single_runs])

    def compute_aggregates(self) -> pd.DataFrame:
        """
        Compute mean and std across seeds for each
        (genus, material, metric, tau, gamma) combination.
        """
        df_single = self.to_single_run_df()
        if df_single.empty:
            self.aggregated_results = pd.DataFrame()
            return self.aggregated_results

        # build aggregation config
        agg_config = {
            "silhouette": ["mean", "std"],
            "ics_score": ["mean", "std"],
            "n_clusters": ["mean", "std"],
            "stability": ["mean"],  # same across seeds, but we keep it
            "hierarchical_score": ["mean", "std"],
        }

        # add all external_* columns
        for col in df_single.columns:
            if col.startswith("external_"):
                agg_config[col] = ["mean", "std"]

        group_cols = ["genus", "material", "metric", "tau", "gamma"]

        self.aggregated_results = (
            df_single.groupby(group_cols, as_index=True)
            .agg(agg_config)
        )

        return self.aggregated_results
