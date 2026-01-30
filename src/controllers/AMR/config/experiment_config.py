# src/controllers/AMR/config/experiment_config.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# ------------------------------------------------------------------ #
# Data configuration
# ------------------------------------------------------------------ #

@dataclass
class DataConfig:
    """
    Configuration for loading and filtering the AMR dataset.
    """
    data_path: Path

    # Default genera / materials as in your previous code
    genera: List[str] = field(
        default_factory=lambda: [
            "Escherichia",
            "Staphylococcus",
            "Klebsiella",
            "Pseudomonas",
            "Proteus",
            "Streptococcus",
        ]
    )
    materials: List[str] = field(
        default_factory=lambda: [
            "Blood Culture",
            "Urine",
        ]
    )

    # If empty, we use all "tested" antibiotics from the DataLoader
    antibiotic_columns: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Parameter grid configuration
# ------------------------------------------------------------------ #

@dataclass
class ParameterConfig:
    """
    Configuration for the τ / γ / metric grid search.
    """

    # τ (similarity threshold) range
    tau_range: List[float] = field(
        default_factory=lambda: [x * 0.1 for x in range(1, 5)]
        # -> [0.1, 0.2, 0.3, 0.4]
    )

    # γ (Louvain resolution) range
    gamma_range: List[float] = field(
        default_factory=lambda: [0.5 + 0.25 * x for x in range(3)]
        # -> [0.5, 0.75, 1.0]
    )

    # how many random seeds / iterations when running Louvain
    n_iterations: int = 10

    # similarity metrics to evaluate
    similarity_metrics: List[str] = field(
        default_factory=lambda: ["jaccard", "dice", "cosine", "phi"]
    )


# ------------------------------------------------------------------ #
# Evaluation configuration
# ------------------------------------------------------------------ #

@dataclass
class EvaluationConfig:
    """
    Configuration for computing external and internal evaluation metrics.
    """

    # Paths to JSON files with ground-truth label mappings
    # keys: "broad", "fine", "who"
    ground_truth_paths: Dict[str, Path] = field(default_factory=dict)

    compute_ics: bool = True
    compute_stability: bool = True
    compute_silhouette: bool = True

    # Optional: how to weight label levels in a single hierarchical score
    # e.g. {"fine": 0.4, "broad": 0.3, "who": 0.3}
    label_hierarchy_weights: Optional[Dict[str, float]] = None


# ------------------------------------------------------------------ #
# Visualization configuration (FDR + styling)
# ------------------------------------------------------------------ #

@dataclass
class VisualizationConfig:
    """
    Configuration for visualization styling and edge significance pruning (FDR).
    """

    # Optional custom colours per similarity metric.
    # Example:
    #   {"cosine": "#1f77b4", "dice": "#ff7f0e", "jaccard": "#2ca02c", "phi": "#d62728"}
    metric_colors: Dict[str, str] = field(default_factory=dict)

    # --- FDR-based edge pruning (for networks) ---
    # If True, you will apply Fisher's exact test + FDR to prune edges
    # before building / visualizing networks (via EdgeSignificancePruner).
    use_fdr_edge_pruning: bool = False

    # FDR control level (e.g. Benjamini–Hochberg)
    fdr_alpha: float = 0.05

    # Only test edges where we have at least this many total isolates
    fdr_min_total: int = 20

    # Only test edges where the “positive” count (e.g. co-resistance)
    # is at least this threshold
    fdr_min_positive: int = 3

    # Alternative hypothesis for Fisher’s test: "two-sided", "greater", or "less"
    fdr_alternative: str = "two-sided"


# ------------------------------------------------------------------ #
# Top-level experiment configuration
# ------------------------------------------------------------------ #

@dataclass
class ExperimentConfig:
    """
    Top-level configuration passed around the AMR analysis pipeline.
    """

    data: DataConfig
    parameters: ParameterConfig
    evaluation: EvaluationConfig

    # Where to write CSVs, plots, networks, etc.
    output_dir: Path = Path("./outputs/sensitivity_runs")

    # For global reproducibility (Louvain, numpy, etc.)
    random_seed: int = 100

    # Optional visualization-related settings (FDR + colours, etc.)
    visualization: Optional[VisualizationConfig] = None

    # -------------------------------------------------------------- #
    # Convenience constructor with sensible defaults
    # -------------------------------------------------------------- #

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """
        Build a default configuration with dataset paths and label JSONs
        matching your current project structure.
        """
        data_cfg = DataConfig(
            data_path=Path("./datasets/WHO_Aware_data"),
            # If you want to explicitly restrict antibiotics, list them here.
            # Otherwise leave empty and DataLoader.abx_tested_cols will be used.
            antibiotic_columns=[],
        )

        param_cfg = ParameterConfig()

        eval_cfg = EvaluationConfig(
            ground_truth_paths={
                "broad": Path("datasets/antibiotic_broad_class_grouping.json"),
                "fine": Path("datasets/antibiotic_class_grouping.json"),
                "who": Path("datasets/antibiotic_class.json"),
            },
            label_hierarchy_weights={"fine": 0.4, "broad": 0.3, "who": 0.3},
        )

        # You can fill metric_colors here if you want fixed colours in all plots
        viz_cfg = VisualizationConfig(
            metric_colors={
                # Example (uncomment / edit as needed):
                # "jaccard": "#1f77b4",
                # "dice":    "#ff7f0e",
                # "cosine":  "#2ca02c",
                # "phi":     "#d62728",
            },
            use_fdr_edge_pruning=True,  # switch to True when you wire in EdgeSignificancePruner
            fdr_alpha=0.05,
            fdr_min_total=20,
            fdr_min_positive=3,
            fdr_alternative="two-sided",
        )

        return cls(
            data=data_cfg,
            parameters=param_cfg,
            evaluation=eval_cfg,
            output_dir=Path("./outputs/sensitivity_runs"),
            random_seed=100,
            visualization=viz_cfg,
        )
