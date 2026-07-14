#!/usr/bin/env python3

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.controllers.AMR.use_cases.run import run_two_key_use_cases
from src.controllers.AMR.config.experiment_config import ExperimentConfig
from src.controllers.AMR.experiments.grid_search_runner import GridSearchRunner
from src.controllers.AMR.visualization.visualization_manager import VisualizationManager
from src.controllers.AMR.data.pairwise_aggregate import (
    is_pairwise_df,
    parquet_files,
    summarize_pairwise_aggregate,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def log_config_summary(config: ExperimentConfig, logger: logging.Logger):
    """
    Small helper to log the most relevant parts of the ExperimentConfig,
    including the FDR / visualization settings if present.
    """
    logger.info("=== Experiment configuration summary ===")
    logger.info("Data path:        %s", config.data.data_path)
    logger.info("Genera:           %s", ", ".join(config.data.genera))
    logger.info("Materials:        %s", ", ".join(config.data.materials))
    logger.info(
        "Antibiotic cols:  %s",
        "ALL loader-tested" if not config.data.antibiotic_columns
        else ", ".join(config.data.antibiotic_columns),
    )
    logger.info(
        "Tau range:        %s",
        ", ".join(f"{t:.2f}" for t in config.parameters.tau_range),
    )
    logger.info(
        "Gamma range:      %s",
        ", ".join(f"{g:.2f}" for g in config.parameters.gamma_range),
    )
    logger.info(
        "Similarity metrics: %s",
        ", ".join(config.parameters.similarity_metrics),
    )
    logger.info("Random seed:      %d", config.random_seed)

    # Evaluation pieces
    eval_cfg = config.evaluation
    logger.info("Compute ICS:          %s", eval_cfg.compute_ics)
    logger.info("Compute stability:    %s", eval_cfg.compute_stability)
    logger.info("Compute silhouette:   %s", eval_cfg.compute_silhouette)
    if eval_cfg.label_hierarchy_weights:
        logger.info(
            "Label hierarchy weights: %s",
            eval_cfg.label_hierarchy_weights,
        )

    # Visualization / FDR summary (if config.visualization exists)
    viz_cfg = getattr(config, "visualization", None)
    if viz_cfg is not None:
        logger.info("=== Visualization / FDR settings ===")
        metric_colors = getattr(viz_cfg, "metric_colors", None)
        if metric_colors:
            logger.info("Custom metric colours: %s", metric_colors)

        use_fdr = getattr(viz_cfg, "use_fdr_edge_pruning", False)
        alpha = getattr(viz_cfg, "fdr_alpha", 0.05)
        min_total = getattr(viz_cfg, "fdr_min_total", 20)
        min_positive = getattr(viz_cfg, "fdr_min_positive", 3)
        alternative = getattr(viz_cfg, "fdr_alternative", "greater")

        logger.info("Use FDR edge pruning: %s", use_fdr)
        logger.info("  FDR alpha:          %.4f", alpha)
        logger.info("  FDR min_total:      %d", min_total)
        logger.info("  FDR min_positive:   %d", min_positive)
        logger.info("  FDR alternative:    %s", alternative)
    else:
        logger.info("No explicit visualization config attached to ExperimentConfig.")

    logger.info("=====================================")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def write_run_manifest(
    *,
    config: ExperimentConfig,
    data_loader,
    output_dir: Path,
    status: str,
    outputs: dict | None = None,
) -> Path:
    data_path = Path(config.data.data_path)
    data_files = parquet_files(data_path)
    data_hashes = {str(path): _sha256_file(path) for path in data_files}

    data_summary = None
    if is_pairwise_df(data_loader.df):
        data_summary = summarize_pairwise_aggregate(data_loader.df).as_dict()

    manifest = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": {
            "data_path": str(config.data.data_path),
            "configured_genera": list(config.data.genera),
            "configured_materials": list(config.data.materials),
            "strict_expected_cohorts": config.data.strict_expected_cohorts,
            "similarity_metrics": list(config.parameters.similarity_metrics),
            "tau_range": list(config.parameters.tau_range),
            "gamma_range": list(config.parameters.gamma_range),
            "n_iterations": config.parameters.n_iterations,
            "random_seed": config.random_seed,
            "output_dir": str(config.output_dir),
            "fdr": {
                "use_fdr_edge_pruning": bool(getattr(config.visualization, "use_fdr_edge_pruning", False)),
                "alpha": float(getattr(config.visualization, "fdr_alpha", 0.05)),
                "min_total": int(getattr(config.visualization, "fdr_min_total", 20)),
                "min_positive": int(getattr(config.visualization, "fdr_min_positive", 3)),
                "alternative": str(getattr(config.visualization, "fdr_alternative", "greater")),
            },
        },
        "data_files": data_hashes,
        "data_summary": data_summary,
        "outputs": outputs or {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # 1. Load default config (adapt paths + antibiotic_columns in ExperimentConfig.default)
    config = ExperimentConfig.default()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", config.output_dir)

    # Log a short summary of what will be run (including FDR settings)
    log_config_summary(config, logger)

    # 2. Run full τ–γ–metric grid search
    logger.info("Starting grid search over tau / gamma / metric...")
    runner = GridSearchRunner(config)
    manifest_path = write_run_manifest(
        config=config,
        data_loader=runner.data_loader,
        output_dir=config.output_dir,
        status="started",
    )
    logger.info("Wrote run manifest to %s", manifest_path)
    results = runner.run()
    logger.info("Grid search complete.")

    # 3. Save single-run and aggregated results
    single_df = results.to_single_run_df()
    agg_df = results.aggregated_results

    single_path = config.output_dir / "single_run_results.csv"
    agg_path = config.output_dir / "aggregated_results.csv"

    single_df.to_csv(single_path, index=False)
    logger.info(
        "Saved single-run results to %s (n=%d rows)",
        single_path,
        len(single_df),
    )

    if agg_df is not None and not agg_df.empty:
        # For MultiIndex columns we keep index in CSV to preserve structure
        agg_df.to_csv(agg_path)
        logger.info(
            "Saved aggregated results to %s (n=%d rows)",
            agg_path,
            len(agg_df),
        )
    else:
        logger.warning(
            "Aggregated results are empty or None. Skipping aggregated_results.csv saving."
        )

    manifest_path = write_run_manifest(
        config=config,
        data_loader=runner.data_loader,
        output_dir=config.output_dir,
        status="grid_search_complete",
        outputs={
            "single_run_results": str(single_path),
            "aggregated_results": str(agg_path) if agg_df is not None and not agg_df.empty else None,
        },
    )
    logger.info("Updated run manifest at %s", manifest_path)

    # 4. Generate publication-ready visualizations
    logger.info("Creating visualization dashboard (Plotly + networks)...")
    viz_manager = VisualizationManager(config, results)
    viz_manager.create_comprehensive_dashboard(config.output_dir)
    logger.info("Visualization dashboard created under %s", config.output_dir)

    logger.info("Analysis complete.")
    
    logger.info("++++++++++++++++++++++++ RUNNING USE CASES ++++++++++++++++++++++++++")
    new_df = runner.data_loader.get_combined()
    run_two_key_use_cases(new_df, Path("./outputs/use_cases"))
    logger.info("++++++++++++++++++++++++ DONE RUNNING USE CASES +++++++++++++++++++++")

    manifest_path = write_run_manifest(
        config=config,
        data_loader=runner.data_loader,
        output_dir=config.output_dir,
        status="complete",
        outputs={
            "single_run_results": str(single_path),
            "aggregated_results": str(agg_path) if agg_df is not None and not agg_df.empty else None,
            "dashboard_dir": str(config.output_dir),
            "use_cases_dir": "./outputs/use_cases",
        },
    )
    logger.info("Final run manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
