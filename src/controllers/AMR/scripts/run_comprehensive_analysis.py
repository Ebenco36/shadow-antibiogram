#!/usr/bin/env python3

import logging
from pathlib import Path

import pandas as pd  # kept in case you want to do quick ad-hoc checks
from src.controllers.AMR.use_cases.run import run_two_key_use_cases
from src.controllers.AMR.experiments.temporal_analysis import run_main_temporal
from src.controllers.AMR.config.experiment_config import ExperimentConfig
from src.controllers.AMR.experiments.grid_search_runner import GridSearchRunner
from src.controllers.AMR.visualization.visualization_manager import VisualizationManager


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
        alternative = getattr(viz_cfg, "fdr_alternative", "two-sided")

        logger.info("Use FDR edge pruning: %s", use_fdr)
        logger.info("  FDR alpha:          %.4f", alpha)
        logger.info("  FDR min_total:      %d", min_total)
        logger.info("  FDR min_positive:   %d", min_positive)
        logger.info("  FDR alternative:    %s", alternative)
    else:
        logger.info("No explicit visualization config attached to ExperimentConfig.")

    logger.info("=====================================")


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
    
    # logger.info("++++++++++++++++++++++++ RUNNING TEMPORAL  ++++++++++++++++++++++++++")
    # run_main_temporal(df=new_df, base_dir = "./outputs/temporal_analysis")
    # logger.info("+++++++++++++++++++++ DONE RUNNING TEMPORAL  ++++++++++++++++++++++++")

    ##########################################################################
    ########### RUN THE CODE FOR CONTINUOUS PARTICIPATION USE CASE ###########
    ##########################################################################
    
    # from src.controllers.AMR.use_cases.helper import filter_continuous_organisations
    # df = new_df
    # if "NumberOrganisation" in df.columns.to_list():
    #     res = filter_continuous_organisations(
    #         df,
    #         org_col="NumberOrganisation",
    #         year_col="Year",       # will use if present
    #         date_col="Date",       # used only if Year missing
    #         min_year=2019,
    #         max_year=2023,
    #         verbose=True,
    #     )

    #     df_cont = res.df_continuous
    #     orgs = res.continuous_orgs

    #     print(f"Continuous organisations: {len(orgs)}")
    #     print(f"Isolates retained: {len(df_cont):,}")
    #     continuous_participation_percentage = len(df_cont)/len(df) * 100
    #     print(f"Percentage of isolates retained after accounting for continuous participation: {continuous_participation_percentage:.2f}%")

    #     run_two_key_use_cases(df_cont, Path("./outputs/use_cases_continuous"))
        # run_main_temporal(df=df_cont, base_dir="./outputs/temporal_analysis_for_continuous")

if __name__ == "__main__":
    main()
