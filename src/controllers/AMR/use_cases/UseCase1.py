from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# === project imports ==========================================================
from src.controllers.AMR.use_cases.helper import build_and_save_network_for_cohort, save_raw_and_pruned_networks
from src.controllers.cohort_analyzer.CohortSelection import (
    ProductionCohortGenerator,
    CohortConfig,
)
from pathlib import Path
# =============================================================================
# 2. LOGGING
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file + stdout for this pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"

    logger = logging.getLogger("UseCase1Temporal")
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers if re-imported
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# =============================================================================
# 4. TEMPORAL COHORT CREATION (E. coli; urine / blood)
# =============================================================================

@dataclass
class SpecimenContext:
    """Configuration for each specimen type in this use case."""
    key: str
    specimens: List[str]
    description: str
    min_sample_size: int = 1000  # can tune per specimen


class TemporalCohortCreator:
    """
    Creates E. coli temporal cohorts (per year × specimen).

    Uses your ProductionCohortGenerator to enforce all the filtering logic
    and metadata handling you defined earlier.
    """

    def __init__(
        self,
        generator: ProductionCohortGenerator,
        logger: Optional[logging.Logger] = None,
        pathogen_genus: str = "Escherichia",
    ):
        self.generator = generator
        self.logger = logger or logging.getLogger("UseCase1Temporal")
        self.pathogen_genus = pathogen_genus

        self.specimen_contexts: Dict[str, SpecimenContext] = {
            "urine": SpecimenContext(
                key="urine",
                specimens=["Urine"],
                description="Community-acquired UTI context",
                min_sample_size=1000,
            ),
            "blood": SpecimenContext(
                key="blood",
                specimens=["Blood Culture"],
                description="Bloodstream infection context",
                min_sample_size=1000,
            ),
        }

        self.temporal_cohorts: Dict[str, pd.DataFrame] = {}

    def _get_available_years(self) -> List[int]:
        if "Year" not in self.generator.df.columns:
            raise ValueError("Column 'Year' not found in dataframe.")
        years = (
            pd.to_numeric(self.generator.df["Year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        print(years)
        return sorted(years)

    def create_all_temporal_cohorts(self) -> Dict[str, pd.DataFrame]:
        """
        Build E. coli cohorts for each year × specimen where N ≥ min_sample_size.
        Cohort name template:
            temporal_ecoli_{specimen_key}_{year}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("USE CASE 1 – STEP 1: CREATING TEMPORAL COHORTS")
        self.logger.info("=" * 80)

        available_years = self._get_available_years()
        self.logger.info(f"Available years: {available_years}")
        for year in available_years:
            for specimen_key, context in self.specimen_contexts.items():
                # Pre-check sample size in the *raw* df to avoid calling generator unnecessarily
                year_mask = (
                    pd.to_numeric(self.generator.df["Year"], errors="coerce")
                    .astype("Int64")
                    .eq(year)
                )
                genus_mask = self.generator.df["PathogenGenus"].eq(self.pathogen_genus)
                specimen_mask = self.generator.df["TextMaterialgroupRkiL0"].isin(
                    context.specimens
                )

                sample_size = int((year_mask & genus_mask & specimen_mask).sum())
                self.logger.info(
                    f"DEBUG 2021: specimen_key={specimen_key} specimens={context.specimens} "
                    f"genus={self.pathogen_genus} sample_size={sample_size} min_required={context.min_sample_size}"
                )
                if sample_size < context.min_sample_size:
                    self.logger.info(
                        f"  ✗ Skipping {year} {specimen_key}: N={sample_size} "
                        f"< {context.min_sample_size}"
                    )
                    continue

                # Build CohortConfig to let ProductionCohortGenerator do the heavy lifting
                config = CohortConfig(
                    name=f"temporal_ecoli_{specimen_key}_{year}",
                    pathogen_genus=[self.pathogen_genus],
                    specimens=context.specimens,
                    isolate_group=None,  # or ['CSY'] if you want 'Erstisolat'
                    years=[str(year)],
                    min_sample_size=context.min_sample_size,
                    remove_untested_abx=True,
                    description=f"{self.pathogen_genus} in {specimen_key} ({year})",
                )

                cohort = self.generator.create_cohort(config)

                if cohort is not None:
                    cohort_name = config.name
                    self.temporal_cohorts[cohort_name] = cohort
                    self.logger.info(
                        f"  ✓ Created: {year} {specimen_key} N={len(cohort)} "
                        f"({context.description})"
                    )
                else:
                    self.logger.warning(
                        f"  ✗ Generator returned None for {year} {specimen_key}"
                    )

        self.logger.info(
            f"\n✓ Created {len(self.temporal_cohorts)} temporal cohorts "
            f"for E. coli ({', '.join(self.specimen_contexts.keys())})"
        )
        return self.temporal_cohorts

    def get_cohort_summary(self, data_summary_path: Optional[str] = None) -> pd.DataFrame:
        """Generate summary table for created temporal cohorts."""
        rows = []
        for name, cohort in self.temporal_cohorts.items():
            parts = name.split("_")
            # name template: temporal_ecoli_{specimen}_{year}
            specimen = parts[2]
            year = int(parts[3])

            rows.append(
                {
                    "cohort_name": name,
                    "specimen_type": specimen,
                    "year": year,
                    "n_isolates": len(cohort),
                    "n_features": cohort.shape[1],
                }
            )

        df = pd.DataFrame(rows).sort_values(["specimen_type", "year"])

        if data_summary_path:
            Path(data_summary_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_summary_path, index=False)
            self.logger.info(f"✓ Saved temporal cohort summary → {data_summary_path}")

        return df

def debug_year_presence(df, label):
    print(f"\n--- {label} ---")
    print("years:", sorted(df["Year"].dropna().unique().tolist()))
    print("counts by year:")
    print(df["Year"].value_counts(dropna=False).sort_index())


def run_temporal_use_case(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    UC1: For each temporal cohort (E. coli × specimen × year),
    build both:
      - raw Jaccard (τ=0.3, γ=1.0) network
      - FDR-pruned Jaccard (τ=0.3, γ=1.0) network
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Base generator
    generator = ProductionCohortGenerator(df)
    
    debug_year_presence(df, "RAW INPUT")
    df_prod = generator.df
    debug_year_presence(df_prod, "AFTER ProductionCohortGenerator")

    print("2021 raw matching counts:")
    print(
        df_prod.assign(YearN=pd.to_numeric(df_prod["Year"], errors="coerce").astype("Int64"))
            .query("YearN == 2021")
            .groupby(["PathogenGenus","TextMaterialgroupRkiL0"]) 
            .size()
            .sort_values(ascending=False)
            .head(30)
    )
    # 2) Temporal cohort creator
    logger = setup_logging(output_dir / "temporal")
    temporal_creator = TemporalCohortCreator(generator, logger=logger)

    # 3) Build cohorts
    temporal_cohorts = temporal_creator.create_all_temporal_cohorts()
    temporal_creator.get_cohort_summary(
        data_summary_path=output_dir / "temporal" / "temporal_cohort_summary.csv"
    )

    # 4) Build networks (raw + FDR) for each cohort
    networks_dir = output_dir / "temporal_networks"
    for cohort_name, cohort_df in temporal_cohorts.items():
        # cohort_name template: temporal_ecoli_{specimen_key}_{year}
        parts = cohort_name.split("_")
        # ["temporal", "ecoli", "{specimen_key}", "{year}"]
        specimen_key = parts[2]
        year = int(parts[3])

        # Look up context info from the creator
        ctx = temporal_creator.specimen_contexts.get(specimen_key)

        if ctx is not None:
            # Example title:
            # "Escherichia – Urine – Community-acquired UTI context – 2019"
            specimen_label = ", ".join(ctx.specimens)
            base_title = (
                f"{temporal_creator.pathogen_genus} – {specimen_label} – "
                f"{ctx.description} – {year}"
            )
        else:
            # Fallback if something unexpected happens
            base_title = f"{temporal_creator.pathogen_genus} – {cohort_name} – {year}"
        
        base_title = ""

        save_raw_and_pruned_networks(
            cohort_df=cohort_df,
            cohort_name=cohort_name,
            output_dir=networks_dir,
            title_prefix=base_title,
        )
