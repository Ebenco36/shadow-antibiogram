from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from shadow_antibiogram.controllers.AMR.use_cases.helper import build_and_save_network_for_cohort, save_raw_and_pruned_networks
# === project imports ==========================================================
from shadow_antibiogram.controllers.cohort_analyzer.CohortSelection import (
    ProductionCohortGenerator,
    CohortConfig,
)
from pathlib import Path

# =============================================================================
# 1. LOGGING
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file + stdout for this pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"

    logger = logging.getLogger("UseCase2Divergence")
    logger.setLevel(logging.INFO)

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
# 3. CONFIGURATION FOR CONTEXT COMPARISONS
# =============================================================================


@dataclass
class ContextComparison:
    """Configuration for paired context comparison."""
    name: str
    pathogen_genus: str
    specimen_type: str
    context_dimension: str          # column name in df (e.g. 'ARS_WardType')
    context_values: Tuple[str, str] # raw values in that column
    context_labels: Tuple[str, str] # nice labels for plots/tables
    description: str
    min_sample_size: int = 500


# =============================================================================
# 4. PAIRED CONTEXT COHORT CREATOR
# =============================================================================


class DiagnosticDivergenceCohortCreator:
    """
    Creates paired context cohorts for Use Case 2 using ProductionCohortGenerator.

    Example comparisons (configured below):
      - E. coli urine: ICU vs ward
      - E. coli urine: inpatient vs outpatient
      - E. coli blood: ICU vs ward
      - Staph blood: ICU vs ward
      - Klebsiella blood: tertiary vs primary/secondary
      - E. coli urine: women vs men
    """

    def __init__(
        self,
        generator: ProductionCohortGenerator,
        logger: Optional[logging.Logger] = None,
    ):
        self.generator = generator
        self.logger = logger or logging.getLogger("UseCase2Divergence")

        self.comparisons: Dict[str, ContextComparison] = {
            # E. coli urine in different acuity levels
            "ecoli_urine_ward_divergence": ContextComparison(
                name="ecoli_urine_ward_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Urine",
                context_dimension="ARS_WardType",
                context_values=("Intensive Care Unit", "Normal Ward"),
                context_labels=("ICU (high-acuity)", "General Ward (routine)"),
                description="E. coli UTIs: Intensive vs. routine care",
                min_sample_size=1000,
            ),
            # E. coli urine in inpatient vs outpatient
            "ecoli_urine_care_divergence": ContextComparison(
                name="ecoli_urine_care_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Urine",
                context_dimension="CareType",
                context_values=("In-Patient", "Out-Patient"),
                context_labels=("Hospitalized", "Community"),
                description="E. coli UTIs: Hospital vs. community practice",
                min_sample_size=1000,
            ),
            # E. coli urine by hospital level
            "ecoli_urine_hospital_level": ContextComparison(
                name="ecoli_urine_hospital_level",
                pathogen_genus="Escherichia",
                specimen_type="Urine",
                context_dimension="Care_Complexity",
                context_values=("Tertiary & Specialized", "Primary/Secondary"),
                context_labels=("Tertiary & Specialized", "Primary/Secondary"),
                description="E. coli UTIs: Primary vs. tertiary centers",
                min_sample_size=500,
            ),
            # E. coli blood in ICU vs ward
            "ecoli_blood_ward_divergence": ContextComparison(
                name="ecoli_blood_ward_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Blood Culture",
                context_dimension="ARS_WardType",
                context_values=("Intensive Care Unit", "Normal Ward"),
                context_labels=("ICU (critical)", "General Ward"),
                description="E. coli bacteremia: ICU vs. general ward",
                min_sample_size=300,
            ),
            # E. coli blood inpatient vs outpatient
            "ecoli_blood_care_divergence": ContextComparison(
                name="ecoli_blood_care_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Blood Culture",
                context_dimension="CareType",
                context_values=("In-Patient", "Out-Patient"),
                context_labels=("In-Patient", "Out-Patient"),
                description="E. coli bacteremia: Hospital vs. community",
                min_sample_size=300,
            ),
            # Staph aureus blood in ICU vs ward
            "staph_blood_ward_divergence": ContextComparison(
                name="staph_blood_ward_divergence",
                pathogen_genus="Staphylococcus",
                specimen_type="Blood Culture",
                context_dimension="ARS_WardType",
                context_values=("Intensive Care Unit", "Normal Ward"),
                context_labels=("ICU (sepsis)", "General Ward"),
                description="S. aureus bacteremia: ICU vs. general ward",
                min_sample_size=300,
            ),
            # Klebsiella blood by hospital level
            "klebsiella_blood_hospital_level": ContextComparison(
                name="klebsiella_blood_hospital_level",
                pathogen_genus="Klebsiella",
                specimen_type="Blood Culture",
                context_dimension="Care_Complexity",
                context_values=("Tertiary & Specialized", "Primary/Secondary"),
                context_labels=("Tertiary & Specialized", "Primary/Secondary"),
                description="K. pneumoniae bacteremia: Primary vs. tertiary",
                min_sample_size=200,
            ),
            # Sex-stratified comparison (E. coli UTI)
            "ecoli_urine_sex_divergence": ContextComparison(
                name="ecoli_urine_sex_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Urine",
                context_dimension="Sex",
                context_values=("Woman", "Man"),
                context_labels=("Female (typical)", "Male (complicated)"),
                description="E. coli UTIs: Female vs. male patients",
                min_sample_size=500,
            ),
            
            # Organization-stratified comparison (E. coli Blood)
            "ecoli_blood_organization_type_divergence": ContextComparison(
                name="ecoli_blood_organization_type_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Blood Culture",
                context_dimension="OrgType",
                context_values=("Hospital", "Doctor's office"),
                context_labels=("Hospital", "Doctor's office"),
                description="E. coli Blood Stream Infection: Hospital vs. Doctor's office",
                min_sample_size=500,
            ),
            "ecoli_urine_organization_type_divergence": ContextComparison(
                name="ecoli_urine_organization_type_divergence",
                pathogen_genus="Escherichia",
                specimen_type="Urine",
                context_dimension="OrgType",
                context_values=("Hospital", "Doctor's office"),
                context_labels=("Hospital", "Doctor's office"),
                description="E. coli UTIs: Hospital vs. Doctor's office",
                min_sample_size=500,
            ),
        }

        self.paired_cohorts: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    # --- helper: map context_dimension → CohortConfig field -------------------
    def _build_context_specific_config(
        self,
        base_name: str,
        comparison: ContextComparison,
        context_index: int,
    ) -> CohortConfig:
        """
        Build CohortConfig for a given context (0 or 1) based on context_dimension.
        """
        context_value = comparison.context_values[context_index]

        # Base config shared by both contexts
        cfg_kwargs = dict(
            name=f"{base_name}_context{context_index+1}",
            pathogen_genus=[comparison.pathogen_genus],
            specimens=[comparison.specimen_type],
            isolate_group=None,
            years=[2019, 2020, 2021, 2022, 2023],
            min_sample_size=comparison.min_sample_size,
            remove_untested_abx=True,
            description=(
                f"{comparison.description} - {comparison.context_labels[context_index]}"
            ),
        )

        dim = comparison.context_dimension
        # Map dimension → proper CohortConfig field
        if dim == "ARS_WardType":
            cfg_kwargs["ward_type"] = context_value
        elif dim == "CareType":
            cfg_kwargs["care_type"] = context_value
        elif dim == "Care_Complexity":
            cfg_kwargs["care_complexity"] = context_value
        elif dim == "Sex":
            # CohortConfig.sex expects Optional[List[str]]
            cfg_kwargs["sex"] = [context_value]
        else:
            # Fall back: don't filter on the unknown dimension, just log
            self.logger.warning(
                f"Unknown context_dimension '{dim}' – using no extra filter for {base_name}"
            )

        return CohortConfig(**cfg_kwargs)

    # --- main cohort construction --------------------------------------------
    def create_all_paired_cohorts(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create all paired cohorts for configured comparisons."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("USE CASE 2 – STEP 1: CREATING PAIRED CONTEXT COHORTS")
        self.logger.info("=" * 80)

        for key in sorted(self.comparisons.keys()):
            comparison = self.comparisons[key]

            try:
                cfg1 = self._build_context_specific_config(
                    base_name=key,
                    comparison=comparison,
                    context_index=0,
                )
                cfg2 = self._build_context_specific_config(
                    base_name=key,
                    comparison=comparison,
                    context_index=1,
                )

                cohort1 = self.generator.create_cohort(cfg1)
                cohort2 = self.generator.create_cohort(cfg2)

                if cohort1 is not None and cohort2 is not None:
                    self.paired_cohorts[key] = (cohort1, cohort2)
                    self.logger.info(
                        f"✓ {key}:\n"
                        f"  {comparison.context_labels[0]:30s}: N={len(cohort1)}\n"
                        f"  {comparison.context_labels[1]:30s}: N={len(cohort2)}"
                    )
                else:
                    self.logger.warning(
                        f"✗ Skipping {key}: one or both contexts were < min_sample_size"
                    )

            except Exception as e:
                self.logger.error(f"Error creating paired cohorts for {key}: {e}")

        self.logger.info(
            f"\n✓ Created {len(self.paired_cohorts)} paired comparisons for UC2"
        )
        return self.paired_cohorts

    def get_summary(self, data_summary_path: Optional[str] = None) -> pd.DataFrame:
        """Summarize all paired cohorts."""
        rows = []

        for key, (cohort1, cohort2) in self.paired_cohorts.items():
            comparison = self.comparisons[key]

            rows.append(
                {
                    "comparison": key,
                    "pathogen_genus": comparison.pathogen_genus,
                    "specimen": comparison.specimen_type,
                    "context_dimension": comparison.context_dimension,
                    "context_1": comparison.context_labels[0],
                    "value_1": comparison.context_values[0],
                    "N_1": len(cohort1),
                    "context_2": comparison.context_labels[1],
                    "value_2": comparison.context_values[1],
                    "N_2": len(cohort2),
                    "N_total": len(cohort1) + len(cohort2),
                }
            )

        df = pd.DataFrame(rows)

        if data_summary_path:
            Path(data_summary_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_summary_path, index=False)
            self.logger.info(f"✓ Saved UC2 cohort summary → {data_summary_path}")

        return df

def run_context_divergence_use_case(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    UC2: For each paired context comparison, build two networks per context:
      - raw Jaccard (τ=0.3, γ=1.0)
      - FDR-pruned Jaccard (τ=0.3, γ=1.0)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ProductionCohortGenerator(df)
    logger = setup_logging(output_dir / "context")

    divergence_creator = DiagnosticDivergenceCohortCreator(generator, logger=logger)

    paired_cohorts = divergence_creator.create_all_paired_cohorts()
    divergence_creator.get_summary(
        data_summary_path=output_dir / "context" / "uc2_cohort_summary.csv"
    )

    networks_dir = output_dir / "context_networks"
    # print(paired_cohorts)
    for key, (cohort1, cohort2) in paired_cohorts.items():
        # key like "ecoli_urine_ward_divergence"
        name1 = f"{key}_context1"
        name2 = f"{key}_context2"
        
        comparison = divergence_creator.comparisons[key]
        base_title = (
            f"{comparison.pathogen_genus} – {comparison.specimen_type} – "
            f"{comparison.description}"
        )
        title1 = "" #f"{base_title} – {comparison.context_labels[0]}"
        title2 = "" #f"{base_title} – {comparison.context_labels[1]}"

        save_raw_and_pruned_networks(
            cohort_df=cohort1,
            cohort_name=name1,
            output_dir=networks_dir,
            title_prefix=title1
        )
        save_raw_and_pruned_networks(
            cohort_df=cohort2,
            cohort_name=name2,
            output_dir=networks_dir,
            title_prefix=title2
        )