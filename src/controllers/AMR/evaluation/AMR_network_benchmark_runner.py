from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.controllers.AMR.use_cases.helper import build_and_save_network_for_cohort
from src.controllers.AMR.evaluation.jaccard_network_comparator import CompareConfig, JaccardNetworkComparator


# -----------------------------
# Config + cohort spec
# -----------------------------
@dataclass(frozen=True)
class CohortSpec:
    """Defines a cohort slice (used to filter BOTH full + aggregated datasets)."""
    pathogen_genus: str = "Escherichia"
    specimen: Optional[str] = None   # TextMaterialgroupRkiL0
    care_type: Optional[str] = None  # CareType
    ward_type: Optional[str] = None  # ARS_WardType
    year: Optional[str] = None       # Year treated as STRING

    def to_name(self) -> str:
        parts = [f"PathogenGenus={self.pathogen_genus}"]
        if self.specimen is not None:
            parts.append(f"Specimen={self.specimen}")
        if self.care_type is not None:
            parts.append(f"CareType={self.care_type}")
        if self.ward_type is not None:
            parts.append(f"WardType={self.ward_type}")
        if self.year is not None:
            parts.append(f"Year={self.year}")
        return "__".join(parts)


@dataclass(frozen=True)
class DatasetComparison:
    """
    Compares FULL vs AGGREGATED for the SAME CohortSpec.
    """
    name: str
    spec: CohortSpec


@dataclass
class RunnerConfig:
    base_output_dir: str = "./network_benchmarks"
    min_rows: int = 10  # guardrail: skip comparisons with too-small cohorts

    comparator_cfg: CompareConfig = field(
        default_factory=lambda: CompareConfig(
            tau=0.3,
            n_clusters=5,
            weighted_graph=False,
            out_dir="./_tmp_should_be_overridden",
            dpi=350,
            save_pdf=True,
        )
    )

    preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


# -----------------------------
# Runner
# -----------------------------
class AMRNetworkBenchmarkRunner:
    """
    Runs network build + network comparison across covariate-defined cohorts,
    comparing FULL dataset network vs AGGREGATED dataset network.

    Output layout:
      base_output_dir/
        comparisons/
          <dataset_tag>/
            <comparison_name>/
              full_network/
              aggregated_network/
              compare/
        master_summary.csv
        master_failures.csv
        run_config.json
    """

    def __init__(self, df_full: pd.DataFrame, df_aggregated: pd.DataFrame, cfg: RunnerConfig):
        self.df_full = df_full.copy()
        self.df_agg = df_aggregated.copy()
        self.cfg = cfg
        self.out = Path(cfg.base_output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        if cfg.preprocess:
            self.df_full = cfg.preprocess(self.df_full)
            self.df_agg = cfg.preprocess(self.df_agg)

        # Ensure Year is treated as string in both datasets
        if "Year" in self.df_full.columns:
            self.df_full["Year"] = self.df_full["Year"].astype(str)
        if "Year" in self.df_agg.columns:
            self.df_agg["Year"] = self.df_agg["Year"].astype(str)

    # ------------- filtering -------------
    @staticmethod
    def _apply_spec(df: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
        out = df[df["PathogenGenus"] == spec.pathogen_genus]
        if spec.specimen is not None:
            out = out[out["TextMaterialgroupRkiL0"] == spec.specimen]
        if spec.care_type is not None:
            out = out[out["CareType"] == spec.care_type]
        if spec.ward_type is not None:
            out = out[out["ARS_WardType"] == spec.ward_type]
        if spec.year is not None:
            # compare as string
            out = out[out["Year"].astype(str) == str(spec.year)]

        return out

    # ------------- building networks -------------
    def _build_network(self, cohort_df: pd.DataFrame, cohort_name: str, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        sim, aware_color_map, title = build_and_save_network_for_cohort(
            cohort_df=cohort_df,
            cohort_name=cohort_name,
            output_dir=out_dir,
        )
        return sim, aware_color_map, title

    # ------------- CompareConfig override -------------
    def _with_out_dir(self, out_dir: Path) -> CompareConfig:
        base = asdict(self.cfg.comparator_cfg)
        base["out_dir"] = str(out_dir)
        return CompareConfig(**base)

    # ------------- one comparison -------------
    def _run_one(self, comparison: DatasetComparison, dataset_tag: str) -> Optional[Dict]:
        comp_dir = self.out / "comparisons" / dataset_tag / comparison.name
        full_dir = comp_dir / "full_network"
        agg_dir = comp_dir / "aggregated_network"
        compare_dir = comp_dir / "compare"

        df_full_cohort = self._apply_spec(self.df_full, comparison.spec)
        df_agg_cohort = self._apply_spec(self.df_agg, comparison.spec)

        if len(df_full_cohort) < self.cfg.min_rows or len(df_agg_cohort) < self.cfg.min_rows:
            print(f"Skipping {comparison.name}: full={len(df_full_cohort)}, agg={len(df_agg_cohort)} < {self.cfg.min_rows}")
            return None   # Skip this comparison entirely

        sim_full, _, _ = self._build_network(
            df_full_cohort,
            "FULL__" + comparison.spec.to_name(),
            full_dir,
        )
        sim_agg, _, _ = self._build_network(
            df_agg_cohort,
            "AGGREGATED__" + comparison.spec.to_name(),
            agg_dir,
        )

        comp_cfg = self._with_out_dir(compare_dir)
        comparator = JaccardNetworkComparator(comp_cfg)
        report = comparator.fit(sim_full, sim_agg)
        comparator.save_all()

        report_enriched = {
            **report,
            "comparison_name": comparison.name,
            "dataset_tag": dataset_tag,
            "cohort_spec": asdict(comparison.spec),
            "n_rows_full_dataset": int(len(df_full_cohort)),
            "n_rows_aggregated_dataset": int(len(df_agg_cohort)),
            "output_dir": str(comp_dir.resolve()),
        }
        return report_enriched

    def run(
        self,
        comparisons: List[DatasetComparison],
        dataset_tag: str = "full_vs_aggregated",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        summaries: List[Dict] = []
        failures: List[Dict] = []

        for c in comparisons:
            try:
                rep = self._run_one(c, dataset_tag=dataset_tag)
                if rep is not None:
                    summaries.append(rep)
            except Exception as e:
                failures.append(
                    {
                        "comparison_name": c.name,
                        "dataset_tag": dataset_tag,
                        "cohort_spec": c.spec.to_name(),
                        "error": repr(e),
                    }
                )

        summary_df = pd.DataFrame(summaries)
        failures_df = pd.DataFrame(failures)

        summary_path = self.out / "master_summary.csv"
        fail_path = self.out / "master_failures.csv"
        summary_df.to_csv(summary_path, index=False)
        failures_df.to_csv(fail_path, index=False)

        with open(self.out / "run_config.json", "w") as f:
            json.dump(
                {
                    "runner_config": {
                        "base_output_dir": self.cfg.base_output_dir,
                        "min_rows": self.cfg.min_rows,
                        "comparator_cfg": asdict(self.cfg.comparator_cfg),
                        "preprocess": None,
                    }
                },
                f,
                indent=2,
                default=str,
            )

        print(f"Saved master summary to: {summary_path.resolve()}")
        print(f"Saved failures to: {fail_path.resolve()}")
        return summary_df, failures_df


# -----------------------------
# Helpers to generate comparisons (FULL vs AGG for same cohort spec)
# -----------------------------
def make_temporal_comparisons(specimen: str, years: Iterable[str]) -> List[DatasetComparison]:
    # treat years as strings, preserve exact formatting
    years_str = sorted({str(y) for y in years})
    comps: List[DatasetComparison] = []
    for y in years_str:
        spec = CohortSpec(pathogen_genus="Escherichia", specimen=specimen, year=y)
        comps.append(
            DatasetComparison(
                name=f"Temporal__{specimen}__Year={y}__FULL_vs_AGG",
                spec=spec,
            )
        )
    return comps


def make_context_comparisons_by_caretype(specimen: str) -> List[DatasetComparison]:
    comps: List[DatasetComparison] = []
    for care in ["In-Patient", "Out-Patient"]:
        spec = CohortSpec(pathogen_genus="Escherichia", specimen=specimen, care_type=care)
        comps.append(
            DatasetComparison(
                name=f"Context__CareType__{specimen}__{care}__FULL_vs_AGG",
                spec=spec,
            )
        )
    return comps


def make_context_comparisons_by_ward(specimen: str) -> List[DatasetComparison]:
    comps: List[DatasetComparison] = []
    for ward in ["Intensive Care Unit", "Normal Ward"]:
        spec = CohortSpec(pathogen_genus="Escherichia", specimen=specimen, ward_type=ward)
        comps.append(
            DatasetComparison(
                name=f"Context__WardType__{specimen}__{ward}__FULL_vs_AGG",
                spec=spec,
            )
        )
    return comps


if __name__ == "__main__":
    from src.controllers.DataLoader import DataLoader
    from src.mappers.top_pathogens import ALL_PATHOGENS

    parquet_dir_full = "./datasets/WHO_Aware_data__"
    loader = DataLoader(parquet_dir_full, pathogen_groups_regex=ALL_PATHOGENS)
    df_full = loader.get_combined(return_which="tested")

    parquet_dir_aggregated = "./datasets/WHO_Aware_data"
    loader = DataLoader(parquet_dir_aggregated, pathogen_groups_regex=ALL_PATHOGENS)
    df_aggregated = loader.get_combined(return_which="tested")

    if "Year" in df_aggregated.columns:
        # Remove trailing '.0' and keep as string
        df_aggregated["Year"] = df_aggregated["Year"].astype(str).str.replace(r'\.0$', '', regex=True)
    
    if "Year" in df_full.columns:
        # Remove trailing '.0' and keep as string
        df_full["Year"] = df_full["Year"].astype(str).str.replace(r'\.0$', '', regex=True)
        
    # years as STRINGS (keep exact formatting)
    years = sorted(
        df_full.loc[df_full["PathogenGenus"] == "Escherichia", "Year"]
        .dropna()
        .astype(str)
        .unique()
    )

    runner_cfg = RunnerConfig(
        base_output_dir="./benchmarks_escherichia_full_vs_agg",
        min_rows=1000,
        comparator_cfg=CompareConfig(
            tau=0.3,
            n_clusters=5,
            weighted_graph=False,
            out_dir="unused",
            dpi=350,
            save_pdf=True,
        ),
    )

    runner = AMRNetworkBenchmarkRunner(df_full, df_aggregated, runner_cfg)

    comparisons: List[DatasetComparison] = []
    comparisons += make_temporal_comparisons("Urine", years)
    comparisons += make_temporal_comparisons("Blood Culture", years)

    comparisons += make_context_comparisons_by_caretype("Urine")
    comparisons += make_context_comparisons_by_caretype("Blood Culture")

    comparisons += make_context_comparisons_by_ward("Urine")
    comparisons += make_context_comparisons_by_ward("Blood Culture")

    summary_df, failures_df = runner.run(
        comparisons=comparisons,
        dataset_tag="full_vs_aggregated",
    )
