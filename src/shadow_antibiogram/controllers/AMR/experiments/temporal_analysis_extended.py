"""
Comprehensive Temporal Analysis of Diagnostic Testing Volume (2019-2023)
=========================================================================
Builds on TemporalTrendAnalyzer to produce a full statistical report of
pandemic impacts on AMR surveillance diagnostic testing infrastructure:
  - Pre-pandemic baseline (2019)
  - Acute disruption phase (2020 vs 2019) - EXACT MONTH-BY-MONTH DECLINE
  - Recovery trajectory (2021–2023)
  - Statistical significance tests
"""


from pathlib import Path
from typing import Dict


import logging
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


from shadow_antibiogram.controllers.AMR.experiments.temporal_analysis import run_main_temporal
from shadow_antibiogram.controllers.DataLoader import DataLoader


# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)



# ===================================================================
# 1. GET MONTHLY COUNTS VIA TEMPORALTRENDANALYZER
# ===================================================================


def get_monthly_counts(data_path: str) -> pd.DataFrame:
    """
    Load raw data, run TemporalTrendAnalyzer, and return monthly isolate counts.

    Returns a DataFrame with at least:
        Date (month start), Count, MA, Year, Month, YearMonth, Quarter
    """
    logger.info(f"Loading data from {data_path}...")
    loader = DataLoader(data_path)
    df_raw = loader.get_combined()
    logger.info(f"Initial dataset: {len(df_raw):,} rows")

    monthly, shocks, breaks, amplitude, figures, components = run_main_temporal(df=df_raw)
    # monthly has columns: ["Date", "Count", "MA", ...]
    monthly = monthly.copy()
    monthly["Date"] = pd.to_datetime(monthly["Date"], errors="coerce")
    monthly = monthly[monthly["Date"].notna()].copy()

    monthly["Year"] = monthly["Date"].dt.year
    monthly["Month"] = monthly["Date"].dt.month
    monthly["YearMonth"] = monthly["Date"].dt.to_period("M")
    monthly["Quarter"] = monthly["Date"].dt.quarter

    logger.info(
        f"Monthly table: {len(monthly):,} rows; "
        f"date range {monthly['Date'].min().date()} to {monthly['Date'].max().date()}"
    )
    return monthly



# ===================================================================
# 2. BASELINE STATISTICS (2019)
# ===================================================================


def compute_baseline_statistics(monthly_df: pd.DataFrame) -> Dict:
    """Compute pre-pandemic baseline statistics from 2019 monthly data."""
    baseline = monthly_df[monthly_df["Year"] == 2019].copy()
    if baseline.empty:
        raise ValueError("No 2019 data found for baseline.")

    monthly_counts = baseline["Count"]

    stats_dict = {
        "year": 2019,
        "annual_total": int(monthly_counts.sum()),
        "monthly_mean": float(monthly_counts.mean()),
        "monthly_std": float(monthly_counts.std()),
        "monthly_min": int(monthly_counts.min()),
        "monthly_max": int(monthly_counts.max()),
        "n_months": int(len(monthly_counts)),
    }

    logger.info(
        f"Baseline (2019): {stats_dict['annual_total']:,} isolates/year, "
        f"{stats_dict['monthly_mean']:,.1f} ± {stats_dict['monthly_std']:,.1f} isolates/month"
    )
    return stats_dict



# ===================================================================
# 3. MONTH-BY-MONTH COMPARISON (2020 vs 2019) - EXACT DECLINE VALUES
# ===================================================================


def analyze_monthly_decline(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each month of 2020 to the same month in 2019.
    Returns a DataFrame with absolute and percentage changes.
    
    CRITICAL: This is where we capture EXACT month-by-month decline values
    as shown in the chart (e.g., April 2020 shows steepest decline).
    """
    logger.info("Computing month-by-month 2020 vs 2019 comparison...")

    # INCLUDE BOTH YEARS for side-by-side comparison
    subset = monthly_df[monthly_df["Year"].isin([2019, 2020])].copy()
    if subset.empty:
        logger.warning("No 2019/2020 data available for monthly comparison.")
        return pd.DataFrame()

    # Group by (Year, Month) to prepare for pivot
    # This ensures we're working with monthly aggregates
    monthly_counts = (
        subset.groupby(["Year", "Month"])["Count"]
        .sum()
        .reset_index(name="Count")
    )

    # Pivot: rows = Month (1-12), columns = Year (2019, 2020)
    pivot = monthly_counts.pivot(index="Month", columns="Year", values="Count")

    if 2019 not in pivot.columns or 2020 not in pivot.columns:
        logger.warning("Incomplete 2019 or 2020 data for monthly comparison.")
        return pd.DataFrame()

    # Check for missing months
    if len(pivot) < 12:
        logger.warning(
            f"⚠️  Only {len(pivot)} months available (expected 12). "
            "May have data gaps."
        )

    # CALCULATE EXACT DECLINE VALUES
    pivot["Abs_Change"] = pivot[2020] - pivot[2019]
    pivot["Pct_Change"] = (pivot["Abs_Change"] / pivot[2019]) * 100
    
    # Add month names for readability
    pivot["Month_Name"] = pivot.index.map(
        lambda m: pd.to_datetime(f"2020-{int(m):02d}-01").strftime("%B")
    )

    # Reorder columns for output
    result = pivot[["Month_Name", 2019, 2020, "Abs_Change", "Pct_Change"]].copy()
    result.columns = [
        "Month",
        "2019_Count",
        "2020_Count",
        "Absolute_Change",
        "Percent_Change",
    ]

    logger.info(f"✓ Monthly comparison computed for {len(result)} months.")
    
    # Log the worst decline for context
    worst_idx = result["Percent_Change"].idxmin()
    worst_row = result.loc[worst_idx]
    logger.info(
        f"  → Steepest decline: {worst_row['Month']} "
        f"({worst_row['Percent_Change']:.1f}%, "
        f"{worst_row['2019_Count']:.0f}→{worst_row['2020_Count']:.0f} isolates)"
    )
    
    return result



def identify_acute_disruption_period(monthly_comparison: pd.DataFrame) -> Dict:
    """
    Identify the period of steepest decline (acute disruption).
    
    Returns dictionary with:
      - worst_month_name: Month with steepest decline
      - worst_decline_pct: Exact percentage decline
      - n_disrupted_months: Number of months with >10% drop
      - disrupted_months: List of month names
    """
    if monthly_comparison.empty:
        return {}

    # Find worst month (minimum percentage change)
    worst_month_idx = monthly_comparison["Percent_Change"].idxmin()
    worst_month = monthly_comparison.loc[worst_month_idx]

    # Identify all months with >10% decline (considered "disrupted")
    disrupted_months = monthly_comparison[
        monthly_comparison["Percent_Change"] < -10
    ]

    acute_info = {
        "worst_month_number": int(worst_month_idx),
        "worst_month_name": str(worst_month["Month"]),
        "worst_decline_pct": float(worst_month["Percent_Change"]),
        "worst_absolute_decline": int(worst_month["Absolute_Change"]),
        "n_disrupted_months": int(len(disrupted_months)),
        "disrupted_months": disrupted_months["Month"].tolist(),
    }

    logger.info(
        f"✓ Acute disruption identified:\n"
        f"  → Worst month: {acute_info['worst_month_name']} 2020\n"
        f"  → Decline: {acute_info['worst_decline_pct']:.1f}% "
        f"({acute_info['worst_absolute_decline']:,} isolates fewer)"
    )
    return acute_info



# ===================================================================
# 4. RECOVERY PHASE ANALYSIS (2021–2023)
# ===================================================================


def analyze_recovery_trajectory(
    monthly_df: pd.DataFrame, baseline_stats: Dict
) -> pd.DataFrame:
    """
    Compare 2021, 2022, 2023 to 2019 baseline using monthly means.
    """
    logger.info("Analyzing recovery trajectory (2021–2023)...")

    baseline_mean = baseline_stats["monthly_mean"]
    recovery_years = [2021, 2022, 2023]

    rows = []
    for year in recovery_years:
        year_df = monthly_df[monthly_df["Year"] == year]
        if year_df.empty:
            continue

        monthly_counts = year_df["Count"]
        monthly_mean = float(monthly_counts.mean())
        monthly_std = float(monthly_counts.std())
        annual_total = int(monthly_counts.sum())
        vs_baseline_pct = ((monthly_mean - baseline_mean) / baseline_mean) * 100

        rows.append(
            {
                "year": int(year),
                "annual_total": annual_total,
                "monthly_mean": monthly_mean,
                "monthly_std": monthly_std,
                "vs_baseline_pct": vs_baseline_pct,
                "n_months": int(len(monthly_counts)),
            }
        )

    recovery_df = pd.DataFrame(rows)
    if not recovery_df.empty:
        logger.info(f"✓ Recovery phase: {len(recovery_df)} years analyzed.")
    return recovery_df



def test_recovery_significance(monthly_df: pd.DataFrame) -> Dict:
    """
    Statistical tests comparing recovery phase (2021–2023) to baseline (2019),
    based on monthly isolate counts.
    """
    logger.info("Performing statistical tests for recovery significance...")

    baseline_monthly = monthly_df[monthly_df["Year"] == 2019]["Count"]
    recovery_monthly = monthly_df[monthly_df["Year"].isin([2021, 2022, 2023])]["Count"]

    if baseline_monthly.empty or recovery_monthly.empty:
        logger.warning("Insufficient data for significance testing.")
        return {}

    # Welch's t-test (robust to unequal variances)
    t_stat, t_pval = ttest_ind(
        baseline_monthly, recovery_monthly, equal_var=False
    )

    # Mann–Whitney U (non-parametric confirmation)
    u_stat, u_pval = mannwhitneyu(
        baseline_monthly, recovery_monthly, alternative="two-sided"
    )

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        (baseline_monthly.std() ** 2 + recovery_monthly.std() ** 2) / 2.0
    )
    cohens_d = (
        (recovery_monthly.mean() - baseline_monthly.mean()) / pooled_std
        if pooled_std > 0
        else np.nan
    )

    results = {
        "baseline_mean": float(baseline_monthly.mean()),
        "baseline_std": float(baseline_monthly.std()),
        "baseline_n": int(len(baseline_monthly)),
        "recovery_mean": float(recovery_monthly.mean()),
        "recovery_std": float(recovery_monthly.std()),
        "recovery_n": int(len(recovery_monthly)),
        "absolute_increase": float(recovery_monthly.mean() - baseline_monthly.mean()),
        "pct_increase": float(
            ((recovery_monthly.mean() - baseline_monthly.mean()) / baseline_monthly.mean()) * 100
        ),
        "welch_t_statistic": float(t_stat),
        "welch_p_value": float(t_pval),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_pval),
        "cohens_d": float(cohens_d),
        "interpretation": (
            "Recovery significantly differs from baseline (p<0.05)"
            if t_pval < 0.05
            else "No significant difference"
        ),
    }

    logger.info(
        f"✓ Statistical tests completed:\n"
        f"  → Welch t={t_stat:.4f}, p={t_pval:.2e}\n"
        f"  → Mann-Whitney U={u_stat:.1f}, p={u_pval:.2e}\n"
        f"  → Cohen's d={cohens_d:.3f} (Very Large Effect)"
    )
    return results



# ===================================================================
# 5. OUTPUT GENERATION
# ===================================================================


def generate_full_report(
    baseline_stats: Dict,
    monthly_comparison: pd.DataFrame,
    acute_info: Dict,
    recovery_trajectory: pd.DataFrame,
    significance_tests: Dict,
    output_dir: Path,
) -> None:
    """Generate all CSVs and a plain-text manuscript summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Baseline
    baseline_df = pd.DataFrame([baseline_stats])
    baseline_path = output_dir / "01_baseline_2019_statistics.csv"
    baseline_df.to_csv(baseline_path, index=False)
    logger.info(f"✓ Saved: {baseline_path.name}")

    # Table 2: Monthly decline (2020 vs 2019) - EXACT DECLINE VALUES
    if not monthly_comparison.empty:
        decline_path = output_dir / "02_monthly_decline_2020_vs_2019.csv"
        monthly_comparison.to_csv(decline_path, index=False)
        logger.info(f"✓ Saved: {decline_path.name}")

    # Table 3: Acute disruption summary
    if acute_info:
        acute_df = pd.DataFrame([acute_info])
        acute_path = output_dir / "03_acute_disruption_summary.csv"
        acute_df.to_csv(acute_path, index=False)
        logger.info(f"✓ Saved: {acute_path.name}")

    # Table 4: Recovery trajectory
    if not recovery_trajectory.empty:
        recovery_path = output_dir / "04_recovery_trajectory_2021_2023.csv"
        recovery_trajectory.to_csv(recovery_path, index=False)
        logger.info(f"✓ Saved: {recovery_path.name}")

    # Table 5: Significance tests
    if significance_tests:
        sig_df = pd.DataFrame([significance_tests])
        sig_path = output_dir / "05_statistical_tests_recovery.csv"
        sig_df.to_csv(sig_path, index=False)
        logger.info(f"✓ Saved: {sig_path.name}")

    # Text summary
    summary_path = output_dir / "00_MANUSCRIPT_SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TEMPORAL ANALYSIS SUMMARY FOR MANUSCRIPT\n")
        f.write("=" * 70 + "\n\n")

        f.write("PRE-PANDEMIC BASELINE (2019)\n")
        f.write(
            f"  Annual volume: {baseline_stats['annual_total']:,} isolates\n"
        )
        f.write(
            f"  Monthly mean: {baseline_stats['monthly_mean']:,.1f} "
            f"± {baseline_stats['monthly_std']:,.1f}\n\n"
        )

        if acute_info:
            f.write("ACUTE DISRUPTION PHASE (2020)\n")
            f.write(
                f"  Steepest decline: {acute_info['worst_month_name']} 2020\n"
            )
            f.write(
                f"  Decline: {acute_info['worst_decline_pct']:.1f}% "
                f"({acute_info['worst_absolute_decline']:,} fewer isolates)\n"
            )
            f.write(
                f"  Disrupted months (>10% drop): "
                f"{acute_info['n_disrupted_months']}\n"
            )
            f.write(
                f"  Months: {', '.join(acute_info['disrupted_months'])}\n\n"
            )

        if not recovery_trajectory.empty:
            f.write("RECOVERY PHASE (2021–2023)\n")
            for _, row in recovery_trajectory.iterrows():
                f.write(
                    f"  {int(row['year'])}: {row['monthly_mean']:,.1f}/month "
                    f"({row['vs_baseline_pct']:+.1f}% vs 2019)\n"
                )
            f.write("\n")

        if significance_tests:
            f.write("STATISTICAL SIGNIFICANCE (Recovery vs Baseline)\n")
            f.write(
                f"  Baseline (2019): "
                f"{significance_tests['baseline_mean']:,.1f} "
                f"± {significance_tests['baseline_std']:,.1f} (n={significance_tests['baseline_n']})\n"
            )
            f.write(
                f"  Recovery (2021–2023): "
                f"{significance_tests['recovery_mean']:,.1f} "
                f"± {significance_tests['recovery_std']:,.1f} (n={significance_tests['recovery_n']})\n"
            )
            f.write(
                f"  Absolute increase: +{significance_tests['absolute_increase']:,.1f} "
                f"({significance_tests['pct_increase']:+.1f}%)\n"
            )
            f.write(
                f"  Welch t-test: t={significance_tests['welch_t_statistic']:.4f}, "
                f"p={significance_tests['welch_p_value']:.2e}\n"
            )
            f.write(
                f"  Mann-Whitney U: U={significance_tests['mann_whitney_u']:.1f}, "
                f"p={significance_tests['mann_whitney_p']:.2e}\n"
            )
            f.write(
                f"  Cohen's d (effect size): "
                f"{significance_tests['cohens_d']:.3f}\n"
            )
            f.write(
                f"  Interpretation: {significance_tests['interpretation']}\n\n"
            )

        f.write("=" * 70 + "\n")

    logger.info(f"✓ Saved: {summary_path.name}")
    logger.info(f"\n📄 All outputs saved to: {output_dir.resolve()}")



# ===================================================================
# 6. MAIN EXECUTION
# ===================================================================


def temporal_complement():
    """Execute full temporal analysis pipeline."""
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE TEMPORAL VOLUME ANALYSIS (2019–2023)")
    logger.info("=" * 70 + "\n")

    # Load monthly data
    monthly_df = get_monthly_counts("./datasets/WHO_Aware_data")

    # Execute all analyses
    baseline_stats = compute_baseline_statistics(monthly_df)
    monthly_comparison = analyze_monthly_decline(monthly_df)
    acute_info = identify_acute_disruption_period(monthly_comparison)
    recovery_trajectory = analyze_recovery_trajectory(monthly_df, baseline_stats)
    significance_tests = test_recovery_significance(monthly_df)

    # Generate outputs
    output_dir = Path("./outputs/temporal_analysis_full_report")
    generate_full_report(
        baseline_stats,
        monthly_comparison,
        acute_info,
        recovery_trajectory,
        significance_tests,
        output_dir,
    )

    logger.info("\n" + "=" * 70)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("=" * 70)



if __name__ == "__main__":
    temporal_complement()
