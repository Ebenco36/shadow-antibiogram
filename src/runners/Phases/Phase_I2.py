# src/runners/analysis_pipeline.py
from __future__ import annotations

"""
================================================================================
--- Analysis Pipeline for Antibiotic Testing Disparities & Trends ---
================================================================================
This script orchestrates a complete, publication-ready analysis from start to
finish.

Analysis Chapters
-----------------
1) Overall Data Context:
   Temporal stability, shocks, breakpoints, decomposition; stratified pathogen
   dynamics across GramType, ARS_WardType, Year, Sex, Hospital level,
   Pediatric vs Elderly.

2) Cross-Sectional Disparities:
   Six high-impact scenarios quantifying disparities with publication tables.

3) Temporal Trends in Testing:
   Trends in testing rates by group, Bundesland stratification, and WHO
   co-testing coverage with antibiotic- and class-level grids.

Outputs
-------
• HTML (interactive), PNG/SVG (publication), optional PDF
• CSVs for all tables + JSON manifests/maps
• Human-readable TXT reports per figure/chapter
================================================================================
"""

import os
import re
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
from src.controllers.TemporalTrends import (
    Fig1VolumeContext,
    Fig2PathogenDynamics,
    Fig3TestingPracticeTrends,
    Fig4CoTestingCoverage,
    ExportSpec,  # for fig4 export_all
)
from src.controllers.summary.Disparity import (
    PublicationStats, filter_ab_dict, preflight, build_class_map, prepare_ward_columns,
    run_all_publication_stats, run_publication_stats_over_time,
    collect_variation_trends, plot_trends_as_bars, WHO_CLASSES
)
from src.controllers.DataLoader import DataLoader
from src.utils.LoadClasses import LoadClasses
from src.utils.helpers import filter_antibiotic_group_items

# ------------------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------------------
DATA_FILE_PATH = "./datasets/output/tables/saved_with_test_indicators_tab.csv"
OUTPUT_ROOT = Path("./outputs/Temporal_Analysis")
OUTPUT_ROOT_TEST = Path("./outputs/")
ANTIBIOTIC_SUFFIX = "_Tested"
DISPARITY_MODELS = ("logit", "log", "rd")  # risk-diff for interpretability
FDR_ALPHA = 0.05
PAGE_SIZE = 30

# Publication export defaults
IMG_WIDTH = 1600
IMG_HEIGHT = 900
IMG_SCALE = 6  # high-res
EXPORT_SVG = True
EXPORT_PDF = False  # requires kaleido PDF support

# WHO priority pathogen regex (used in scenarios)
CRITICAL_PATHOGENS = r"\b(?:Enterobacter|Escherichia|Klebsiella|Citrobacter|Serratia|Proteus|Morganella|Providencia|Acinetobacter baumannii|Mycobacterium)\b"
HIGH_PATHOGENS = r"\b(?:Salmonella Typhi|Shigella spp|Enterococcus faecium|Pseudomonas aeruginosa|Non-typhoidal Salmonella|Neisseria gonorrhoeae|Staphylococcus aureus)\b"

# Suppress noisy warnings in notebooks/CI (you can relax this locally)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _slugify(name: str) -> str:
    s = re.sub(r"\s+", "_", str(name).strip().lower())
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"

def _write_html(fig: go.Figure, path: Path) -> None:
    try:
        path.write_text(fig.to_html(full_html=False, include_plotlyjs="cdn"), encoding="utf-8")
    except Exception as e:
        print(f"  - WARNING: failed to save HTML {path.name}: {e}")

def _export_plotly(
    fig: go.Figure,
    base: Path,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    scale: int = IMG_SCALE,
    write_svg: bool = EXPORT_SVG,
    write_pdf: bool = EXPORT_PDF,
    also_html: bool = True
) -> None:
    """
    High-resolution static export (PNG, optional SVG/PDF) with HTML fallback.
    Never raises – logs and continues.
    """
    if also_html:
        _write_html(fig, base.with_suffix(".html"))
    try:
        fig.write_image(str(base.with_suffix(".png")), width=width, height=height, scale=scale)
    except Exception as e:
        print(f"  - WARNING: PNG export failed for {base.name}. Is 'kaleido' installed? Error: {e}")
    if write_svg:
        try:
            fig.write_image(str(base.with_suffix(".svg")), width=width, height=height, scale=max(1, scale // 2))
        except Exception as e:
            print(f"  - WARNING: SVG export failed for {base.name}. Error: {e}")
    if write_pdf:
        try:
            fig.write_image(str(base.with_suffix(".pdf")), width=width, height=height, scale=max(1, scale // 2))
        except Exception as e:
            print(f"  - WARNING: PDF export failed for {base.name}. Error: {e}")

def _save_table(df: Optional[pd.DataFrame], path: Path) -> None:
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(path, index=False)
    except Exception as e:
        print(f"  - WARNING: could not save table {path.name}: {e}")

def _save_text(text: str, path: Path) -> None:
    try:
        path.write_text(text or "", encoding="utf-8")
    except Exception as e:
        print(f"  - WARNING: could not save text {path.name}: {e}")

def _ensure_year(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    if "Year" not in out.columns:
        if date_col not in out.columns:
            raise ValueError(f"`{date_col}` not found to derive Year.")
        dt = pd.to_datetime(out[date_col], errors="coerce", utc=False)
        out = out[~dt.isna()].copy()
        out[date_col] = dt.dropna()
        out["Year"] = out[date_col].dt.year.astype("Int64")
    return out

def _find_bundesland_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Bundesland", "Region"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _compute_disparities(trends_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Range of mean_testing_rate across groups per antibiotic; ranked desc.
    Robust to small tables; returns empty DataFrame if required cols missing.
    """
    if trends_df.empty:
        return pd.DataFrame()
    need = {"group", "antibiotic", "mean_testing_rate"}
    cols = set(trends_df.columns)
    if not need.issubset(cols):
        return pd.DataFrame()
    tmp = trends_df[["group", "antibiotic", "mean_testing_rate"]].copy()
    agg = (tmp.groupby("antibiotic", observed=True)["mean_testing_rate"]
             .agg(["min", "max", "mean", "count"])
             .rename(columns={"min": "min_rate", "max": "max_rate", "mean": "mean_rate", "count": "n_groups"}))
    agg["disparity_range"] = agg["max_rate"] - agg["min_rate"]
    return agg.sort_values("disparity_range", ascending=False).reset_index()

# ------------------------------------------------------------------------------
# WHO map builder (for Fig 4)
# ------------------------------------------------------------------------------

def build_who_map() -> Dict[str, str]:
    """Map Antibiotic_Tested column -> WHO AWaRe category."""
    loader = LoadClasses()
    who_map: Dict[str, str] = {}
    for cls in WHO_CLASSES:
        abx_names = loader.get_antibiotics_by_category([cls])
        tested_cols = loader.convert_to_tested_columns(abx_names)
        for col in tested_cols:
            who_map[str(col)] = str(cls)
    return who_map

# ------------------------------------------------------------------------------
# FIGURE 2 — stratified runner (Chapter 1)
# ------------------------------------------------------------------------------

def run_fig2_pathogen_dynamics_stratified(
    main_df: pd.DataFrame,
    output_root: Path,
    date_col: str = "Date",
):
    """
    Runs Fig2PathogenDynamics across strata: GramType, ARS_WardType, Year
    """
    df = main_df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = _ensure_year(df, date_col=date_col)

    candidate_cols = [
        "GramType",
        "ARS_WardType",
        "Year",
        "Sex",
        "AgeGroup",
        "Care_Complexity",
        "Hospital_Priority",
        "TextMaterialgroupRkiL0",
        "ARS_HospitalLevelManual"
    ]
    group_cols = [c for c in candidate_cols if c and c in df.columns]

    chapter_root = Path(output_root) / "chapter_1_data_context"
    strat_root = _ensure_dir(chapter_root / "fig2_stratified")

    print("\n" + "=" * 80)
    print("### FIG 2: PATHOGEN GROUP DYNAMICS — STRATIFIED ###")
    print("=" * 80)
    print(f"Strata to analyze: {group_cols}")

    for group_col in group_cols:
        print(f"\n--- Running Fig 2 (stratified) by [{group_col}] ---")
        col_slug = _slugify(group_col)
        col_root = _ensure_dir(strat_root / col_slug)
        pub_dir = _ensure_dir(col_root / "pub_outputs")

        sub = df[~df[group_col].isna()].copy()
        if sub.empty or sub[group_col].nunique(dropna=True) < 1:
            print(f"Skipping [{group_col}] — no data.")
            continue

        analyzer = Fig2PathogenDynamics(
            sub, date_col=date_col, group_col=group_col,
            title=f"Pathogen Dynamics by {group_col}"
        )
        monthly = analyzer.prepare()
        trends = analyzer.analyze()

        fig = analyzer.fig_plotly(show_ma=True, annotate_breaks=True)
        _export_plotly(fig, col_root / f"fig2_{col_slug}_overview",
                       width=1200, height=700, also_html=True)

        # Save CSVs
        _save_table(monthly, col_root / f"{col_slug}_monthly_counts.csv")
        _save_table(trends,  col_root / f"{col_slug}_trends.csv")

        # Per-level comprehensive plots
        print(f"Exporting comprehensive group plots for [{group_col}] …")
        for lvl in sub[group_col].dropna().unique():
            try:
                gfig = analyzer.create_group_comprehensive_plot(group=str(lvl), show_ma=True, model="additive", period=12)
            except Exception:
                gfig = None
            if gfig is None:
                continue
            _export_plotly(gfig, pub_dir / f"{_slugify(f'{group_col}_{lvl}')}_comprehensive",
                           width=1000, height=600, also_html=True)

        # Caption + small report
        cap = ""
        try:
            cap = analyzer.caption() or ""
        except Exception:
            pass
        _save_text(cap, col_root / f"{col_slug}_caption.txt")
        try:
            report = [
                f"# Fig 2 Stratified Report: {group_col}",
                "",
                "## Caption",
                cap or "(no caption)",
                "",
                "## Exports",
                str(pub_dir.resolve()),
            ]
            _save_text("\n".join(report), col_root / f"{col_slug}_report.txt")
            print(f"Saved report for [{group_col}]")
        except Exception as e:
            print(f"  - WARNING: could not write report for [{group_col}]: {e}")

    print(f"\nAll stratified outputs under: {str(strat_root.resolve())}")

# ------------------------------------------------------------------------------
# CHAPTER 1 ORCHESTRATOR
# ------------------------------------------------------------------------------

def run_analysis_chapter_1_context(main_df: pd.DataFrame, output_root: Path, show: bool = False):
    print("\n" + "#" * 80)
    print("### CHAPTER 1: ANALYZING OVERALL DATA CONTEXT & DYNAMICS ###")
    print("#" * 80)

    chapter_root = Path(output_root) / "chapter_1_data_context"
    pub_dir = _ensure_dir(chapter_root / "pub_outputs")
    fig1_dir = _ensure_dir(chapter_root / "fig1_volume_context")
    fig2_dir = _ensure_dir(chapter_root / "fig2_pathogen_dynamics")

    # ---------------- Figure 1: Volume context ----------------
    print("\n--- Running Fig 1: Isolate Volume Analysis ---")
    fig1_base = fig1_dir / "fig1_volume_context"

    vc = Fig1VolumeContext(main_df, date_col="Date", title="Monthly Isolate Volume")
    summary1, fig1, caption1 = vc.analyze_with_mk(
        detect_breaks=True, compute_seasonality=True,
        basepath=str(fig1_base), export=True, show=show, comprehensive_view=True
    )
    print(f"Fig 1 Caption: {caption1}")
    print(f"--> Fig 1 results saved to: {fig1_dir}")

    # Optional decomposition (4 panels)
    try:
        fig_dec = vc.decompose_plot(model="additive", period=12, title="Decomposition — Isolate Volume")
        _export_plotly(fig_dec, fig1_base.with_name("fig1_volume_decomposition"),
                       width=1200, height=800, also_html=True)
    except Exception as e:
        print(f"  - WARNING: could not export decomposition figure: {e}")

    # Diagnostics + tables
    try:
        summary_diag, lb_df, comps = vc.decompose_diagnostics(
            model="additive", period=12, lags=(6, 12), alpha=0.05, export_path=str(fig1_base)
        )
        if isinstance(lb_df, pd.DataFrame) and not lb_df.empty:
            _save_table(lb_df, fig1_base.with_name("fig1_lb_table.csv"))
        if isinstance(comps, pd.DataFrame) and not comps.empty:
            _save_table(comps, fig1_base.with_name("fig1_decomposition_components.csv"))
        elif isinstance(comps, dict):
            (fig1_base.with_name("fig1_decomposition_components.json")
             ).write_text(json.dumps(comps, indent=2, default=str), encoding="utf-8")

        # TXT report
        explained_share = (summary_diag or {}).get("explained_share")
        report = [
            "# Figure 1 — Isolate Volume",
            "",
            "## Caption",
            caption1 or "(no caption)",
            "",
            "## Key Diagnostics",
            f"Explained share (~1 - residual variance ratio): {round(float(explained_share), 3) if explained_share is not None else 'NA'}",
            "",
            "## Exported Files",
            f"- Basepath: {fig1_base}",
            "- Decomposition: fig1_volume_decomposition.png/svg/html",
            "- LB table: fig1_lb_table.csv (if available)",
            "- Decomposition components: fig1_decomposition_components.csv/json (if available)",
        ]
        _save_text("\n".join(report), fig1_dir / "fig1_volume_context_report.txt")
        print(f"--> Fig 1 text report saved.")
    except Exception as e:
        print(f"  - WARNING: diagnostics/report failed for Fig 1: {e}")

    # ---------------- Figure 2: GramType overview ----------------
    print("\n--- Running Fig 2: Pathogen Group Dynamics (by GramType) ---")
    try:
        analyzer = Fig2PathogenDynamics(
            main_df, date_col="Date", group_col="GramType", title="Pathogen Dynamics by Gram Type"
        )
        monthly2 = analyzer.prepare()
        trends2  = analyzer.analyze()
        fig2 = analyzer.fig_plotly(show_ma=True, annotate_breaks=True)
        analyzer.export(basepath=str(fig2_dir / "fig2_pathogen_dynamics"), fig=fig2)
        cap2 = analyzer.caption()
        _save_text(cap2, fig2_dir / "fig2_caption.txt")
        print(f"Fig 2 Caption: {cap2}")
        print(f"--> Fig 2 results saved to: {fig2_dir}")
        _save_table(monthly2, fig2_dir / "fig2_monthly_counts_by_group.csv")
        _save_table(trends2,  fig2_dir / "fig2_trends.csv")
    except Exception as e:
        print(f"  - WARNING: Fig 2 GramType run failed: {e}")

    # Chapter-level TXT report
    try:
        lines = [
            "# Chapter 1: Data Context & Dynamics",
            "",
            "## Figure 1 Caption",
            caption1 or "(no caption)",
            "",
            "## Figure 2 Caption",
            cap2 if 'cap2' in locals() else "(unavailable)",
            "",
            "## Exported Figures Directory",
            str(pub_dir.resolve()),
        ]
        _save_text("\n".join(lines), chapter_root / "chapter_1_report.txt")
        _save_text(caption1 or "", chapter_root / "fig1_caption.txt")
        if 'cap2' in locals():
            _save_text(cap2 or "", chapter_root / "fig2_caption.txt")
        print(f"\n--> Chapter 1 TXT report saved.")
    except Exception as e:
        print(f"  - WARNING: could not write Chapter 1 TXT report: {e}")

    # Stratified Fig 2 runs
    run_fig2_pathogen_dynamics_stratified(main_df=main_df, output_root=output_root, date_col="Date")

    print(f"\nAll Chapter 1 outputs under: {chapter_root.resolve()}")

# ------------------------------------------------------------------------------
# CHAPTER 2 — Cross-sectional disparity scenarios
# ------------------------------------------------------------------------------

def run_scenario_icu_vs_ward(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "=" * 80)
    print("--> SCENARIO 2.1: ICU vs. WARD FOR BLOODSTREAM INFECTIONS")
    print("=" * 80)
    scenario_root = _ensure_dir(output_root / "1_icu_vs_ward_bloodstream")

    # Debug: Check data availability at each filtering step
    debug_info = []
    debug_info.append(f"Initial dataset: {len(main_df)} records")
    
    # Step 1: Pathogen filter
    pathogen_group_regex = f"{CRITICAL_PATHOGENS}|{HIGH_PATHOGENS}"
    pathogen_mask = main_df['Pathogen'].astype(str).str.contains(pathogen_group_regex, na=False)
    df_step1 = main_df[pathogen_mask].copy()
    debug_info.append(f"After pathogen filter: {len(df_step1)} records")
    debug_info.append(f"Pathogens found: {df_step1['Pathogen'].unique()[:10]}")  # Show first 10
    
    # Step 2: Blood Culture filter
    blood_mask = df_step1['TextMaterialgroupRkiL0'] == 'Blood Culture'
    df_step2 = df_step1[blood_mask].copy()
    debug_info.append(f"After blood culture filter: {len(df_step2)} records")
    debug_info.append(f"Material groups available: {df_step1['TextMaterialgroupRkiL0'].unique()}")
    
    # Step 3: Care type filter
    care_mask = df_step2['CareType'] == 'In-Patient'
    df_step3 = df_step2[care_mask].copy()
    debug_info.append(f"After stationary care filter: {len(df_step3)} records")
    debug_info.append(f"Care types available: {df_step2['CareType'].unique()}")
    
    # Step 4: Ward columns preparation and filter
    if len(df_step3) > 0:
        df_filtered = prepare_ward_columns(df_step3)
        ward_mask = df_filtered["WardGroup"].isin(['ICU', 'Normal Ward'])
        df_filtered = df_filtered[ward_mask].copy()
        debug_info.append(f"After ward group filter: {len(df_filtered)} records")
        debug_info.append(f"Ward groups available: {df_filtered['WardGroup'].unique() if len(df_filtered) > 0 else 'None'}")
    else:
        df_filtered = pd.DataFrame()
        debug_info.append("Skipping ward filter - no data from previous steps")

    print(f"    Data filtered. Running analysis on {len(df_filtered)} records.")
    
    # Save debug information
    _save_text("\n".join(debug_info), scenario_root / "filtering_debug_info.txt")
    
    # GUARD CLAUSE: Skip analysis if no data
    if len(df_filtered) == 0:
        print("    ⚠️  SKIPPING: No data after filtering. Check filtering_debug_info.txt for details.")
        
        # Create a comprehensive diagnostic report
        diagnostic_report = [
            "ICU vs WARD BLOODSTREAM INFECTION ANALYSIS - DIAGNOSTIC REPORT",
            "=" * 60,
            "",
            "FILTERING STEPS:",
            f"1. Initial dataset: {len(main_df)} records",
            f"2. Critical/High priority pathogens: {len(df_step1)} records",
            f"3. Blood Culture material: {len(df_step2)} records", 
            f"4. Stationary care type: {len(df_step3)} records",
            f"5. ICU/Normal Ward groups: {len(df_filtered)} records",
            "",
            "DATA AVAILABILITY:",
            f"Unique pathogens in data: {main_df['Pathogen'].nunique()}",
            f"Sample pathogens: {list(main_df['Pathogen'].dropna().unique()[:10])}",
            f"Material groups: {list(main_df['TextMaterialgroupRkiL0'].dropna().unique())}",
            f"Care types: {list(main_df['CareType'].dropna().unique())}",
            "",
            "RECOMMENDATIONS:",
            "1. Check if your pathogen names match the regex pattern",
            "2. Verify that 'Blood Culture' exists in TextMaterialgroupRkiL0",
            "3. Ensure there are stationary care records",
            "4. Check if WardGroup column has ICU/Normal Ward values after prepare_ward_columns",
        ]
        
        _save_text("\n".join(diagnostic_report), scenario_root / "scenario_diagnostic_report.txt")
        return
    
    # If we have data, proceed with analysis
    preflight(df_filtered, test_suffix=ANTIBIOTIC_SUFFIX)

    pub = PublicationStats.from_wide(
        df=df_filtered, stratum_col="ARS_HospitalLevelManual", compare_col="WardGroup", antibiotic_suffix=ANTIBIOTIC_SUFFIX
    ).with_class_map(class_map)

    pub.save_all_everything(
        out_dir=str(scenario_root / "comprehensive_analysis"),
        disparity_models=DISPARITY_MODELS, page_size=PAGE_SIZE, classes=list(WHO_CLASSES)
    )

    # Over-time stats (with guard clause)
    time_series_root = _ensure_dir(scenario_root / "time_series_analysis")
    run_publication_stats_over_time(
        df=df_filtered, out_root=time_series_root, stratum_col="ARS_HospitalLevelManual", compare_col="WardGroup",
        antibiotic_suffix=ANTIBIOTIC_SUFFIX, class_map=class_map, who_classes=list(WHO_CLASSES)
    )
    
    # Only try to plot if we have variation trends data
    try:
        var_trends = collect_variation_trends(time_series_root)
        if not var_trends.empty:
            plot_trends_as_bars(var_trends, metric="gini").savefig(
                time_series_root / "trend_gini.png", dpi=300, bbox_inches="tight"
            )
            plot_trends_as_bars(var_trends, metric="median_pct").savefig(
                time_series_root / "trend_median_coverage.png", dpi=300, bbox_inches="tight"
            )
    except Exception as e:
        print(f"    Note: Could not generate trend plots: {e}")
    
    print(f"    Scenario 2.1 complete. Results saved to: {scenario_root}")

def run_scenario_hospital_level(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "=" * 80)
    print("--> SCENARIO 2.2: SPECIALIZED vs. BASIC CARE HOSPITALS")
    print("=" * 80)
    scenario_root = _ensure_dir(output_root / "2_specialized_vs_basic_care")

    pathogen_group_regex = f"{CRITICAL_PATHOGENS}|{HIGH_PATHOGENS}"
    df_filtered = main_df[main_df['Pathogen'].astype(str).str.contains(pathogen_group_regex, na=False)].copy()
    df_filtered = df_filtered[df_filtered['TextMaterialgroupRkiL0'].isin(["Blood Culture", "Urine"])].copy()
    df_filtered = df_filtered[df_filtered['CareType'] == 'In-Patient'].copy()

    df_filtered['hospital_level_group'] = np.where(
        df_filtered['ARS_HospitalLevelManual'].astype('string').str.contains(r'\bLevel\s*[3-7]\b', na=False),
        'Specialized Care', 'Basic Care'
    )

    print(f"    Data prepared. Running analysis on {len(df_filtered)} records.")
    preflight(df_filtered, test_suffix=ANTIBIOTIC_SUFFIX)

    pub = PublicationStats.from_wide(
        df=df_filtered, stratum_col="ARS_HospitalLevelManual", compare_col="hospital_level_group", antibiotic_suffix=ANTIBIOTIC_SUFFIX
    ).with_class_map(class_map)

    pub.save_all_everything(
        out_dir=str(scenario_root), disparity_models=DISPARITY_MODELS, classes=list(WHO_CLASSES)
    )
    print(f"    Scenario 2.2 complete. Results saved to: {scenario_root}")

def run_scenario_pediatric_vs_elderly(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "=" * 80)
    print("--> SCENARIO 2.3: PEDIATRIC vs. ELDERLY CARE")
    print("=" * 80)
    scenario_root = _ensure_dir(output_root / "3_pediatric_vs_elderly")

    df_filtered = main_df[main_df['CareType'] == 'In-Patient'].copy()
    pediatric_groups = ['0 years', '1-4 years', '5-9 years', '10-14 years']
    elderly_groups = ['70-74 years', '75-79 years', '80-84 years', '85-89 years', '90-94 years', '≥95 years']

    df_filtered['age_category'] = np.nan
    df_filtered.loc[df_filtered['AgeRange'].isin(pediatric_groups), 'age_category'] = 'Pediatric'
    df_filtered.loc[df_filtered['AgeRange'].isin(elderly_groups), 'age_category'] = 'Elderly'
    df_filtered.dropna(subset=['age_category'], inplace=True)

    print(f"    Data prepared. Running analysis on {len(df_filtered)} records.")
    preflight(df_filtered, test_suffix=ANTIBIOTIC_SUFFIX)

    pub = PublicationStats.from_wide(
        df=df_filtered, stratum_col="ARS_HospitalLevelManual", compare_col="age_category", antibiotic_suffix=ANTIBIOTIC_SUFFIX
    ).with_class_map(class_map)

    pub.save_all_everything(
        out_dir=str(scenario_root), disparity_models=DISPARITY_MODELS, classes=list(WHO_CLASSES)
    )
    print(f"    Scenario 2.3 complete. Results saved to: {scenario_root}")


def run_scenario_bsi_vs_uti_tiered(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "=" * 80)
    print("--> SCENARIO 2.5: TIERED BSI vs. UTI ANALYSIS BY PATHOGEN")
    print("=" * 80)
    base_df = main_df[
        main_df['TextMaterialgroupRkiL0'].isin(['Blood Culture', 'Urine']) &
        (main_df['CareType'] == 'In-Patient')
    ].copy()

    analysis_targets = {
        "pathogen_Escherichia_coli": ("Pathogen", "Escherichia coli"),
        "pathogen_Staphylococcus_aureus": ("Pathogen", "Staphylococcus aureus"),
        "pathogen_Pseudomonas_aeruginosa": ("Pathogen", "Pseudomonas aeruginosa"),
    }

    for target_name, (filter_col, filter_value) in analysis_targets.items():
        scenario_root = _ensure_dir(output_root / "5_bsi_vs_uti_tiered" / target_name)
        print(f"\n--- Running tiered analysis for: {target_name} ---")

        df_filtered = base_df[base_df[filter_col] == filter_value].copy()
        if len(df_filtered) < 100:
            print(f"    Insufficient data ({len(df_filtered)} records). Skipping.")
            continue

        print(f"    Data prepared. Running analysis on {len(df_filtered)} records.")
        preflight(df_filtered, test_suffix=ANTIBIOTIC_SUFFIX)
        pub = PublicationStats.from_wide(
            df=df_filtered, stratum_col="ARS_HospitalLevelManual", compare_col="TextMaterialgroupRkiL0",
            antibiotic_suffix=ANTIBIOTIC_SUFFIX
        ).with_class_map(class_map)
        pub.save_all_everything(
            out_dir=str(scenario_root), disparity_models=DISPARITY_MODELS, classes=list(WHO_CLASSES)
        )
        print(f"    Analysis for {target_name} saved to: {scenario_root}")
    print("    Scenario 2.5 complete.")

def run_scenario_general_comparisons(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "=" * 80)
    print("--> SCENARIO 2.6: GENERAL DEMOGRAPHIC & CLINICAL COMPARISONS")
    print("=" * 80)

    scenario_root = _ensure_dir(output_root / "6_general_comparisons")
    df_filtered = main_df[main_df['CareType'] == 'In-Patient'].copy()

    print("    Running a broad set of predefined general comparisons...")
    run_all_publication_stats(
        df=df_filtered, out_root=scenario_root, stratum_col="ARS_HospitalLevelManual",
        antibiotic_suffix=ANTIBIOTIC_SUFFIX, class_map=class_map, who_classes=WHO_CLASSES,
        disparity_models=DISPARITY_MODELS, page_size=PAGE_SIZE
    )
    print(f"    Scenario 2.6 complete. Results saved to: {scenario_root}")

def run_analysis_chapter_2_disparities(main_df: pd.DataFrame, class_map: dict, output_root: Path):
    print("\n" + "#" * 80)
    print("### CHAPTER 2: INVESTIGATING CROSS-SECTIONAL DISPARITIES ###")
    print("#" * 80)

    chapter_root = _ensure_dir(output_root / "chapter_2_disparity_scenarios")
    run_scenario_icu_vs_ward(main_df, class_map, chapter_root)
    run_scenario_hospital_level(main_df, class_map, chapter_root)
    run_scenario_pediatric_vs_elderly(main_df, class_map, chapter_root)
    run_scenario_bsi_vs_uti_tiered(main_df, class_map, chapter_root)
    run_scenario_general_comparisons(main_df, class_map, chapter_root)

# ------------------------------------------------------------------------------
# CHAPTER 3 — Temporal testing trends + WHO coverage
# ------------------------------------------------------------------------------

def run_analysis_chapter_3_testing_trends(
    main_df: pd.DataFrame,
    class_map: Dict,
    who_map: Dict,
    output_root: Path,
    *,
    show: bool = False,
    first_n_to_plot: int = 70
):
    print("\n" + "#" * 80)
    print("### CHAPTER 3: ANALYZING TEMPORAL TRENDS IN TESTING PRACTICES ###")
    print("#" * 80)

    chapter_root = Path(output_root) / "chapter_3_testing_trends"
    fig3_care_dir = _ensure_dir(chapter_root / "fig3_testing_practice_trends_caretype")
    fig3_state_dir = _ensure_dir(chapter_root / "fig3_testing_practice_trends_bundesland")
    fig4_dir = _ensure_dir(chapter_root / "fig4_cotesting_coverage")

    # Use only antibiotic cols present in df
    all_abx_cols = sorted({col for cols in class_map.values() for col in cols if col in main_df.columns})
    if not all_abx_cols:
        print("Warning: No antibiotic *_Tested columns found. Skipping Chapter 3.")
        return

    def _run_trends_for_group(group_col: str, out_dir: Path, df: pd.DataFrame, title_prefix: str):
        print(f"\n--- Running {title_prefix}: Testing Practice Trends (by {group_col}) ---")
        analyzer = Fig3TestingPracticeTrends(
            df=df, date_col="Date", group_col=group_col,
            antibiotic_cols=all_abx_cols, title=f"AST Testing by {group_col}"
        )
        monthly = analyzer.prepare()
        trends_df = analyzer.analyze_trends()
        if trends_df.empty:
            print(f"Warning: Trend analysis for {group_col} produced no results.")
            return

        _save_table(monthly,   out_dir / f"fig3_monthly_{_slugify(group_col)}.csv")
        _save_table(trends_df, out_dir / f"fig3_testing_trends_{_slugify(group_col)}.csv")
        print(f"--> Trend analysis saved for {group_col}.")

        # Per-antibiotic interactive + static plots
        plots_html = _ensure_dir(out_dir / "plots_html")
        plots_png  = _ensure_dir(out_dir / "plots_png")
        plots_svg  = _ensure_dir(out_dir / "plots_svg")

        for abx in all_abx_cols[:max(0, int(first_n_to_plot))]:
            try:
                fig = analyzer.fig_plotly(antibiotic=abx)
                if fig is None:
                    continue
                if show:
                    try: fig.show()
                    except Exception: pass
                safe = _slugify(abx.replace("_Tested", ""))
                _write_html(fig, plots_html / f"testing_rate_{safe}.html")
                _export_plotly(fig, plots_png / f"testing_rate_{safe}",
                               width=1200, height=700, scale=4, write_svg=True, also_html=False)
                # Duplicate SVG to its folder
                try:
                    (plots_svg / f"testing_rate_{safe}.svg").write_bytes(
                        (plots_png / f"testing_rate_{safe}.svg").read_bytes()
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"  - WARNING for {abx}: plotting/export failed: {e}")

        # Disparity summary
        disparities = _compute_disparities(trends_df, group_col)
        if not disparities.empty:
            try:
                disp_path = out_dir / f"fig3_biggest_disparities_{_slugify(group_col)}.csv"
                disparities.to_csv(disp_path, index=False)
                report = [
                    f"# Fig 3 Report: {group_col}",
                    "",
                    f"Antibiotics plotted: {min(first_n_to_plot, len(all_abx_cols))}",
                    f"Trends CSV: fig3_testing_trends_{_slugify(group_col)}.csv",
                    f"Disparities CSV: {disp_path.name}",
                ]
                _save_text("\n".join(report), out_dir / f"fig3_report_{_slugify(group_col)}.txt")
                print(f"--> Disparity analysis and report saved for {group_col}.")
            except Exception as e:
                print(f"  - WARNING: could not save disparities/report for {group_col}: {e}")

    # Fig 3A: CareType (if present)
    if "CareType" in main_df.columns:
        _run_trends_for_group("CareType", fig3_care_dir, main_df, "Fig 3A")
    else:
        print("\n--- Skipping Fig 3A: No 'CareType' column found. ---")

    # Fig 3B: Bundesland (heuristic)
    bl_col = _find_bundesland_col(main_df)
    if bl_col:
        _run_trends_for_group(bl_col, fig3_state_dir, main_df, "Fig 3B")
    else:
        print("\n--- Skipping Fig 3B: No 'Bundesland' or equivalent column found. ---")

    # Fig 4: Co-testing coverage (WHO)
    print("\n--- Running Fig 4: Co-Testing Coverage Analysis ---")
    try:
        coverage = Fig4CoTestingCoverage(
            df=main_df,
            class_map=class_map,
            focus_classes=[
                "Carbapenem (β-lactam)",
                "Fluoroquinolone",
                "Third-gen cephalosporin (β-lactam)",
                "Fourth-gen cephalosporin (β-lactam)",
                "Macrolide",
                "Glycopeptide",
            ],
            date_col="Date",
            title="Coverage of Key Antibiotic Classes"
        )
        m_class = coverage.prepare()
        m_abx   = coverage.prepare_antibiotic_level(who_map=who_map)
        m_who   = coverage.prepare_who_level(who_map=who_map)
        _save_table(m_class, fig4_dir / "fig4_coverage_by_class.csv")
        _save_table(m_abx,   fig4_dir / "fig4_coverage_by_antibiotic.csv")
        _save_table(m_who,   fig4_dir / "fig4_coverage_by_who.csv")

        manifest = coverage.export_all(
            who_map_used=who_map,
            export=ExportSpec(out_dir=str(fig4_dir), basename="fig4_coverage",
                              image_scale=IMG_SCALE, image_width=IMG_WIDTH, image_height=900,
                              export_svg=EXPORT_SVG, export_pdf=EXPORT_PDF),
            fig_who_kwargs={"top_n": 10}
        )
        _save_text(json.dumps(manifest, indent=2), fig4_dir / "fig4_export_manifest.json")
        _save_text(coverage.caption(), fig4_dir / "fig4_caption.txt")
        print("--> Fig 4 coverage analysis and plots saved.")
    except Exception as e:
        print(f"  - WARNING: Fig 4 coverage analysis failed: {e}")

    print(f"\nAll Chapter 3 outputs under: {chapter_root.resolve()}")

# ------------------------------------------------------------------------------
# Master Orchestrator
# ------------------------------------------------------------------------------

def run_phase_I2(main_df: pd.DataFrame, show: bool = False):
    """
    Main function to orchestrate the entire analysis pipeline (Chapters 1–3).
    Expects a tidy DataFrame with at least a temporal column from which 'Date' can be created.
    """
    _ensure_dir(OUTPUT_ROOT)

    # Build/confirm Date column robustly
    if "Date" not in main_df.columns:
        # Try common alternatives
        candidates = ["YearMonth", "SampleMonth", "CollectionMonth"]
        found = None
        for c in candidates:
            if c in main_df.columns:
                found = c; break
        if not found:
            raise KeyError("No 'Date' column present and no fallback monthly column found.")
        main_df = main_df.copy()
        main_df["Date"] = pd.to_datetime(main_df[found], errors="coerce", utc=False)

    main_df["Date"] = pd.to_datetime(main_df["Date"], errors="coerce", utc=False)
    main_df = main_df.dropna(subset=["Date"]).sort_values("Date")

    print("--- Building WHO AWaRe class and WHO maps... ---")
    class_map = build_class_map()
    who_map = build_who_map()

    # Optional: external antibiotic grouping (ensure compatibility)
    try:
        with open("./datasets/antibiotic_class_grouping.json", "r") as f:
            loaded_antibiotic_classes = json.load(f)
        kept, missing = filter_antibiotic_group_items(main_df, loaded_antibiotic_classes)
        if missing:
            miss_p = OUTPUT_ROOT / "missing_antibiotics_in_data.json"
            _save_text(json.dumps({"missing_in_df": missing}, indent=2), miss_p)
    except Exception:
        kept = class_map  # fallback to build_class_map()

    # ---------------- Run Chapters ----------------
    run_analysis_chapter_1_context(main_df, OUTPUT_ROOT, show=show)
    # run_analysis_chapter_2_disparities(main_df, class_map, OUTPUT_ROOT_TEST)
    run_analysis_chapter_3_testing_trends(main_df, kept, who_map, OUTPUT_ROOT, show=show, first_n_to_plot=70)

    print("\n" + "=" * 80)
    print("MASTER ANALYSIS PIPELINE COMPLETE!")
    print(f"All results have been saved to: {OUTPUT_ROOT.resolve()}")
    print("=" * 80)


# if __name__ == "__main__":
#     from src.controllers.DataLoader import DataLoader
#     loader = DataLoader("./datasets/WHO_Aware_data")
#     df_combined = loader.get_combined()
#     main_df = df_combined
#     run_phase_I2(main_df=main_df)