"""
================================================================================
--- Unified Visual Analytics Pipeline for Antibiotic Testing Coverage ---
================================================================================
This script provides a structured, class-based, and pipeline-driven approach 
to generating a comprehensive suite of publication-quality visualizations for 
analyzing antibiotic testing coverage. It combines a robust class structure 
with a declarative analysis pipeline for maximum clarity and flexibility.

Date: 26. September 2025
Location: Berlin, Germany
================================================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import altair as alt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ==============================================================================
# 0. IMPORTS
# ==============================================================================
from shadow_antibiogram.controllers.summary.AntibioticCoverageSummary import AntibioticCoverageSummary
from shadow_antibiogram.controllers.AMRMultiAnalyzer import AMRMultiAnalyzer
from shadow_antibiogram.utils.helpers import plot_clustergram_with_dendrograms, plot_tests_boxplot
from shadow_antibiogram.controllers.interfaces.data import (
    PctMetric, SuffixAntibioticDetector, DefaultAntibioticNameParser
)
from shadow_antibiogram.controllers.StratifiedFilter import StratifiedFilter
from shadow_antibiogram.utils.LoadClasses import LoadClasses

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class AnalysisConfig:
    """Holds all configuration for the visual analytics pipeline."""
    # --- File Output Settings ---
    output_dir: str
    image_formats: List[str] = field(default_factory=lambda: ["html", "png", "svg", "pdf"])
    image_scale: int = 3
    
    # --- The Declarative Analysis Pipeline ---
    analysis_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    # General list of stratifiers used by multiple chapters
    stratifier_candidates: List[str] = field(default_factory=lambda: [
        "Bundesland", "CareType", "ARS_WardType", "Pediatric_vs_Elderly", "Sex",
        "TextMaterialgroupRkiL0", "GramType", "ARS_HospitalLevelManual", "Anonymized_Lab", 
        "PathogenGenus", "Year", "BroadAgeGroup", "ARS_Region"
    ])
    # Specific scenarios for the layered boxplot analysis
    boxplot_comparison_scenarios: List[Dict] = field(default_factory=lambda: [
        {"compare_col": "Year", "values": None},
        {"compare_col": "CareType", "values": ["In-Patient", "Out-Patient"]},
        {"compare_col": "Sex", "values": ["Man", "Woman", "Others"]},
        {"compare_col": "ARS_WardType", "values": ["Intensive Care Unit", "Normal Ward"]},
        {"compare_col": "Pediatric_vs_Elderly", "values": ["Pediatric", "Elderly"]}
    ])
    # Columns to use for faceting in the bar chart comparisons
    barchart_comparison_cols: List[str] = field(default_factory=lambda: ["Sex", "CareType"])

# ==============================================================================
# 2. THE CORE PIPELINE ORCHESTRATOR CLASS
# ==============================================================================

class VisualAnalyticsPipeline:
    """
    Orchestrates the generation of a complete suite of visualizations based on a 
    declarative analysis pipeline.
    """
    def __init__(self, df: pd.DataFrame, config: AnalysisConfig):
        self.df = df.copy()
        self.config = config
        self.loader = LoadClasses()
        
        # --- Setup AWaRe Classifications ---
        self.aware_lists = self._build_aware_map()
        self.abx_to_class = {c: cls for cls, cols in self.aware_lists.items() for c in cols}
        self.all_aware_antibiotics = sorted(list(self.abx_to_class.keys()))
        self.class_palette = {"Access": "#56B4E9", "Watch": "#E69F00", "Reserve": "#D55E00"}
        
        # --- Instantiate Analyzer for Clustergrams ---
        self.amr_analyzer = AMRMultiAnalyzer(
            self.df, SuffixAntibioticDetector("_Tested"), DefaultAntibioticNameParser("_Tested")
        )
        os.makedirs(self.config.output_dir, exist_ok=True)
        print("VisualAnalyticsPipeline initialized successfully.")

    def _build_aware_map(self) -> Dict[str, List[str]]:
        """Builds a dictionary of AWaRe classes and their antibiotic columns."""
        aware_map = {"All": []}
        for aclass in ["Access", "Watch", "Reserve"]:
            abx = self.loader.get_antibiotics_by_category([aclass])
            cols = self.loader.convert_to_tested_columns(abx)
            present_cols = [c for c in cols if c in self.df.columns]
            if present_cols:
                aware_map[aclass] = present_cols
                aware_map["All"].extend(present_cols)
        aware_map["All"] = sorted(list(set(aware_map["All"])))
        return aware_map

    def _safe_filename(self, title: str) -> str:
        """Creates a safe filename from a plot title."""
        s = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
        return re.sub(r'\s+', '_', s).lower()

    def _safe_name(self, s: str) -> str:
        s = re.sub(r"\s+", "_", str(s).strip())
        return re.sub(r"[^A-Za-z0-9_.-]", "", s)


    def _save_viz(self, fig: object, base_path: Path):
        """Saves a visualization object, handling Plotly and Altair, including PDF export."""
        
        base_path.parent.mkdir(parents=True, exist_ok=True)
        # ---------------------------
        # Plotly Figures
        # ---------------------------
        if hasattr(fig, "write_html"):

            for fmt in self.config.image_formats:
                out_path = base_path.with_suffix(f".{fmt}")
                try:
                    if fmt == "html":
                        fig.write_html(str(out_path))
                    else:
                        # PNG, SVG, PDF via kaleido
                        fig.write_image(
                            str(out_path),
                            scale=self.config.image_scale
                        )
                except Exception as e:
                    print(
                        f"[Warning] Failed to save Plotly figure as '{fmt}' "
                        f"for '{base_path.name}': {e}"
                    )
                    if fmt in {"png", "svg", "pdf"}:
                        print(
                            "          -> Ensure Kaleido is installed: "
                            "pip install -U kaleido"
                        )
                        
        elif isinstance(fig, alt.TopLevelMixin):
            for fmt in self.config.image_formats:
                out_path = base_path.with_suffix(f".{fmt}")
                try:
                    fig.save(str(out_path))
                except Exception as e:
                    print(
                        f"[Warning] Failed to save Altair chart as '{fmt}' "
                        f"for '{base_path.name}': {e}"
                    )

                    if fmt in {"png", "svg", "pdf"}:
                        print(
                            "          -> Ensure you have one of the following installed:\n"
                            "             pip install altair_saver vl-convert-python\n"
                            "             OR\n"
                            "             pip install pyppeteer"
                        )
        else:
            print(
                f"[Error] Could not save figure '{base_path.name}': "
                f"unsupported type '{type(fig).__name__}'."
            )


    def _generate_clustergram(self, dataframe: pd.DataFrame, title: str, output_path: Path):
        """Generates and saves a clustergram."""
        if dataframe.empty:
            print(f"Skipping clustergram '{title}' due to empty data.")
            return
        fig = self.amr_analyzer.clustergram(
            ["Pathogen"], PctMetric(), plot_func=plot_clustergram_with_dendrograms, df=dataframe,
            plot_kwargs=dict(
                title=title, interactive=True, cell_height=12, row_panel_width=200,
                heatmap_height=1200, heatmap_width=1300, row_metric='correlation', col_metric='euclidean'
            )
        )
        self._save_viz(fig, output_path)
        print(f"   -> Saved: {output_path.name}")


    def _generate_boxplot(self, dataframe: pd.DataFrame, compare_by: str, antibiotics: List[str], title: str, output_path: Path):
        """Generates and saves a comparative boxplot."""
        if dataframe.empty:
            print(f"Skipping boxplot '{title}' due to empty data.")
            return
        
        df_plot = dataframe.copy()

        if pd.api.types.is_categorical_dtype(df_plot[compare_by]):
            df_plot[compare_by] = df_plot[compare_by].cat.remove_unused_categories()
    
        fig = plot_tests_boxplot(
            test_data_df=df_plot, group_col="PathogenGenus", compare_col=compare_by,
            antibiotics_to_plot=antibiotics, antibiotic_class_map=self.abx_to_class,
            class_palette=self.class_palette, sort_stat="q3", height=650,
            width=max(1200, 40 * len(antibiotics)), title=title,
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
            show_threshold_line=False, export_csv_path=output_path.with_suffix(".csv")
        )
        self._save_viz(fig, output_path)
        print(f"   -> Saved: {output_path.name}")

    def run(self):
        """Executes all tasks defined in the analysis pipeline configuration."""
        print(f"\nStarting execution of {len(self.config.analysis_pipeline)} analysis tasks...")
        output_root = Path(self.config.output_dir)

        for i, task in enumerate(self.config.analysis_pipeline):
            task_type = task.get("type")
            section_folder = task.get("section", "miscellaneous")
            title_template = task.get("title_template", task.get("title", "Untitled_Analysis"))
            print(f"\n--- Running Task {i+1}/{len(self.config.analysis_pipeline)}: {title_template.format('*')} ({section_folder}) ---")
            
            # --- Handle Filtering ---
            filtered_df = self.df.copy()
            if "filter_on" in task:
                for f in task["filter_on"]:
                    if f["column"] in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[f["column"]].isin(f["values"])]
            
            # --- Execute Plotting ---
            if task_type == "clustergram":
                if "filter_by" in task:
                    for category in task["categories"]:
                        category_df = filtered_df[filtered_df[task["filter_by"]] == category]
                        title = title_template.format(category)
                        output_path = output_root / section_folder / self._safe_filename(title)
                        self._generate_clustergram(category_df, title, output_path)
                else: # Global plot
                    output_path = output_root / section_folder / self._safe_filename(title_template)
                    self._generate_clustergram(filtered_df, title_template, output_path)

            elif task_type == "boxplot":
                comparison_variable = task.get("compare_by")
                
                # --- 1. Collect all antibiotics for the single plot ---
                antibiotics_to_plot = []
                if "All" in task["aware_classes"]:
                    antibiotics_to_plot = self.all_aware_antibiotics
                else:
                    temp_list = []
                    for aware_class in task["aware_classes"]:
                        temp_list.extend(self.aware_lists.get(aware_class, []))
                    # Remove duplicates and sort for a consistent order
                    antibiotics_to_plot = sorted(list(set(temp_list)))

                if not antibiotics_to_plot:
                    continue # Skip if no antibiotics were found for the given classes

                # --- 2. Create a single, combined title ---
                class_names_str = " & ".join(task["aware_classes"])
                title = task["title_template"].format(class_names_str)
                output_path = output_root / section_folder / self._safe_filename(title)
                # Ensure directory exists for both CSV and figures
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # --- 3. Generate one plot with the combined list ---
                self._generate_boxplot(
                    filtered_df, 
                    comparison_variable, 
                    antibiotics_to_plot, 
                    title, 
                    output_path
                )

        print("\nVisual analytics pipeline complete!")

    def _generate_and_save_boxplot(self, df_subset: pd.DataFrame, strat_col: str, 
                                   antibiotics: List[str], aware_class_name: str, 
                                   output_path: Path, compare_col: Optional[str] = None):
        """Central plotting function to generate and save a single boxplot figure."""
        if df_subset.empty or strat_col not in df_subset.columns or df_subset[strat_col].nunique() < 2:
            return
        title = f"{aware_class_name} Coverage Across {strat_col}" + (f" by {compare_col}" if compare_col else "")
        abx_to_class_map = {col: aw_class for aw_class, cols in self.aware_lists.items() for col in cols if aw_class != "All"}
        threshold_line = None if abx_to_class_map else True
        try:
            fig = plot_tests_boxplot(
                test_data_df=df_subset, group_col=strat_col, compare_col=compare_col,
                antibiotics_to_plot=antibiotics, antibiotic_class_map=abx_to_class_map,
                class_palette=self.class_palette, width=max(1500, 30 * len(antibiotics)),
                height=900 if compare_col else 750, title=title, threshold_line=threshold_line,
                legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
            )
            self._save_viz(fig, output_path)
            print(f"    -> Saved: {output_path.name}.{self.config.image_formats[0]}")
        except Exception as e:
            print(f"[Error] Failed to generate plot for {output_path.name}: {e}")

    def run_per_antibiotic_boxplots(self):
        """CHAPTER 1: Runs the full layered boxplot analysis."""
        print("\n--- Running Chapter 1: Per-Antibiotic Coverage Boxplots ---")
        chapter_dir = Path(self.config.output_dir) / "WHO_AWaRe_on_Coverage"
        
        # Phase 1: Baseline Analysis
        print("  Phase 1: Generating baseline variation plots...")
        for strat_col in self.config.stratifier_candidates:
            if strat_col not in self.df.columns: continue
            for aware_class, abx_list in self.aware_lists.items():
                if not abx_list: continue
                folder_name = "all_aware" if aware_class == "All" else f"by_class_{aware_class.lower()}"
                output_dir = chapter_dir / "baseline" / folder_name
                self._generate_and_save_boxplot(
                    self.df, strat_col, abx_list, f"{aware_class} Class",
                    output_dir / f"{aware_class.lower()}_by_{self._safe_name(strat_col)}"
                )
        
        # Phase 2: Comparative Analysis
        print("  Phase 2: Generating comparative variation plots...")
        for scenario in self.config.boxplot_comparison_scenarios:
            compare_col, values = scenario["compare_col"], scenario["values"]
            if compare_col not in self.df.columns: continue
            df_filtered = self.df[self.df[compare_col].isin(values)] if values else self.df.copy()

            for strat_col in self.config.stratifier_candidates:
                if strat_col not in df_filtered.columns or strat_col == compare_col: continue
                for aware_class, abx_list in self.aware_lists.items():
                    if not abx_list: continue
                    folder_name = "all_aware" if aware_class == "All" else f"by_class_{aware_class.lower()}"
                    output_dir = chapter_dir / "comparative" / folder_name
                    self._generate_and_save_boxplot(
                        df_filtered, strat_col, abx_list, f"{aware_class} Class",
                        output_dir / f"{aware_class.lower()}_by_{self._safe_name(strat_col)}_vs_{self._safe_name(compare_col)}",
                        compare_col=compare_col
                    )
    
    def run_group_comparison_barcharts(self):
        """CHAPTER 2: Generates bar charts comparing mean coverage for AWaRe classes."""
        print("\n--- Running Chapter 2: Group Comparison Bar Charts ---")
        chapter_dir = Path(self.config.output_dir) / "2_group_comparisons"
        
        summary_analyzer = AntibioticCoverageSummary(df=self.df)
        for strat_col in self.config.stratifier_candidates:
            if strat_col not in self.df.columns: continue
            for compare_col in self.config.barchart_comparison_cols:
                if compare_col not in self.df.columns or strat_col == compare_col: continue
                
                print(f"  Generating bar chart stratified by {strat_col}, compared by {compare_col}")
                for aware_key, abx_list in self.aware_lists.items():
                    if aware_key == "All" or not abx_list: continue
                    try:
                        fig = summary_analyzer.plot_group_comparison(
                            compare_col=compare_col, group_by=strat_col,
                            variation_mode="iqr", antibiotics_to_plot=abx_list,
                            title=f"Mean {aware_key} Coverage Across {strat_col} by {compare_col}"
                        )
                        fig.update_layout(width=1600, height=900, title_x=0.5)
                        base_name = f"barchart_{self._safe_name(strat_col)}_by_{self._safe_name(compare_col)}_{aware_key}"
                        self._save_viz(fig, chapter_dir / base_name)
                    except Exception as e:
                        print(f"[Error] Failed for {strat_col}/{compare_col}/{aware_key}: {e}")


    def run_clustergrams(self):
        """CHAPTER 3: Generates clustergrams to visualize testing patterns."""
        print("\n--- Running Chapter 3: Coverage Clustergrams ---")
        chapter_dir = Path(self.config.output_dir) / "3_clustergrams"
        
        an = AMRMultiAnalyzer(
            self.df, SuffixAntibioticDetector("_Tested"), DefaultAntibioticNameParser("_Tested")
        )
        for strat_col in self.config.stratifier_candidates:
            if strat_col not in self.df.columns: continue
            print(f"  Generating clustergram for: {strat_col}")
            try:
                sf = StratifiedFilter(self.df)
                prefilter = sf.build_prefilter(strat_col, {"not in": ["Unknown", "", None, np.nan]})
                subset = an.get_data(copy=True, prefilter=prefilter)
                
                if subset.empty or subset[strat_col].nunique() < 2:
                    print(f"    Skipping {strat_col}, not enough data or distinct groups after filtering.")
                    continue

                fig = an.clustergram(
                    [strat_col], PctMetric(), df=subset,
                    plot_func=plot_clustergram_with_dendrograms,
                    plot_kwargs=dict(
                        title=f"Antibiotic Testing Coverage (%) by {strat_col}",
                        heatmap_height=min(2000, 30 * subset[strat_col].nunique()), 
                        heatmap_width=1200
                    )
                )
                self._save_viz(fig, chapter_dir / f"clustergram_{self._safe_name(strat_col)}")
            except Exception as e:
                print(f"[Error] Failed to generate clustergram for {strat_col}: {e}")

    def run_others(self):
        """Runs all chapters of the visual analysis pipeline."""
        self.run_per_antibiotic_boxplots()
        # self.run_group_comparison_barcharts()
        # self.run_clustergrams()
        print("\nVisual analytics pipeline complete!")

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
def main(main_df:pd.DataFrame):
    """Main function to load data, define the pipeline, and run the analysis."""
    # --- B. Perform one-time feature engineering ---
    print("Engineering 'Pediatric_vs_Elderly' feature...")
    pediatric_groups = ['0 years', '1-4 years', '5-9 years', '10-14 years']
    elderly_groups = ['70-74 years', '75-79 years', '80-84 years', '85-89 years', '90-94 years', '≥95 years']
    main_df['Pediatric_vs_Elderly'] = pd.NA
    main_df.loc[main_df['AgeRange'].isin(pediatric_groups), 'Pediatric_vs_Elderly'] = 'Pediatric'
    main_df.loc[main_df['AgeRange'].isin(elderly_groups), 'Pediatric_vs_Elderly'] = 'Elderly'
    
    # --- C. Define the Analysis Pipeline ---
    publication_pipeline: List[Dict[str, Any]] = [
        # SECTION 1: THE PRIMARY DIVIDE - INPATIENT VS. OUTPATIENT CARE
        # {
        #     "section": "care_setting_comparison",
        #     "type": "clustergram", "filter_by": "CareType", "categories": ["In-Patient", "Out-Patient"],
        #     "title_template": "Figure 1. Divergent Antimicrobial Testing Panels in {} Settings",
        #     "rationale": "To establish the foundational differences in testing formularies between hospital and community care."
        # },
        {
            "section": "care_setting_comparison",
            "type": "boxplot", "compare_by": "CareType",
            "filter_on": [{"column": "CareType", "values": ["In-Patient", "Out-Patient"]}],
            "aware_classes": ["Watch", "Reserve"],
            "title_template": "Figure 2. Testing Escalation for 'Watch' & 'Reserve' Classes in Inpatient vs. Outpatient Settings",
            "rationale": "To quantify the difference in testing for high-priority antibiotics, a proxy for perceived infection severity and resistance risk."
        },
        # SECTION 2: STRATIFICATION BY INFECTION PROFILE
        # {
        #     "section": "infection_source_stratification",
        #     "type": "clustergram", "filter_by": "TextMaterialgroupRkiL0", "categories": ["Blood Culture", "Urine"],
        #     "title_template": "Figure 3. Comparison of Testing Panels by Sample Source - {}",
        #     "rationale": "To visually compare the entire testing formulary for bloodstream infections versus urinary tract infections."
        # },
        {
            "section": "infection_source_stratification",
            "type": "boxplot", "compare_by": "TextMaterialgroupRkiL0",
            "filter_on": [
                {"column": "CareType", "values": ["In-Patient"]},
                {"column": "TextMaterialgroupRkiL0", "values": ["Blood Culture", "Urine"]}
            ],
            "aware_classes": ["All"],
            "title_template": "Figure 4. INPATIENT Testing Panels by Infection Source (Blood Culture vs. Urine)",
            "rationale": "To assess how hospitals adapt testing strategies for a systemic bloodstream infection versus a localized UTI."
        },
        # SECTION 3: INFLUENCE OF INSTITUTIONAL AND GEOGRAPHIC FACTORS
        # {
        #     "section": "institutional_and_geographic_context",
        #     "type": "clustergram", "filter_by": "Hospital_Priority", "categories": ["High", "Medium"],
        #     "title_template": "Figure 5. Comparison of Testing Panels by Hospital Priority - {} Priority",
        #     "rationale": "To provide a detailed visual overview of how the entire testing formulary differs between high- and medium-priority hospitals."
        # },
        {
            "section": "institutional_and_geographic_context",
            "type": "boxplot", "compare_by": "Facility_Function",
            "filter_on": [
                {"column": "CareType", "values": ["In-Patient"]},
                {"column": "Facility_Function", "values": ["General Hospital", "Referral/Advanced Hospital"]}
            ],
            "aware_classes": ["Reserve"],
            "title_template": "Figure 6. 'Reserve' Class Testing by Hospital Type (General vs. Advanced)",
            "rationale": "To investigate institutional practice variation by comparing last-resort testing between standard and high-level hospitals."
        },
        # {
        #     "section": "institutional_and_geographic_context",
        #     "type": "clustergram", "filter_by": "ARS_Region", "categories": ["West", "North East"],
        #     "title_template": "Figure 7. Regional Practice Variation: Comparison of Testing Panels in {} Region",
        #     "rationale": "To visually identify geographic heterogeneity in standard testing protocols."
        # },
        # SECTION 4: DEEP DIVE INTO HIGH-CONSEQUENCE CLINICAL SCENARIOS
        {
            "section": "high_risk_clinical_scenarios",
            "type": "boxplot", "compare_by": "ARS_WardType",
            "filter_on": [
                {"column": "PathogenGenus", "values": ["Acinetobacter", "Pseudomonas"]},
                {"column": "TextMaterialgroupRkiL0", "values": ["Blood Culture"]},
                {"column": "ARS_WardType", "values": ["Intensive Care Unit", "Normal Ward"]}
            ],
            "aware_classes": ["All"],
            "title_template": "Figure 8. Testing Panels for Critical Pathogen Bacteremia: ICU vs. Normal Ward",
            "rationale": "To analyze testing for highly resistant pathogens in the most critical clinical scenario."
        },
    ]
    
    # --- D. Configure and Run the Pipeline ---
    config = AnalysisConfig(
        output_dir="./outputs/boxplots",
        analysis_pipeline=publication_pipeline
    )
    pipeline = VisualAnalyticsPipeline(df=main_df, config=config)
    pipeline.run()
    pipeline.run_others()

def run_phase_I1(main_df:pd.DataFrame):
    main(main_df)

