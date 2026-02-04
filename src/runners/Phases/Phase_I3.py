# ====================================================================
# Pipeline for AST Panel Breadth Analysis — Plotly-native edition
# ====================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import warnings

import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp, kruskal

import plotly.graph_objects as go
import plotly.express as px

from src.controllers.summary.ASTPanelBreadthAnalyzer import ASTPanelBreadthAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================
# Storyline Test Harness (Plotly-native) with ops applied
# ==============================================================

# Optional ordered-trend helper
try:
    import scikit_posthocs as sp
    _HAS_SCPH = True
except Exception:
    _HAS_SCPH = False

def fdr_bh(pvals: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    if not pvals:
        return np.array([], dtype=bool), np.array([], dtype=float)
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    p[np.isnan(p)] = 1.0
    reject, q, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return reject, q

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0: return np.nan
    # rank approach
    from scipy.stats import rankdata
    allv = np.concatenate([a, b])
    r = rankdata(allv, method="average")
    ra = r[:n1].sum()
    return (2.0 * (ra - n1*(n1+1)/2.0) / (n1*n2)) - 1.0

def jonckheere_trend(groups_in_order: List[np.ndarray]) -> Optional[Dict[str, float]]:
    if not _HAS_SCPH:
        return None
    try:
        data = [np.asarray(g, dtype=float)[~np.isnan(g)] for g in groups_in_order]
        if sum(len(g) for g in data) < 3 or sum(len(g) > 0 for g in data) < 2:
            return None
        meds = [np.nanmedian(g) if len(g) else np.nan for g in data]
        ascending = all(meds[i] <= meds[i+1] for i in range(len(meds)-1))
        grp = np.concatenate([[k] * len(v) for k, v in enumerate(data)])
        vals = np.concatenate(data)
        df_long = pd.DataFrame({"grp": grp, "val": vals})
        dunn = sp.posthoc_dunn(df_long, val_col="val", group_col="grp", p_adjust=None)
        adj_p = [dunn.iloc[i, i+1] if i+1 < dunn.shape[1] else np.nan for i in range(dunn.shape[0])]
        min_adj = float(np.nanmin(adj_p)) if adj_p else np.nan
        return {"ascending_medians": bool(ascending), "min_adjacent_p": min_adj}
    except Exception:
        return None

def _format_block_summary(name: str,
                          group_stats: pd.DataFrame,
                          ks_df: Optional[pd.DataFrame],
                          cliffs_df: Optional[pd.DataFrame],
                          kruskal_dict: Optional[Dict],
                          jt_dict: Optional[Dict],
                          decisions: Dict[str, bool],
                          fdr_alpha: float) -> str:
    lines = [f"# {name} — Summary"]
    if isinstance(group_stats, pd.DataFrame) and not group_stats.empty:
        gcol = group_stats.columns[0]
        med_part = "; ".join([f"{r[0]}: med={r[1]:.1f} (n={int(r[2])})"
                              for r in group_stats[[gcol, "median", "n"]].values])
        lines.append(f"Medians by group → {med_part}")
    if kruskal_dict and not np.isnan(kruskal_dict.get("H", np.nan)):
        lines.append(f"Kruskal–Wallis: H={kruskal_dict['H']:.2f}, p={kruskal_dict['p_value']:.3g} (k={kruskal_dict['n_groups']})")
    if jt_dict is not None:
        asc = "ascending" if jt_dict.get("ascending_medians") else "not ascending"
        ptxt = jt_dict.get("min_adjacent_p")
        pstr = f"{ptxt:.3g}" if ptxt is not None and not np.isnan(ptxt) else "NA"
        lines.append(f"Ordered trend (adjacent Dunn heuristic): {asc}, min adjacent p={pstr}")
    if isinstance(ks_df, pd.DataFrame) and not ks_df.empty and "q_value" in ks_df.columns:
        sig = int((ks_df["q_value"] < fdr_alpha).sum())
        lines.append(f"KS pairwise: {sig}/{len(ks_df)} significant after FDR (q<{fdr_alpha}).")
    if isinstance(cliffs_df, pd.DataFrame) and not cliffs_df.empty:
        top = cliffs_df.iloc[0]
        lines.append(f"Strongest effect (Cliff’s δ): {top['g1']} vs {top['g2']} → δ={top['delta']:.2f}.")
    if decisions:
        flags = [k for k, v in decisions.items() if v]
        lines.append("Decisions: " + (", ".join(flags) if flags else "No rule triggered."))
    return "\n".join(lines)

def run_storyline_with_ops(
    analyzer: ASTPanelBreadthAnalyzer,
    story: List[Dict],
    output_root: Path | str,
    *,
    default_alpha: float = 0.05,
    save_pdf: bool = True
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"RUNNING STORYLINE ({len(story)} blocks)")
    print("=" * 80)

    for i, block in enumerate(story, 1):
        name = block["name"]
        chapter = block.get("chapter", "uncategorized")
        group_col = block["group_col"]
        ops = block.get("ops", {})
        alpha = float(ops.get("fdr_alpha", default_alpha))

        block_dir = output_root / chapter / name
        block_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n({i}/{len(story)}) {name} …")

        # Run analyzer (Plotly figs)
        results = analyzer.analyze_stratum(block, create_ks_heatmap=True)

        # Save tables
        if isinstance(results.get("summary"), pd.DataFrame):
            results["summary"].to_csv(block_dir / "summary_stats.csv", index=False)

        ks_raw = results.get("ks_results")
        if isinstance(ks_raw, pd.DataFrame) and not ks_raw.empty and "p_value" in ks_raw.columns:
            rej, q = fdr_bh(ks_raw["p_value"].tolist(), alpha=alpha)
            ks_out = ks_raw.copy()
            ks_out["q_value"] = q
            ks_out["reject_fdr"] = rej
            ks_out.to_csv(block_dir / "ks_test_results.csv", index=False)
        else:
            pd.DataFrame(columns=["Comparison", "KS_statistic", "p_value", "q_value", "reject_fdr"]).to_csv(
                block_dir / "ks_test_results.csv", index=False
            )

        if isinstance(results.get("kruskal_result"), dict):
            pd.DataFrame([results["kruskal_result"]]).to_csv(block_dir / "kruskal_test_result.csv", index=False)

        # Ordered trend (optional per ops)
        jt_info = None
        ordered_levels = ops.get("ordered_levels")
        if ordered_levels:
            df_f = results["filtered_df"]
            series_by_order = []
            for lvl in ordered_levels:
                vals = df_f.loc[df_f[group_col]==lvl, "antibiotics_tested_count"].astype(float).values
                if len(vals) > 0: series_by_order.append(vals)
            jt_info = jonckheere_trend(series_by_order)

        # Cliff’s δ pairs
        cliffs_rows = []
        try:
            df_f = results["filtered_df"]
            levels = df_f[group_col].dropna().unique().tolist()
            for i1 in range(len(levels)):
                a = df_f.loc[df_f[group_col]==levels[i1], "antibiotics_tested_count"].astype(float).values
                for i2 in range(i1+1, len(levels)):
                    b = df_f.loc[df_f[group_col]==levels[i2], "antibiotics_tested_count"].astype(float).values
                    delta = cliffs_delta(a, b)
                    cliffs_rows.append({"g1": str(levels[i1]), "g2": str(levels[i2]),
                                        "delta": delta, "n1": len(a), "n2": len(b)})
            cliffs_df = pd.DataFrame(cliffs_rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)
            if not cliffs_df.empty:
                cliffs_df.to_csv(block_dir / "cliffs_delta_pairs.csv", index=False)
        except Exception:
            cliffs_df = pd.DataFrame()

        # Decision rules (ops)
        decisions = {}
        dr = ops.get("decision_rules", {}) if isinstance(ops.get("decision_rules"), dict) else {}
        if dr.get("delta_median_min") is not None and isinstance(results.get("summary"), pd.DataFrame):
            s = results["summary"]
            if "median" in s.columns and not s.empty:
                dmed = float(s["median"].max() - s["median"].min())
                decisions["delta_median_rule"] = dmed >= float(dr["delta_median_min"])
        if dr.get("ks_frac_sig_min") is not None and isinstance(ks_raw, pd.DataFrame) and not ks_raw.empty and "p_value" in ks_raw.columns:
            _, q = fdr_bh(ks_raw["p_value"].tolist(), alpha=alpha)
            frac = float((q < alpha).mean()) if len(q) else 0.0
            decisions["ks_fraction_rule"] = frac >= float(dr["ks_frac_sig_min"])
        if dr.get("abs_delta_min") is not None and not cliffs_df.empty:
            decisions["effect_size_rule"] = bool((cliffs_df["delta"].abs() >= float(dr["abs_delta_min"])).any())

        # Save Plotly figs (HTML + PNG/SVG/PDF)
        if results.get("density_fig") is not None:
            analyzer.save_plotly(results["density_fig"], block_dir / "density_ecdf",
                                 width=1400, height=800, scale=4, write_svg=True, write_pdf=save_pdf)
        if results.get("violin_fig") is not None:
            analyzer.save_plotly(results["violin_fig"], block_dir / "violin_box",
                                 width=1400, height=800, scale=4, write_svg=True, write_pdf=save_pdf)
        if results.get("ks_heatmap_fig") is not None:
            analyzer.save_plotly(results["ks_heatmap_fig"], block_dir / "ks_heatmap",
                                 width=1200, height=900, scale=4, write_svg=True, write_pdf=save_pdf)

        # Compact text report
        try:
            group_stats = results["summary"]
            ks_df = pd.read_csv(block_dir / "ks_test_results.csv") if (block_dir / "ks_test_results.csv").exists() else None
            summary_txt = _format_block_summary(
                name=name, group_stats=group_stats, ks_df=ks_df, cliffs_df=cliffs_df,
                kruskal_dict=results.get("kruskal_result"), jt_dict=jt_info,
                decisions=decisions, fdr_alpha=alpha
            )
            (block_dir / "report.txt").write_text(summary_txt, encoding="utf-8")
        except Exception as e:
            (block_dir / "report.txt").write_text(f"# {name} — Summary\n(summary unavailable: {e})", encoding="utf-8")

        print(f"→ Saved to: {block_dir}")

# ==============================================================
# Storyline definition (ops applied to ALL blocks)
# ==============================================================

OUTPUT_ROOT   = Path("./outputs/panel_breadth_analysis")
TEST_SUFFIX   = "_Tested"
SAVE_PDF      = True

STORYLINE = [
    # --- CHAPTER 1: THE STEWARDSHIP PARADOX (Core Finding) ---
    {
        "name": "1.1_Patient_Acuity_Gradient",
        "chapter": "1_Stewardship_Paradox",
        "group_col": "CareType_Ward_Group",
        "why_important": "Panel breadth increases with patient acuity.",
        "filter_dict": {"CareType_Ward_Group": ["Outpatient", "Normal Ward", "Intermediate Care", "Intensive Care Unit"]},
    },
    {
        "name": "1.2_Stewardship_Gap_Ecoli_UTI",
        "chapter": "1_Stewardship_Paradox",
        "group_col": "CareType",
        "pathogen_genus": "Escherichia",
        "why_important": "Guideline non-adherence for E. coli UTIs.",
        "filter_dict": {"TextMaterialgroupRkiL0": "Urine", "CareType": ["Out-Patient", "In-Patient"]},
    },

    # --- CHAPTER 2: SYSTEMIC DRIVERS ---
    {
        "name": "2.1_Regional_And_Hospital_Styles",
        "chapter": "2_Systemic_Drivers",
        "group_col": "ARS_HospitalLevelManual",
        "why_important": "Variation by hospital resources/level.",
        "filter_dict": {"CareType": "In-Patient"},
    },
    {
        "name": "2.2_Equity_For_Vulnerable_Populations",
        "chapter": "2_Systemic_Drivers",
        "group_col": "AgeGroup",
        "why_important": "Pediatric vs elderly breadth.",
        "filter_dict": {"CareType": "In-Patient"},
    },

    # --- CHAPTER 3: MICROBIOLOGY PERSPECTIVE ---
    {
        "name": "3.1_Pathogen_GramType_Logic",
        "chapter": "3_Microbiology_Perspective",
        "group_col": "GramType",
        "why_important": "Gram-negative vs Gram-positive.",
        "filter_dict": {"GramType": ["Gram-negative", "Gram-positive"]},
    },
    {
        "name": "3.2_Specimen_Source_Logic",
        "chapter": "3_Microbiology_Perspective",
        "group_col": "TextMaterialgroupRkiL0",
        "why_important": "Sterile vs non-sterile sources.",
        "filter_dict": {"TextMaterialgroupRkiL0": ["Blood Culture", "Urine", "Respiratory", "Wound"]},
    },
]

# ---- Default ops to every block (applied here) ----
DEFAULT_OPS = {
    # Add ordered_levels per block only if you want trend testing on that block.
    # "ordered_levels": ["Outpatient","Normal Ward","Intermediate Care","Intensive Care Unit"],
    "decision_rules": {
        "delta_median_min": 2,   # minimum median gap (antibiotics)
        "ks_frac_sig_min": 0.2,  # ≥20% of KS pairs significant after FDR
        "abs_delta_min": 0.33    # at least one pair with |Cliff's δ| ≥ 0.33
    },
    "fdr_alpha": 0.05
}

def _apply_ops_to_storyline(story, default_ops):
    out = []
    for b in story:
        bb = dict(b)
        bb["ops"] = {**default_ops, **(b.get("ops", {}) or {})}
        out.append(bb)
    return out

STORYLINE = _apply_ops_to_storyline(STORYLINE, DEFAULT_OPS)

# ==============================================================
# Main runner
# ==============================================================

def run_phase_I3(main_df: pd.DataFrame):
    """
    Feature-engineer a few helper columns, run the Plotly-native storyline with
    decision rules applied to every block, and export high-resolution outputs.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---- feature engineering for scenarios ----
    df = main_df.copy()

    # 1) CareType_Ward_Group
    df["CareType_Ward_Group"] = df["CareType"].astype(str)
    ward_map = {
        "Intensive Care Unit": "Intensive Care Unit",
        "Normal Ward": "Normal Ward",
        "Intermediate Care/Awake Station": "Intermediate Care",
        "Early Rehabilitation": "Rehabilitation",
        "Outpatient": "Outpatient",
        "Day Clinic": "Day Clinic",
        "Operating Room": "Operating Room",
        "Other Treatment Type": "Others",
        "Unknown": "Others",
        "Rehabilitation": "Rehabilitation"
    }
    if "ARS_WardType" in df.columns:
        df.loc[df["CareType"] == "In-Patient", "CareType_Ward_Group"] = (
            df["ARS_WardType"].map(ward_map).fillna("Normal Ward")
        )

    # 2) (Optional) AgeGroup already present; if not, build Age_Category elsewhere

    # ---- initialize analyzer ----
    analyzer = ASTPanelBreadthAnalyzer(df=df, test_col_suffix=TEST_SUFFIX)

    # ---- execute storyline with Plotly. Please do not stress meeeeeeeeee ----
    run_storyline_with_ops(
        analyzer=analyzer,
        story=STORYLINE,
        output_root=OUTPUT_ROOT,
        default_alpha=0.05,
        save_pdf=SAVE_PDF
    )

    print("\n" + "="*80)
    print("AST PANEL BREADTH ANALYSIS PIPELINE COMPLETE!")
    print(f"All results saved to: {OUTPUT_ROOT.resolve()}")
    print("="*80)
