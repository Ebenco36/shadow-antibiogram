# src/runners/run_cotesting.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from src.controllers.DataLoader import DataLoader
from src.utils.LoadClasses import LoadClasses
from src.controllers.CoTestAnalyzer import CoTestAnalyzer
from src.utils.helpers import (
    get_label,
    plot_clustergram_with_dendrograms,
    visualize_antibiotic_network,
)
from src.mappers.pathogen_genus import (
    pathogen_genus_critical,
    pathogen_genus_other,
    pathogen_genus_medium,
    pathogen_genus_high,
)

# --------------------------- CONFIG ---------------------------

DATA_CSV = "./datasets/WHO_Aware_data"
CLASS_JSON = "./datasets/antibiotic_class_grouping.json"
WHO_JSON = "./datasets/antibiotic_class.json"

OUT_CLUSTERGRAM = "./datasets/output/old_but_relevant/clustergrams"
OUT_NETWORK = "./datasets/output/old_but_relevant/network_outputs"
OUT_SWEEP = "./datasets/output/old_but_relevant/co_testing_network_outputs"

os.makedirs(OUT_CLUSTERGRAM, exist_ok=True)
os.makedirs(OUT_NETWORK, exist_ok=True)
os.makedirs(OUT_SWEEP, exist_ok=True)

# Default network threshold
DEFAULT_THRESHOLD = 0.3

# Colors for community highlighting
MY_CLUSTER_COLORS = {
    0: "#a6cee3", 1: "#1f78b4", 2: "#b2df8a", 3: "#33a02c", 4: "#fb9a99",
    5: "#e31a1c", 6: "#fdbf6f", 7: "#ff7f00", 8: "#cab2d6", 9: "#6a3d9a",
}

# For the automated sweep (phase II style)
GROUP_VARS = [
    "ARS_HospitalLevelManual",
    "Hospital_Priority",
    "HighLevelAgeRange",
    "ARS_WardType",
    "CareType",
    "AgeGroup",
    "AgeRange",
    "GramType"
]
MIN_ROWS = 5
NETWORK_THRESHOLD_SWEEP = 0.2


# ------------------------- UTILITIES --------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def clean_filename(x: Any) -> str:
    return (
        str(x)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(":", "-")
        .replace("|", "-")
    )

def pick_abx_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_Tested")]

def build_class_map(load: LoadClasses, df: pd.DataFrame, class_names: List[str]) -> Dict[str, List[str]]:
    """Build {class: [*_Tested cols present in df]} from a list of class names."""
    out: Dict[str, List[str]] = {}
    for cls in class_names:
        cols = load.convert_to_tested_columns(
            load.get_antibiotics_by_class(antibiotic_classes=[cls])
        )
        cols = [c for c in cols if c in df.columns]
        if cols:
            out[cls] = cols
    return out

def build_who_map(load: LoadClasses, df: pd.DataFrame, categories: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cat in categories:
        cols = load.convert_to_tested_columns(
            load.get_antibiotics_by_category(categories=[cat])
        )
        cols = [c for c in cols if c in df.columns]
        if cols:
            out[f"{cat}List"] = cols
    return out

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    filters: {col: value or [values]} or {"__callable__": lambda df: df[...]}
    """
    if "__callable__" in filters:
        return filters["__callable__"](df).copy()

    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(list(val))
        else:
            mask &= (df[col] == val)
    return df[mask].copy()

def label_map_for(analyzer: CoTestAnalyzer,
                  antibiotic_class_map: Dict[str, List[str]],
                  include_class: bool) -> Dict[str, str]:
    # get_label expects a list of antibiotic columns
    return get_label(
        analyzer.abx_cols,
        antibiotic_class_map=antibiotic_class_map,
        format_type="abbr",
        enrich=True,
        include_class=include_class,
    )

def set_analyzer(analyzer: CoTestAnalyzer,
                 subset: pd.DataFrame,
                 abx_cols: List[str]) -> None:
    analyzer._set_transaction(subset, antibiotic_cols=abx_cols)
    analyzer.remove_zero_columns()

def save_clustergram(fig, basename: str, out_dir: str,
                     png_scale: float = 4.0,
                     html=True, svg=True, png=True, pdf=True) -> None:
    ensure_dir(out_dir)
    base = os.path.join(out_dir, basename)
    if html:
        fig.save(base + ".html")
    if svg:
        fig.save(base + ".svg")
    if png:
        fig.save(base + ".png", scale_factor=png_scale)
    if pdf:
        fig.save(base + ".pdf")

def is_square(df_sim: pd.DataFrame) -> bool:
    return df_sim.shape[0] == df_sim.shape[1] and set(df_sim.index) == set(df_sim.columns)

# -------------- Pre-flight network validator / preparer --------------

def _make_unique(idx: pd.Index) -> pd.Index:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for s in idx.astype(str):
        n = seen.get(s, 0) + 1
        seen[s] = n
        out.append(f"{s} ({n})" if n > 1 else s)
    return pd.Index(out, name=idx.name)

def _find_label_collisions(label_map: Dict[str, str]) -> Dict[str, List[str]]:
    inv: Dict[str, List[str]] = {}
    for k, v in label_map.items():
        inv.setdefault(v, []).append(k)
    return {lbl: srcs for lbl, srcs in inv.items() if len(srcs) > 1}

def prepare_similarity_for_network(
    sim_df: pd.DataFrame,
    label_map: Optional[Dict[str, str]] = None,
    *,
    require_symmetric: bool = True,
    auto_reorder: bool = True,
    auto_dedup_labels: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Returns (prepared_df_or_None, report). If None, caller should skip network.
    - Applies label_map (index+columns if square, index-only if rectangular)
    - Ensures numeric dtype
    - Optionally reorders columns to match index when label sets are identical
    - Optionally de-dups labels if collisions exist after renaming
    - Validates symmetry (if required), NaNs, etc.
    """
    rep: Dict[str, Any] = {}
    df = sim_df.copy()

    # 1) Apply labels safely
    if label_map:
        rep["label_collisions"] = _find_label_collisions(label_map)
        if df.shape[0] == df.shape[1]:
            df = df.rename(index=label_map, columns=label_map)
        else:
            df = df.rename(index=label_map)  # rectangular: index-only

    # 2) Basic shape/labels checks
    rep["square"] = df.shape[0] == df.shape[1]
    rep["dup_index"] = bool(df.index.has_duplicates)
    rep["dup_columns"] = bool(df.columns.has_duplicates)
    rep["same_set"] = rep["square"] and (set(df.index) == set(df.columns))
    rep["identical_order"] = rep["square"] and df.index.equals(df.columns)

    # 3) Optional fixes: de-dup then reorder
    fixes: List[str] = []
    if auto_dedup_labels and (rep["dup_index"] or rep["dup_columns"]):
        df.index = _make_unique(df.index)
        if rep["square"]:
            df.columns = _make_unique(df.columns)
        fixes.append("deduplicated labels")
        rep["dup_index"] = bool(df.index.has_duplicates)
        rep["dup_columns"] = bool(df.columns.has_duplicates)

    if rep["square"] and rep["same_set"] and not rep["identical_order"] and auto_reorder:
        df = df.reindex(columns=df.index)  # align columns to index order
        fixes.append("aligned column order to index")
        rep["identical_order"] = df.index.equals(df.columns)

    rep["fixes"] = fixes

    # 4) Hard fail cases (skip plotting but allow CSV)
    if not rep["square"]:
        rep["reason"] = "rectangular matrix (to-groups); not suitable for network"
        return None, rep
    if not rep["same_set"]:
        rep["reason"] = "row/column label sets differ after renaming"
        return None, rep
    if df.index.has_duplicates or df.columns.has_duplicates:
        rep["reason"] = "duplicate labels remain after de-dup"
        return None, rep

    # 5) Numeric / NaN / symmetry diagnostics
    rep["has_nan"] = bool(df.isna().any().any())
    if rep["has_nan"]:
        df = df.fillna(0.0)  # policy: fill NaN with 0
        fixes.append("filled NaN with 0.0")

    # ensure numeric
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except Exception:
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        fixes.append("coerced non-numeric to numeric (NaN->0)")

    rep["val_min"] = float(df.values.min()) if df.size else np.nan
    rep["val_max"] = float(df.values.max()) if df.size else np.nan

    if require_symmetric:
        sym = np.allclose(df.values, df.values.T, equal_nan=True)
        rep["symmetric"] = bool(sym)
        if not sym:
            rep["reason"] = "matrix not symmetric"
            return None, rep

    # 6) Diagonal sanity (optional)
    diag = np.diag(df.values)
    rep["diag_all_zero"] = bool(np.allclose(diag, 0.0))
    rep["diag_all_one"] = bool(np.allclose(diag, 1.0))

    return df, rep


# -------------------- CLUSTERGRAM + NETWORK --------------------

def draw_cluster_and_network(
    analyzer: CoTestAnalyzer,
    label_map: Dict[str, str],
    antibiotic_class_map: Dict[str, List[str]],
    title: str,
    basename: str,
    out_cluster_dir: str,
    out_network_dir: str,
    *,
    include_class_in_matrix: bool = True,
    heatmap_size: Tuple[int, int] = (1000, 1000),
    dendro_row: bool = True,
    dendro_col: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    col_metric: str = "correlation",
    row_metric: str = "correlation",
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    # Similarity (symmetric antibiotics × antibiotics)
    sim = analyzer.jaccard()

    # Rename axes for display
    if include_class_in_matrix:
        label_map_with_class = get_label(
            analyzer.abx_cols,
            antibiotic_class_map=antibiotic_class_map,
            format_type="abbr",
            enrich=True,
            include_class=True
        )
        sim_named = sim.rename(index=label_map_with_class, columns=label_map_with_class)
    else:
        sim_named = sim.rename(index=label_map, columns=label_map)

    # Clustergram
    cluster = plot_clustergram_with_dendrograms(
        sim_named,
        include_row_dendrogram=dendro_row,
        include_col_dendrogram=dendro_col,
        interactive=True,
        row_dendrogram_side="left",
        cell_height=12,
        gutter_px=0, gutter_py=0,
        title=title,
        row_panel_width=200, col_panel_height=200,
        heatmap_width=heatmap_size[0], heatmap_height=heatmap_size[1],
        source="Antibiotics", target="Antibiotics",
        col_metric=col_metric, row_metric=row_metric,
    )
    save_clustergram(cluster, basename, out_cluster_dir)

    # Network preflight
    prepped, rep = prepare_similarity_for_network(sim_named, label_map=None)
    if prepped is None:
        # Always save the CSV even if network skipped
        ensure_dir(out_network_dir)
        csv_path = os.path.join(out_network_dir, f"{basename}.csv")
        sim_named.to_csv(csv_path)
        print(f"[SKIP NETWORK] {basename} — {rep.get('reason', 'pre-check failed')}; report={rep}")
        return

    # Network
    ensure_dir(out_network_dir)
    visualize_antibiotic_network(
        data_input=prepped,
        threshold=threshold,
        output_dir=out_network_dir + "/",
        output_image=f"{basename}.png",
        output_html=f"{basename}.html",
        title=title,
        cluster_colors=cluster_colors or MY_CLUSTER_COLORS,
    )


# ------------------------ LOAD DATA & MAPS --------------------

if __name__ == "__main__":
    # Prefer the project DataLoader so you keep any in-pipeline cleaning/typing
    loader = DataLoader(DATA_CSV)
    df = loader.get_combined()

    load = LoadClasses()
    abx_cols_all = pick_abx_cols(df)

    # 1) Build class map (deduped) from a single list of class names
    CLASS_NAMES = [
        "Fluoroquinolone", "Aminoglycoside", "Penicillin (β-lactam)",
        "β-lactam/β-lactamase inhibitor", "Monobactam (β-lactam)",
        "Third-gen cephalosporin (β-lactam)", "Fourth-gen cephalosporin (β-lactam)",
        "First-gen cephalosporin (β-lactam)", "Pleuromutilin",
        "Siderophore cephalosporin (β-lactam)", "Tetracycline derivative",
        "Macrocyclic", "Aminocyclitol", "Amphenicol", "Tetracycline",
        "Lincosamide", "Fifth-gen cephalosporin (β-lactam)", "Polymyxin",
        "Glycopeptide", "Streptogramin", "Lipopeptide", "Lipoglycopeptide",
        "Phosphonic acid derivative", "Second-gen cephalosporin (β-lactam)",
        "Oxazolidinone", "Macrolide", "Nitrofuran", "Rifamycin", "Glycylcycline",
        "Carbapenem (β-lactam)", "Steroid antibiotic", "Pseudomonic acid",
        "Sulfonamide/Trimethoprim combo", "Quinolone", "Sulfonamide",
        "Dihydrofolate reductase inhibitor", "Polyene", "Polypeptide",
        "Echinocandin", "Azole", "Oxacephem (β-lactam)", "Nitroimidazole",
        "Quinolone derivative", "Streptogramin combo", "Antimetabolite", "Other Class",
        "Ketolide (macrolide derivative)", "Carbacephem (β-lactam)", "β-lactamase inhibitor",
    ]
    CLASS_NAMES = list(dict.fromkeys(CLASS_NAMES))  # dedupe, preserve order
    class_map_present = build_class_map(load, df, CLASS_NAMES)
    save_json(class_map_present, CLASS_JSON)
    print("Saved antibiotic classes:", CLASS_JSON)

    # 2) Build WHO map
    who_map_present = build_who_map(load, df, ["Watch", "Access", "Reserve", "Not Set"])
    save_json(who_map_present, WHO_JSON)
    print("Saved WHO categories:", WHO_JSON)

    # 3) Prepare analyzer once (we'll reset per scenario)
    analyzer = CoTestAnalyzer(transactions=df, antibiotic_cols=abx_cols_all)
    analyzer.remove_zero_columns()

    # Load the class map we just saved (for downstream labels)
    with open(CLASS_JSON, "r") as f:
        loaded_antibiotic_classes = json.load(f)

    # ---------------------- DECLARE SCENARIOS ---------------------

    Scenario = Dict[str, Any]
    SCENARIOS: List[Scenario] = [
        # Overview (whole dataset)
        dict(
            name="overview_all",
            filters={},
            title="Antibiotic Co-Testing Patterns Overview",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.5,
        ),
        # Q2: E. coli (Urine)
        dict(
            name="ecoli_urine",
            filters={"PathogenGenus": "Escherichia", "TextMaterialgroupRkiL0": "Urine"},
            title="Antibiotic Co-Testing Patterns for Escherichia coli in Urine Isolates",
            include_class=False,
            heatmap=(1000, 1000),
            threshold=0.5,
        ),
        # Q2 part 2: E. coli (Blood Culture)
        dict(
            name="ecoli_blood",
            filters={"PathogenGenus": "Escherichia", "TextMaterialgroupRkiL0": "Blood Culture"},
            title="Antibiotic Co-Testing Patterns for Escherichia coli in Blood Culture Isolates",
            include_class=False,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        # Q3: E. coli Urine by age groups
        dict(
            name="ecoli_urine_elderly",
            filters={"PathogenGenus": "Escherichia", "TextMaterialgroupRkiL0": "Urine", "AgeGroup": "Elderly"},
            title="Antibiotic Co-Testing Patterns for Escherichia coli in Urine Isolates in Elderly",
            include_class=False,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="ecoli_urine_pediatric",
            filters={"PathogenGenus": "Escherichia", "TextMaterialgroupRkiL0": "Urine", "AgeGroup": "Pediatric"},
            title="Antibiotic Co-Testing Patterns for Escherichia coli in Urine Isolates in Infants",
            include_class=False,
            heatmap=(500, 500),
            threshold=0.3,
        ),
        dict(
            name="ecoli_urine_adult",
            filters={"PathogenGenus": "Escherichia", "TextMaterialgroupRkiL0": "Urine", "AgeGroup": "Adult"},
            title="Antibiotic Co-Testing Patterns for Escherichia coli in Urine Isolates in Adults",
            include_class=False,
            heatmap=(500, 500),
            threshold=0.3,
        ),
        # Q4: Staphylococcus overall
        dict(
            name="staph_all",
            filters={"PathogenGenus": "Staphylococcus"},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        # Q5: Staphylococcus per year
        dict(
            name="staph_2019",
            filters={"PathogenGenus": "Staphylococcus", "Year": 2019},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Year 2019",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_2020",
            filters={"PathogenGenus": "Staphylococcus", "Year": 2020},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Year 2020",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_2021",
            filters={"PathogenGenus": "Staphylococcus", "Year": 2021},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Year 2021",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_2022",
            filters={"PathogenGenus": "Staphylococcus", "Year": 2022},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Year 2022",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_2023",
            filters={"PathogenGenus": "Staphylococcus", "Year": 2023},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Year 2023",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        
        
        
        # Q4: Ecoli overall
        dict(
            name="ecoli_all",
            filters={"PathogenGenus": "Escherichia"},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        # Q5: Escherichia per year
        dict(
            name="ecoli_2019",
            filters={"PathogenGenus": "Escherichia", "Year": 2019},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates in Year 2019",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="ecoli_2020",
            filters={"PathogenGenus": "Escherichia", "Year": 2020},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates in Year 2020",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="ecoli_2021",
            filters={"PathogenGenus": "Escherichia", "Year": 2021},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates in Year 2021",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_2022",
            filters={"PathogenGenus": "Escherichia", "Year": 2022},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates in Year 2022",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="ecoli_2023",
            filters={"PathogenGenus": "Escherichia", "Year": 2023},
            title="Antibiotic Co-Testing Patterns for Escherichia Isolates in Year 2023",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        
        
        
        # Q6: Staphylococcus by lab
        dict(
            name="staph_lab_28_berlin",
            filters={"PathogenGenus": "Staphylococcus", "Anonymized_Lab": "Lab 10"},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Lab 28 Berlin",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
        dict(
            name="staph_amedes_goettingen",
            filters={"PathogenGenus": "Staphylococcus", "Anonymized_Lab": "Lab 1"},
            title="Antibiotic Co-Testing Patterns for Staphylococcus Isolates in Amedes Lab",
            include_class=True,
            heatmap=(1000, 1000),
            threshold=0.3,
        ),
    ]

    # ------------------------ RUN SCENARIOS -----------------------

    for sc in SCENARIOS:
        subset = apply_filters(df, sc["filters"])
        if subset.shape[0] < MIN_ROWS:
            print(f"↪ Skip {sc['name']} (n={subset.shape[0]} < {MIN_ROWS})")
            continue

        abx_cols = pick_abx_cols(subset)
        if not abx_cols:
            print(f"↪ Skip {sc['name']} (no *_Tested columns)")
            continue

        set_analyzer(analyzer, subset, abx_cols)
        lbl = label_map_for(analyzer, loaded_antibiotic_classes, include_class=False)

        base = f"clustergram_{sc['name']}"
        draw_cluster_and_network(
            analyzer=analyzer,
            label_map=lbl,
            antibiotic_class_map=loaded_antibiotic_classes,
            title=sc["title"],
            basename=base,
            out_cluster_dir=OUT_CLUSTERGRAM,
            out_network_dir=OUT_NETWORK,
            include_class_in_matrix=bool(sc["include_class"]),
            heatmap_size=tuple(sc["heatmap"]),
            threshold=float(sc["threshold"]),
            cluster_colors=MY_CLUSTER_COLORS,
        )
        print(f"Wrote outputs for {sc['name']}")

    # ---------------- WHO PATHOGEN LIST & BY-CARE SWEEPS ----------

    who_pathogens = [
        "Escherichia", "Klebsiella", "Acinetobacter", "Pseudomonas", "Staphylococcus",
        "Enterococcus", "Neisseria", "Salmonella", "Shigella", "Mycobacterium",
    ]

    for pathogen in who_pathogens:
        sub = df[df["PathogenGenus"] == pathogen].copy()
        if sub.shape[0] < MIN_ROWS:
            continue
        abx_cols = pick_abx_cols(sub)
        set_analyzer(analyzer, sub, abx_cols)
        lbl = label_map_for(analyzer, loaded_antibiotic_classes, include_class=False)

        name = f"pathogen_{clean_filename(pathogen)}"
        draw_cluster_and_network(
            analyzer, lbl, loaded_antibiotic_classes,
            title=f"Antibiotic Co-Testing Patterns for Pathogen: {pathogen}",
            basename=f"clustergram_{name}",
            out_cluster_dir=OUT_CLUSTERGRAM,
            out_network_dir=OUT_NETWORK,
            include_class_in_matrix=True,
            heatmap_size=(1000, 1000),
            threshold=0.3,
        )
        print(f"WHO pathogen sweep: {pathogen}")

    # CareType × WHO pathogens
    care_types = sorted([c for c in df["CareType"].dropna().unique()])
    for care in care_types:
        for pathogen in who_pathogens:
            sub = df[(df["CareType"] == care) & (df["PathogenGenus"] == pathogen)].copy()
            if sub.shape[0] < MIN_ROWS:
                continue
            abx_cols = pick_abx_cols(sub)
            set_analyzer(analyzer, sub, abx_cols)
            lbl = label_map_for(analyzer, loaded_antibiotic_classes, include_class=False)
            text = "OutPatients" if care == "ambulant" else "InPatients"
            base = f"in_{clean_filename(text)}_pathogen_{clean_filename(pathogen)}"
            draw_cluster_and_network(
                analyzer, lbl, loaded_antibiotic_classes,
                title=f"Antibiotic Co-Testing Patterns in {text} for Pathogen: {pathogen}",
                basename=f"clustergram_{base}",
                out_cluster_dir=OUT_CLUSTERGRAM,
                out_network_dir=OUT_NETWORK,
                include_class_in_matrix=True,
                heatmap_size=(1000, 1000),
                threshold=0.3,
            )
            print(f"CareType×Pathogen: {care} × {pathogen}")

    # ------------- PHASE II STYLE: MULTI-METRIC SWEEP -------------

    def compute_metrics(an: CoTestAnalyzer, right: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        if right is None:
            return {
                "jaccard": an.jaccard(),
                "cfws": an.cfws(),
                "idf_cfws": an.idfcfws(),
                "cosine": an.cos(),
                "cosine_idf_fws": an.cosineidffws(),
            }
        else:
            return {
                "jaccard": an.jaccard(right=right),
                "cfws": an.cfws(right=right),
                "idf_cfws": an.idfcfws(right=right),
                "cosine": an.cos(right=right),
                "cosine_idf_fws": an.cosineidffws(right=right),
            }

    def save_all_outputs(
        pathogen: str,
        group_var: str,
        group_value: Optional[str],
        metric_name: str,
        sim_df: pd.DataFrame,
        label_map: Dict[str, str],
        out_root: str,
        title_prefix: str,
        threshold: float = NETWORK_THRESHOLD_SWEEP,
        draw_network: bool = True,
    ) -> None:
        if group_value is None:
            subdir = os.path.join(out_root, f"{group_var}", f"_to_groups", metric_name)
            fname_stub = f"{pathogen}-to-{group_var}"
            title = f"{title_prefix}: {pathogen} → {group_var}"
        else:
            safe_val = clean_filename(group_value)
            subdir = os.path.join(out_root, f"{group_var}", safe_val, metric_name)
            fname_stub = f"{pathogen}-in-{safe_val}"
            title = f"{title_prefix}: {pathogen} in {group_value}"

        ensure_dir(subdir)

        # Rename for saving
        sim_named = sim_df.rename(index=label_map, columns=label_map) if is_square(sim_df) \
            else sim_df.rename(index=label_map)

        # Always save CSV
        csv_path = os.path.join(subdir, f"similarity_dataframe-{fname_stub}.csv")
        sim_named.to_csv(csv_path)

        if draw_network:
            # Pre-flight: only plot if square + good labels + symmetric, etc.
            prepped, rep = prepare_similarity_for_network(sim_named, label_map=None)
            if prepped is None:
                print(f"[SKIP NETWORK] {fname_stub} — {rep.get('reason', 'pre-check failed')}; report={rep}")
                return
            visualize_antibiotic_network(
                data_input=prepped,
                threshold=threshold,
                output_dir=subdir + "/",
                output_image=f"network-{fname_stub}.png",
                output_html=f"network-{fname_stub}.html",
                title=title,
                cluster_colors=MY_CLUSTER_COLORS,
            )

    # Build genus universe
    genus_universe = (
        pathogen_genus_critical + pathogen_genus_high + pathogen_genus_medium + pathogen_genus_other
    )

    for pathogen in genus_universe:
        if (df["PathogenGenus"] == pathogen).sum() < MIN_ROWS:
            continue

        # per-group value (symmetric)
        for group_var in GROUP_VARS:
            values = (
                df.loc[df["PathogenGenus"] == pathogen, group_var]
                .dropna()
                .astype(str)
                .replace({"^\s+$": ""}, regex=True)
                .unique()
                .tolist()
            )
            for val in values:
                subset = df[(df["PathogenGenus"] == pathogen) & (df[group_var] == val)]
                if subset.shape[0] < MIN_ROWS:
                    continue
                abx_cols = pick_abx_cols(subset)
                set_analyzer(analyzer, subset, abx_cols)
                lbl = label_map_for(analyzer, loaded_antibiotic_classes, include_class=False)
                sims = compute_metrics(analyzer, right=None)
                for metric_name, sim_mat in sims.items():
                    if sim_mat.empty:
                        continue
                    save_all_outputs(
                        pathogen=pathogen,
                        group_var=group_var,
                        group_value=val,
                        metric_name=metric_name,
                        sim_df=sim_mat,
                        label_map=lbl,
                        out_root=OUT_SWEEP,
                        title_prefix="Antibiotic Co-Testing Patterns",
                        threshold=NETWORK_THRESHOLD_SWEEP,
                        draw_network=True,
                    )

        # to-groups (rectangular; antibiotic × all categories) — CSV only, network skipped by preflight
        for group_var in GROUP_VARS:
            subset_all = df[df["PathogenGenus"] == pathogen]
            if subset_all.shape[0] < MIN_ROWS:
                continue
            abx_cols = pick_abx_cols(subset_all)
            set_analyzer(analyzer, subset_all, abx_cols)
            lbl = label_map_for(analyzer, loaded_antibiotic_classes, include_class=False)
            sims_rect = compute_metrics(analyzer, right=group_var)
            for metric_name, sim_mat in sims_rect.items():
                if sim_mat.empty:
                    continue
                save_all_outputs(
                    pathogen=pathogen,
                    group_var=group_var,
                    group_value=None,
                    metric_name=metric_name,
                    sim_df=sim_mat,
                    label_map=lbl,
                    out_root=OUT_SWEEP,
                    title_prefix="Antibiotic Co-Testing vs Groups",
                    threshold=NETWORK_THRESHOLD_SWEEP,
                    draw_network=True,  # preflight will skip network for rectangular
                )

    print("Finished generating all co-testing outputs.")
