import os
import math
import json
import textwrap
import numpy as np
import pandas as pd
import altair as alt
import networkx as nx
from pathlib import Path
from typing import Mapping
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from difflib import SequenceMatcher
from scipy.spatial.distance import pdist
from src.utils.LoadClasses import LoadClasses
from src.utils.network import visualize_antibiotic_network
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from src.mappers.antibiotic_to_grams import ABX_TARGET_MAP, CATALOG
from typing import (Dict, List, Optional, Sequence, Tuple, Union, Mapping, Any)

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

try:
    from community import community_louvain  # python-louvain
except Exception as e:
    raise ImportError(
        "This function requires 'python-louvain'. Install with: pip install python-louvain") from e


alt.data_transformers.enable("json")



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

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
    
def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def build_broad_class_map(load: LoadClasses, df: pd.DataFrame, broad_names: List[str]) -> Dict[str, List[str]]:
    """
    Build {broad_class: [*_Tested columns in df]} using the LoadClasses broad mappings.
    """
    out: Dict[str, List[str]] = {}

    for broad in broad_names:
        cols = load.convert_to_tested_columns(
            load.get_antibiotics_by_broad_class(broad_classes=[broad])
        )
        cols = [c for c in cols if c in df.columns]
        if cols:
            out[broad] = cols

    return out


def merge_antibiotic_data(
    existing_file: str,
    who_file: str,
    output_file: str,
    class_column: str = "Class",
    broad_class_column: str = "Broad Class",
):
    """
    DYNAMIC merge with fuzzy matching for antibiotic names.
    
    Features:
    - Case-insensitive + normalization (remove spaces, dashes, slashes)
    - Fuzzy string matching (difflib) for near-misses
    - Comprehensive synonym mapping (1000+ aliases)
    - Manual brand name cross-reference
    - Collision detection and reporting
    - Preserves all rows from existing_file
    - WHO data filled with 'Not Set' where missing
    
    Args:
        existing_file: Path to existing antibiotic classification CSV
        who_file: Path to WHO AWaRe classification CSV
        output_file: Path to save merged result
        class_column: Column name for antibiotic class (default: "Class")
        broad_class_column: Column name for broad class output (default: "Broad Class")
    
    Returns:
        Merged DataFrame with all records
    """
    
    # Load datasets
    existing_df = pd.read_csv(existing_file)
    who_df = pd.read_csv(who_file)
    
    print(f"\n{'='*70}")
    print(f"ANTIBIOTIC MERGE: Dynamic Matching with Fuzzy Logic (EXPANDED)")
    print(f"{'='*70}")
    
    print(f"\nExisting file: {len(existing_df)} records")
    print(f"WHO file: {len(who_df)} records")
    
    # ===== STEP 1: NORMALIZATION FUNCTION =====
    def normalize_name(name):
        """Normalize antibiotic names for matching."""
        if pd.isna(name):
            return ""
        
        name = str(name).strip().lower()
        # Remove common separators and spaces
        name = name.replace("/", "").replace("-", "").replace(" ", "")
        # Remove leading/trailing underscores
        name = name.replace("_iv", "").replace("_oral", "")
        return name
    
    # ===== STEP 2: COMPREHENSIVE SYNONYM MAPPING (1000+ entries) =====
    
    from src.mappers.synonym_mapping import synonym_mapping
    # Create normalized → original mapping from WHO file
    who_normalized = {}
    for idx, row in who_df.iterrows():
        original_name = row["Antibiotic"]
        normalized = normalize_name(original_name)
        who_normalized[normalized] = original_name
    
    print(f"\nWHO file normalized entries: {len(who_normalized)}")
    print(f"Synonym mapping size: {len(synonym_mapping)} aliases")
    
    # ===== FUZZY MATCHING FUNCTION =====
    def fuzzy_match(existing_name, who_dict, threshold=0.80):
        """Find best match using similarity ratio."""
        normalized = normalize_name(existing_name)
        
        # Direct match
        if normalized in who_dict:
            return who_dict[normalized], 1.0
        
        # Synonym mapping
        if normalized in synonym_mapping:
            syn = normalize_name(synonym_mapping[normalized])
            if syn in who_dict:
                return who_dict[syn], 1.0
        
        # Fuzzy matching fallback
        best_match = None
        best_score = 0
        
        for who_normalized_name, who_original_name in who_dict.items():
            score = SequenceMatcher(None, normalized, who_normalized_name).ratio()
            if score > best_score:
                best_score = score
                best_match = who_original_name
        
        if best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    # ===== PERFORM MERGE =====
    print(f"\nPerforming fuzzy merge with {len(existing_df)} existing antibiotics...")
    
    merge_results = []
    matched_count = 0
    perfect_matches = 0
    fuzzy_matches = 0
    unmatched = []
    
    for idx, row in existing_df.iterrows():
        existing_name = row["Antibiotic Name"]
        matched_who_name, similarity = fuzzy_match(existing_name, who_normalized, threshold=0.80)
        
        if matched_who_name:
            matched_count += 1
            
            if similarity == 1.0:
                perfect_matches += 1
                match_type = "PERFECT"
            else:
                fuzzy_matches += 1
                match_type = f"FUZZY({similarity:.2f})"
            
            # Get WHO row
            who_row = who_df[who_df["Antibiotic"] == matched_who_name].iloc[0]
            
            # Build merged row
            merged_row = row.copy()
            merged_row["WHO_Match"] = matched_who_name
            merged_row["Match_Type"] = match_type
            merged_row["WHO_Class"] = who_row.get("WHO_Class", "Not Set")
            merged_row["WHO_ATC_code"] = who_row.get("WHO_ATC_code", "Not Set")
            merged_row["Category"] = who_row.get("Category", "Not Set")
            merged_row["Listed_on_EML_2019"] = who_row.get("Listed_on_EML_2019", "Not Set")
            
            merge_results.append(merged_row)
        else:
            unmatched.append(existing_name)
            merged_row = row.copy()
            merged_row["WHO_Match"] = "NO MATCH"
            merged_row["Match_Type"] = "UNMATCHED"
            merged_row["WHO_Class"] = "Not Set"
            merged_row["WHO_ATC_code"] = "Not Set"
            merged_row["Category"] = "Not Set"
            merged_row["Listed_on_EML_2019"] = "Not Set"
            merge_results.append(merged_row)
    
    merged_df = pd.DataFrame(merge_results)
    
    # ===== ADD BROAD CLASS =====
    def compute_broad_class(row):
        """Collapse all β-lactams into one broad category."""
        if class_column not in row or pd.isna(row[class_column]):
            return "Not Set"
        
        cls = str(row[class_column]).lower()
        if "β-lactam" in str(row[class_column]) or "beta-lactam" in cls:
            return "β-lactam"
        return row[class_column]
    
    if class_column in merged_df.columns:
        merged_df[broad_class_column] = merged_df.apply(compute_broad_class, axis=1)
    
    # ===== SAVE AND REPORT =====
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"MERGE RESULTS")
    print(f"{'='*70}")
    print(f"\nExisting antibiotics matched: {matched_count}/{len(existing_df)} ({100*matched_count/len(existing_df):.1f}%)")
    print(f"  - Perfect matches: {perfect_matches}")
    print(f"  - Fuzzy matches: {fuzzy_matches}")
    print(f"  - Unmatched: {len(unmatched)} ({100*len(unmatched)/len(existing_df):.1f}%)")
    
    if unmatched:
        print(f"\n⚠ UNMATCHED ANTIBIOTICS ({len(unmatched)}):")
        for ab in sorted(unmatched)[:100]:
            print(f"  - {ab}")
        if len(unmatched) > 100:
            print(f"  ... and {len(unmatched) - 100} more")
    
    matched_who_names = set(merged_df[merged_df["WHO_Match"] != "NO MATCH"]["WHO_Match"])
    unmatched_who = set(who_df["Antibiotic"]) - matched_who_names
    
    if unmatched_who:
        print(f"\n⚠ WHO ANTIBIOTICS NOT IN EXISTING FILE ({len(unmatched_who)}):")
        for ab in sorted(unmatched_who)[:100]:
            print(f"  - {ab}")
        if len(unmatched_who) > 100:
            print(f"  ... and {len(unmatched_who) - 100} more")
    
    print(f"\n✓ Merged dataset saved to: {output_file}")
    print(f"  Total records: {len(merged_df)}")
    print(f"  New columns: WHO_Match, Match_Type, WHO_Class, WHO_ATC_code, Category, Listed_on_EML_2019, {broad_class_column}")
    
    return merged_df

def compute_row_features(row, antibiotics, class_map, who_class_map):
    """
    Uses the *_Tested flags for each antibiotic to avoid mutating raw R/I/S values.
    """
    tested_abx = [abx for abx in antibiotics if row.get(f"{abx}_Tested", 0) == 1]
    classes = set(class_map.get(abx) for abx in tested_abx if abx in class_map)
    who_flags = [who_class_map.get(abx) for abx in tested_abx if abx in who_class_map]

    num_classes = len(classes)
    is_critical = int(any(cls in ['Watch', 'Reserve'] for cls in who_flags))
    is_reserve  = int(any(cls == 'Reserve' for cls in who_flags))

    return pd.Series([num_classes, is_critical, is_reserve])

def prepare_feature_inputs(ars_data: pd.DataFrame, who_data: pd.DataFrame):
    """
    - Returns antibiotic names and mappings.
    - Creates <abx>_Tested flags WITHOUT touching the raw antibiotic result columns (R/I/S/NaN).
    """
    antibiotics = [col for col in ars_data.columns if col in who_data['Full Name'].values]
    who_filtered = who_data[who_data['Full Name'].isin(antibiotics)]

    class_map = dict(zip(who_filtered['Full Name'], who_filtered['Class']))
    who_class_map = dict(zip(who_filtered['Full Name'], who_filtered['Category']))

    # Create _Tested flags in a non-destructive way
    for abx in antibiotics:
        tested_col = f"{abx}_Tested"
        if tested_col not in ars_data.columns:
            ars_data[tested_col] = ars_data[abx].notna().astype("Int8")

    return antibiotics, class_map, who_class_map


########### DENDROGRAM IMPLEMENTATION STARTS HERE USIN ALTAIR ########

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _validate_and_clean_matrix(df: pd.DataFrame,
                               *,
                               clean: str = 'auto',
                               na_fill: float = 0.0,
                               diag_fill: float = 1.0) -> pd.DataFrame:
    df_num = df.copy().apply(pd.to_numeric, errors='coerce')
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    r0, c0 = df_num.shape

    if clean in ('auto', 'fill'):
        if df_num.shape[0] == df_num.shape[1] and list(df_num.index) == list(df_num.columns):
            np.fill_diagonal(df_num.values, diag_fill)

    if clean == 'drop':
        keep_r, keep_c = df_num.notna().all(axis=1), df_num.notna().all(axis=0)
        df_num = df_num.loc[keep_r, keep_c]
    elif clean == 'auto':
        keep_r, keep_c = ~df_num.isna().all(axis=1), ~df_num.isna().all(axis=0)
        df_num = df_num.loc[keep_r, keep_c]
        if df_num.isna().any().any():
            df_num.fillna(na_fill, inplace=True)
    elif clean == 'fill':
        df_num.fillna(na_fill, inplace=True)
    elif clean != 'none':
        raise ValueError(f"Unknown clean mode: {clean!r}")

    row_var, col_var = df_num.var(axis=1), df_num.var(axis=0)
    if (row_var == 0).any() or (col_var == 0).any():
        print(
            f"[clustergram] warning: {(row_var == 0).sum()} constant rows, {(col_var == 0).sum()} constant cols.")
    if df_num.shape != (r0, c0):
        print(f"[clustergram] cleaned shape {df_num.shape} (was {r0}x{c0}).")
    if not np.isfinite(df_num.values).all():
        raise ValueError("Cleaning failed: non-finite values remain.")
    return df_num


def _distance_vector_from_similarity(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("‘1-sim’ requires square matrix.")
    if not np.allclose(arr, arr.T, equal_nan=True):
        arr = (arr + arr.T) / 2
    d = 1.0 - arr
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def _compute_linkage(df: pd.DataFrame, *, rows: bool, metric: str,
                     method: str = 'average', metric_kwargs: Optional[dict] = None):
    if df.shape[0] < 2:
        raise ValueError(
            "Insufficient observations to cluster (need ≥2 rows).")
    if metric == '1-sim':
        dist = _distance_vector_from_similarity(df)
        if dist.size < 1:
            raise ValueError("Insufficient data for 1-sim.")
        return linkage(dist, method=method)
    data = df.values if rows else df.values.T
    if data.shape[1] < 1:
        raise ValueError("No features remain for correlation.")
    kw = {} if metric_kwargs is None else metric_kwargs
    dist = pdist(data, metric=metric, **kw)
    if dist.size < 1 or not np.isfinite(dist).all():
        raise ValueError("Cannot compute finite correlation distances.")
    return linkage(dist, method=method)


def _leaf_order(Z):
    return dendrogram(Z, no_plot=True, orientation='top')['leaves']


def _tree_members(Z):
    _, nodes = to_tree(Z, rd=True)

    def leaves(nd):
        return [nd.id] if nd.is_leaf() else leaves(nd.left) + leaves(nd.right)
    return {nd.id: leaves(nd) for nd in nodes}


def _node_heights(Z):
    n = Z.shape[0] + 1
    h = np.zeros(2 * n - 1)
    for i in range(n - 1):
        h[n + i] = Z[i, 2]
    return h


def _empty_axis_data(labels):
    n = len(labels)
    nodes = [{'node': i, 'lo': i, 'hi': i, 'count': 1, 'height': 0.0,
              'pos': float(i), 'leaf': True} for i in range(n)]
    return pd.DataFrame([]), pd.DataFrame(nodes), labels


def _axis_has_structure(n_labels: int) -> bool:
    return n_labels >= 2


# ----------------------------------------------------------------------
# Build dendrogram data
# ----------------------------------------------------------------------

def _build_row_dendro_data(df: pd.DataFrame, metric: str,
                           metric_kwargs: Optional[dict] = None):
    try:
        Z = _compute_linkage(df, rows=True, metric=metric,
                             method='average', metric_kwargs=metric_kwargs)
    except Exception as e:
        print(f"[clustergram] row clustering failed ({e}); identity order.")
        return _empty_axis_data(list(df.index))
    order = _leaf_order(Z)
    labels = [df.index[i] for i in order]
    pos = {orig: disp for disp, orig in enumerate(order)}
    members, heights = _tree_members(Z), _node_heights(Z)
    segs, nodes = [], []
    n = len(order)
    for i in range(n):
        nodes.append({'node': i, 'lo': i, 'hi': i, 'count': 1,
                      'height': 0.0, 'pos': float(i), 'leaf': True})
    for i in range(n - 1):
        p, nL, nR = n + i, int(Z[i, 0]), int(Z[i, 1])
        Ls, Rs = members[nL], members[nR]
        pL, pR = [pos[x] for x in Ls], [pos[x] for x in Rs]
        lo, hi = min(pL + pR), max(pL + pR)
        cL, cR = float(np.mean(pL)), float(np.mean(pR))
        hL, hR, hP = heights[nL], heights[nR], heights[p]
        segs.extend([
            {'x': hL, 'y': cL, 'x2': hP, 'y2': cL},
            {'x': hR, 'y': cR, 'x2': hP, 'y2': cR},
            {'x': hP, 'y': cL, 'x2': hP, 'y2': cR}
        ])
        nodes.append({'node': p, 'lo': lo, 'hi': hi, 'count': len(pL) + len(pR),
                      'height': hP, 'pos': (cL + cR) / 2, 'leaf': False})
    return pd.DataFrame(segs), pd.DataFrame(nodes), labels


def _build_col_dendro_data(df: pd.DataFrame, metric: str,
                           metric_kwargs: Optional[dict] = None):
    try:
        Z = _compute_linkage(df, rows=False, metric=metric,
                             method='average', metric_kwargs=metric_kwargs)
    except Exception as e:
        print(f"[clustergram] col clustering failed ({e}); identity order.")
        return _empty_axis_data(list(df.columns))
    order = _leaf_order(Z)
    labels = [df.columns[i] for i in order]
    pos = {orig: disp for disp, orig in enumerate(order)}
    members, heights = _tree_members(Z), _node_heights(Z)
    segs, nodes = [], []
    n = len(order)
    for i in range(n):
        nodes.append({'node': i, 'lo': i, 'hi': i, 'count': 1,
                      'height': 0.0, 'pos': float(i), 'leaf': True})
    for i in range(n - 1):
        p, nL, nR = n + i, int(Z[i, 0]), int(Z[i, 1])
        Ls, Rs = members[nL], members[nR]
        pL, pR = [pos[x] for x in Ls], [pos[x] for x in Rs]
        lo, hi = min(pL + pR), max(pL + pR)
        cL, cR = float(np.mean(pL)), float(np.mean(pR))
        hL, hR, hP = heights[nL], heights[nR], heights[p]
        segs.extend([
            {'x': cL, 'y': hL, 'x2': cL, 'y2': hP},
            {'x': cR, 'y': hR, 'x2': cR, 'y2': hP},
            {'x': cL, 'y': hP, 'x2': cR, 'y2': hP}
        ])
        nodes.append({'node': p, 'lo': lo, 'hi': hi, 'count': len(pL) + len(pR),
                      'height': hP, 'pos': (cL + cR) / 2, 'leaf': False})
    return pd.DataFrame(segs), pd.DataFrame(nodes), labels

# ----------------------------------------------------------------------
# Render dendrogram charts
# ----------------------------------------------------------------------


def _row_dendrogram(seg_df, node_df, n_rows, side, rowHover, rowSelect,
                    panel_width, panel_height, interactive, *, row_flip: bool = False):
    yscale = alt.Scale(domain=[-0.5, n_rows-0.5], reverse=(not row_flip))
    xscale = alt.Scale(reverse=(side == 'left'))
    lines = alt.Chart(seg_df).mark_rule(color='#999999').encode(
        x=alt.X('x:Q', axis=None, scale=xscale),
        x2='x2:Q',
        y=alt.Y('y:Q', axis=None, scale=yscale),
        y2='y2:Q'
    )
    enable = (
        interactive and rowHover and rowSelect and not node_df.empty and n_rows >= 2)
    if not enable:
        return lines.properties(width=panel_width, height=panel_height)
    pts = alt.Chart(node_df).mark_point(opacity=0, size=600).encode(
        x=alt.X('height:Q', scale=xscale),
        y=alt.Y('pos:Q', scale=yscale),
        tooltip=[alt.Tooltip(f, title=f.capitalize())
                 for f in ('node', 'count', 'lo', 'hi')]
    ).add_params(rowHover, rowSelect)
    hover = alt.Chart(node_df).mark_point(color='#555555', size=70).encode(
        x=alt.X('height:Q', scale=xscale), y=alt.Y('pos:Q', scale=yscale)
    ).transform_filter(rowHover)
    select = alt.Chart(node_df).mark_point(color='#000000', size=90).encode(
        x=alt.X('height:Q', scale=xscale), y=alt.Y('pos:Q', scale=yscale)
    ).transform_filter(rowSelect)
    return (lines+pts+hover+select).properties(width=panel_width, height=panel_height)


def _col_dendrogram(seg_df, node_df, n_cols, colHover, colSelect, panel_height, interactive):
    xscale = alt.Scale(domain=[-0.5, n_cols-0.5])
    lines = alt.Chart(seg_df).mark_rule(color='#999999').encode(
        x=alt.X('x:Q', axis=None, scale=xscale),
        x2='x2:Q',
        y=alt.Y('y:Q', axis=None),
        y2='y2:Q'
    )
    enable = (
        interactive and colHover and colSelect and not node_df.empty and n_cols >= 2)
    if not enable:
        return lines.properties(height=panel_height)
    pts = alt.Chart(node_df).mark_point(opacity=0, size=600).encode(
        x=alt.X('pos:Q', scale=xscale),
        y='height:Q',
        tooltip=[alt.Tooltip(f, title=f.capitalize())
                 for f in ('node', 'count', 'lo', 'hi')]
    ).add_params(colHover, colSelect)
    hover = alt.Chart(node_df).mark_point(color='#555555', size=70).encode(
        x=alt.X('pos:Q', scale=xscale), y='height:Q'
    ).transform_filter(colHover)
    select = alt.Chart(node_df).mark_point(color='#000000', size=90).encode(
        x=alt.X('pos:Q', scale=xscale), y='height:Q'
    ).transform_filter(colSelect)
    return (lines+pts+hover+select).properties(height=panel_height)

# ----------------------------------------------------------------------
# Full clustergram
# ----------------------------------------------------------------------

def plot_clustergram_with_dendrograms(
    sim_matrix: pd.DataFrame,
    *,
    include_row_dendrogram: bool = True,
    include_col_dendrogram: bool = True,
    interactive: bool = False,
    metric: str = 'auto',
    row_metric: Optional[str] = None,        # NEW
    col_metric: Optional[str] = None,        # NEW
    clean: str = 'auto',
    na_fill: float = 0.0,
    diag_fill: float = 1.0,
    # ---- sorting controls ----
    sort_cols_by: Optional[str] = None,
    sort_rows_by: Optional[str] = None,
    force_col_sort_over_cluster: bool = False,
    force_row_sort_over_cluster: bool = False,
    # ---- color controls ----
    color_domain: Optional[List[float]] = None,
    color_range: Optional[List[str]] = None,
    legend_min_ticks: int = 6,
    zero_anchor: float = 0.0,
    neg_color: str = "#d98b3a",
    zero_color: str = "#ffffff",
    pos_color: str = "#2c68af",
    # ---- layout / labels ----
    row_dendrogram_side: str = 'left',
    row_flip: bool = False,
    cell_height: int = 14,
    cell_width: Optional[int] = None,
    heatmap_height: Optional[int] = None,
    heatmap_width: Optional[int] = None,
    row_panel_width: int = 60,
    col_panel_height: int = 60,
    highlight_mode: str = 'and',
    gutter_px: int = 0,
    gutter_py: int = 0,
    title: str = 'Interactive Clustergram',
    source: str = "Source",
    target: str = "Target",
    apply_config: bool = True,
    legend_title: str = "Similarity",
    metric_kwargs: Optional[dict] = None,
    axis_label_font_size: int = 14,
    axis_title_font_size: int = 16,
    legend_label_font_size: int = 12,
    legend_title_font_size: int = 13,
):
    import numpy as np
    import altair as alt
    from scipy.spatial.distance import pdist

    if not include_row_dendrogram and not include_col_dendrogram:
        interactive = False

    cluster_rows_when_hidden = True
    cluster_cols_when_hidden = False
    respect_side_when_hidden = False

    sim_matrix = _validate_and_clean_matrix(sim_matrix,
                                            clean=clean,
                                            na_fill=na_fill,
                                            diag_fill=diag_fill)

    def _agg_order(df: pd.DataFrame, axis: int, how: str) -> list[str]:
        stat = getattr(df, how)(axis=axis)
        return stat.sort_values(ascending=False).index.tolist()

    desired_col_order = list(sim_matrix.columns)
    desired_row_order = list(sim_matrix.index)
    if sort_cols_by:
        desired_col_order = _agg_order(sim_matrix, axis=0, how=sort_cols_by)
    if sort_rows_by:
        desired_row_order = _agg_order(sim_matrix, axis=1, how=sort_rows_by)

    if not include_col_dendrogram:
        sim_matrix = sim_matrix.loc[:, desired_col_order]
    if not include_row_dendrogram:
        sim_matrix = sim_matrix.loc[desired_row_order, :]

    # pick default metric
    if metric == 'auto':
        r_const = (sim_matrix.var(axis=1) == 0).any()
        c_const = (sim_matrix.var(axis=0) == 0).any()
        if r_const or c_const:
            metric_eff = '1-sim'
        else:
            try:
                d = pdist(sim_matrix.values, 'correlation')
                metric_eff = 'correlation' if np.isfinite(d).all() else '1-sim'
            except Exception:
                metric_eff = '1-sim'
    else:
        metric_eff = metric

    # per-axis overrides
    r_metric = row_metric or metric_eff
    c_metric = col_metric or metric_eff

    if r_metric == 'correlation' or c_metric == 'correlation':
        rv, cv = sim_matrix.var(axis=1) > 0, sim_matrix.var(axis=0) > 0
        if (~rv).sum() or (~cv).sum():
            sim_matrix = sim_matrix.loc[rv, cv]
        if sim_matrix.shape[0] < 2 or sim_matrix.shape[1] < 2:
            r_metric = c_metric = '1-sim'

    if include_row_dendrogram or cluster_rows_when_hidden:
        row_seg, row_nodes, row_labels = _build_row_dendro_data(sim_matrix, metric=r_metric,
                                                                metric_kwargs=metric_kwargs)
        if force_row_sort_over_cluster and sort_rows_by:
            row_labels = desired_row_order
    else:
        row_labels = list(sim_matrix.index)
        row_seg = row_nodes = pd.DataFrame([])

    if include_col_dendrogram or cluster_cols_when_hidden:
        col_seg, col_nodes, col_labels = _build_col_dendro_data(sim_matrix, metric=c_metric,
                                                                metric_kwargs=metric_kwargs)
        if force_col_sort_over_cluster and sort_cols_by:
            col_labels = desired_col_order
    else:
        col_labels = list(sim_matrix.columns)
        col_seg = col_nodes = pd.DataFrame([])

    nr, nc = len(row_labels), len(col_labels)

    if cell_width is None:
        cell_width = cell_height
    heat_w = heatmap_width if heatmap_width is not None else nc * cell_width
    heat_h = heatmap_height if heatmap_height is not None else nr * cell_height

    mat = sim_matrix.loc[row_labels, col_labels]
    m = (mat.reset_index()
            .melt(id_vars='index', var_name='target', value_name='similarity')
            .rename(columns={'index': 'source'}))
    m['row_idx'] = m['source'].map(
        {lab: i for i, lab in enumerate(row_labels)})
    m['col_idx'] = m['target'].map(
        {lab: i for i, lab in enumerate(col_labels)})

    row_sig = interactive and include_row_dendrogram and _axis_has_structure(
        nr)
    col_sig = interactive and include_col_dendrogram and _axis_has_structure(
        nc)

    rowHover = rowSelect = colHover = colSelect = None
    if row_sig:
        rowHover = alt.selection_point(name='rowHover',  fields=['node', 'lo', 'hi'],
                                       on='mouseover', clear='mouseout')
        rowSelect = alt.selection_point(name='rowSelect', fields=['node', 'lo', 'hi'],
                                        on='click', toggle=False, clear='dblclick')
    if col_sig:
        colHover = alt.selection_point(name='colHover',  fields=['node', 'lo', 'hi'],
                                       on='mouseover', clear='mouseout')
        colSelect = alt.selection_point(name='colSelect', fields=['node', 'lo', 'hi'],
                                        on='click', toggle=False, clear='dblclick')

    row_chart = (_row_dendrogram(row_seg, row_nodes, nr, row_dendrogram_side,
                                 rowHover, rowSelect,
                                 row_panel_width, heat_h, row_sig,
                                 row_flip=row_flip)
                 if include_row_dendrogram else None)

    col_chart = (_col_dendrogram(col_seg, col_nodes, nc,
                                 colHover, colSelect,
                                 col_panel_height, col_sig)
                 .properties(width=heat_w)
                 if include_col_dendrogram else None)

    if include_row_dendrogram and respect_side_when_hidden:
        y_orient = 'right' if row_dendrogram_side == 'left' else 'left'
        label_align = 'left' if row_dendrogram_side == 'left' else 'right'
    elif include_row_dendrogram:
        y_orient = 'right' if row_dendrogram_side == 'left' else 'left'
        label_align = 'left' if row_dendrogram_side == 'left' else 'right'
    else:
        y_orient = 'left'
        label_align = 'right'

    y_ax = alt.Axis(
        orient=y_orient, title=source,
        labelAlign=label_align, labelPadding=1,
        tickSize=0,
        labelFontSize=axis_label_font_size,
        titleFontSize=axis_title_font_size,
        # grid=True, gridColor='lightgray'
    )
    x_ax = alt.Axis(
        orient='bottom', labelAngle=270,
        labelPadding=2, tickSize=0, title=target,
        labelFontSize=axis_label_font_size,
        titleFontSize=axis_title_font_size,
        # grid=True, gridColor='lightgray'
    )

    x_scale = alt.Scale(domain=col_labels, paddingInner=0, paddingOuter=0)
    y_scale = alt.Scale(domain=row_labels, paddingInner=0, paddingOuter=0)

    tol = 1e-9
    m["similarity"] = np.where(np.isclose(m["similarity"], zero_anchor, atol=tol),
                               zero_anchor,
                               m["similarity"])

    data_min = float(m["similarity"].min())
    data_max = float(m["similarity"].max())

    if (color_domain is not None) and (color_range is not None):
        dom = color_domain
        rng = color_range
    else:
        if color_domain is None:
            dom_lo = min(data_min, zero_anchor)
            dom_hi = max(data_max, zero_anchor)
            dom = [dom_lo, dom_hi]
        else:
            dom = color_domain

        if color_range is None:
            if dom[0] < zero_anchor < dom[-1]:
                eps = max((dom[-1] - dom[0]) * 1e-6, 1e-9)
                dom = [dom[0], zero_anchor - eps,
                       zero_anchor, zero_anchor + eps, dom[-1]]
                rng = [neg_color, neg_color, zero_color, pos_color, pos_color]
            else:
                rng = ['#fdfdfd', pos_color]
        else:
            rng = color_range

    color_scale = alt.Scale(domain=dom, range=rng, clamp=True)

    span_min, span_max = (
        dom[0], dom[-1]) if len(dom) >= 2 else (dom[0], dom[0])
    if span_min == span_max:
        ticks = [span_min] * legend_min_ticks
    else:
        ticks = list(np.linspace(span_min, span_max, max(legend_min_ticks, 2)))
        if span_min < zero_anchor < span_max and not np.any(np.isclose(ticks, zero_anchor)):
            ticks.append(zero_anchor)
        ticks = sorted(set(np.round(ticks, 12)))

    legend_obj = alt.Legend(
        title=legend_title,
        values=ticks,
        gradientOpacity=1.0, format=".2f",
        labelFontSize=legend_label_font_size,
        titleFontSize=legend_title_font_size,
    )

    base_heat = alt.Chart(m).mark_rect().encode(
        x=alt.X('target:N', sort=None, axis=x_ax, scale=x_scale),
        y=alt.Y('source:N', sort=None, axis=y_ax, scale=y_scale),
        color=alt.Color('similarity:Q', scale=color_scale, legend=legend_obj),
        opacity=alt.value(0.25 if interactive else 1.0),
        tooltip=[
            alt.Tooltip('source:N', title=source),
            alt.Tooltip('target:N', title=target),
            alt.Tooltip('similarity:Q', title=legend_title)
        ]
    ).properties(width=heat_w, height=heat_h)

    if interactive:
        row_expr = ("(isDefined(rowSelect.lo) ? datum.row_idx>=rowSelect.lo&&datum.row_idx<=rowSelect.hi : "
                    "(isDefined(rowHover.lo)? datum.row_idx>=rowHover.lo&&datum.row_idx<=rowHover.hi : true))") if row_sig else "true"
        col_expr = ("(isDefined(colSelect.lo) ? datum.col_idx>=colSelect.lo&&datum.col_idx<=colSelect.hi : "
                    "(isDefined(colHover.lo)? datum.col_idx>=colHover.lo&&datum.col_idx<=colHover.hi : true))") if col_sig else "true"

        if highlight_mode == 'row':
            expr = row_expr
        elif highlight_mode == 'col':
            expr = col_expr
        elif highlight_mode == 'union':
            expr = f"({row_expr})||({col_expr})"
        else:
            expr = f"({row_expr})&&({col_expr})"

        heat_high = alt.Chart(m).transform_filter(expr).mark_rect().encode(
            x=alt.X('target:N', sort=None, axis=x_ax, scale=x_scale),
            y=alt.Y('source:N', sort=None, axis=y_ax, scale=y_scale),
            color=alt.Color('similarity:Q', scale=color_scale, legend=None),
            opacity=alt.value(1.0)
        ).properties(width=heat_w, height=heat_h)

        heatmap = base_heat + heat_high
    else:
        heatmap = base_heat

    spacer = None
    if include_col_dendrogram and include_row_dendrogram:
        spacer = alt.Chart(pd.DataFrame({'_': [0]})).mark_rect(opacity=0).properties(
            width=row_panel_width, height=col_panel_height
        )

    if include_row_dendrogram and include_col_dendrogram:
        if row_dendrogram_side == 'left':
            left = alt.vconcat(spacer, row_chart, spacing=gutter_py)
            right = alt.vconcat(col_chart, heatmap, spacing=gutter_py)
            outer = alt.hconcat(left, right, spacing=gutter_px)
        else:
            left = alt.vconcat(col_chart, heatmap, spacing=gutter_py)
            right = alt.vconcat(spacer, row_chart, spacing=gutter_py)
            outer = alt.hconcat(left, right, spacing=gutter_px)
    elif include_row_dendrogram:
        outer = (alt.hconcat(row_chart, heatmap, spacing=gutter_px)
                 if row_dendrogram_side == 'left'
                 else alt.hconcat(heatmap, row_chart, spacing=gutter_px))
    elif include_col_dendrogram:
        outer = alt.vconcat(col_chart, heatmap, spacing=gutter_py)
    else:
        outer = heatmap

    if not apply_config:
        return outer

    return (outer
            .configure_concat(spacing=0)
            .configure_view(stroke=None)
            .configure_axis(labelFontSize=10, titleFontSize=12, labelLimit=0)
            .properties(title=title)
            .resolve_scale(color='independent', y='independent')
            .configure_title(fontSize=18, anchor='middle'))


def plot_tests_boxplot(
    test_data_df,
    antibiotic_col: str = "Antibiotic",
    antibiotic_classes_title: str = "Antibiotic Class",
    group_col: str = "Bundesland",
    compare_col: Optional[str] = None,
    population_data_path: Optional[str] = None,
    population_option: str = "latest",
    custom_years: Optional[List[int]] = None,
    antibiotic_classes: Optional[Dict[str, List[str]]] = None,
    extent: float = 1.5,
    sort_stat: str = "median",            # one of: median, mean, q1, q3, iqr
    antibiotics_to_plot: Optional[List[str]] = None,
    box_colors: Optional[Dict[str, str]] = None,
    width: int = 1300,
    height: int = 500,
    yaxis_range: Optional[tuple] = None,
    threshold_line: Optional[float] = 1.0,
    threshold_line_color: str = "red",
    threshold_line_dash: str = "dash",
    show_threshold_line: bool = True,
    export_csv_path: Optional[str] = None,
    min_group_size: int = 1,
    box_color: str = "#5580B0",
    outlier_color: str = "crimson",
    show_outliers: bool = False,
    # "outliers" | "suspectedoutliers" | "all" | False
    box_point_option: str = "all",
    xaxis_label_format: str = "full",     # "abbr" | "both" | "full"
    legend: dict = dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),
    title: str = "",
    # optional class awareness (antibiotic-level mode only)
    # maps raw *_Tested column -> class name
    antibiotic_class_map: Optional[Dict[str, str]] = None,
    # maps class name -> color
    class_palette: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Build an interactive Plotly box plot for antibiotic testing coverage.

    Modes
    -----
    1) Class level:
       pass `antibiotic_classes={class_name: [*_Tested, ...], ...}` to aggregate columns
       into classes; x-axis will be class names.

    2) Antibiotic level:
       omit `antibiotic_classes` to plot each *_Tested column individually.
       If `antibiotic_class_map` is provided, antibiotics on the x-axis are grouped
       into contiguous blocks by class (Access -> Watch -> Reserve -> Unknown),
       and traces are colored by that class.

    Normalization
    -------------
    - If (compare_col is set) and not (Bundesland+population mode), Value = Count / N * 100.
    - If (group_col == 'Bundesland' and population_data_path is provided), Value = Count / Population * 100000.
    - Otherwise Value = Count / N * 100.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # ---------- load / copy ----------
    if isinstance(test_data_df, str):
        df = pd.read_csv(test_data_df)
    else:
        df = test_data_df.copy()

    if df.empty:
        raise ValueError("test_data_df is empty.")
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not found in dataframe.")
    if compare_col and compare_col not in df.columns:
        raise KeyError(f"compare_col '{compare_col}' not found in dataframe.")

    # default palette for AWaRe classes
    if class_palette is None:
        class_palette = {"Access": "#56B4E9", "Watch": "#E69F00",
                         "Reserve": "#D55E00", "Unknown": "#999999"}

    # ---------- long form prep ----------
    id_vars = [group_col] + ([compare_col] if compare_col else [])
    group_fields = [antibiotic_col, group_col] + \
        ([compare_col] if compare_col else [])

    if antibiotic_classes:
        # aggregate to class level
        df_agg = df[id_vars].copy()
        class_cols: List[str] = []
        for class_name, abx_list in antibiotic_classes.items():
            existing_cols = [c for c in abx_list if c in df.columns]
            if existing_cols:
                # sum of tested indicators across all antibiotics in the class
                df_agg[class_name] = df[existing_cols].sum(axis=1)
                class_cols.append(class_name)
        if not class_cols:
            raise ValueError(
                "None of the antibiotic class columns exist in the dataframe.")

        df_long = df_agg.melt(id_vars=id_vars, value_vars=class_cols,
                              var_name=antibiotic_col, value_name="Count_per_sample")
        grouped = (df_long.groupby(group_fields, observed=False)["Count_per_sample"]
                   .sum().reset_index(name="Count"))
    else:
        # antibiotic level
        indicator_cols = [c for c in df.columns if c.endswith("_Tested")]
        if not indicator_cols:
            raise ValueError("No antibiotic *_Tested columns were found.")
        if antibiotics_to_plot:
            missing = [c for c in antibiotics_to_plot if c not in df.columns]
            if missing:
                print(f"[warn] antibiotics_to_plot not in df: {missing}")
            indicator_cols = [
                c for c in indicator_cols if c in set(antibiotics_to_plot)]
            if not indicator_cols:
                raise ValueError(
                    "After filtering, no antibiotic columns remain to plot.")

        df_long = df.melt(id_vars=id_vars, value_vars=indicator_cols,
                          var_name=antibiotic_col, value_name="Tested")

        grouped = (df_long[df_long["Tested"] == 1]
                   .groupby(group_fields, observed=False)["Tested"]
                   .sum().reset_index(name="Count"))

    # ---------- normalization ----------
    if compare_col and compare_col != "Year" and group_col != "Bundesland":
        base_group = [group_col, compare_col]
        base_sizes = df.groupby(
            base_group, observed=False).size().reset_index(name="N")
        merged = grouped.merge(base_sizes, on=base_group, how="inner")
        merged["Value"] = merged["Count"] / merged["N"] * 100
        ylabel = "Mean Test Coverage (%)"
    else:
        if group_col == "Bundesland" and population_data_path:
            pop_df = pd.read_csv(population_data_path)
            pop_df.columns = [c.strip() for c in pop_df.columns]
            required = {"bundesland", "Year", "total"}
            if not required.issubset(set(c.lower() for c in pop_df.columns)):
                raise KeyError(
                    "population_data must include 'bundesland', 'Year', 'total' columns.")
            pop_df["Bundesland"] = pop_df["bundesland"]

            if population_option == "custom" and custom_years:
                pop_df = pop_df[pop_df["Year"].isin(custom_years)]
                if pop_df.empty:
                    raise ValueError(
                        "No population rows remain for specified custom_years.")
            else:
                latest = pop_df.groupby("Bundesland", observed=False)[
                    "Year"].max().reset_index()
                pop_df = pop_df.merge(
                    latest, on=["Bundesland", "Year"], how="inner")

            pop_summary = (pop_df.groupby("Bundesland", observed=False)["total"]
                           .sum().reset_index(name="Population"))
            merged = grouped.merge(pop_summary, on="Bundesland", how="left")
            merged["Value"] = merged["Count"] / merged["Population"] * 100000
            ylabel = "Tests per 100k Population"
        else:
            group_sizes = df.groupby(
                group_col, observed=False).size().rename("N").reset_index()
            if min_group_size > 1:
                valid = group_sizes[group_sizes["N"]
                                    >= min_group_size][group_col]
                group_sizes = group_sizes[group_sizes[group_col].isin(valid)]
            merged = grouped.merge(group_sizes, on=group_col, how="inner")
            merged["Value"] = merged["Count"] / merged["N"] * 100
            ylabel = "Mean Test Coverage (%)"

    # ---------- sorting + labels ----------
    merged[antibiotic_col] = merged[antibiotic_col].astype(str)

    # stats per antibiotic/class for ordering
    group_stats = merged.groupby(antibiotic_col, observed=False)["Value"].agg(
        median="median",
        mean="mean",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )
    group_stats["iqr"] = group_stats["q3"] - group_stats["q1"]
    if sort_stat not in group_stats.columns:
        raise ValueError(
            f"sort_stat must be one of {list(group_stats.columns)}")

    # Build label map and ordered x category list
    label_map: Dict[str, str] = {}
    xaxis_title = antibiotic_col

    if antibiotic_classes:
        # class level
        nonempty = merged[antibiotic_col].dropna().unique()
        order = group_stats[sort_stat].sort_values(
            ascending=False).index.tolist()
        new_order = [abx for abx in order if abx in nonempty]
        xaxis_title = antibiotic_classes_title
    else:
        # antibiotic level – pretty labels + class blocks
        all_keys_sorted = group_stats[sort_stat].sort_values(
            ascending=False).index.tolist()
        for original_name in all_keys_sorted:
            cleaned = original_name.replace("_Tested", "")
            parts = cleaned.split(" - ")
            if len(parts) == 2:
                abbr, full = parts[0].strip(), parts[1].strip()
                label_map[original_name] = (
                    abbr if xaxis_label_format == "abbr"
                    else f"{abbr} ({full})" if xaxis_label_format == "both"
                    else full
                )
            else:
                label_map[original_name] = cleaned

        merged["_PrettyLabel"] = merged[antibiotic_col].map(label_map)

        # derive class per raw key; fall back to "Unknown"
        abx_to_class = {str(k): str(v)
                        for k, v in (antibiotic_class_map or {}).items()}
        order_classes = ["Access", "Watch", "Reserve", "Unknown"]

        class_to_members = {c: [] for c in order_classes}
        for abx in all_keys_sorted:
            cls = abx_to_class.get(abx, "Unknown")
            if cls not in class_to_members:
                cls = "Unknown"
            class_to_members[cls].append(abx)

        # choose within-class order by selected stat
        abx_order_raw: List[str] = []
        for cls in order_classes:
            members = class_to_members[cls]
            if not members:
                continue
            members_sorted = sorted(
                members, key=lambda k: float(group_stats.loc[k, sort_stat]), reverse=True
            )
            abx_order_raw.extend(members_sorted)

        new_order = [label_map[k] for k in abx_order_raw]
        merged[antibiotic_col] = merged["_PrettyLabel"]

    # Make x categorical in the exact order we want (must be unique)
    new_order = pd.unique(pd.Series(new_order)).tolist()
    merged[antibiotic_col] = pd.Categorical(merged[antibiotic_col],
                                            categories=new_order, ordered=True)

    # ---------- outliers (IQR) ----------
    stats_group_by = [antibiotic_col] + ([compare_col] if compare_col else [])
    stats = (merged.groupby(stats_group_by, observed=False)["Value"]
             .agg(q1=lambda x: x.quantile(0.25),
                  q3=lambda x: x.quantile(0.75))
             .assign(iqr=lambda d: d.q3 - d.q1,
                     lower=lambda d: d.q1 - extent * d.iqr,
                     upper=lambda d: d.q3 + extent * d.iqr)
             .reset_index())
    merged = merged.merge(stats, on=stats_group_by, how="left")
    merged["is_outlier"] = (merged["Value"] < merged["lower"]) | (
        merged["Value"] > merged["upper"])

    # export tidy data if requested
    if export_csv_path:
        merged.to_csv(export_csv_path, index=False)

    # denominator label for hover
    if "Population" in merged.columns:
        merged["Denominator"] = merged["Population"]
        denominator_label = "Population"
    elif "N" in merged.columns:
        merged["Denominator"] = merged["N"]
        denominator_label = "N (Samples)"
    else:
        merged["Denominator"] = "N/A"
        denominator_label = "N/A"

    # ---------- plot ----------
    fig = go.Figure()

    if compare_col:
        # grouped boxes colored by compare_col
        unique_groups = merged[compare_col].dropna().unique()
        num_groups = len(unique_groups)
        if not box_colors:
            default_palette = ["#a6cee3", "#1f78b4",
                               "#b2df8a", "#33a02c", "#fb9a99"]
            box_colors = {group: default_palette[i % len(default_palette)]
                          for i, group in enumerate(unique_groups)}
        offsets = np.linspace(-0.3, 0.3, num_groups)
        for i, group in enumerate(unique_groups):
            sub_df = merged[merged[compare_col] == group]
            custom = sub_df[[group_col, "Count", "Denominator"]]
            fig.add_trace(go.Box(
                x=sub_df[antibiotic_col],
                y=sub_df["Value"],
                name=str(group),
                marker_color=box_colors.get(group, box_color),
                boxpoints=box_point_option,
                jitter=0.4,
                pointpos=offsets[i],
                customdata=custom,
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    f"Antibiotic: %{{x}}<br>"
                    f"{group_col}: %{{customdata[0]}}<br>"
                    f"Value: %{{y:.2f}}<br><br>"
                    f"<b>Count: %{{customdata[1]}}</b><br>"
                    f"<b>{denominator_label}: %{{customdata[2]}}</b>"
                    f"<extra></extra>"
                )
            ))
        fig.update_layout(boxmode="group")
        
        if antibiotic_class_map and not antibiotic_classes:
            x_cursor = 0
            ordered_classes = ["Access", "Watch", "Reserve", "Unknown"]
            class_boundaries = {}
            class_midpoints = {}

            # First, determine the position and size of each class block on the x-axis
            for cls in ordered_classes:
                # Find which members of this class are actually in the final plot
                members_in_plot = [
                    label_map[k] for k in class_to_members.get(cls, []) if label_map.get(k) in new_order
                ]
                if not members_in_plot:
                    continue
                
                num_members = len(members_in_plot)
                class_midpoints[cls] = x_cursor + (num_members / 2.0) - 0.5
                x_cursor += num_members
                class_boundaries[cls] = x_cursor - 0.5

            # Second, add the annotations and separator lines to the figure
            for i, cls in enumerate(ordered_classes):
                # Add text annotation for the class
                if cls in class_midpoints:
                    fig.add_annotation(
                        x=class_midpoints[cls], y=1.06, xref="x", yref="paper",
                        text=f"<b>{cls} Antibiotics</b>", showarrow=False,
                        font=dict(color=class_palette.get(cls, "#888888"), size=18, family="Arial"),
                        align="center"
                    )
                
                # Add a separator line *after* the current class block
                if i < len(ordered_classes) - 1 and cls in class_boundaries:
                    next_cls = ordered_classes[i+1]
                    if next_cls in class_boundaries: # Only draw a line if the next class exists
                        fig.add_shape(
                            type="line", x0=class_boundaries[cls], x1=class_boundaries[cls],
                            y0=0, y1=1, xref="x", yref="paper",
                            line=dict(color=class_palette.get(next_cls, "#888888"), width=2, dash="dot"),
                            layer="below" # Place lines behind the data
                        )

    else:
        # Antibiotic-level: split one trace per class (keeps contiguous class blocks)
        if antibiotic_classes:
            custom = merged[[group_col, "Count", "Denominator"]]
            fig.add_trace(go.Box(
                x=merged[antibiotic_col], y=merged["Value"], width=0.6,
                marker_color=box_color, boxpoints=box_point_option, name="Distribution",
                jitter=0.0, pointpos=0, customdata=custom,
                hovertemplate=("Antibiotic Class: %{x}<br>"
                               f"{group_col}: %{{customdata[0]}}<br>"
                               "Value: %{y:.2f}<br><br>"
                               "Count: %{customdata[1]}<br>"
                               f"{denominator_label}: %{{customdata[2]}}"
                               "<extra></extra>")
            ))
        else:
            # create class per row from map (Unknown if absent)
            if antibiotic_class_map:
                # pretty label -> raw key
                # (labels = new_order, map back to original for class lookup)
                rev_label_map = {v: k for k, v in label_map.items()}
                merged["_Class"] = merged[antibiotic_col].map(
                    lambda lab: antibiotic_class_map.get(
                        rev_label_map.get(lab, lab), "Unknown")
                )
            else:
                merged["_Class"] = "Unknown"

            for cls in ["Access", "Watch", "Reserve", "Unknown"]:
                sub = merged[merged["_Class"] == cls]
                if sub.empty:
                    continue
                custom = sub[[group_col, "Count", "Denominator"]]
                fig.add_trace(go.Box(
                    x=sub[antibiotic_col],
                    y=sub["Value"],
                    name=cls,
                    marker_color=class_palette.get(cls, "#888888"),
                    boxpoints=box_point_option,
                    jitter=0.35,
                    pointpos=0,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cls}</b><br>"
                        "Antibiotic: %{x}<br>"
                        f"{group_col}: %{{customdata[0]}}<br>"
                        "Value: %{y:.2f}<br><br>"
                        "Count: %{customdata[1]}<br>"
                        f"{denominator_label}: %{{customdata[2]}}"
                        "<extra></extra>"
                    )
                ))
            # traces share x; category order enforces contiguous class blocks
            fig.update_layout(boxmode="overlay")

    # optional outlier markers
    if show_outliers:
        outliers = merged[merged["is_outlier"]]
        if not outliers.empty:
            fig.add_trace(go.Scatter(
                x=outliers[antibiotic_col], y=outliers["Value"], mode="markers",
                marker=dict(color=outlier_color, size=10,
                            symbol="circle-open"),
                showlegend=False, name="Outlier", hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=outlier_color,
                            symbol="circle-open"),
                showlegend=True, name=f"Outlier ( > {extent} × IQR )"
            ))

    # optional vertical threshold line (based on x cutoff by sort_stat)
    if show_threshold_line and threshold_line is not None:
        below_thresh = group_stats[sort_stat][group_stats[sort_stat]
                                              < threshold_line]
        if not below_thresh.empty:
            original_cutoff = below_thresh.index[0]
            cutoff_label = original_cutoff if antibiotic_classes else label_map.get(
                original_cutoff, original_cutoff)
            if cutoff_label in new_order:
                try:
                    cutoff_idx = new_order.index(cutoff_label)
                    fig.add_shape(
                        type="line", x0=cutoff_idx - 0.5, x1=cutoff_idx - 0.5, y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color=threshold_line_color,
                                  width=3, dash=threshold_line_dash),
                        layer="above"
                    )
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode="lines",
                        line=dict(color=threshold_line_color,
                                  width=3, dash=threshold_line_dash),
                        showlegend=True, name=f"Threshold ({sort_stat} < {threshold_line:.1f})"
                    ))
                except ValueError:
                    pass
    
    # fig.update_layout(
    #     title=dict(
    #         text=(title or "").replace("\n", "<br>"),
    #         x=0.5,
    #         y=0.98,
    #         font=dict(size=28, family="Arial", color="black")
    #     ),
    #     xaxis_title=xaxis_title,
    #     yaxis_title=ylabel,
    #     width=width,
    #     height=height,
    #     margin=dict(t=120),
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    #     legend_title_text="Legend",
    #     legend=dict(
    #         **legend,
    #         font=dict(size=18),
    #         title=dict(font=dict(size=20)),
    #         y=legend.get("y", -0.25) - 0.08,
    #     )
    # )
    
    
    # Make a safe copy of legend and adjust y
    legend_cfg = dict(legend)
    legend_cfg["y"] = legend_cfg.get("y", -0.25) - 0.70  # move legend down

    fig.update_layout(
        title=dict(
            text=(title or "").replace("\n", "<br>"),
            x=0.5,
            y=0.98,
            font=dict(size=28, family="Arial", color="black")
        ),
        xaxis_title=xaxis_title,
        yaxis_title=ylabel,
        width=width,
        height=height,
        margin=dict(t=120),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="Legend",
        legend=dict(
            **legend_cfg,
            font=dict(size=20),
            title=dict(font=dict(size=24)),
        )
    )


    fig.update_xaxes(
        tickangle=45, categoryorder="array",
        categoryarray=new_order, automargin=True,
        tickfont=dict(size=20),
        title_font=dict(size=24)
    )
    
    fig.update_yaxes(
        zeroline=True, zerolinewidth=2, zerolinecolor="LightGrey",
        range=yaxis_range, rangemode="tozero" if yaxis_range is None else "normal",
        tickfont=dict(size=20),
        title_font=dict(size=24)
    )
    
    return fig

########################## OTHER HELPERS ##################################################

def _ensure_list(x):
    return list(x) if isinstance(x, (list, tuple, set, pd.Index)) else [x]


def save_chart(
    chart: Union["alt.Chart", go.Figure],
    basename: Union[str, Path],
    formats: Sequence[str] = ("png", "svg", "pdf", "html"),
    *,
    scale: float = 3.0,              # for high-res raster
    width: Optional[int] = None,     # override size if wanted
    height: Optional[int] = None,
    plotly_engine: str = "kaleido",  # or "orca" if you use it
    plotly_html_full: bool = True,
    plotly_html_js: str = "cdn",     # 'cdn' | 'inline'
    altair_embed_options: Optional[dict] = None,
) -> Mapping[str, Path]:
    """
    Save a Plotly or Altair chart to multiple formats.

    Parameters
    ----------
    chart : Union[alt.Chart, go.Figure]
        The figure object to persist.
    basename : str|Path
        Path without extension. The function will append .png / .svg / ...
    formats : Sequence[str]
        Iterable of formats to export. Valid: 'png','svg','pdf','html'
    scale : float
        Resolution multiplier for raster images (png) or vector scaling.
    width, height : int|None
        Override figure size on export (Plotly only).
    plotly_engine : str
        Static export engine for Plotly (default 'kaleido').
    plotly_html_full : bool
        If True, write a full HTML doc; otherwise, a snippet.
    plotly_html_js : str
        Where to load plotly.js from.
    altair_embed_options : dict|None
        Passed to Altair's HTML saver (e.g., {"actions": False})
    """
    out_paths: dict[str, Path] = {}
    base = Path(basename)
    base.parent.mkdir(parents=True, exist_ok=True)

    is_plotly = isinstance(chart, go.Figure)
    is_altair = alt is not None and isinstance(chart, alt.TopLevelMixin)

    if not (is_plotly or is_altair):
        raise TypeError(f"Unsupported chart type: {type(chart)}")

    for fmt in formats:
        ext = fmt.lower().lstrip(".")
        out_file = base.with_suffix(f".{ext}")

        if is_plotly:
            if ext == "html":
                chart.write_html(
                    out_file,
                    full_html=plotly_html_full,
                    include_plotlyjs=plotly_html_js,
                )
            elif ext in {"png", "svg", "pdf"}:
                chart.write_image(
                    out_file,
                    format=ext,
                    scale=scale,
                    width=width,
                    height=height,
                    engine=plotly_engine
                )
            else:
                raise ValueError(f"Format '{fmt}' not supported for Plotly.")
        else:  # Altair
            if ext == "html":
                chart.save(out_file, format="html",
                           embed_options=altair_embed_options or {})
            elif ext in {"png", "svg", "pdf"}:
                # altair_saver or vl-convert node is required for png/svg/pdf
                chart.save(out_file, format=ext, scale_factor=scale)
            else:
                raise ValueError(f"Format '{fmt}' not supported for Altair.")

        out_paths[ext] = out_file

    return out_paths


def _ci_rename(df: pd.DataFrame, wanted: Mapping[str, str]) -> pd.DataFrame:
    lut = {c.lower(): c for c in df.columns}
    missing = [k for k in wanted if k not in lut]
    if missing:
        # don't crash, just skip missing
        wanted = {k: v for k, v in wanted.items() if k in lut}
    ren = {lut[k]: v for k, v in wanted.items() if lut[k] != v}
    return df.rename(columns=ren)


def format_antibiotic_label(abbr, name, style='abbr'):
    if style == 'abbr':
        return abbr
    elif style == 'name':
        return name
    elif style == 'both':
        return f"{abbr} ({name})"
    else:
        raise ValueError(
            "Invalid style. Choose from 'abbr', 'name', or 'both'.")


def build_mappings(class_dict, catalog_str):
    # Parse CATALOG into {abbr: full_name}
    abbr2full = {}
    for item in catalog_str.split(","):
        parts = item.strip().split(" - ")
        if len(parts) == 2:
            abbr, fullname = parts
            abbr2full[abbr.strip()] = fullname.strip()

    # Build {abbr: class_name} from existing dict
    class_mapping = {}
    for class_name, abx_list in class_dict.items():
        for abx in abx_list:
            abbr = abx.split(" - ")[0].replace("_Tested", "").strip()
            class_mapping[abbr] = class_name

    return class_mapping, abbr2full


def decorate_label_dynamic(label, abbr2full, class_mapping, target_map, format_type="full", include_class=False):
    """
    Decorates label based on format_type ('full' or 'abbr').
    Example outputs:
      - full: 'Amoxicillin/clavulanic acid [Penicillin + Beta-lactamase inhibitor] (mixed)'
      - abbr: 'AMC [Penicillin + Beta-lactamase inhibitor] (mixed)'
    """
    label_clean = label.replace("_Tested", "").strip()

    if format_type == "abbr":
        abbr = label_clean
        full = abbr2full.get(abbr, None)
    else:
        # Try to find abbreviation from full name
        reverse_map = {v: k for k, v in abbr2full.items()}
        abbr = reverse_map.get(label_clean, None)
        full = label_clean

    # If we fail to find abbreviation, try partial match
    if not abbr:
        match = next((a for a, f in abbr2full.items()
                      if label_clean.lower() in f.lower()), None)
        if match:
            abbr = match
            full = abbr2full[abbr]
        else:
            return label  # No match

    abx_class = class_mapping.get(abbr, "Unknown")
    tag = target_map.get(abbr, "null")
    if include_class:
        return f"{full if format_type == 'full' else abbr} [{abx_class}] ({tag})"
    else:
        return f"{full if format_type == 'full' else abbr} ({tag})"


def get_label(abx_cols=[], antibiotic_class_map="", format_type="abbr", enrich=False, include_class=False):
    label_map = create_label_mapping(
        abx_cols=abx_cols, format_type=format_type)

    if enrich:
        CLASS_MAPPING, ABBR2FULL = build_mappings(
            antibiotic_class_map, CATALOG)

        label_map = {
            abbr: decorate_label_dynamic(
                full_name, ABBR2FULL,
                CLASS_MAPPING, ABX_TARGET_MAP,
                format_type=format_type,
                include_class=include_class
            )
            for abbr, full_name in label_map.items()
        }

    return label_map


def create_label_mapping(abx_cols=[], format_type: str = 'combined', remove_suffix: str = "_Tested") -> dict:
    """
    Creates a dictionary for cleaner labels in various formats.

    Args:
        format_type (str): The desired format for the labels. 
                        Options: 'abbr', 'full', 'combined'.
        remove_suffix (str): The suffix to remove from column names.

    Returns:
        dict: A mapping from old labels to new, cleaner labels.
    """
    label_map = {}
    for col in abx_cols:
        # First, remove the suffix to clean up the string
        cleaned_col = col.replace(remove_suffix, "")

        # Try to split the string into abbreviation and full name
        parts = cleaned_col.split(' - ', 1)

        if len(parts) == 2:
            abbr, full_name = parts
            if format_type == 'abbr':
                label_map[col] = abbr
            elif format_type == 'full':
                label_map[col] = full_name
            elif format_type == 'combined':
                # Keep the "Abbr - Full Name" format
                label_map[col] = cleaned_col
            else:
                # Default to combined if format is invalid
                label_map[col] = cleaned_col
        else:
            # If the label doesn't contain " - ", just use the cleaned version
            label_map[col] = cleaned_col

    return label_map


def filter_antibiotic_group_items(df, groups):
    """
    Filter out antibiotics not found in our dataset but correctly place in the class
    this is needed to avoid errors
    """
    present = {g: [a for a in ab if a in df.columns]
               for g, ab in groups.items()}
    missing = {g: [a for a in ab if a not in df.columns]
               for g, ab in groups.items()}
    return {g: v for g, v in present.items() if v}, missing


def antibiotic_aggregate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    tested_cols = [c for c in df.columns if c.endswith("_Tested")]
    lab_testing_summary = (
        df.groupby(group_col)[tested_cols]
        .sum()
    )
    lab_testing_summary["Total_Tests"] = lab_testing_summary.sum(axis=1)
    lab_testing_summary_sorted = lab_testing_summary.sort_values(
        by="Total_Tests",
        ascending=False
    )
    return lab_testing_summary_sorted.head(50)