# -*- coding: utf-8 -*-
"""
Similarity network plotter with robust antibiotic label↔code mapping,
data-driven class abbreviation, high-visibility class bubbles,
intra-cluster spacing control, and spatially split, purity-guarded class bubbles.

Key features:
- Robust label_to_code that handles: "CIP - Ciprofloxacin_Tested", "CIP (mixed)", "Ciprofloxacin".
- AbbrevConfig: if contains_map is empty, original class names are kept.
- Legend (bottom): shows classes present (count) and up to N representative codes per class.
  - Optional priority codes (e.g., ["SXT"]) always included if present.
  - Code selection by node degree (default), or alphabetical.
- Class bubbles: soft fill under nodes + bold outline on top (with halo), slightly expanded
  so outlines remain visible around nodes.
- Intra-cluster spacing (minimum distance) via a small collision-avoidance relaxer.
- NEW: Spatial sub-clusters per class + purity guard so a big bubble can’t cover other groups.

Author: you 🛠️
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Mapping, Sequence, Dict, Callable, Literal, Iterable, List, Union
import re, unicodedata, itertools
import matplotlib as mpl
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------- robust label→code plumbing ---------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _canon_name(s: str) -> str:
    # normalize common spelling variants you have in your data
    s = _strip_accents(str(s)).lower()
    s = s.replace("ceftriaxon", "ceftriaxone").replace("cefotaxim", "cefotaxime")
    s = s.replace("doxycyclin", "doxycycline").replace("nitroxolin", "nitroxoline")
    s = re.sub(r"\bco[-\s]?trimoxazol(e)?\b", "cotrimoxazole", s)
    s = re.sub(r"[_-](tested|test|result|ast)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def build_name_to_code(lookup_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a normalized name→code map from a lookup table.
    Required columns: 'antibiotic_code'
    Optional columns: 'canonical_display_name', 'alt_names', 'synonyms' (semicolon-separated)
    """
    m: Dict[str, str] = {}
    if lookup_df is None or lookup_df.empty:
        return m
    need = {"antibiotic_code"}
    if not need.issubset(set(lookup_df.columns)):
        raise KeyError("lookup_df must contain at least 'antibiotic_code'.")
    for _, r in lookup_df.dropna(subset=["antibiotic_code"]).iterrows():
        code = str(r["antibiotic_code"]).strip().upper()
        # Include the code itself as a resolvable token
        m[code.lower()] = code
        for col in ("canonical_display_name", "alt_names", "synonyms"):
            if col in lookup_df.columns and pd.notna(r.get(col)):
                for nm in str(r[col]).split(";"):
                    key = _canon_name(nm)
                    if key:
                        m[key] = code
    return m

def label_to_code(
    label: str,
    *,
    known_codes: Optional[Iterable[str]] = None,
    name_to_code: Optional[Dict[str, str]] = None,
    code_aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Extract a canonical antibiotic code from many label formats:
      'NOR - Norfloxacin_Tested', 'NOR (mixed)', 'NOR', 'Norfloxacin'
    """
    s = str(label).strip()

    # drop trailing parentheses like '(+ve)', '(-ve)', '(mixed)'
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s)

    # CODE at start (e.g., 'NOR ...')
    m = re.match(r"^([A-Z0-9]{2,6})\b", s)
    if m:
        code = m.group(1).upper()
        if known_codes is None or code in set(known_codes):
            if code_aliases and code in code_aliases:
                code = code_aliases[code]
            return code

    # Split 'CODE - Name'
    m2 = re.match(r"^([A-Z0-9]{2,6})\s*-\s*(.+)$", s)
    if m2:
        code = m2.group(1).upper()
        if known_codes is None or code in set(known_codes):
            if code_aliases and code in code_aliases:
                code = code_aliases[code]
            return code
        # else fall through to name side

    # Name-only (or take the 'Name' side after 'CODE - Name')
    name = s.split(" - ", 1)[-1]
    key = _canon_name(name)
    if name_to_code and key in name_to_code:
        code = name_to_code[key]
        if code_aliases and code in code_aliases:
            code = code_aliases[code]
        return code

    return None

def build_code_to_class(
    class_to_items: Mapping[str, Sequence[str]],
    *,
    known_codes: Optional[Iterable[str]] = None,
    name_to_code: Optional[Dict[str, str]] = None,
    code_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Convert JSON mapping of class→[strings] into a robust code→class map.
    Strings can be 'NOR - Norfloxacin_Tested', 'NOR', 'Norfloxacin', etc.
    """
    code2class: Dict[str, str] = {}
    if not class_to_items:
        return code2class
    for cls, items in class_to_items.items():
        for it in items:
            c = label_to_code(it, known_codes=known_codes, name_to_code=name_to_code, code_aliases=code_aliases)
            if c:
                code2class[c] = cls
    return code2class

# --------------------------- abbreviation helpers & engine ---------------------------

def _norm(s: str) -> str:
    s = _strip_accents(str(s)).strip().lower()
    # normalize common punctuation and greek symbols
    s = (s.replace("—", "-").replace("–", "-")
           .replace("β", "beta"))  # allow ASCII 'beta' rules
    return s

def _uniqueify(labels: List[str]) -> List[str]:
    """Ensure list of labels is unique by adding -2, -3... suffixes on collisions."""
    seen: Dict[str, int] = {}
    out = []
    for lab in labels:
        base = lab or ""
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}-{seen[base]}")
    return out

@dataclass
class AbbrevConfig:
    """
    Rules to abbreviate class names. If `contains_map` is empty → keep originals.
    Keys are matched on normalized (lowercased, de-accented) text.
    """
    contains_map: Dict[str, str] = field(default_factory=dict)  # keep originals by default
    generation_map: Dict[str, str] = field(default_factory=lambda: {
        "first-gen": "1st Gen", "1st gen": "1st Gen", "g1": "1st Gen",
        "second-gen": "2nd Gen", "2nd gen": "2nd Gen", "g2": "2nd Gen",
        "third-gen": "3rd Gen", "3rd gen": "3rd Gen", "g3": "3rd Gen",
        "fourth-gen": "4th Gen", "4th gen": "4th Gen", "g4": "4th Gen",
        "fifth-gen": "5th Gen", "5th gen": "5th Gen", "g5": "5th Gen",
    })
    regex_rules: List[Tuple[re.Pattern, Union[str, Callable[[re.Match], str]]]] = field(default_factory=list)
    keep_paren_suffix: bool = True
    joiner: str = "/"

class AbbrevEngine:
    def __init__(self, config: Optional[AbbrevConfig] = None):
        self.cfg = config or AbbrevConfig()
        if not self.cfg.regex_rules:
            self.cfg.regex_rules = [
                (re.compile(r"\b(cephalosporin)s?\b", re.I), "cephalosporin"),
            ]

    def _apply_regex_rules(self, text: str) -> str:
        out = text
        for pat, repl in self.cfg.regex_rules:
            out = pat.sub(repl if not callable(repl) else (lambda m: repl(m)), out)
        return out

    def _extract_paren_suffix(self, text: str) -> Tuple[str, Optional[str]]:
        s = text.strip()
        m = re.search(r"\(([^)]+)\)\s*$", s)
        if not m:
            return s, None
        head = s[:m.start()].strip()
        par = m.group(1).strip()
        return head, par

    def _abbr_single(self, piece: str) -> str:
        if not piece:
            return piece
        if not self.cfg.contains_map:
            return piece.strip()

        raw = piece.strip()
        head_raw, par_raw = self._extract_paren_suffix(raw)
        head_norm = _norm(self._apply_regex_rules(head_raw))
        suffix = ""

        gen = ""
        for k, v in self.cfg.generation_map.items():
            if k in head_norm:
                gen = v; break

        base = None
        if "cephalosporin" in head_norm:
            base = f"{gen} Ceph".strip() if gen else "Ceph"
        else:
            for k, v in self.cfg.contains_map.items():
                if k in head_norm:
                    base = v; break
        if not base:
            base = head_raw.strip()

        if self.cfg.keep_paren_suffix and par_raw:
            par_norm = _norm(self._apply_regex_rules(par_raw))
            par_abbr = None
            for k, v in self.cfg.contains_map.items():
                if k in par_norm:
                    par_abbr = v; break
            suffix = f" ({par_abbr})" if par_abbr else f" ({par_raw.strip()})"

        return f"{base}{suffix}"

    def abbreviate(self, label: str) -> str:
        if not label:
            return ""
        if not self.cfg.contains_map:
            return str(label).strip()

        # exact full-string match (after normalization)
        full_norm = _norm(self._apply_regex_rules(label))
        if full_norm in self.cfg.contains_map:
            return self.cfg.contains_map[full_norm]

        # split on "/" but avoid splits inside parentheses
        parts: List[str] = []
        depth = 0
        buff = []
        for ch in str(label):
            if ch == "(": depth += 1
            elif ch == ")": depth = max(0, depth - 1)
            if ch == "/" and depth == 0:
                parts.append("".join(buff)); buff = []
            else:
                buff.append(ch)
        if buff: parts.append("".join(buff))

        abbrs = [self._abbr_single(p) for p in parts]
        return self.cfg.joiner.join(a.strip() for a in abbrs if a.strip())

    def make_map(self, class_to_items: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, str]:
        if not class_to_items:
            return {}
        original_classes = [str(k) for k in class_to_items.keys()]
        abbrs = [self.abbreviate(k) for k in original_classes]
        abbrs_unique = _uniqueify(abbrs)
        return {orig: ab for orig, ab in zip(original_classes, abbrs_unique)}

def make_class_abbrev_map(
    class_to_items: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    config: Optional[AbbrevConfig] = None,
    extra_contains: Optional[Dict[str, str]] = None,
    extra_generations: Optional[Dict[str, str]] = None,
    extra_regex_rules: Optional[List[Tuple[re.Pattern, Union[str, Callable[[re.Match], str]]]]] = None
) -> Dict[str, str]:
    """
    Backwards-compatible wrapper that returns {class_name -> abbreviated_label}.
    If `contains_map` is empty (default), labels are returned unchanged.
    """
    if not class_to_items:
        return {}
    cfg = config or AbbrevConfig()
    if extra_contains:
        cfg.contains_map = {**cfg.contains_map, **{_norm(k): v for k, v in extra_contains.items()}}
    if extra_generations:
        cfg.generation_map = {**cfg.generation_map, **{_norm(k): v for k, v in extra_generations.items()}}
    if extra_regex_rules:
        compiled: List[Tuple[re.Pattern, Union[str, Callable[[re.Match], str]]]] = []
        for pat, repl in extra_regex_rules:
            compiled.append((re.compile(pat, re.I), repl))
        cfg.regex_rules = list(cfg.regex_rules) + compiled
    return AbbrevEngine(cfg).make_map(class_to_items)

# --------------------------- simple collision-avoidance (intra-cluster spacing) -----------

def _separate_points(pos_dict: Dict[str, Tuple[float, float]],
                     min_dist: float,
                     iters: int = 40,
                     lr: float = 0.45,
                     cool: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Simple O(n^2) collision-avoidance to enforce a minimum distance among points.
    Keeps the cluster centroid fixed to avoid drift.
    """
    if min_dist is None or min_dist <= 0 or len(pos_dict) <= 1:
        return pos_dict
    keys = list(pos_dict.keys())
    P = np.array([pos_dict[k] for k in keys], dtype=float)

    for _ in range(max(1, iters)):
        P -= P.mean(axis=0, keepdims=True)  # recenter (avoid drift)
        for i in range(len(P) - 1):
            for j in range(i + 1, len(P)):
                diff = P[j] - P[i]
                d = float(np.hypot(diff[0], diff[1])) + 1e-12
                if d < min_dist:
                    push = (min_dist - d) / d
                    delta = diff * (0.5 * lr * push)
                    P[i] -= delta
                    P[j] += delta
        lr *= cool
    P -= P.mean(axis=0, keepdims=True)
    return {k: (float(P[i, 0]), float(P[i, 1])) for i, k in enumerate(keys)}

# --------------------------- bubble helpers: spatial clusters + purity guard --------------

def _clusters_by_distance(node_ids: List[str],
                          pos: Mapping[str, Tuple[float,float]],
                          eps: float) -> List[List[str]]:
    """Union-find style grouping: edges between nodes ≤ eps → connected components."""
    if len(node_ids) <= 1:
        return [node_ids[:]]
    parent = list(range(len(node_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(node_ids) - 1):
        xi, yi = pos[node_ids[i]]
        for j in range(i+1, len(node_ids)):
            xj, yj = pos[node_ids[j]]
            if (xi-xj)**2 + (yi-yj)**2 <= eps*eps:
                union(i, j)

    roots = {}
    for i, n in enumerate(node_ids):
        r = find(i)
        roots.setdefault(r, []).append(n)
    return list(roots.values())

def _ellipse_contains(cx, cy, w, h, ang_deg, pt):
    """Point-in-rotated-ellipse test."""
    x, y = pt
    ang = np.deg2rad(ang_deg)
    ca, sa = np.cos(ang), np.sin(ang)
    xr =  ca*(x-cx) + sa*(y-cy)
    yr = -sa*(x-cx) + ca*(y-cy)
    return (xr/(w/2.0))**2 + (yr/(h/2.0))**2 <= 1.0 + 1e-9

# --------------------------- main plotter (robust + legend below) ---------------------------

def plot_similarity_network_nested(
    sim: pd.DataFrame,
    *,
    # pruning
    min_weight: float = 0.40,
    top_m_per_node: Optional[int] = 5,
    mutual_top_m: bool = True,
    max_edges_overall: Optional[int] = None,
    # layout
    outer_layout: Literal["spring","kamada"] = "spring",
    inner_layout: Literal["spring","kamada"] = "kamada",
    k_outer: float = 4.0, k_inner: float = 2.4,
    scale_outer: float = 8.0, scale_inner: float = 3.0,
    # NEW: intra-cluster spacing controls
    inner_min_dist: Optional[float] = None,   # if None => auto ~ 0.25*scale_inner
    inner_separation_iters: int = 40,
    inner_separation_lr: float = 0.45,
    inner_separation_cool: float = 0.95,
    inner_scale_up: float = 1.0,              # >1.0 radially expands each cluster
    seed: int = 42,
    pos_fixed: Optional[Mapping[str, Tuple[float, float]]] = None,
    # classes / mapping
    class_to_items: Optional[Mapping[str, Sequence[str]]] = None,
    item_to_class: Optional[Mapping[str, str]] = None,    # legacy direct mapping
    class_colors: Optional[Mapping[str, str]] = None,
    class_abbrev: Optional[Mapping[str, str]] = None,
    # robust code mapping
    code_to_class: Optional[Mapping[str,str]] = None,
    label_code_resolver: Optional[Callable[..., Optional[str]]] = None,
    label_code_kwargs: Optional[dict] = None,
    # bubbles
    show_class_bubbles: bool = True,
    min_nodes_for_class_bubble: int = 1,
    label_classes_on_bubbles: bool = False,
    # high-visibility bubble styling
    class_bubble_expand: float = 1.10,
    class_bubble_fill_alpha: float = 0.14,
    class_bubble_edge_alpha: float = 0.95,
    class_bubble_edge_lw: float = 2.6,
    class_bubble_outline_on_top: bool = True,
    class_bubble_outline_style: str = "-",
    class_bubble_outline_halo: bool = True,
    # NEW: spatial subclusters + purity guard (prevents overreach)
    class_bubble_mode: Literal["spatial","all"] = "spatial",
    class_bubble_eps: Optional[float] = None,      # auto from layout if None
    class_bubble_purity_min: float = 0.70,         # % inside bubble that must be this class
    class_bubble_shrink_factor: float = 0.92,
    class_bubble_shrink_iters: int = 6,
    class_bubble_trim_outliers: bool = True,
    class_bubble_trim_max: int = 3,
    # community labels/hulls
    annotate_communities: bool = False,
    community_summary_topk: int = 8,
    # legends
    show_class_legend: bool = False,
    show_edge_weight_legend: bool = True,
    show_class_legend_below: bool = True,
    legend_below_max_cols: int = 4,
    legend_below_fontsize: int = 9,
    legend_below_pad: float = 0.15,
    class_legend_min_count: int = 1,
    class_legend_codes_max: int = 10,
    class_legend_codes_separator: str = ", ",
    legend_priority_codes: Optional[Iterable[str]] = None,  # e.g., ["SXT"]
    legend_code_sort: Literal["auto","degree","alphabetical"] = "auto",
    # styling
    base_node_size: float = 260.0,
    size_per_edge: float = 140.0,
    use_weighted_degree: bool = False,
    node_ring_linewidth: float = 0.8,
    edge_color: str = "#666666",
    edge_alpha_range: Tuple[float, float] = (0.35, 0.90),
    repel_labels: bool = True,
    class_label_fontsize: int = 11,
    class_label_box_alpha: float = 0.88,
    shade_communities: bool = True,
    figsize: Tuple[int, int] = (13, 12),
    title: str = "Similarity network — nested class grouping (node color = community)",
    save_path: Optional[str] = None,
) -> Dict[str, object]:

    # ---------- 0) validate & normalize ----------
    if sim.shape[0] != sim.shape[1] or not sim.index.equals(sim.columns):
        raise ValueError("sim must be square with identical index and columns.")
    sim = sim.astype(float)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim.values, 1.0)
    nodes = list(sim.index.astype(str))
    sim.index = sim.columns = nodes

    # ---------- 0b) build item_to_class (prefer code-based) ----------
    if code_to_class is not None:
        if label_code_resolver is None:
            label_code_resolver = label_to_code
        label_code_kwargs = label_code_kwargs or {}
        item_to_class = {n: code_to_class.get(label_code_resolver(n, **label_code_kwargs)) for n in nodes}
    else:
        if item_to_class and class_to_items:
            raise ValueError("Provide either class_to_items or item_to_class, not both.")
        if item_to_class is None and class_to_items is not None:
            if label_code_resolver is None:
                label_code_resolver = label_to_code
            label_code_kwargs = label_code_kwargs or {}
            code2class = build_code_to_class(class_to_items, **label_code_kwargs)
            item_to_class = {n: code2class.get(label_code_resolver(n, **label_code_kwargs)) for n in nodes}
        elif item_to_class is None:
            item_to_class = {n: None for n in nodes}

    # Abbreviations for legend text
    if class_abbrev is None and class_to_items:
        class_abbrev = make_class_abbrev_map(class_to_items)

    # ---------- 1) prune edges ----------
    G = nx.Graph(); G.add_nodes_from(nodes)
    strong_lists: Dict[str, pd.Series] = {}
    for u in nodes:
        row = sim.loc[u].drop(u)
        strong = row[row >= min_weight].sort_values(ascending=False)
        if top_m_per_node:
            strong = strong.head(top_m_per_node)
        strong_lists[u] = strong

    edge_strength: Dict[Tuple[str, str], float] = {}
    if mutual_top_m:
        for u in nodes:
            for v, w in strong_lists[u].items():
                if u == v: continue
                if u in strong_lists.get(v, pd.Series(index=[])).index:
                    a, b = (u, v) if u < v else (v, u)
                    edge_strength[(a, b)] = max(edge_strength.get((a, b), 0.0),
                                                float(min(w, strong_lists[v][u])))
    else:
        for u in nodes:
            for v, w in strong_lists[u].items():
                if u == v: continue
                a, b = (u, v) if u < v else (v, u)
                edge_strength[(a, b)] = max(edge_strength.get((a, b), 0.0), float(w))

    edges_sorted = sorted(edge_strength.items(), key=lambda kv: kv[1], reverse=True)
    if max_edges_overall is not None:
        edges_sorted = edges_sorted[:max_edges_overall]
    for (u, v), w in edges_sorted:
        G.add_edge(u, v, weight=float(w))
    iso = [n for n in G.nodes() if G.degree(n) == 0]
    if iso: G.remove_nodes_from(iso)
    if G.number_of_edges() == 0:
        raise RuntimeError("No edges after pruning. Relax min_weight or increase top_m_per_node.")

    # ---------- 2) communities ----------
    import community as community_louvain
    part = community_louvain.best_partition(G, random_state=seed)
    comm_ids = sorted(set(part.values()))
    comm_nodes = {c: [n for n, cid in part.items() if cid == c] for c in comm_ids}

    # ---------- 3) layout ----------
    def build_comm_graph():
        from collections import Counter
        H = nx.Graph()
        for c in comm_ids: H.add_node(c)
        inter = Counter()
        for u, v, _ in G.edges(data=True):
            cu, cv = part[u], part[v]
            if cu != cv: inter[tuple(sorted((cu, cv)))] += 1
        for (cu, cv), w in inter.items():
            H.add_edge(cu, cv, weight=w)
        return H

    if pos_fixed is None:
        H = build_comm_graph()
        if outer_layout == "spring":
            pos_outer = nx.spring_layout(H, k=k_outer, seed=seed, scale=scale_outer, weight="weight")
        else:
            pos_outer = nx.kamada_kawai_layout(H, weight="weight")
            for c in pos_outer:
                pos_outer[c] = (pos_outer[c][0]*scale_outer, pos_outer[c][1]*scale_outer)

        pos: Dict[str, Tuple[float,float]] = {}
        for c in comm_ids:
            sub = G.subgraph(comm_nodes[c])
            if len(sub) == 1:
                pos[next(iter(sub))] = pos_outer[c]; continue

            # Inner layout
            if inner_layout == "spring":
                sub_pos = nx.spring_layout(sub, k=k_inner, seed=seed, scale=scale_inner, weight="weight")
            else:
                sub_pos = nx.kamada_kawai_layout(sub, weight="weight")
                xs = np.array([xy[0] for xy in sub_pos.values()])
                ys = np.array([xy[1] for xy in sub_pos.values()])
                xs = (xs - xs.mean())/(xs.std()+1e-9) * (scale_inner/2.0)
                ys = (ys - ys.mean())/(ys.std()+1e-9) * (scale_inner/2.0)
                sub_pos = {n:(x,y) for n,(x,y) in zip(sub_pos.keys(), zip(xs,ys))}

            # (a) optional radial scale-up
            if inner_scale_up and inner_scale_up != 1.0:
                cx0 = np.mean([xy[0] for xy in sub_pos.values()])
                cy0 = np.mean([xy[1] for xy in sub_pos.values()])
                sub_pos = {n: (cx0 + (x - cx0)*inner_scale_up, cy0 + (y - cy0)*inner_scale_up)
                           for n, (x, y) in sub_pos.items()}

            # (b) enforce minimum intra-cluster distance
            auto_min = 0.25 * scale_inner
            mind = float(inner_min_dist) if inner_min_dist is not None else auto_min
            sub_pos = _separate_points(sub_pos, min_dist=mind,
                                       iters=inner_separation_iters,
                                       lr=inner_separation_lr,
                                       cool=inner_separation_cool)

            # Shift to outer anchor
            cx, cy = pos_outer[c]
            for n,(x,y) in sub_pos.items():
                pos[n] = (cx+x, cy+y)
    else:
        pos = {n: tuple(pos_fixed[n]) for n in G.nodes() if n in pos_fixed}

    # ---------- 4) robust ellipse helper ----------
    from matplotlib.patches import Ellipse
    if G.number_of_edges() > 0:
        edge_lengths = [np.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]) for u,v in G.edges()]
        d_med = float(np.median(edge_lengths))
    else:
        d_med = 1.0
    PAD_MAJOR = 0.28 * d_med
    PAD_MINOR = 0.14 * d_med
    MIN_DIAM  = 0.35 * d_med

    def robust_ellipse(points):
        pts = np.array(points, float)
        n = len(pts)
        if n == 1:
            cx, cy = pts[0]
            w = h = max(MIN_DIAM, 2*PAD_MINOR)
            return cx, cy, w, h, 0.0
        if n == 2:
            (x1,y1),(x2,y2) = pts
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            dx, dy = (x2-x1), (y2-y1)
            dist = float(np.hypot(dx, dy))
            ang  = np.degrees(np.arctan2(dy, dx))
            major = max(dist + 2*PAD_MAJOR, MIN_DIAM)
            minor = max(2*PAD_MINOR, MIN_DIAM*0.6)
            return cx, cy, major, minor, ang
        cx, cy = pts.mean(axis=0)
        X = pts - [cx, cy]
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        rx, ry = (2.0*s/np.sqrt(n))
        rx = max(rx + PAD_MAJOR, MIN_DIAM/2.0)
        ry = max(ry + PAD_MINOR, MIN_DIAM/2.0)
        ang = np.degrees(np.arctan2(Vt[0,1], Vt[0,0]))
        return cx, cy, rx*2, ry*2, ang

    # ---------- 5) colors, sizes ----------
    comm_palette = itertools.cycle(
        ["#67a9cf","#ef8a62","#91cf60","#998ec3","#f1a340",
         "#7fbf7b","#af8dc3","#fddbc7","#a6dba0","#fee090"]
    )
    color_by_comm = {cid: next(comm_palette) for cid in comm_ids}
    node_fill = {n: color_by_comm[part[n]] for n in G.nodes()}

    # --- define once (near your plotting helpers) ---
    def get_cmap_compat(name: str):
        """
        Compatible colormap getter.
        Uses the modern registry (Matplotlib ≥3.6), falls back to legacy on older versions.
        """
        try:
            return mpl.colormaps.get_cmap(name)   # preferred API
        except Exception:
            from matplotlib.cm import get_cmap as _legacy_get_cmap
            return _legacy_get_cmap(name)
    
    classes_present = sorted(set(c for n,c in item_to_class.items() if c))
    if class_colors is None and classes_present:
        # from matplotlib.cm import get_cmap
        cmap = get_cmap_compat("tab20")
        class_colors = {cls: cmap(i % 20) for i, cls in enumerate(classes_present)}
    elif class_colors is None:
        class_colors = {}

    if use_weighted_degree:
        deg = pd.Series(dict(G.degree(weight="weight")))
        degn = (deg - deg.min())/(deg.max()-deg.min()+1e-9)
        node_size = {n: base_node_size + size_per_edge*float(degn[n]*(len(G)/8)) for n in G.nodes()}
    else:
        deg = pd.Series(dict(G.degree()))
        node_size = {n: base_node_size + size_per_edge*float(deg[n]) for n in G.nodes()}

    weights = np.array([d["weight"] for *_ , d in G.edges(data=True)])
    wmin, wmax = (weights.min(), weights.max()) if len(weights) else (0.0, 1.0)
    width = {(u,v): 1 + 6*(d["weight"]-wmin)/(wmax-wmin+1e-9) for u,v,d in G.edges(data=True)}
    a0, a1 = edge_alpha_range
    alpha = {(u,v): float(a0 + (a1-a0)*(d["weight"]-wmin)/(wmax-wmin+1e-9)) for u,v,d in G.edges(data=True)}

    # ---------- 6) draw ----------
    fig, ax = plt.subplots(figsize=figsize)

    # community hulls (very soft)
    if shade_communities:
        for cid in comm_ids:
            pts = [pos[n] for n in comm_nodes[cid]]
            cx, cy, w, h, ang = robust_ellipse(pts)
            ax.add_patch(Ellipse((cx, cy), w, h, angle=ang,
                                 facecolor="#00000012", edgecolor="#00000025", lw=1, zorder=0.2))

    # ---------------- class bubbles (spatial clusters + purity guard) ----------------
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe

    def _draw_class_ellipse(ax, cx, cy, w, h, ang, col_rgb, *, z_fill=0.9, z_edge=3.2):
        from matplotlib.patches import Ellipse as _Ellipse
        ax.add_patch(_Ellipse(
            (cx, cy), w, h, angle=ang,
            facecolor=(col_rgb[0], col_rgb[1], col_rgb[2], class_bubble_fill_alpha),
            edgecolor="none", lw=0, zorder=z_fill
        ))
        edge = _Ellipse(
            (cx, cy), w, h, angle=ang,
            facecolor="none",
            edgecolor=(col_rgb[0], col_rgb[1], col_rgb[2], class_bubble_edge_alpha),
            lw=class_bubble_edge_lw, ls=class_bubble_outline_style,
            zorder=z_edge
        )
        if class_bubble_outline_halo:
            edge.set_path_effects([
                pe.withStroke(linewidth=class_bubble_edge_lw + 2.0, foreground="white", alpha=0.9),
                pe.Normal()
            ])
        ax.add_patch(edge)

    if show_class_bubbles:
        # auto eps from inner scale / median edge length
        auto_eps = 0.22 * (scale_inner if isinstance(scale_inner, (int, float)) else 3.0)
        if G.number_of_edges() > 0:
            auto_eps = max(auto_eps, 0.40 * float(np.median([np.hypot(pos[u][0]-pos[v][0],
                                                                       pos[u][1]-pos[v][1]) for u,v in G.edges()])))
        bubble_eps = float(class_bubble_eps) if class_bubble_eps is not None else auto_eps

        for cid in comm_ids:
            # collect nodes by class within this community
            by_cls: Dict[str, list] = {}
            for n in comm_nodes[cid]:
                cls = item_to_class.get(n)
                if cls: by_cls.setdefault(cls, []).append(n)

            # all nodes in this community (for purity check)
            comm_nodes_set = set(comm_nodes[cid])

            for cls, members in by_cls.items():
                # choose clusters
                if class_bubble_mode == "spatial":
                    clusters = _clusters_by_distance(members, pos, bubble_eps)
                else:
                    clusters = [members]

                base_col = np.array(mcolors.to_rgb(class_colors.get(cls, "#888888")))

                for cluster_nodes in clusters:
                    if len(cluster_nodes) < max(1, min_nodes_for_class_bubble):
                        continue

                    # initial ellipse
                    pts = [pos[n] for n in cluster_nodes]
                    cx, cy, w, h, ang = robust_ellipse(pts)
                    w *= class_bubble_expand; h *= class_bubble_expand

                    # purity guard: ratio of nodes inside that belong to this class
                    def purity(cx, cy, w, h, ang):
                        inside = []
                        for n in comm_nodes_set:
                            if _ellipse_contains(cx, cy, w, h, ang, pos[n]):
                                inside.append(n)
                        if not inside:
                            return 1.0, inside
                        good = sum(1 for n in inside if item_to_class.get(n) == cls)
                        return good / max(1, len(inside)), inside

                    pur, inside_nodes = purity(cx, cy, w, h, ang)

                    # (a) shrink if impure
                    it = 0
                    while pur < class_bubble_purity_min and it < max(0, int(class_bubble_shrink_iters)):
                        w *= class_bubble_shrink_factor
                        h *= class_bubble_shrink_factor
                        pur, inside_nodes = purity(cx, cy, w, h, ang)
                        it += 1

                    # (b) optionally trim farthest cluster points and retry
                    trims = 0
                    while pur < class_bubble_purity_min and class_bubble_trim_outliers and \
                          trims < max(0, int(class_bubble_trim_max)) and len(cluster_nodes) > max(1, min_nodes_for_class_bubble):
                        # remove the farthest point from current center
                        dists = [(n, np.hypot(pos[n][0]-cx, pos[n][1]-cy)) for n in cluster_nodes]
                        far_n, _ = max(dists, key=lambda t: t[1])
                        cluster_nodes = [n for n in cluster_nodes if n != far_n]
                        pts = [pos[n] for n in cluster_nodes]
                        cx, cy, w, h, ang = robust_ellipse(pts)
                        w *= class_bubble_expand; h *= class_bubble_expand
                        pur, inside_nodes = purity(cx, cy, w, h, ang)
                        trims += 1

                    # skip if still impure
                    if pur < class_bubble_purity_min:
                        continue

                    # draw
                    _draw_class_ellipse(ax, cx, cy, w, h, ang, base_col,
                                        z_fill=0.9,
                                        z_edge=(3.2 if class_bubble_outline_on_top else 1.3))

                    if label_classes_on_bubbles:
                        lbl = class_abbrev.get(cls, cls) if class_abbrev else cls
                        ax.text(cx, cy, f"{lbl} (n={len(cluster_nodes)})",
                                fontsize=class_label_fontsize, fontweight="bold",
                                color=(*base_col, 0.85), ha="center", va="center",
                                bbox=dict(facecolor="white", edgecolor="none",
                                          alpha=class_label_box_alpha, pad=0.35),
                                zorder=3.6)

    # edges
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=float(width[(u, v)]),
                               edge_color=edge_color,
                               alpha=float(alpha[(u, v)]),
                               ax=ax)

    # nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=[node_fill[n] for n in G.nodes()],
        node_size=[node_size[n] for n in G.nodes()],
        edgecolors="#333", linewidths=node_ring_linewidth, alpha=0.98, ax=ax
    )

    # node labels
    if repel_labels:
        try:
            from adjustText import adjust_text
            texts = []
            for n,(x,y) in pos.items():
                texts.append(ax.text(x, y, n, fontsize=8, ha="center", va="center",
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25),
                                     zorder=4.0))
            adjust_text(texts, ax=ax,
                        expand_points=(1.2,1.4), expand_text=(1.2,1.4),
                        force_text=(0.6,0.9), force_points=(0.6,0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#bbb"))
        except Exception:
            nx.draw_networkx_labels(G, pos, font_size=8,
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25), ax=ax)
    else:
        nx.draw_networkx_labels(G, pos, font_size=8,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25), ax=ax)

    # ---------- legends ----------
    from matplotlib.patches import Patch
    from collections import Counter

    cls_counts = Counter([item_to_class[n] for n in G.nodes() if item_to_class.get(n)])

    # node code helper
    def _node_code(label: str) -> str:
        if label_code_resolver is not None:
            c = label_code_resolver(label, **(label_code_kwargs or {}))
            if c:
                return str(c)
        m = re.match(r"^([A-Z0-9]{2,6})\b", str(label).strip())
        return m.group(1) if m else str(label)

    # collect codes & degrees per class
    codes_by_class: Dict[str, List[str]] = {}
    deg_by_code: Dict[str, float] = {}
    for n in G.nodes():
        cls = item_to_class.get(n)
        if not cls:
            continue
        code = _node_code(n)
        codes_by_class.setdefault(cls, []).append(code)
        # representativeness: use degree (weighted optional)
        deg_by_code[code] = float(G.degree(n, weight=("weight" if use_weighted_degree else None)))

    # dedupe codes preserving appearance order
    for cls, codes in list(codes_by_class.items()):
        seen = set(); uniq = []
        for c in codes:
            if c not in seen:
                uniq.append(c); seen.add(c)
        codes_by_class[cls] = uniq

    priority = [c.upper() for c in (legend_priority_codes or [])]

    def _rank_codes(codes: List[str]) -> List[str]:
        if legend_code_sort in ("auto","degree"):
            ranked = sorted(codes, key=lambda c: (-deg_by_code.get(c, 0.0), c))
        else:
            ranked = sorted(codes)
        # put priority codes first if present
        pr = [c for c in ranked if c.upper() in priority]
        rest = [c for c in ranked if c.upper() not in priority]
        return pr + rest

    handles = []
    for cls, cnt in cls_counts.most_common():
        if cnt < class_legend_min_count:
            continue
        lbl = class_abbrev.get(cls, cls) if class_abbrev else cls
        codes = _rank_codes(codes_by_class.get(cls, []))
        shown = codes[:class_legend_codes_max]
        overflow = max(0, len(codes) - len(shown))
        codes_txt = class_legend_codes_separator.join(shown)
        if overflow > 0:
            codes_txt += f" +{overflow} more"
        display_label = f"{lbl} ({cnt})" + (f" — {codes_txt}" if codes_txt else "")
        handles.append(Patch(facecolor=class_colors.get(cls, "#cccccc"),
                             edgecolor="#333", label=display_label))

    # right-side edge legend (skip if class legend below)
    if show_edge_weight_legend and len(weights) > 0 and not show_class_legend_below:
        import matplotlib.lines as mlines
        q = np.quantile(weights, [0.25, 0.50, 0.90])
        ex = []
        for val, name in zip(q, ["25th", "50th", "90th"]):
            lw = 1 + 6*(val - wmin)/(wmax - wmin + 1e-9)
            ex.append(mlines.Line2D([], [], color="#333333", linewidth=lw, label=f"{name}: {val:.2f}"))
        leg2 = ax.legend(handles=ex, title="Edge weight examples",
                         loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        ax.add_artist(leg2)

    # class legend (below or right)
    if handles:
        if show_class_legend_below:
            n_classes = len(handles)
            ncol = min(legend_below_max_cols, max(1, int(np.ceil(np.sqrt(n_classes)))))
            plt.subplots_adjust(bottom=max(0.08, legend_below_pad))
            fig.legend(handles=handles, loc="lower center", ncol=ncol,
                       frameon=True, fontsize=legend_below_fontsize, title="Classes")
            plt.tight_layout(rect=(0, legend_below_pad, 1, 1))
        elif show_class_legend:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      frameon=True, title="Classes")

    ax.set_title(title); ax.set_axis_off()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return {"fig": fig, "ax": ax, "positions": pos, "communities": part, "graph": G}

