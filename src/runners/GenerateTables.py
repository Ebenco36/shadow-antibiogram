#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build AST tables (with *_Tested indicators) — programmatic API (no CLI).

Primary entry points:
- build_publication_tables(...)
- build_publication_tables_with_options(Options(...))

Returns: list[str] of produced file paths (CSV and any requested conversions)
"""

from __future__ import annotations
import logging
import os
import re
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from src.controllers.DataLoader import DataLoader, read_any

# -------------------------
# Defaults / Constants
# -------------------------
DEFAULT_INPUT = "datasets/output/tables/saved_with_test_indicators_tab.csv"
DEFAULT_OUT_DIR = "./outputs/publication_tables"

AGE_BUCKETS = ['<15', '15–29', '30–44', '45–59', '60–74', '75–89', '90+']
WARD_CANDIDATES = ['ARS_WardType', 'WardType', 'Ward', 'ARS_Wardtype']
SPECIMEN_CANDIDATES = ['TextMaterialgroupRkiL0', 'Specimen', 'Material', 'TextMaterial']

# -------------------------
# WHO regex lists (user-provided)
# -------------------------
CRITICAL_PATHOGENS = r"\b(?:Enterobacter|Escherichia|Klebsiella|Citrobacter|Serratia|Proteus|Morganella|Providencia|Acinetobacter baumannii|Mycobacterium)\b"
HIGH_PATHOGENS     = r"\b(?:Salmonella Typhi|Shigella spp|Enterococcus faecium|Pseudomonas aeruginosa|Non-typhoidal Salmonella|Neisseria gonorrhoeae|Staphylococcus aureus)\b"
MEDIUM_PATHOGENS   = r"\b(?:Group A Streptococci|Streptococcus pneumoniae|Haemophilus influenzae|Group B Streptococci)\b"
ALL_PATHOGENS = f"{CRITICAL_PATHOGENS}|{HIGH_PATHOGENS}|{MEDIUM_PATHOGENS}"

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("pubtables")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------------
# Options (programmatic)
# -------------------------
@dataclass
class Options:
    input: str = DEFAULT_INPUT
    out_dir: str = DEFAULT_OUT_DIR

    table1_top_n: Optional[int] = None
    table2_top_n: Optional[int] = None
    table2_top_mode: str = "per_gram"   # "per_gram" or "overall"
    appendix_top_n: Optional[int] = 20

    restrict_genera: Optional[List[str]] = None
    who_focus: bool = False             # filter to rows matching WHO regex lists

    # Output formats and compression
    formats: List[str] = field(default_factory=lambda: ["csv"])
    compression: str = "none"           # "none", "gzip", "bz2", "zstd"

    # LaTeX knobs (used if "latex" in formats)
    latex_longtable: bool = True
    latex_landscape: bool = False
    latex_fontsize: str = "scriptsize"

    # Optional Overleaf + bundle
    overleaf_project: Optional[str] = None
    overleaf_title: str = "AST Tables"
    zip_bundle: Optional[str] = None

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def antimicrobial_display(tested_col: str) -> str:
    """'AMK - Amikacin_Tested' -> 'Amikacin'."""
    m = re.match(r"^[^-]+-\s*(.+?)_Tested$", tested_col)
    return m.group(1).strip() if m else tested_col.replace("_Tested", "").strip()

def fmt_cell(count: int, row_total: int) -> str:
    return f"{int(count)} ({(count/row_total*100):.1f})" if row_total > 0 else "0 (0.0)"

def fmt_total(count: int, grand_total: int) -> str:
    return f"{int(count)} ({(count/grand_total*100):.1f})" if grand_total > 0 else "0 (0.0)"

def gram_key(gt: str) -> int:
    s = (gt or "").lower()
    if "gram-positive" in s:
        return 0
    if "gram-negative" in s:
        return 1
    return 2

def coalesce_columns(df: pd.DataFrame, candidates: Iterable[str], default_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    df[default_name] = pd.NA
    return default_name

def _norm(s): 
    return str(s).strip().lower() if pd.notna(s) else ""

def _is_salmonella_typhi(pathogen: str) -> bool:
    p = _norm(pathogen)
    return ("salmonella" in p) and ("typhi" in p)

def _is_salmonella_paratyphi(pathogen: str) -> bool:
    p = _norm(pathogen)
    return ("salmonella" in p) and ("paratyphi" in p)

def detect_antibiotic_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.endswith('_Tested')]
    if not cols:
        raise ValueError("No antibiotic *_Tested columns found.")
    return cols

def safe_sex_levels(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    levels = [x for x in df[col].dropna().unique().tolist()]
    preferred = ['Woman', 'Man', 'Others', 'Unknown']
    return [x for x in preferred if x in levels] + [x for x in levels if x not in preferred]

def derive_highlevel_age(df: pd.DataFrame) -> pd.Series:
    """Create HighLevelAgeRange if missing using AgeRange/AgeGroup semantics."""
    if 'HighLevelAgeRange' in df.columns:
        return df['HighLevelAgeRange']

    def bucket_from_text(x: str):
        if not isinstance(x, str):
            return None
        m = re.search(r"(\d+)", x)
        if not m:
            return None
        age = int(m.group(1))
        if age < 15:
            return '<15'
        if age < 30:
            return '15–29'
        if age < 45:
            return '30–44'
        if age < 60:
            return '45–59'
        if age < 75:
            return '60–74'
        if age < 90:
            return '75–89'
        return '90+'

    if 'AgeRange' in df.columns:
        est = df['AgeRange'].apply(bucket_from_text)
    elif 'AgeGroup' in df.columns:
        est = df['AgeGroup'].apply(bucket_from_text)
    else:
        est = pd.Series(pd.NA, index=df.index)

    if 'AgeGroup' in df.columns:
        est = est.fillna(df['AgeGroup'].apply(bucket_from_text))

    return est.fillna('Unknown')

# -------------------------
# WHO tagging/filter (regex-driven)
# -------------------------
def tag_who_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag rows with WHO_Tier and WHO_Target based on regex patterns applied to 'Pathogen'.
    """
    if 'Pathogen' not in df.columns:
        raise KeyError("WHO tagging requires column: Pathogen")

    # Compile with case-insensitive matching
    RE_CRIT = re.compile(CRITICAL_PATHOGENS, re.IGNORECASE)
    RE_HIGH = re.compile(HIGH_PATHOGENS, re.IGNORECASE)
    RE_MED  = re.compile(MEDIUM_PATHOGENS, re.IGNORECASE)

    # Sub-patterns to set consistent WHO_Target for Critical
    RE_ENTB_GENS = re.compile(
        r"\b(?:Enterobacter|Escherichia|Klebsiella|Citrobacter|Serratia|Proteus|Morganella|Providencia)\b",
        re.IGNORECASE
    )
    RE_AB   = re.compile(r"\bAcinetobacter baumannii\b", re.IGNORECASE)
    RE_MYCO = re.compile(r"\bMycobacterium\b", re.IGNORECASE)

    # High targets
    RE_SAL_TY   = re.compile(r"\bSalmonella Typhi\b", re.IGNORECASE)
    RE_SHIG_SPP = re.compile(r"\bShigella spp\b", re.IGNORECASE)
    RE_EFM      = re.compile(r"\bEnterococcus faecium\b", re.IGNORECASE)
    RE_PAE      = re.compile(r"\bPseudomonas aeruginosa\b", re.IGNORECASE)
    RE_NTS      = re.compile(r"\bNon-typhoidal Salmonella\b", re.IGNORECASE)
    RE_NG       = re.compile(r"\bNeisseria gonorrhoeae\b", re.IGNORECASE)
    RE_SA       = re.compile(r"\bStaphylococcus aureus\b", re.IGNORECASE)

    # Medium targets
    RE_GAS = re.compile(r"\bGroup A Streptococci\b", re.IGNORECASE)
    RE_SPN = re.compile(r"\bStreptococcus pneumoniae\b", re.IGNORECASE)
    RE_HI  = re.compile(r"\bHaemophilus influenzae\b", re.IGNORECASE)
    RE_GBS = re.compile(r"\bGroup B Streptococci\b", re.IGNORECASE)

    who_tier = []
    who_target = []

    for name in df['Pathogen'].astype(str):
        tier = pd.NA
        target = pd.NA

        # Critical
        if RE_CRIT.search(name):
            if RE_AB.search(name):
                tier, target = "Critical", "Acinetobacter baumannii"
            elif RE_MYCO.search(name):
                tier, target = "Critical", "Mycobacterium"
            elif RE_ENTB_GENS.search(name):
                tier, target = "Critical", "Enterobacterales"

        # High
        if pd.isna(tier) and RE_HIGH.search(name):
            if RE_PAE.search(name):
                tier, target = "High", "Pseudomonas aeruginosa"
            elif RE_SA.search(name):
                tier, target = "High", "Staphylococcus aureus"
            elif RE_EFM.search(name):
                tier, target = "High", "Enterococcus faecium"
            elif RE_NG.search(name):
                tier, target = "High", "Neisseria gonorrhoeae"
            elif RE_SHIG_SPP.search(name):
                tier, target = "High", "Shigella spp"
            elif RE_SAL_TY.search(name):
                tier, target = "High", "Salmonella Typhi"
            elif RE_NTS.search(name):
                tier, target = "High", "Non-typhoidal Salmonella"

        # Medium
        if pd.isna(tier) and RE_MED.search(name):
            if RE_SPN.search(name):
                tier, target = "Medium", "Streptococcus pneumoniae"
            elif RE_HI.search(name):
                tier, target = "Medium", "Haemophilus influenzae"
            elif RE_GAS.search(name):
                tier, target = "Medium", "Group A Streptococci"
            elif RE_GBS.search(name):
                tier, target = "Medium", "Group B Streptococci"

        who_tier.append(tier)
        who_target.append(target)

    out = df.copy()
    out['WHO_Tier'] = who_tier
    out['WHO_Target'] = who_target
    return out

def filter_who_simple(df: pd.DataFrame) -> pd.DataFrame:
    tagged = tag_who_simple(df)
    return tagged[tagged['WHO_Tier'].notna()].copy()

def filter_who_regex(df: pd.DataFrame) -> pd.DataFrame:
    if 'Pathogen' not in df.columns:
        raise KeyError("WHO filtering requires column: Pathogen")
    re_all = re.compile(ALL_PATHOGENS, re.IGNORECASE)
    return df[df['Pathogen'].astype(str).str.contains(re_all, na=False)].copy()

# -------------------------
# Table builders
# -------------------------
def build_table1_bacteria_summary(df_raw: pd.DataFrame, out_dir: str, top_n: Optional[int]) -> str:
    req_cols = ['Pathogen', 'PathogenGenus', 'Sex']
    miss = [c for c in req_cols if c not in df_raw.columns]
    if miss:
        raise KeyError(f"Missing columns for Table 1: {miss}")

    df = df_raw.copy()
    df['Bacteria'] = df['PathogenGenus']
    df['HighLevelAgeRange'] = derive_highlevel_age(df)

    age_counts = df.groupby(['Bacteria', 'HighLevelAgeRange']).size().unstack(fill_value=0)
    age_cols = [c for c in AGE_BUCKETS if c in age_counts.columns]
    age_counts = age_counts.reindex(columns=age_cols, fill_value=0)

    sex_levels = safe_sex_levels(df, 'Sex')
    sex_counts = df.groupby(['Bacteria', 'Sex']).size().unstack(fill_value=0)
    sex_counts = sex_counts.reindex(columns=sex_levels, fill_value=0)

    table = pd.concat([age_counts, sex_counts], axis=1).fillna(0).astype(int)
    table['Total'] = table.sum(axis=1)
    table = table.sort_values('Total', ascending=False)
    if top_n is not None:
        table = table.head(int(top_n))

    grand = int(table['Total'].sum())
    formatted = table.copy()
    for col in age_cols + sex_levels:
        formatted[col] = table.apply(lambda r: fmt_cell(r[col], r['Total']), axis=1)
    formatted['Total'] = table['Total'].apply(lambda x: fmt_total(x, grand))
    formatted = formatted.reindex(columns=age_cols + sex_levels + ['Total'])

    out_csv = os.path.join(out_dir, "bacteria_summary.csv")
    ensure_dir(os.path.dirname(out_csv))
    formatted.to_csv(out_csv, index_label="Different Types of Bacteria")
    logger.info(f"Saved Table 1 -> {out_csv}")
    return out_csv

def build_table2_specimen_single(df_raw: pd.DataFrame, out_dir: str, top_n: Optional[int], top_mode: str) -> str:
    specimen_col = coalesce_columns(df_raw, SPECIMEN_CANDIDATES, "Specimen")
    req = ['PathogenGenus', 'GramType', specimen_col]
    miss = [c for c in req if c not in df_raw.columns]
    if miss:
        raise KeyError(f"Missing columns for Table 2: {miss}")

    df = df_raw.copy()
    df['Bacteria'] = df['PathogenGenus']
    df['Specimen'] = df[specimen_col]
    df['GramType'] = df['GramType'].fillna('Unknown')
    grand_total = len(df)

    specimen_order = df['Specimen'].dropna().unique().tolist()
    specimen_totals = df['Specimen'].value_counts()
    specimen_header_map = {s: f"{s} (n={int(specimen_totals.get(s, 0))})" for s in specimen_order}
    total_header = f"Total (n={grand_total})"

    def make_group_pivot(sub: pd.DataFrame) -> pd.DataFrame:
        p = sub.groupby(['Bacteria', 'Specimen']).size().unstack(fill_value=0)
        p = p.reindex(columns=specimen_order, fill_value=0)
        p['Total'] = p.sum(axis=1)
        return p.sort_values('Total', ascending=False)

    groups = sorted(df.groupby('GramType'), key=lambda x: gram_key(x[0]))
    raw_blocks: List[Tuple[str, pd.DataFrame]] = [(gt, make_group_pivot(sub)) for gt, sub in groups]

    keep_index: Optional[set] = None
    if top_mode == "overall" and top_n is not None:
        combined = pd.concat([p for _, p in raw_blocks], axis=0)
        keep_index = set(combined.nlargest(int(top_n), 'Total').index.tolist())

    def format_counts(p: pd.DataFrame) -> pd.DataFrame:
        pf = p.copy()
        for col in specimen_order:
            pf[col] = [fmt_cell(x, rt) for x, rt in zip(p[col], p['Total'])]
        pf['Total'] = [fmt_total(x, grand_total) for x in p['Total']]
        return pf

    blocks: List[pd.DataFrame] = []
    for gt, p in raw_blocks:
        n_gram = int(df[df['GramType'] == gt].shape[0])
        header_row = {'Gram-stain': f"{gt} (n={n_gram})", 'Different Types of Bacteria': ''}
        for s in specimen_order:
            header_row[s] = ''
        header_row['Total'] = ''
        blocks.append(pd.DataFrame([header_row]))

        q = p.copy()
        if keep_index is not None:
            q = q.loc[[idx for idx in q.index if idx in keep_index]]
        elif top_mode == "per_gram" and top_n is not None:
            q = q.nlargest(int(top_n), 'Total')

        qf = format_counts(q)
        qf.insert(0, 'Different Types of Bacteria', qf.index)
        qf.insert(0, 'Gram-stain', '')
        blocks.append(qf.reset_index(drop=True))

    out = pd.concat(blocks, axis=0, ignore_index=True)
    rename_cols = {s: specimen_header_map[s] for s in specimen_order}
    out = out.rename(columns=rename_cols).rename(columns={'Total': total_header})
    final_cols = ['Gram-stain', 'Different Types of Bacteria'] + [specimen_header_map[s] for s in specimen_order] + [total_header]
    out = out[final_cols]

    out_csv = os.path.join(out_dir, "bacteria_by_specimen_SINGLE_TABLE.csv")
    ensure_dir(os.path.dirname(out_csv))
    out.to_csv(out_csv, index=False)
    logger.info(f"Saved Table 2 -> {out_csv}")
    return out_csv

def build_table4_gn_antimicrobial_tested(df_raw: pd.DataFrame, out_dir: str) -> str:
    meta_cols = ["Pathogen", "PathogenGenus", "GramType"]
    miss = [c for c in meta_cols if c not in df_raw.columns]
    if miss:
        raise KeyError(f"Missing columns for Table 4: {miss}")

    df = df_raw[meta_cols + [c for c in detect_antibiotic_columns(df_raw)]].copy()
    df["GramType"] = df["GramType"].astype(str)
    df = df[df["GramType"].str.contains("Gram-negative", case=False, na=False)].copy()

    def is_salmonella_typhi(row) -> bool:
        p = _norm(row.get("Pathogen", ""))
        g = _norm(row.get("PathogenGenus", ""))
        return ("salmonella" in g) and ("typhi" in p)

    organism_specs = [
        ("Pseudomonas spp.", lambda r: _norm(r.get("PathogenGenus", "")) == "pseudomonas"),
        ("Klebsiella spp.",  lambda r: _norm(r.get("PathogenGenus", "")) == "klebsiella"),
        ("Proteus spp.",     lambda r: _norm(r.get("PathogenGenus", "")) == "proteus"),
        ("E. coli",          lambda r: _norm(r.get("PathogenGenus", "")) == "escherichia"),
        ("Enterobacter spp.",lambda r: _norm(r.get("PathogenGenus", "")) == "enterobacter"),
        ("Salmonella typhi", is_salmonella_typhi),
    ]

    antibiotics = [(tc, antimicrobial_display(tc)) for tc in detect_antibiotic_columns(df)]
    rows = []
    for tested_col, ab_name in antibiotics:
        row = {"Antimicrobials": ab_name}
        total_T = 0
        for org_label, selector in organism_specs:
            sub = df[df.apply(selector, axis=1)]
            T = pd.to_numeric(sub.get(tested_col, 0), errors="coerce").fillna(0).sum()
            row[f"{org_label} ∑T"] = int(T)
            total_T += int(T)
        row["Total ∑T"] = int(total_T)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Total ∑T", ascending=False).reset_index(drop=True)
    ordered_cols = ["Antimicrobials"] + [f"{lbl} ∑T" for lbl, _ in organism_specs] + ["Total ∑T"]
    out = out.reindex(columns=ordered_cols)

    out_csv = os.path.join(out_dir, "table4_gram_negative_antimicrobial_tested.csv")
    ensure_dir(os.path.dirname(out_csv))
    out.to_csv(out_csv, index=False)
    logger.info(f"Saved Table 4 -> {out_csv}")
    return out_csv

def build_pub_pivot(df_in: pd.DataFrame, category_col: str, out_dir: str, outfile: str, top_n: Optional[int]) -> str:
    p = df_in.groupby(['Bacteria', category_col]).size().unstack(fill_value=0)
    col_order = df_in[category_col].value_counts().index.tolist()
    p = p.reindex(columns=col_order, fill_value=0)
    p['Total'] = p.sum(axis=1)
    p = p.sort_values('Total', ascending=False)
    if top_n is not None:
        p = p.head(int(top_n))

    grand = int(p['Total'].sum())
    col_totals = df_in[category_col].value_counts()
    header_map = {c: f"{c} (n={int(col_totals.get(c, 0))})" for c in col_order}
    total_header = f"Total (n={grand})"

    pf = p.copy()
    for c in col_order:
        pf[c] = [fmt_cell(x, rt) for x, rt in zip(p[c], p['Total'])]
    pf['Total'] = [fmt_total(x, grand) for x in p['Total']]
    pf = pf.rename(columns={**header_map, 'Total': total_header})
    pf = pf[[*header_map.values(), total_header]]

    out_path = os.path.join(out_dir, outfile)
    ensure_dir(os.path.dirname(out_path))
    pf.to_csv(out_path, index_label="Different Types of Bacteria")
    logger.info(f"Saved -> {out_path}")
    return out_path

def build_appendix_tables(df_raw: pd.DataFrame, out_dir: str, top_n: Optional[int]) -> List[str]:
    saved: List[str] = []

    overall = df_raw.groupby(df_raw['PathogenGenus']).size().sort_values(ascending=False)
    if top_n is not None:
        overall = overall.head(int(top_n))
    grand_total_overall = int(overall.sum())
    overall_tbl = pd.DataFrame({
        'Total (n)': overall.astype(int),
        'Total (%)': (overall / grand_total_overall * 100).round(1)
    })
    overall_tbl['Total'] = overall_tbl.apply(lambda r: f"{int(r['Total (n)'])} ({r['Total (%)']:.1f})", axis=1)
    overall_tbl = overall_tbl[['Total']]
    overall_path = os.path.join(out_dir, "tableA_top_pathogens_overall.csv")
    ensure_dir(os.path.dirname(overall_path))
    overall_tbl.to_csv(overall_path, index_label="Different Types of Bacteria")
    logger.info(f"Saved -> {overall_path}")
    saved.append(overall_path)

    df = df_raw.copy()
    df['Bacteria'] = df['PathogenGenus']

    if 'SeasonName' in df.columns:
        saved.append(build_pub_pivot(df, 'SeasonName', out_dir, "tableB_pathogens_by_season.csv", top_n))

    ward_col = None
    for c in WARD_CANDIDATES:
        if c in df.columns:
            ward_col = c
            break
    if ward_col:
        saved.append(build_pub_pivot(df, ward_col, out_dir, "tableC_pathogens_by_wardtype.csv", top_n))

    return saved

def build_who_top_table_simple(df: pd.DataFrame, out_dir: str) -> List[str]:
    if 'WHO_Tier' not in df.columns or 'WHO_Target' not in df.columns:
        df = tag_who_simple(df)

    saved = []

    fam = df['WHO_Target'].value_counts(dropna=True)
    grand = int(fam.sum())
    overall = pd.DataFrame({'Total (n)': fam.astype(int), 'Total (%)': (fam / grand * 100).round(1)})
    overall['Total'] = overall.apply(lambda r: f"{int(r['Total (n)'])} ({r['Total (%)']:.1f})", axis=1)
    overall = overall[['Total']]
    p_overall = os.path.join(out_dir, "who_top_pathogens_simple.csv")
    ensure_dir(os.path.dirname(p_overall))
    overall.to_csv(p_overall, index_label="WHO target")
    saved.append(p_overall)

    tier_counts = df.groupby(['WHO_Tier', 'WHO_Target']).size().reset_index(name='n')
    blocks = []
    for tier in ['Critical', 'High', 'Medium']:
        sub = tier_counts[tier_counts['WHO_Tier'] == tier].sort_values('n', ascending=False)
        if sub.empty:
            continue
        n_tier = int(sub['n'].sum())
        header = pd.DataFrame({'WHO tier/target': [f"{tier} (n={n_tier})"], 'Total': ['']})
        sub['Total'] = sub.apply(lambda r: f"{int(r['n'])} ({(r['n']/n_tier*100):.1f})", axis=1)
        sub = sub[['WHO_Target', 'Total']].rename(columns={'WHO_Target': 'WHO tier/target'})
        blocks += [header, sub]
    by_tier = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=['WHO tier/target', 'Total'])
    p_tier = os.path.join(out_dir, "who_top_pathogens_by_tier_simple.csv")
    by_tier.to_csv(p_tier, index=False)
    saved.append(p_tier)

    logger.info(f"Saved WHO tables -> {p_overall}, {p_tier}")
    return saved

# -------------------------
# Output writers / converters (multi-format + LaTeX + Overleaf)
# -------------------------
TEXT_COMP_MAP = {
    "none": None,
    "gzip": "gzip",
    "bz2": "bz2",
    "zstd": "zstd",
}

def _latex_safe_label(stem: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]+', '-', stem).strip('-').lower()
    return f"tab:{s}" if s else "tab:table"

def _latex_caption_from_filename(stem: str) -> str:
    title = re.sub(r'[_\-]+', ' ', stem).strip().title()
    return title

def write_latex_table(
    df: pd.DataFrame,
    base_path_no_ext: str,
    *,
    caption: str | None = None,
    label: str | None = None,
    longtable: bool = True,
    landscape: bool = False,
    fontsize: str | None = "scriptsize",
) -> str:
    out = f"{base_path_no_ext}.tex"
    stem = os.path.basename(base_path_no_ext)

    if caption is None:
        caption = _latex_caption_from_filename(stem)
    if label is None:
        label = _latex_safe_label(stem)

    try:
        latex_body = df.to_latex(
            index=False, escape=True, longtable=longtable,
            bold_rows=False, multicolumn=False, multicolumn_format='c',
            na_rep="", caption=None, label=None, buf=None, booktabs=True,
        )
    except TypeError:
        latex_body = df.to_latex(
            index=False, escape=True, longtable=longtable,
            bold_rows=False, multicolumn=False, multicolumn_format='c',
            na_rep="", caption=None, label=None, buf=None,
        )

    if longtable:
        env_open = textwrap.dedent(f"""
        % Auto-generated table: {stem}
        \\begin{{longtable}}{{{'l' * len(df.columns)}}}
        \\caption{{{caption}}}\\label{{{label}}}\\\\
        """).strip("\n")
        env_close = "\\end{longtable}\n"
    else:
        env_open = textwrap.dedent(f"""
        % Auto-generated table: {stem}
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        """).strip("\n")
        env_close = "\\end{table}\n"

    prefix = []
    suffix = []
    if landscape:
        prefix.append("\\begin{landscape}")
        suffix.insert(0, "\\end{landscape}")
    if fontsize:
        prefix.append(f"\\begin{{{fontsize}}}")
        suffix.insert(0, f"\\end{{{fontsize}}}")

    content = "\n".join(prefix + [env_open, latex_body, env_close] + suffix)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(content)
    return out

def _write_df(
    df: pd.DataFrame,
    base_path_no_ext: str,
    fmt: str,
    compression: str | None,
    *,
    latex_longtable=True,
    latex_landscape=False,
    latex_fontsize="scriptsize"
):
    fmt = fmt.lower()
    comp = None if not compression or compression == "none" else compression

    if fmt == "latex":
        return write_latex_table(
            df, base_path_no_ext,
            longtable=latex_longtable,
            landscape=latex_landscape,
            fontsize=latex_fontsize,
        )

    if fmt == "csv":
        out = f"{base_path_no_ext}.csv" + (".gz" if comp == "gzip" else ".bz2" if comp == "bz2" else ".zst" if comp == "zstd" else "")
        df.to_csv(out, index=False, compression=comp)
        return out

    if fmt == "tsv":
        out = f"{base_path_no_ext}.tsv" + (".gz" if comp == "gzip" else ".bz2" if comp == "bz2" else ".zst" if comp == "zstd" else "")
        df.to_csv(out, sep="\t", index=False, compression=comp)
        return out

    if fmt == "parquet":
        out = f"{base_path_no_ext}.parquet"
        df.to_parquet(out, index=False)
        return out

    if fmt == "feather":
        out = f"{base_path_no_ext}.feather"
        df.to_feather(out)
        return out

    if fmt == "json":
        out = f"{base_path_no_ext}.json"
        df.to_json(out, orient="records", lines=False)
        return out

    if fmt == "html":
        out = f"{base_path_no_ext}.html"
        df.to_html(out, index=False)
        return out

    if fmt == "xlsx":
        out = f"{base_path_no_ext}.xlsx"
        with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
            sheet = os.path.basename(base_path_no_ext)[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
        return out

    raise ValueError(f"Unsupported format: {fmt}")

def convert_outputs(
    csv_paths: List[str],
    formats: List[str],
    compression: str,
    *,
    latex_longtable: bool,
    latex_landscape: bool,
    latex_fontsize: str
) -> List[str]:
    produced: List[str] = []
    fmts = [f.lower() for f in formats]
    _ = TEXT_COMP_MAP.get(compression.lower(), None)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        base_no_ext = os.path.splitext(csv_path)[0]

        for fmt in fmts:
            if fmt == "csv":
                out = _write_df(
                    df, base_no_ext, "csv", compression,
                    latex_longtable=latex_longtable,
                    latex_landscape=latex_landscape,
                    latex_fontsize=latex_fontsize
                )
            else:
                out = _write_df(
                    df, base_no_ext, fmt, None,
                    latex_longtable=latex_longtable,
                    latex_landscape=latex_landscape,
                    latex_fontsize=latex_fontsize
                )
            logger.info(f"Written {fmt}: {out}")
            produced.append(out)

    return produced

def zip_bundle(file_paths: List[str], zip_path: str):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            arcname = os.path.basename(p)
            zf.write(p, arcname)
    logger.info(f"ZIP bundle written -> {zip_path}")

def build_overleaf_project(tex_files: List[str], project_dir: str, title: str = "AST Tables") -> str:
    """
    Create an Overleaf-ready project:
      project_dir/
        main.tex
        tables/*.tex
    """
    tables_dir = os.path.join(project_dir, "tables")
    ensure_dir(tables_dir)

    rel_inputs = []
    for t in tex_files:
        dst = os.path.join(tables_dir, os.path.basename(t))
        if os.path.abspath(t) != os.path.abspath(dst):
            with open(t, "r", encoding="utf-8") as src, open(dst, "w", encoding="utf-8") as out:
                out.write(src.read())
        rel_inputs.append(os.path.join("tables", os.path.basename(t)))

    preamble = textwrap.dedent(r"""
        % Auto-generated Overleaf project
        \documentclass[11pt,a4paper]{article}
        \usepackage[margin=2.2cm]{geometry}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{lscape}
        \usepackage{caption}
        \usepackage{xcolor}
        \captionsetup[table]{skip=6pt}
        \title{{TITLE}}
        \date{}
        \begin{document}
        \maketitle

        %%% Tables
    """).lstrip("\n")
    preamble = preamble.replace("{TITLE}", title)

    main_tex = os.path.join(project_dir, "main.tex")
    ensure_dir(project_dir)
    with open(main_tex, "w", encoding="utf-8") as fh:
        fh.write(preamble)
        for path in rel_inputs:
            fh.write("\\input{" + path + "}\n\n")
        fh.write("\\end{document}\n")

    logger.info(f"Overleaf project created -> {project_dir}")
    return main_tex

# -------------------------
# Public API (no CLI)
# -------------------------
def build_publication_tables(
    input: str = DEFAULT_INPUT,
    out_dir: str = DEFAULT_OUT_DIR,
    *,
    table1_top_n: Optional[int] = None,
    table2_top_n: Optional[int] = None,
    table2_top_mode: str = "per_gram",
    appendix_top_n: Optional[int] = 20,
    restrict_genera: Optional[List[str]] = None,
    who_focus: bool = False,
    formats: Optional[List[str]] = None,
    compression: str = "none",
    latex_longtable: bool = True,
    latex_landscape: bool = False,
    latex_fontsize: str = "scriptsize",
    overleaf_project: Optional[str] = None,
    overleaf_title: str = "AST Tables",
    zip_bundle_path: Optional[str] = None,
) -> List[str]:
    """
    Programmatic entry point to produce tables and optional conversions/Overleaf/ZIP.
    """
    ensure_dir(out_dir)
    if not os.path.exists(input):
        raise FileNotFoundError(f"Input file not found: {input}")

    df = read_any(input)

    # Minimal schema check
    need = ['Pathogen', 'PathogenGenus']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns in input: {miss}")

    # Tag WHO columns upfront
    df = tag_who_simple(df)

    # Optional genus restriction
    if restrict_genera:
        before = len(df)
        df = df[df['PathogenGenus'].isin(restrict_genera)].copy()
        logger.info(f"Restrict genera -> rows: {before} -> {len(df)}")

    # Optional WHO filter
    if who_focus:
        before = len(df)
        df = df[df['WHO_Tier'].notna()].copy()
        # Alternatively: df = filter_who_regex(df)
        logger.info(f"WHO focus -> rows: {before} -> {len(df)}")

    produced_csv: List[str] = []
    produced_csv.append(build_table1_bacteria_summary(df, out_dir, top_n=table1_top_n))
    produced_csv.append(build_table2_specimen_single(df, out_dir, top_n=table2_top_n, top_mode=table2_top_mode))
    produced_csv.append(build_table4_gn_antimicrobial_tested(df, out_dir))
    produced_csv += build_appendix_tables(df, out_dir, top_n=appendix_top_n)
    produced_csv += build_who_top_table_simple(df, out_dir)

    logger.info("\nSaved tables (CSV):")
    for pth in produced_csv:
        logger.info(f"- {pth}")

    # Validate formats
    valid = {"csv", "tsv", "parquet", "feather", "json", "html", "xlsx", "latex"}
    req_formats = [f.lower() for f in (formats or ["csv"])]
    bad = [f for f in req_formats if f not in valid]
    if bad:
        raise ValueError(f"Unsupported format(s): {bad}. Choose from {sorted(valid)}")

    # Convert to requested formats (including csv with compression if requested)
    all_paths = list(produced_csv)
    if req_formats:
        converted = convert_outputs(
            produced_csv, req_formats, compression=compression,
            latex_longtable=latex_longtable,
            latex_landscape=latex_landscape,
            latex_fontsize=latex_fontsize
        )
        all_paths = sorted(set(all_paths + converted))

    # Overleaf project creation (if latex files exist)
    if overleaf_project:
        tex_files = [p for p in all_paths if p.lower().endswith(".tex")]
        if not tex_files:
            logger.warning("No .tex files found for Overleaf. Did you include 'latex' in formats?")
        ensure_dir(overleaf_project)
        main_tex = build_overleaf_project(tex_files, overleaf_project, title=overleaf_title)
        all_paths.append(main_tex)

    # Optional: zip bundle
    if zip_bundle_path:
        ensure_dir(os.path.dirname(zip_bundle_path) or ".")
        zip_bundle(all_paths, zip_bundle_path)

    return all_paths

def build_publication_tables_with_options(opts: Options) -> List[str]:
    return build_publication_tables(
        input=opts.input,
        out_dir=opts.out_dir,
        table1_top_n=opts.table1_top_n,
        table2_top_n=opts.table2_top_n,
        table2_top_mode=opts.table2_top_mode,
        appendix_top_n=opts.appendix_top_n,
        restrict_genera=opts.restrict_genera,
        who_focus=opts.who_focus,
        formats=opts.formats,
        compression=opts.compression,
        latex_longtable=opts.latex_longtable,
        latex_landscape=opts.latex_landscape,
        latex_fontsize=opts.latex_fontsize,
        overleaf_project=opts.overleaf_project,
        overleaf_title=opts.overleaf_title,
        zip_bundle_path=opts.zip_bundle,
    )


if __name__ == "__main__":
    files = build_publication_tables(
        input="./datasets/WHO_Aware_data",
        out_dir="./outputs/publication_tables",
        formats=["csv", "latex"],
        who_focus=True
    )

    # Or with Options
    opts = Options(
        input="./datasets/WHO_Aware_data",
        out_dir="./outputs/publication_tables",
        formats=["csv", "latex"],
        overleaf_project="./outputs/publication_tables/overleaf_project"
    )
    files = build_publication_tables_with_options(opts)

