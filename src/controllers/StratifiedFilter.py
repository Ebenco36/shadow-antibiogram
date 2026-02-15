import re
import numpy as np
import pandas as pd
import unicodedata as _ud
from typing import Any, Dict, Iterable, List, Tuple, TypeAlias, Union, Optional, Callable, Generator

Spec = Union[str, Dict[str, Any]]
KeyLevel = Tuple[str, Any]
PrefilterT = Callable[[pd.DataFrame], pd.DataFrame]


class StratifiedFilter:
    """
    Build subsets for each stratifier with optional per-key 'in' / 'not in' filtering.
    Supports extra metadata fields (e.g., size) that can be returned alongside subsets.
    """

    def __init__(self, df: pd.DataFrame, *, dropna_levels: bool = True):
        self.df = df
        self.dropna_levels_default = dropna_levels

    # ---------- public API ----------

    def stratify(
        self,
        stratifiers: Iterable[Spec],
        *,
        dropna_levels: Optional[bool] = None,
        with_meta: bool = False
    ) -> Dict[KeyLevel, Any]:
        """
        Return a dict mapping (key, level) -> subset DataFrame (or (df, meta) if with_meta=True).
        """
        dropna_lvls = self.dropna_levels_default if dropna_levels is None else dropna_levels
        out: Dict[KeyLevel, Any] = {}

        for key, fdict, extras in self._normalize_with_meta(stratifiers):
            if key not in self.df.columns:
                continue  # skip gracefully

            mask = self._mask_for_key(key, fdict)
            dff = self.df[mask]

            levels = dff[key].dropna().unique(
            ) if dropna_lvls else dff[key].unique()
            for lvl in levels:
                sub = dff[dff[key] == lvl].copy()
                out[(key, lvl)] = (sub, extras.copy()) if with_meta else sub

        return out

    def iter_subsets(
        self,
        stratifiers: Iterable[Spec],
        *,
        dropna_levels: Optional[bool] = None,
        with_meta: bool = False
    ) -> Generator[Any, None, None]:
        """
        Generator: yields (key, level, subset_df) or (key, level, subset_df, meta) if with_meta=True.
        """
        subsets = self.stratify(
            stratifiers, dropna_levels=dropna_levels, with_meta=with_meta)
        for (key, level), val in subsets.items():
            if with_meta:
                df, meta = val
                yield key, level, df, meta
            else:
                yield key, level, val

    # inside StratifiedFilter
    def iter_specs(self, stratifiers, with_meta: bool = False):
        """
        Public wrapper around normalization.
        Yields (key, fdict) or (key, fdict, extras) if with_meta=True.
        """
        for key, fdict, extras in self._normalize_with_meta(stratifiers):
            if with_meta:
                yield key, fdict, extras
            else:
                yield key, fdict

    def get_subset(
        self,
        stratifiers: Iterable[Spec],
        key: str,
        level: Any,
        *,
        dropna_levels: Optional[bool] = None,
        with_meta: bool = False
    ) -> Any:
        """
        Convenience accessor: run stratify() and return the specific (key, level) subset.
        Returns df or (df, meta) depending on with_meta.
        """
        subsets = self.stratify(
            stratifiers, dropna_levels=dropna_levels, with_meta=with_meta)
        return subsets.get((key, level))

    # ---------- prefilter builders ----------

    def build_prefilter(self, key: str, filter_dict: Optional[Dict[str, Any]] = None) -> PrefilterT:
        fdict = (filter_dict or {}).copy()

        def _pf(df: pd.DataFrame) -> pd.DataFrame:
            if key not in df.columns:
                return df
            mask = pd.Series(True, index=df.index)
            if "in" in fdict:
                mask &= df[key].isin(self._as_list(fdict["in"]))
            if "not in" in fdict:
                mask &= ~df[key].isin(self._as_list(fdict["not in"]))
            return df[mask]
        return _pf

    def build_prefilters(self, stratifiers: Iterable[Spec]) -> List[PrefilterT]:
        return [self.build_prefilter(key, fdict) for key, fdict, _extras in self._normalize_with_meta(stratifiers)]

    # ---------- helpers ----------

    @staticmethod
    def _as_list(x: Any) -> List[Any]:
        if isinstance(x, (list, tuple, set, pd.Index)):
            return list(x)
        return [x]

    def _normalize_with_meta(
        self,
        stratifiers: Iterable[Spec]
    ) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        norm: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for item in stratifiers:
            if isinstance(item, str):
                norm.append((item, {}, {}))
            else:
                key = item.get("key")
                fdict = item.get("filter_dict", {}) or {}
                extras = {k: v for k, v in item.items(
                ) if k not in ("key", "filter_dict")}
                norm.append((key, fdict, extras))
        return norm

    def _mask_for_key(self, key: str, fdict: Dict[str, Any]) -> pd.Series:
        mask = pd.Series(True, index=self.df.index)
        if "in" in fdict:
            mask &= self.df[key].isin(self._as_list(fdict["in"]))
        if "not in" in fdict:
            mask &= ~self.df[key].isin(self._as_list(fdict["not in"]))
        return mask


# ------------------------ Filtering utilities for Pathogens with regex ------------------------ #

def make_pathogen_pattern(names: List[str]) -> str:
    """Build r'\b(?:name1|name2|...)\b' safely from a list of pathogen names."""
    escaped = [re.escape(n)
               for n in names]  # handles spaces, hyphens, dots, etc.
    return r"\b(?:" + "|".join(escaped) + r")\b"


def normalize_pathogen_filters(research_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every {'column': 'Pathogen', 'values': ...} uses the \b(?:...)\b regex format."""
    for rq in research_questions:
        for f in rq.get("filters", []):
            if f.get("column") != "Pathogen" or "values" not in f:
                continue
            vals = f["values"]
            # List-like -> convert to regex
            if isinstance(vals, (list, tuple, set)):
                f["values"] = make_pathogen_pattern(list(vals))
            # String -> ensure it is wrapped with \b(?: ... )\b
            elif isinstance(vals, str):
                s = vals.strip()
                if not (s.startswith(r"\b(?:") and s.endswith(r")\b")):
                    # strip any surrounding parentheses once before wrapping
                    inner = s
                    if inner.startswith("(?:") and inner.endswith(")"):
                        inner = inner[3:-1]
                    elif inner.startswith("(") and inner.endswith(")"):
                        inner = inner[1:-1]
                    f["values"] = r"\b(?:" + inner + r")\b"
            # Anything else -> leave as-is (scalar equality case)
    return research_questions


def _fold_string(s: Any) -> str:
    """lower+strip+remove accents (so 'In-Patient' == 'In-Patient')."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = _ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.casefold().strip()


FilterSpec: TypeAlias = Dict[str, Any]


def apply_filters(df: pd.DataFrame, filters: List[FilterSpec]) -> pd.DataFrame:
    """
    Apply list of filters. Supports:
      - {"column": "...", "values": [..]}        -> isin (case/diacritic-insensitive for strings)
      - {"column": "...", "values": "regex"}     -> regex contains
      - {"column": "...", "regex": "regex"}      -> regex contains
      - {"column": "...", "values": scalar}      -> equality
      Optional: {"case_insensitive": True/False} (default True for regex).
    """
    mask = pd.Series(True, index=df.index)

    for f in filters:
        col = f.get("column")
        if not col or col not in df.columns:
            print(
                f"[WARN] Filter column '{col}' not in DataFrame. Skipping this filter.")
            continue

        # defaults
        ci = f.get("case_insensitive", True)

        if "values" in f:
            vals = f["values"]

            # list-like -> isin (fold strings if possible)
            if isinstance(vals, (list, tuple, set, pd.Series, np.ndarray)):
                if all(isinstance(v, str) for v in vals):
                    want = {_fold_string(v) for v in vals}
                    have = df[col].map(_fold_string)
                    mask &= have.isin(want)
                else:
                    mask &= df[col].isin(vals)

            # string -> treat as regex
            elif isinstance(vals, str):
                flags = re.IGNORECASE if ci else 0
                pattern = re.compile(vals, flags)
                mask &= df[col].astype(str).str.contains(pattern, na=False)

            # scalar -> equality
            else:
                mask &= (df[col] == vals)

        elif "regex" in f and f["regex"] is not None:
            flags = re.IGNORECASE if ci else 0
            pattern = f["regex"]
            if isinstance(pattern, str):
                pattern = re.compile(pattern, flags)
            mask &= df[col].astype(str).str.contains(pattern, na=False)

        else:
            print(
                f"[WARN] Filter spec has neither 'values' nor 'regex': {f}. Skipping.")

    return df.loc[mask].copy()
