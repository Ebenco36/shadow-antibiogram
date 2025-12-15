from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

import pandas as pd

from src.mappers.top_pathogens import ALL_PATHOGENS
from src.utils.LoadClasses import LoadClasses


# -------- helpers --------
_UNNAMED_RE = re.compile(r"^Unnamed(?::\s*\d+)?$")

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if _UNNAMED_RE.match(str(c))]
    return df.drop(columns=cols) if cols else df

def read_any(path: Union[str, Path]) -> pd.DataFrame:
    """
    Auto-detect reader:
      - Parquet directory  -> pd.read_parquet(dir, engine="pyarrow")
      - .parquet           -> pd.read_parquet(file, engine="pyarrow")
      - .feather/.ft       -> pd.read_feather(file)
      - otherwise          -> pd.read_csv(file)
    """
    p = Path(path)
    if p.is_dir():
        return _drop_unnamed(pd.read_parquet(p, engine="pyarrow"))
    suf = p.suffix.lower()
    if suf == ".parquet":
        return _drop_unnamed(pd.read_parquet(p, engine="pyarrow"))
    if suf in (".feather", ".ft"):
        return _drop_unnamed(pd.read_feather(p))
    # CSV fallback
    try:
        # pandas >= 2.0: lower memory via arrow dtypes if available
        return _drop_unnamed(pd.read_csv(p, low_memory=False, dtype_backend="pyarrow"))
    except TypeError:
        return _drop_unnamed(pd.read_csv(p, low_memory=False))


# -------- main class --------
# -------- main class --------
class DataLoader:
    def __init__(self, filepath: str, pathogen_groups_regex: Sequence[str] | str = ()):
        self.filepath = filepath
        self.load = LoadClasses()

        # Read any format (parquet dir/file, feather, csv)
        self.df = read_any(filepath)

        # Optional normalization if column exists
        if "ARS_WardType" in self.df.columns:
            self.df = self.df.assign(
                ARS_WardType=self.df["ARS_WardType"].replace({
                    "Early Rehabilitation": "Rehabilitation",
                    "Rehabilitation": "Rehabilitation",
                })
            )

        # Ensure Pathogen column exists
        if "Pathogen" not in self.df.columns:
            raise ValueError("Expected 'Pathogen' column not found in the input data.")

        # Build regex from list/tuple or use provided string
        if pathogen_groups_regex:
            if isinstance(pathogen_groups_regex, (list, tuple, set)):
                pattern = "|".join(f"(?:{p})" for p in pathogen_groups_regex if p)
            else:
                pattern = str(pathogen_groups_regex)
        else:
            pattern = ALL_PATHOGENS

        # Filter pathogens of interest
        self.df = self.df[self.df["Pathogen"].astype("string")
                          .str.contains(pattern, case=False, na=False, regex=True)]

        # ---- Artifact cleanup (defensive) ----
        artifact_cols = [c for c in self.df.columns if c.endswith("_Tested_Outcome")]
        if artifact_cols:
            # These should never exist; drop if present
            self.df = self.df.drop(columns=artifact_cols, errors="ignore")

        # ---- Column bookkeeping ----
        self.all_cols = self.df.columns.to_list()

        # core suffixes we support
        self.TESTED_SUFFIX = "_Tested"
        self.OUTCOME_SUFFIX = "_Outcome"

        # All tested/outcome columns present
        self.abx_tested_cols = sorted([c for c in self.all_cols if c.endswith(self.TESTED_SUFFIX)])
        self.abx_outcome_cols = sorted([c for c in self.all_cols if c.endswith(self.OUTCOME_SUFFIX) and " - " in c])

        # Antibiotic "base names" (e.g., "CIP - Ciprofloxacin")
        self.tested_bases  = [c[: -len(self.TESTED_SUFFIX)] for c in self.abx_tested_cols]
        self.outcome_bases = [c[: -len(self.OUTCOME_SUFFIX)] for c in self.abx_outcome_cols]

        # Pairs that exist for both tested & outcome
        self.paired_bases = sorted(set(self.tested_bases).intersection(self.outcome_bases))

        # Meta columns = everything else
        self.meta_cols = [c for c in self.all_cols if c not in (self.abx_tested_cols + self.abx_outcome_cols)]

        # Optional sanity check (default on): ensure outcome <NA> whenever tested == 0
        self._sanity_check_outcomes()

    # ---------- internals ----------
    def _present(self, cols: List[str]) -> List[str]:
        """Return only the columns that exist in dataframe."""
        return [c for c in cols if c in self.df.columns]

    def _tested_col(self, base: str) -> str:
        return f"{base}{self.TESTED_SUFFIX}"

    def _outcome_col(self, base: str) -> str:
        return f"{base}{self.OUTCOME_SUFFIX}"

    def _bases_to_cols(self, bases: List[str], return_which: str = "tested") -> List[str]:
        """
        Map a list of antibiotic base names to specific columns.
          return_which: "tested" | "outcome" | "both"
        """
        if return_which == "tested":
            return self._present([self._tested_col(b) for b in bases])
        elif return_which == "outcome":
            return self._present([self._outcome_col(b) for b in bases])
        elif return_which == "both":
            cols = []
            for b in bases:
                t = self._tested_col(b)
                y = self._outcome_col(b)
                if t in self.df.columns: cols.append(t)
                if y in self.df.columns: cols.append(y)
            return cols
        else:
            raise ValueError("return_which must be 'tested', 'outcome', or 'both'")

    def _sanity_check_outcomes(self, raise_on_violation: bool = False):
        """
        Enforce: if Tested == 0 then Outcome must be NaN.
        If raise_on_violation=False, will fix by setting those outcomes to NaN.
        """
        # Only check pairs that exist
        for base in self.paired_bases:
            tcol = self._tested_col(base)
            ycol = self._outcome_col(base)
            # guard types
            t = self.df[tcol].astype("Int8").fillna(0)
            y = self.df[ycol]
            bad_mask = (t == 0) & y.notna()
            if bad_mask.any():
                if raise_on_violation:
                    n_bad = int(bad_mask.sum())
                    raise AssertionError(f"[{base}] Found {n_bad} rows where {tcol}==0 but {ycol} is not NaN.")
                # fix in-place
                self.df.loc[bad_mask, ycol] = pd.NA

    # ---------- public API ----------
    def load_abx_classes(self):
        return self.load.antibiotic_class_list

    def _get_bases_by_categories(self, categories: List[str], include_not_set: bool = False) -> List[str]:
        """
        Use your LoadClasses to fetch *base* antibiotic names by AWaRe categories.
        We then intersect with what exists in the dataframe (by either tested or outcome).
        """
        bases_from_categories = self.load.get_antibiotics_by_category(categories)
        if include_not_set:
            bases_from_categories += self.load.get_antibiotics_by_category(["Not Set"])

        # Keep only bases that actually exist in either tested or outcome
        present_bases = set(self.tested_bases) | set(self.outcome_bases)
        bases = [b for b in bases_from_categories if b in present_bases]
        return sorted(bases)

    def _get_bases_by_class(self, classes: List[str]) -> List[str]:
        bases_from_classes = self.load.get_antibiotics_by_class(classes)
        present_bases = set(self.tested_bases) | set(self.outcome_bases)
        bases = [b for b in bases_from_classes if b in present_bases]
        return sorted(bases)

    def get_abx_by_category(self, categories: List[str], *, return_which: str = "tested",
                            use_not_set: bool = False) -> List[str]:
        """
        Fetch *columns* (not bases) for antibiotics in certain AWaRe categories.
        return_which ∈ {"tested","outcome","both"} controls which columns you get.
        """
        bases = self._get_bases_by_categories(categories, include_not_set=use_not_set)
        return self._bases_to_cols(bases, return_which=return_which)

    def get_abx_by_class(self, classes: List[str], *, return_which: str = "tested") -> List[str]:
        """
        Fetch *columns* (not bases) for antibiotics in certain drug classes.
        return_which ∈ {"tested","outcome","both"}.
        """
        bases = self._get_bases_by_class(classes)
        return self._bases_to_cols(bases, return_which=return_which)

    def get_combined(self, *, return_which: str = "tested", use_not_set: bool = False) -> pd.DataFrame:
        """
        Return dataframe with metadata + selected antibiotic columns.

        Parameters
        ----------
        return_which : {"tested","outcome","both"}, default="tested"
            Which side to include.
        use_not_set : bool, default=False
            If True, include antibiotics categorized as "Not Set".
        """
        access  = self.get_abx_by_category(["Access"],  return_which=return_which, use_not_set=False)
        watch   = self.get_abx_by_category(["Watch"],   return_which=return_which, use_not_set=False)
        reserve = self.get_abx_by_category(["Reserve"], return_which=return_which, use_not_set=False)

        abx_selected = access + watch + reserve
        if use_not_set:
            abx_selected += self.get_abx_by_category(["Not Set"], return_which=return_which, use_not_set=True)

        return self.df[self._present(self.meta_cols + abx_selected)]
