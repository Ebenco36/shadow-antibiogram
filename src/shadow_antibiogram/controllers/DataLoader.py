from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union, Optional, Iterable

import pandas as pd

from shadow_antibiogram.mappers.top_pathogens import ALL_PATHOGENS
from shadow_antibiogram.utils.LoadClasses import LoadClasses


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
        return _drop_unnamed(pd.read_csv(p, low_memory=False, dtype_backend="pyarrow"))
    except TypeError:
        return _drop_unnamed(pd.read_csv(p, low_memory=False))


class DataLoader:
    """
    Backwards-compatible loader supporting TWO dataset layouts:

    1) WIDE (old isolate-level):
        - Antibiotics are columns ending with _Tested / _Outcome

    2) PAIRWISE (new aggregated pairs):
        - Columns: ab_1, ab_2, a, b, c, d
        - Antibiotics live as values inside ab_1/ab_2

    This keeps your old API intact.
    In PAIRWISE mode:
        - get_abx_by_* returns antibiotic VALUE labels (e.g. "..._Tested")
        - get_combined returns the filtered PAIRWISE table (meta + core)
        - get_pairwise returns the raw/filtered PAIRWISE table
    """

    TESTED_SUFFIX = "_Tested"
    OUTCOME_SUFFIX = "_Outcome"
    PAIRWISE_CORE = {"ab_1", "ab_2", "a", "b", "c", "d"}

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
        self.df = self.df[
            self.df["Pathogen"].astype("string").str.contains(
                pattern, case=False, na=False, regex=True
            )
        ]

        # ---- Artifact cleanup (defensive) ----
        artifact_cols = [c for c in self.df.columns if str(c).endswith("_Tested_Outcome")]
        if artifact_cols:
            self.df = self.df.drop(columns=artifact_cols, errors="ignore")

        # ---- Column bookkeeping ----
        self.all_cols = self.df.columns.to_list()

        # ---- Detect format ----
        self.is_pairwise: bool = self.PAIRWISE_CORE.issubset(set(self.all_cols))

        # ---- Init branch ----
        if self.is_pairwise:
            self._init_pairwise()
        else:
            self._init_wide()

    # ----------------------------
    # Init branches
    # ----------------------------

    def _init_wide(self) -> None:
        """Original behavior (wide isolate-level)."""

        # Antibiotics as columns
        self.abx_tested_cols = sorted([c for c in self.all_cols if str(c).endswith(self.TESTED_SUFFIX)])
        self.abx_outcome_cols = sorted(
            [c for c in self.all_cols if str(c).endswith(self.OUTCOME_SUFFIX) and " - " in str(c)]
        )

        # Base names
        self.tested_bases = [c[: -len(self.TESTED_SUFFIX)] for c in self.abx_tested_cols]
        self.outcome_bases = [c[: -len(self.OUTCOME_SUFFIX)] for c in self.abx_outcome_cols]

        # Paired bases (tested + outcome exist)
        self.paired_bases = sorted(set(self.tested_bases).intersection(self.outcome_bases))

        # For unified API: “tested antibiotics identifiers”
        self.abx_tested = list(self.abx_tested_cols)  # identifiers are column names in wide mode

        # Meta columns
        self.meta_cols = [c for c in self.all_cols if c not in (self.abx_tested_cols + self.abx_outcome_cols)]

        # Sanity check outcomes
        self._sanity_check_outcomes()

    def _init_pairwise(self) -> None:
        """New behavior for aggregated pairwise data."""

        # No antibiotic columns exist
        self.abx_tested_cols = []
        self.abx_outcome_cols = []
        self.outcome_bases = []
        self.paired_bases = []

        # Antibiotics are values in ab_1/ab_2
        ab1 = self.df["ab_1"].astype("string") if "ab_1" in self.df.columns else pd.Series([], dtype="string")
        ab2 = self.df["ab_2"].astype("string") if "ab_2" in self.df.columns else pd.Series([], dtype="string")
        self.abx_tested_values = sorted(set(ab1.dropna().astype(str)).union(set(ab2.dropna().astype(str))))

        # Base names (strip suffix)
        self.tested_bases = sorted({v.replace(self.TESTED_SUFFIX, "") for v in self.abx_tested_values})

        # For unified API: “tested antibiotics identifiers”
        self.abx_tested = list(self.abx_tested_values)  # identifiers are values in pairwise mode

        # Meta columns = everything except pairwise core
        self.meta_cols = [c for c in self.all_cols if c not in self.PAIRWISE_CORE]

    # ----------------------------
    # internal helpers
    # ----------------------------

    def _present(self, cols: List[str]) -> List[str]:
        return [c for c in cols if c in self.df.columns]

    def _tested_label(self, base: str) -> str:
        return f"{base}{self.TESTED_SUFFIX}"

    def _outcome_label(self, base: str) -> str:
        return f"{base}{self.OUTCOME_SUFFIX}"

    def _sanity_check_outcomes(self, raise_on_violation: bool = False) -> None:
        """
        Enforce: if Tested == 0 then Outcome must be NaN.
        Only relevant for WIDE datasets with outcomes.
        """
        if self.is_pairwise:
            return

        for base in self.paired_bases:
            tcol = self._tested_label(base)
            ycol = self._outcome_label(base)
            if tcol not in self.df.columns or ycol not in self.df.columns:
                continue
            t = self.df[tcol].astype("Int8").fillna(0)
            y = self.df[ycol]
            bad_mask = (t == 0) & y.notna()
            if bad_mask.any():
                if raise_on_violation:
                    n_bad = int(bad_mask.sum())
                    raise AssertionError(f"[{base}] Found {n_bad} rows where {tcol}==0 but {ycol} is not NaN.")
                self.df.loc[bad_mask, ycol] = pd.NA

    # ----------------------------
    # public API (same signatures)
    # ----------------------------

    def load_abx_classes(self):
        return self.load.antibiotic_class_list

    def _get_bases_by_categories(self, categories: List[str], include_not_set: bool = False) -> List[str]:
        bases = self.load.get_antibiotics_by_category(categories)
        if include_not_set:
            bases += self.load.get_antibiotics_by_category(["Not Set"])

        present = set(self.tested_bases) | set(getattr(self, "outcome_bases", []))
        return sorted([b for b in bases if b in present])

    def _get_bases_by_class(self, classes: List[str]) -> List[str]:
        bases = self.load.get_antibiotics_by_class(classes)
        present = set(self.tested_bases) | set(getattr(self, "outcome_bases", []))
        return sorted([b for b in bases if b in present])

    def get_abx_by_category(
        self,
        categories: List[str],
        *,
        return_which: str = "tested",
        use_not_set: bool = False,
    ) -> List[str]:
        """
        WIDE:
          return_which: tested/outcome/both -> returns column names

        PAIRWISE:
          return_which must be 'tested' -> returns antibiotic value labels
          (e.g. "AMC - ..._Tested")
        """
        bases = self._get_bases_by_categories(categories, include_not_set=use_not_set)

        if self.is_pairwise:
            if return_which != "tested":
                raise ValueError("PAIRWISE format supports return_which='tested' only (no outcomes in this schema).")
            wanted = [self._tested_label(b) for b in bases]
            present = set(self.abx_tested_values)
            return [w for w in wanted if w in present]

        # WIDE behavior
        if return_which == "tested":
            return self._present([self._tested_label(b) for b in bases])
        if return_which == "outcome":
            return self._present([self._outcome_label(b) for b in bases])
        if return_which == "both":
            cols: List[str] = []
            for b in bases:
                t = self._tested_label(b)
                y = self._outcome_label(b)
                if t in self.df.columns:
                    cols.append(t)
                if y in self.df.columns:
                    cols.append(y)
            return cols
        raise ValueError("return_which must be 'tested', 'outcome', or 'both'")

    def get_abx_by_class(self, classes: List[str], *, return_which: str = "tested") -> List[str]:
        bases = self._get_bases_by_class(classes)

        if self.is_pairwise:
            if return_which != "tested":
                raise ValueError("PAIRWISE format supports return_which='tested' only (no outcomes in this schema).")
            wanted = [self._tested_label(b) for b in bases]
            present = set(self.abx_tested_values)
            return [w for w in wanted if w in present]

        # WIDE behavior
        if return_which == "tested":
            return self._present([self._tested_label(b) for b in bases])
        if return_which == "outcome":
            return self._present([self._outcome_label(b) for b in bases])
        if return_which == "both":
            cols: List[str] = []
            for b in bases:
                t = self._tested_label(b)
                y = self._outcome_label(b)
                if t in self.df.columns:
                    cols.append(t)
                if y in self.df.columns:
                    cols.append(y)
            return cols
        raise ValueError("return_which must be 'tested', 'outcome', or 'both'")

    def get_combined(self, *, return_which: str = "tested", use_not_set: bool = False) -> pd.DataFrame:
        """
        WIDE:
          returns meta + antibiotic columns

        PAIRWISE:
          returns meta + (ab_1,ab_2,a,b,c,d) filtered to selected antibiotics
        """
        access = self.get_abx_by_category(["Access"], return_which=return_which, use_not_set=False)
        watch = self.get_abx_by_category(["Watch"], return_which=return_which, use_not_set=False)
        reserve = self.get_abx_by_category(["Reserve"], return_which=return_which, use_not_set=False)

        abx_selected = access + watch + reserve
        if use_not_set:
            abx_selected += self.get_abx_by_category(["Not Set"], return_which=return_which, use_not_set=True)

        if not self.is_pairwise:
            # original wide return
            return self.df[self._present(self.meta_cols + abx_selected)]

        # pairwise return
        cols = self._present(self.meta_cols + ["ab_1", "ab_2", "a", "b", "c", "d"])
        out = self.df[cols].copy()

        if abx_selected:
            s = set(abx_selected)
            out = out[out["ab_1"].astype(str).isin(s) & out["ab_2"].astype(str).isin(s)]
        return out

    def get_pairwise(self, *, filter_abx: Optional[List[str]] = None) -> pd.DataFrame:
        """
        PAIRWISE-only convenience accessor.
        filter_abx expects antibiotic labels like "..._Tested"
        """
        if not self.is_pairwise:
            raise ValueError("This dataset is not pairwise (missing ab_1/ab_2/a/b/c/d).")

        df = self.df.copy()
        if filter_abx is not None:
            s = set(filter_abx)
            df = df[df["ab_1"].astype(str).isin(s) & df["ab_2"].astype(str).isin(s)]
        return df

    # ----------------------------
    # extra helpers (optional, nice)
    # ----------------------------

    @property
    def antibiotic_identifiers(self) -> List[str]:
        """
        Unified: returns antibiotic identifiers for this dataset.
        - WIDE: column names "..._Tested"
        - PAIRWISE: values in ab_1/ab_2 "..._Tested"
        """
        return list(self.abx_tested)
