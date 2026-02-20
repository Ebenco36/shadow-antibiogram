# ===================== Interfaces =====================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union
import pandas as pd

from shadow_antibiogram.utils.helpers import _ci_rename

class AntibioticNameParser(ABC):
    @abstractmethod
    def parse(self, column: str) -> Tuple[str, str]:
        ...

class AntibioticDetector(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> List[str]:
        ...

class ThresholdPolicy(ABC):
    @abstractmethod
    def filter(self, mat: pd.DataFrame) -> pd.DataFrame:
        ...

class PopulationProvider(ABC):
    @abstractmethod
    def get_population(self, key_values: pd.DataFrame) -> pd.Series:
        ...

class MetricStrategy(ABC):
    @abstractmethod
    def compute(self,
                grouped: pd.core.groupby.generic.DataFrameGroupBy,
                antibiotics: Sequence[str],
                df_all: pd.DataFrame) -> pd.DataFrame:
        ...
        
        
# ================= Default implementations ==============
@dataclass(frozen=True)
class DefaultAntibioticNameParser(AntibioticNameParser):
    suffix: str = "_Tested"

    def parse(self, column: str) -> Tuple[str, str]:
        core = column.replace(self.suffix, "")
        if " - " in core:
            abbr, longn = core.split(" - ", 1)
        else:
            abbr, longn = core, core
        return longn, abbr

@dataclass(frozen=True)
class SuffixAntibioticDetector(AntibioticDetector):
    suffix: str = "_Tested"

    def detect(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.endswith(self.suffix)]

@dataclass
class PctThresholdPolicy(ThresholdPolicy):
    min_threshold: float = 0.0
    scope: str = "any"        # any | all | global
    global_vector: Optional[pd.Series] = None

    def filter(self, mat: pd.DataFrame) -> pd.DataFrame:
        if self.min_threshold <= 0:
            return mat
        if self.scope == "any":
            keep = (mat >= self.min_threshold).any(axis=0)
        elif self.scope == "all":
            keep = (mat >= self.min_threshold).all(axis=0)
        elif self.scope == "global":
            if self.global_vector is None:
                raise ValueError("global_vector must be set for 'global' scope")
            keep = self.global_vector >= self.min_threshold
        else:
            raise ValueError("scope must be 'any'|'all'|'global'")
        return mat.loc[:, keep]

@dataclass
class BundeslandPopulationProvider(PopulationProvider):
    pop_df: pd.DataFrame
    bundesland_col: str = "Bundesland"
    year_col: str = "Year"
    pop_col: str = "total"
    agg: str = "mean"
    use_year: Union[bool, Literal['auto']] = 'auto'
    year_filter: Optional[Iterable[int]] = None

    # DEBUG STASHES
    preagg_source_frame: pd.DataFrame = field(init=False, default=None, repr=False)
    postmerge_frame:   pd.DataFrame = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.pop_df = _ci_rename(
            self.pop_df, {"bundesland": self.bundesland_col,
                          "year": self.year_col,
                          "total": self.pop_col}
        )

    def _decide_use_year(self, keys: pd.DataFrame) -> bool:
        if self.use_year in (True, False):
            return self.use_year
        return (self.year_col in keys.columns) and (self.year_col in self.pop_df.columns)

    def get_population(self,
                       key_values: pd.DataFrame,
                       *,
                       years_hint: Optional[Iterable[int]] = None) -> pd.Series:
        key_values = _ci_rename(key_values,
                                {"bundesland": self.bundesland_col,
                                 "year": self.year_col})

        if self.bundesland_col not in key_values.columns:
            raise KeyError(f"'{self.bundesland_col}' missing in key_values: {list(key_values.columns)}")

        use_year = self._decide_use_year(key_values)

        # Decide year subset ONLY if we won't use year explicitly
        yr_subset = None
        if not use_year and self.year_col in self.pop_df.columns:
            if self.year_filter is not None:
                yr_subset = set(self.year_filter)
            elif years_hint is not None:
                yr_subset = set(years_hint)

        pop_df = self.pop_df
        if yr_subset is not None:
            pop_df = pop_df[pop_df[self.year_col].isin(yr_subset)]

        if use_year:
            # nothing aggregated – just merge
            cols = [self.bundesland_col, self.year_col, self.pop_col]
            self.preagg_source_frame = pop_df[cols].copy()  # <-- SAVED BEFORE ANYTHING
            merged = key_values.merge(self.preagg_source_frame.rename(columns={self.pop_col: "pop"}),
                                      on=[self.bundesland_col, self.year_col],
                                      how="left")
            out = merged["pop"]
            self.postmerge_frame = merged  # <-- after merge
        else:
            # SAVE *before* we aggregate
            cols = [self.bundesland_col, self.year_col, self.pop_col] if self.year_col in pop_df.columns else [self.bundesland_col, self.pop_col]
            self.preagg_source_frame = pop_df[cols].copy()  # <-- SAVED

            pop_agg = (pop_df
                       .groupby(self.bundesland_col, as_index=False)[self.pop_col]
                       .agg(self.agg)
                       .rename(columns={self.pop_col: "pop"}))

            merged = key_values.merge(pop_agg, on=self.bundesland_col, how="left")
            out = merged["pop"]
            self.postmerge_frame = merged  # <-- after merge

        # Fallback if NA
        if out.isna().any():
            fb = (pop_df.groupby(self.bundesland_col)[self.pop_col]
                        .agg(self.agg).rename("fallback_pop"))
            fb_merge = key_values.merge(fb, on=self.bundesland_col, how="left")
            out = out.fillna(fb_merge["fallback_pop"])
            # keep fallback info too
            self.postmerge_frame = self.postmerge_frame.join(
                fb_merge["fallback_pop"], how="left"
            )
        
        return out



# @dataclass
# class BundeslandPopulationProvider(PopulationProvider):
#     pop_df: pd.DataFrame
#     bundesland_col: str = "Bundesland"
#     year_col: str = "Year"
#     pop_col: str = "total"
#     agg: Literal["mean", "median", "sum"] = "mean"
#     use_year: Literal["auto", True, False] = "auto"
#     year_filter: Optional[Iterable[int]] = None   # <--- NEW
#     _norm_done: bool = field(init=False, default=False)

#     def __post_init__(self):
#         self._normalize_cols()

#     def _normalize_cols(self):
#         if self._norm_done:
#             return
#         self.pop_df = _ci_rename(self.pop_df,
#                                  {"bundesland": self.bundesland_col,
#                                   "year": self.year_col,
#                                   "total": self.pop_col})
#         self._norm_done = True

#     def _decide_use_year(self, keys: pd.DataFrame) -> bool:
#         if self.use_year in (True, False):
#             return self.use_year
#         return (self.year_col in keys.columns) and (self.year_col in self.pop_df.columns)

#     def get_population(self,
#                        key_values: pd.DataFrame,
#                        *,
#                        years_hint: Optional[Iterable[int]] = None) -> pd.Series:
#         key_values = _ci_rename(key_values,
#                                 {"bundesland": self.bundesland_col,
#                                  "year": self.year_col})

#         if self.bundesland_col not in key_values.columns:
#             raise KeyError(f"'{self.bundesland_col}' missing in key_values: {list(key_values.columns)}")

#         use_year = self._decide_use_year(key_values)
#         # decide which years we’re allowed to use when NOT using year
#         yr_subset: Optional[pd.Series] = None
#         if not use_year:
#             # choose: explicit filter > hint from caller > all years
#             if self.year_filter is not None:
#                 yr_subset = pd.Series(self.year_filter)
#             elif years_hint is not None:
#                 yr_subset = pd.Series(list(years_hint))

#         pop_df = self.pop_df
#         if yr_subset is not None and self.year_col in pop_df.columns:
#             pop_df = pop_df[pop_df[self.year_col].isin(yr_subset)]

#         if use_year:
#             cols = [self.bundesland_col, self.year_col, self.pop_col]
#             merged = key_values.merge(pop_df[cols].rename(columns={self.pop_col: "pop"}),
#                                       on=[self.bundesland_col, self.year_col],
#                                       how="left")
#             out = merged["pop"]
#         else:
#             pop_agg = (pop_df.groupby(self.bundesland_col, as_index=False)[self.pop_col]
#                              .agg(self.agg)
#                              .rename(columns={self.pop_col: "pop"}))
#             out = key_values.merge(pop_agg, on=self.bundesland_col, how="left")["pop"]

#         # fallback to overall agg per Bundesland if any NA
#         if out.isna().any():
#             fb = (pop_df.groupby(self.bundesland_col)[self.pop_col]
#                         .agg(self.agg).rename("fallback_pop"))
#             out = out.fillna(key_values.merge(fb, on=self.bundesland_col, how="left")["fallback_pop"])
#         return out


# -------- Metric strategies --------
@dataclass(frozen=True)
class PctMetric(MetricStrategy):
    def compute(self, grouped, antibiotics, df_all) -> pd.DataFrame:
        return grouped[antibiotics].mean(numeric_only=True) * 100
    
    def legend_title(self) -> str:
        return "Percent tested (%)"

@dataclass(frozen=True)
class MeanMetric(MetricStrategy):
    def compute(self, grouped, antibiotics, df_all) -> pd.DataFrame:
        return grouped[antibiotics].mean(numeric_only=True)
    
    def legend_title(self) -> str:
        return "Mean Test Covarage"

@dataclass(frozen=True)
class CountMetric(MetricStrategy):
    def compute(self, grouped, antibiotics, df_all) -> pd.DataFrame:
        return grouped[antibiotics].sum(numeric_only=True)
    
    def legend_title(self) -> str:
        return "Total Test Coverage"


@dataclass
class Per100kMetric(MetricStrategy):
    population_provider: PopulationProvider
    scale: float = 1e5
    year_range: Optional[Tuple[int, int]] = None

    def compute(self, grouped, antibiotics, df_all) -> pd.DataFrame:
        group_cols = list(grouped.keys)  # avoids the deprecated .grouper
        counts = grouped[antibiotics].sum(numeric_only=True).reset_index()
        
        # If "Year" isn't in the group keys, we may want to hint a range
        years_hint = None
        if "Year" not in group_cols and self.year_range is not None and "Year" in df_all.columns:
            lo, hi = self.year_range
            years_hint = range(lo, hi + 1)

        pop = self.population_provider.get_population(counts[group_cols], years_hint=years_hint)

        counts["__pop__"] = pop
        for a in antibiotics:
            # if a == "SXT - Co-Trimoxazol_Tested":
            #     print(counts[a], counts["__pop__"])
            counts[a] = counts[a] / counts["__pop__"] * self.scale
        return counts.set_index(group_cols)[antibiotics]
        # ---- compute all rates at once and SAVE them
        # counts["__pop__"] = pop
        # rates = counts[antibiotics].div(counts["__pop__"], axis=0) * self.scale
        # self.saved_rates = pd.concat([counts[group_cols], rates], axis=1)
        # pd.concat([counts[group_cols], counts], axis=1).to_csv("bbbbbbbbbbb.csv")
        # # return in the expected wide shape
        # return self.saved_rates.set_index(group_cols)[antibiotics]
    

    def legend_title(self) -> str:
        return f"Tests per {self._humanize(self.scale)}"

    @staticmethod
    def _humanize(n: float) -> str:
        if n >= 1e9: return f"{n/1e9:g}B"
        if n >= 1e6: return f"{n/1e6:g}M"
        if n >= 1e3: return f"{n/1e3:g}k"
        return f"{n:g}"


# @dataclass
# class Per100kMetric(MetricStrategy):
#     population_provider: PopulationProvider
#     scale: float = 1e5

#     def compute(self, grouped, antibiotics, df_all) -> pd.DataFrame:
#         counts = grouped[antibiotics].sum(numeric_only=True).reset_index()
#         pop = self.population_provider.get_population(counts[grouped.keys])
#         counts = counts.copy()
#         counts["__pop__"] = pop
#         for a in antibiotics:
#             counts[a] = counts[a] / counts["__pop__"] * self.scale
#         return counts.set_index(grouped.keys)[antibiotics]

#     def legend_title(self) -> str:
#         return f"Tests per {self.humanize_scale(self.scale)} population"
    
    
#     # ------------- tiny helper --------------
#     def humanize_scale(self, n: float) -> str:
#         """
#         1e3 -> '1k', 1e5 -> '100k', 1e6 -> '1M', 2.5e6 -> '2.5M', etc.
#         """
#         if n >= 1e9:
#             return f"{n/1e9:g}B"
#         if n >= 1e6:
#             return f"{n/1e6:g}M"
#         if n >= 1e3:
#             return f"{n/1e3:g}k"
#         return f"{n:g}"
