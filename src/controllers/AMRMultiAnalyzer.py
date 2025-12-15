# amr_oop.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
                    Union, ClassVar)
from pathlib import Path
import warnings
import pandas as pd

# local imports
from src.controllers.viz.PlotlyVizMixin import PlotlyVizMixin
from src.controllers.interfaces.data import (
    AntibioticDetector, AntibioticNameParser,
    CountMetric, MeanMetric, MetricStrategy,
    PctMetric, PctThresholdPolicy, Per100kMetric,
    PopulationProvider, ThresholdPolicy
)
from src.utils.helpers import save_chart

# --- type alias -------------------------------------------------------
PrefilterT = Callable[[pd.DataFrame], pd.DataFrame]


# ===================== Facade =====================
@dataclass
class AMRMultiAnalyzer(PlotlyVizMixin):
    df: pd.DataFrame
    antibiotic_detector: AntibioticDetector
    name_parser: AntibioticNameParser
    antibiotic_suffix: str = "_Tested"
    population_provider: Optional[PopulationProvider] = None
    charts_dir: Optional[Path] = None

    # NEW: prefilter hook(s)
    prefilter: Optional[Union[PrefilterT, Sequence[PrefilterT]]] = None

    # caches / internals
    _geojson_cache: ClassVar[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = {}
    _raw_df: pd.DataFrame = field(init=False, repr=False)
    _last_prefilter_error: Optional[Exception] = field(init=False, default=None, repr=False)

    # -------------------------------------------------------------
    # lifecycle
    def __post_init__(self):
        self._raw_df = self.df.copy()
        self.df = self._apply_prefilter(self._raw_df, self.prefilter)

    # -------------------------------------------------------------
    # public helpers
    def get_antibiotics(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        data = self._resolve_df(df)
        return self.antibiotic_detector.detect(data)

    def set_prefilter(self,
                      prefilter: Optional[Union[PrefilterT, Sequence[PrefilterT]]],
                      *,
                      apply_now: bool = True):
        self.prefilter = prefilter
        if apply_now:
            self.df = self._apply_prefilter(self._raw_df, prefilter)

    def reapply_prefilter(self):
        self.df = self._apply_prefilter(self._raw_df, self.prefilter)

    def get_data(self,
                 *,
                 prefilter: Optional[Union[PrefilterT, Sequence[PrefilterT]]] = None,
                 copy: bool = True) -> pd.DataFrame:
        base = self._raw_df if prefilter is not None else self.df
        base = base.copy() if copy else base
        return self._apply_prefilter(base, prefilter) if prefilter is not None else base

    # -------------------------------------------------------------
    # internal
    def _apply_prefilter(self, df: pd.DataFrame,
                         prefilter: Optional[Union[PrefilterT, Sequence[PrefilterT]]]) -> pd.DataFrame:
        if prefilter is None:
            return df

        funcs: List[PrefilterT]
        if callable(prefilter):
            funcs = [prefilter]
        else:
            funcs = list(prefilter)

        for i, fn in enumerate(funcs, 1):
            try:
                out = fn(df)
                if not isinstance(out, pd.DataFrame):
                    raise TypeError(f"Prefilter #{i} returned {type(out)}, expected DataFrame")
                df = out
            except Exception as e:
                self._last_prefilter_error = e
                warnings.warn(f"[prefilter #{i}] skipped due to error: {e}", RuntimeWarning)
        return df

    def _resolve_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Use provided df or fall back to the analyzer's current df."""
        return df if df is not None else self.df

    
    def compute_metric(
        self,
        groupby: Union[str, Iterable[str]],
        metric_strategy: MetricStrategy,
        *,
        antibiotics: Optional[Iterable[str]] = None,
        threshold_policy: Optional[ThresholdPolicy] = None,
        drop_zero: bool = True,
        return_wide: bool = False,
        observed: bool = False,
        df: Optional[pd.DataFrame] = None,
        include_components: bool = True,
        num_label: str = "Numerator",
        den_label: str = "Denominator"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        data = self._resolve_df(df)

        if isinstance(groupby, str):
            groupby = [groupby]

        if antibiotics is None:
            antibiotics = self.get_antibiotics(data)
        antibiotics = list(antibiotics)

        g = data.groupby(groupby, observed=observed)

        # -------- primary metric --------
        mat = metric_strategy.compute(g, antibiotics, data)

        # -------- optional components (num/den) --------
        num_df = den_df = None
        if include_components:
            if isinstance(metric_strategy, PctMetric):
                num_df = g[antibiotics].sum(numeric_only=True)
                den_s  = g.size()
                den_df = pd.DataFrame(
                    {a: den_s for a in antibiotics}
                )  # broadcast same denom to all abs
            elif isinstance(metric_strategy, CountMetric):
                num_df = g[antibiotics].sum(numeric_only=True)
                den_df = None   # count already a numerator; leave denom empty or group size
            elif isinstance(metric_strategy, MeanMetric):
                num_df = g[antibiotics].sum(numeric_only=True)
                den_df = g[antibiotics].count()
            elif isinstance(metric_strategy, Per100kMetric):
                num_df = g[antibiotics].sum(numeric_only=True)
                # population per group
                key_frame = num_df.reset_index()[groupby]
                pop = metric_strategy.population_provider.get_population(
                    key_frame,
                    years_hint=getattr(metric_strategy, "year_range", None) and
                            range(metric_strategy.year_range[0], metric_strategy.year_range[1]+1)
                )
                den_df = pd.DataFrame({a: pop.values for a in antibiotics},
                                    index=num_df.index)
            else:
                # fallback: can't infer
                include_components = False

        # -------- thresholding --------
        if threshold_policy is not None:
            if isinstance(threshold_policy, PctThresholdPolicy) and threshold_policy.scope == "global":
                glob = data[antibiotics].mean(numeric_only=True)
                if isinstance(metric_strategy, PctMetric):
                    glob = glob * 100
                threshold_policy.global_vector = glob
            mat = threshold_policy.filter(mat)
            if include_components and num_df is not None:
                num_df = num_df.loc[mat.index, mat.columns]
                if den_df is not None:
                    den_df = den_df.loc[mat.index, mat.columns]

        if drop_zero:
            keep = (mat != 0).any(axis=0)
            mat = mat.loc[:, keep]
            if include_components and num_df is not None:
                num_df = num_df.loc[:, keep]
                if den_df is not None:
                    den_df = den_df.loc[:, keep]

        metric_name = self._metric_name(metric_strategy)

        # -------- melt to long --------
        long_df = (
            mat.reset_index()
            .melt(id_vars=groupby, var_name="Antibiotic_col", value_name=metric_name)
        )
        if include_components and num_df is not None:
            n_long = (
                num_df.reset_index()
                    .melt(id_vars=groupby, var_name="Antibiotic_col", value_name=num_label)
            )
            long_df = long_df.merge(n_long, on=groupby+["Antibiotic_col"], how="left")
            if den_df is not None:
                d_long = (
                    den_df.reset_index()
                        .melt(id_vars=groupby, var_name="Antibiotic_col", value_name=den_label)
                )
                long_df = long_df.merge(d_long, on=groupby+["Antibiotic_col"], how="left")

        long_df[["Antibiotic", "Abbreviation"]] = long_df["Antibiotic_col"].apply(
            lambda c: pd.Series(self.name_parser.parse(c))
        )

        cols = list(groupby) + ["Antibiotic", "Abbreviation", metric_name, num_label, den_label, "Antibiotic_col"]
        # if components were disabled or missing, drop them safely
        cols = [c for c in cols if c in long_df.columns]
        long_df = (long_df[cols]
                .sort_values(groupby + [metric_name], ascending=[True]*len(groupby) + [False]))

        if not return_wide:
            return long_df

        wide_df = (
            long_df.pivot_table(index=groupby, columns="Antibiotic", values=metric_name, aggfunc="first")
                    .astype(float)
                    .sort_index()
        )
        return long_df, wide_df


    def metric_matrix(
        self,
        row_group: Union[str, Iterable[str]],
        metric_strategy: MetricStrategy,
        *,
        antibiotics: Optional[Iterable[str]] = None,
        threshold_policy: Optional[ThresholdPolicy] = None,
        drop_zero: bool = True,
        scale_to_0_1: bool = False,
        observed: bool = False,
        df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        _, wide = self.compute_metric(
            row_group,
            metric_strategy,
            antibiotics=antibiotics,
            threshold_policy=threshold_policy,
            drop_zero=drop_zero,
            return_wide=True,
            observed=observed,
            df=df
        )

        if scale_to_0_1 and wide.max().max() > 1.0:
            wide = wide / 100.0

        if isinstance(row_group, Iterable) and not isinstance(row_group, str) and len(row_group) > 1:
            wide.index = wide.index.to_flat_index().map(lambda t: " | ".join(map(str, t)))
            wide.index.name = "Group"
        else:
            wide.index.name = row_group if isinstance(row_group, str) else row_group[0]
        wide.columns.name = "Antibiotic"
        return wide

    @staticmethod
    def _metric_name(strategy: MetricStrategy) -> str:
        if isinstance(strategy, PctMetric):     return "Pct_tested"
        if isinstance(strategy, MeanMetric):    return "Mean_tested"
        if isinstance(strategy, CountMetric):   return "Count_tested"
        if isinstance(strategy, Per100kMetric): return "Rate_per100k"
        return "Value"

    def clustergram(
        self,
        group_cols: Union[str, Iterable[str]],
        metric_strategy: MetricStrategy,
        *,
        plot_func: Callable,
        plot_kwargs: Optional[dict] = None,
        antibiotics: Optional[Iterable[str]] = None,
        threshold_policy: Optional[ThresholdPolicy] = None,
        drop_zero: bool = True,
        scale_to_0_1: bool = False,
        observed: bool = False,
        source_label: Optional[str] = None,
        target_label: str = "Antibiotic",
        df: Optional[pd.DataFrame] = None
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        # print(df)
        mat = self.metric_matrix(
            row_group=group_cols,
            metric_strategy=metric_strategy,
            antibiotics=antibiotics,
            threshold_policy=threshold_policy,
            drop_zero=drop_zero,
            scale_to_0_1=scale_to_0_1,
            observed=observed,
            df=df
        )

        if "legend_title" not in plot_kwargs or plot_kwargs["legend_title"] is None:
            plot_kwargs["legend_title"] = metric_strategy.legend_title()

        mat = mat.fillna(0).astype(float).copy()
        mat.index.name = None
        mat.columns.name = None

        if source_label is None:
            source_label = " | ".join(map(str, group_cols)) if isinstance(group_cols, (list, tuple)) else str(group_cols)

        order = mat.max(axis=0).sort_values(ascending=True).index
        mat_sorted = mat.loc[:, order]
        return plot_func(mat_sorted, source=source_label, target=target_label, **plot_kwargs)

    def save(self, chart, basename, formats=("png", "svg", "pdf", "html"), **kwargs):
        return save_chart(chart, basename, formats, **kwargs)
