from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, levene
import statsmodels.formula.api as smf


class AntibioticCoverageSummary:
    # -------------------------- INIT & LABEL SYSTEM --------------------------
    def __init__(self, df, antibiotic_suffix="_Tested", population_df=None, label_mode="name"):
        """
        label_mode: 'abbr' -> 'AMK'
                    'full' -> 'AMK - Amikacin'
                    'name' -> 'Amikacin'   (full name ONLY; no abbreviation)
        """
        self.df = df.copy()
        self.antibiotic_suffix = antibiotic_suffix
        self.abx_cols = [c for c in self.df.columns if c.endswith(antibiotic_suffix)]
        self.coverage_label = "Coverage (%)"
        self.population_df = population_df
        self.label_mode = label_mode  # 'abbr' | 'full' | 'name'

        # Parse antibiotic names and build resolvers
        self.abx_info = {}
        self.abbr_to_col = {}
        self.fullname_to_col = {}
        self.full_to_col = {}  # alias for fullname_to_col
        self.name_to_col = {}

        for col in self.abx_cols:
            core = col[: -len(antibiotic_suffix)]
            if " - " in core:
                abbr, fullname = core.split(" - ", 1)
            else:
                parts = core.split("_", 1)
                abbr = parts[0]
                fullname = parts[1] if len(parts) > 1 else parts[0]
            abbr = abbr.strip()
            fullname = fullname.strip()

            # name (requested) is *without* abbreviation
            name_only = fullname
            full_display = f"{abbr} - {fullname}"

            self.abx_info[col] = {
                "abbr": abbr,
                "name": name_only,
                "full": full_display
            }
            self.abbr_to_col[abbr] = col
            self.fullname_to_col[fullname] = col
            self.full_to_col[fullname] = col
            self.name_to_col[name_only] = col

    def set_label_mode(self, mode: str):
        if mode not in ("abbr", "full", "name"):
            raise ValueError("label_mode must be 'abbr', 'full', or 'name'")
        self.label_mode = mode

    def get_label(self, abx_col: str, mode: str = None) -> str:
        if mode is None:
            mode = self.label_mode
        info = self.abx_info.get(abx_col)
        if not info:
            return abx_col
        return info[mode]

    def resolve_to_columns(self, keys):
        """
        Accept keys as:
          - raw col names (e.g., 'AMK - Amikacin_Tested')
          - abbreviation ('AMK')
          - full drug name only ('Amikacin')
          - full display ('AMK - Amikacin')
        Return list of raw column names (unknown items passed through).
        """
        resolved = []
        for k in keys:
            if k in self.abx_cols:
                resolved.append(k)
            elif k in self.abbr_to_col:
                resolved.append(self.abbr_to_col[k])
            elif k in self.full_to_col:
                resolved.append(self.full_to_col[k])
            elif k in self.name_to_col:
                resolved.append(self.name_to_col[k])
            else:
                resolved.append(k)
        return resolved

    def _attach_labels(self, df_like: pd.DataFrame, src_col="Antibiotic"):
        """Add Abbreviation / FullName / AntibioticLabel columns based on label_mode."""
        out = df_like.copy()
        out["Abbreviation"] = out[src_col].map(lambda c: self.abx_info.get(c, {}).get("abbr", c))
        out["FullName"] = out[src_col].map(lambda c: self.abx_info.get(c, {}).get("name", c))   # only the drug name
        out["FullDisplay"] = out[src_col].map(lambda c: self.abx_info.get(c, {}).get("full", c))  # 'ABBR - Full'
        # Chosen label for plotting:
        out["AntibioticLabel"] = out[src_col].map(lambda c: self.get_label(c))
        return out

    def calculate_variation_stats(self, base_long, cmp_key, variation_mode="ci"):
        """
        Calculate variation statistics with proper confidence interval handling
        
        Parameters:
        base_long: DataFrame in long format
        cmp_key: Column name for grouping/comparison
        variation_mode: Type of variation measure
        
        Returns:
        DataFrame with mean and error bar values
        """
        
        agg = base_long.groupby(["Antibiotic", cmp_key], observed=False)["Value"]
        mean_vals = agg.mean().rename("Mean").reset_index()
        
        if variation_mode == "iqr":
            q1 = agg.quantile(0.25).rename("Q1")
            q3 = agg.quantile(0.75).rename("Q3")
            var_df = pd.concat([q1, q3], axis=1).reset_index()
            merged = mean_vals.merge(var_df, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = (merged["Mean"] - merged["Q1"]).clip(lower=0)
            merged["err_high"] = (merged["Q3"] - merged["Mean"]).clip(lower=0)
            
        elif variation_mode == "stddev":
            sd = agg.std().rename("SD").reset_index()
            merged = mean_vals.merge(sd, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = merged["SD"]
            merged["err_high"] = merged["SD"]
            
        elif variation_mode == "sem":
            sd = agg.std()
            n = agg.count()
            sem = (sd / np.sqrt(n)).rename("SEM").reset_index()
            merged = mean_vals.merge(sem, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = merged["SEM"]
            merged["err_high"] = merged["SEM"]
            
        elif variation_mode == "ci":
            # Proper 95% confidence interval calculation
            sd = agg.std()
            n = agg.count()
            
            # Handle cases where n=0 or sd is NaN
            ci95 = 1.96 * (sd / np.sqrt(n))
            ci95 = ci95.fillna(0).rename("CI95").reset_index()
            
            merged = mean_vals.merge(ci95, on=["Antibiotic", cmp_key], how="left")
            
            # For confidence intervals, both error bars should be the same value
            # representing the margin of error
            merged["err_low"] = merged["CI95"]
            merged["err_high"] = merged["CI95"]
            
        elif variation_mode == "minmax":
            mn = agg.min().rename("Min")
            mx = agg.max().rename("Max")
            var_df = pd.concat([mn, mx], axis=1).reset_index()
            merged = mean_vals.merge(var_df, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = (merged["Mean"] - merged["Min"]).clip(lower=0)
            merged["err_high"] = (merged["Max"] - merged["Mean"]).clip(lower=0)
            
        elif variation_mode == "mad":
            mad = agg.apply(lambda s: (s - s.mean()).abs().mean()).rename("MAD").reset_index()
            merged = mean_vals.merge(mad, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = merged["MAD"]
            merged["err_high"] = merged["MAD"]
            
        elif variation_mode == "gini":
            def _gini(a):
                a = np.asarray(a)
                a = a[np.isfinite(a)]
                a = a[a >= 0]
                if a.size == 0 or a.sum() == 0:
                    return 0.0
                a = np.sort(a)
                n = a.size
                idx = np.arange(1, n + 1)
                return (np.sum((2 * idx - n - 1) * a) / (n * a.sum())) * 100
            
            g = agg.apply(_gini).rename("Gini").reset_index()
            merged = mean_vals.merge(g, on=["Antibiotic", cmp_key], how="left")
            merged["err_low"] = merged["Gini"]
            merged["err_high"] = merged["Gini"]
            
        else:
            raise ValueError("variation_mode must be one of: iqr, stddev, sem, ci, minmax, mad, gini")
        
        return merged

    def plot_group_comparison(
        self,
        df_override: Optional[pd.DataFrame] = None,
        *,
        compare_col: Optional[str],                    # e.g., "TextMaterialgroupRkiL0" or None
        values: Optional[List[str]] = None,            # e.g., ["Urine", "Blood Culture"]
        group_by: Optional[str] = None,                # e.g., "Bundesland" (for error bars across strata)
        variation_mode: str = "iqr",                   # "iqr" | "stddev" | "sem" | "ci" | "minmax" | "mad" | "gini"
        min_group_size: int = 1,                       # ignore tiny strata when group_by is used
        year: Optional[int] = None,                    # optional year filter
        label_mode_override: Optional[str] = None,     # 'abbr' | 'full' | 'name'
        vertical: bool = True,
        title: Optional[str] = None,
        output_prefix: Optional[str] = None,
        # ---- sorting controls ----
        sort_mode: str = "overall_mean",               # 'overall_mean' | 'max' | 'by_group'
        sort_group_value: Optional[Union[str, int]] = None,  # required if sort_mode=='by_group'
        ascending: bool = False,
        # ---- class aggregation / filtering ----
        antibiotic_classes: Optional[Dict[str, List[str]]] = None,  # {"Beta-lactams": ["AMX - Amoxicillin_Tested", ...]}
        antibiotics_to_plot: Optional[List[str]] = None,            # accepts abbr/full/name/raw; will be resolved
        # ---- normalization options ----
        per_100k_population: bool = False,             # if True: denom=population (national or by Bundesland)
        export_csv_path: Optional[str] = None,
        # ---- legend & colors ----
        show_legend: Optional[bool] = None,            # default: False if compare_col None, else True
        bar_color: str = "#0072B2",                    # Okabe–Ito blue (single series)
        color_map: Optional[Dict[Union[str, int], str]] = None      # map compare value -> color (Okabe–Ito fallback)
    ):
        """
        Side-by-side grouped bars for each antibiotic / class comparing categories in `compare_col`.
        If `group_by` is provided, bars show mean across strata with error bars from `variation_mode`.
        If `compare_col` is None, show a single series (no legend) using `bar_color`.

        Hover shows top/bottom `group_by` (when provided) with values + denominators.
        """
        import numpy as np
        import pandas as pd
        import plotly.express as px

        df = (df_override.copy() if df_override is not None else self.df.copy())
        if year is not None and "Year" in df.columns:
            df = df[df["Year"] == year]
        if compare_col is not None and values is not None:
            df = df[df[compare_col].isin(values)]
        if df.empty:
            raise ValueError("No data after filtering (check year/filters).")

        lab_mode = label_mode_override or getattr(self, "label_mode", "name")

        # Color-blind friendly palette (Okabe–Ito)
        OKABE_ITO = [
            "#8c510a",
            "#bf812d",
            "#dfc27d",
            "#f6e8c3",
            "#c7eae5",
            "#80cdc1",
            "#35978f",
            "#01665e"
        ]

        if show_legend is None:
            show_legend = compare_col is not None

        # ---------- build indicator matrix or class totals ----------
        if antibiotic_classes:
            class_cols = {}
            for cls, cols in antibiotic_classes.items():
                existing = [c for c in cols if c in df.columns]
                if existing:
                    class_cols[cls] = df[existing].sum(axis=1)
            if not class_cols:
                raise ValueError("None of the class columns exist in df for the provided antibiotic_classes.")
            id_cols = ([compare_col] if compare_col else []) + ([group_by] if group_by else [])
            work = pd.concat([df[id_cols], pd.DataFrame(class_cols, index=df.index)], axis=1)
            abx_cols_like = list(class_cols.keys())   # already clean labels
            labels_are_clean = True
        else:
            id_cols = ([compare_col] if compare_col else []) + ([group_by] if group_by else [])
            work = df[id_cols + self.abx_cols].copy()
            abx_cols_like = self.abx_cols[:]
            labels_are_clean = False

        # Optional filter list
        if antibiotics_to_plot:
            if labels_are_clean:
                keep = [a for a in antibiotics_to_plot if a in abx_cols_like]
            else:
                keep = self.resolve_to_columns(antibiotics_to_plot)
            abx_cols_like = [c for c in abx_cols_like if c in keep]
            if not abx_cols_like:
                pass
                # raise ValueError("None of the requested antibiotics/classes matched the dataframe.")

        # ---------- helpers ----------
        def _national_population_for_year():
            if self.population_df is None:
                raise ValueError("Population data required for per_100k_population.")
            pop = self.population_df.copy()
            pop.columns = [c.strip() for c in pop.columns]
            if "Bundesland" not in pop.columns and "bundesland" in pop.columns:
                pop["Bundesland"] = pop["bundesland"]
            if year is not None and "Year" in pop.columns:
                pop = pop[pop["Year"] == year]
            elif "Year" in pop.columns:
                latest = pop.groupby("Bundesland")["Year"].max().reset_index()
                pop = pop.merge(latest, on=["Bundesland", "Year"], how="inner")
            total = float(pop["total"].sum())
            if total <= 0:
                raise ValueError("Population total must be > 0.")
            return total

        # ---------- compute per-level coverage ----------
        cmp_key = compare_col or "__cmp"

        if group_by is None:
            # NATIONAL MODE
            if per_100k_population:
                nat_pop = _national_population_for_year()
                tested = (work.groupby(compare_col, observed=False)[abx_cols_like].sum().reset_index()
                        if compare_col else work[abx_cols_like].sum().to_frame().T.assign(__dummy="All"))
                long_df = tested.melt(id_vars=[compare_col] if compare_col else ["__dummy"],
                                    value_vars=abx_cols_like,
                                    var_name="Antibiotic", value_name="Count")
                long_df["Value"] = long_df["Count"] / nat_pop * 100000
                long_df["Denominator"] = nat_pop
                ylabel = "Tests per 100k Population"
                if "__dummy" in long_df.columns:
                    long_df[cmp_key] = "All"
                    long_df.drop(columns=["__dummy"], inplace=True)
            else:
                pct = (work.groupby(compare_col, observed=False)[abx_cols_like].mean() * 100
                    if compare_col else pd.DataFrame(work[abx_cols_like].mean() * 100).T)
                Ns = (work.groupby(compare_col, observed=False).size().rename("N").reset_index()
                    if compare_col else pd.DataFrame({"N": [len(work)]}))
                
                
                long_df = pct.reset_index().melt(
                    id_vars=[compare_col] if compare_col else None,
                    value_vars=abx_cols_like,
                    var_name="Antibiotic",
                    value_name="Value"
                )
                if compare_col is None:
                    long_df[cmp_key] = "All"
                long_df = long_df.merge(Ns,
                                        left_on=[compare_col] if compare_col else None,
                                        right_on=[compare_col] if compare_col else None,
                                        how="left")
                long_df.rename(columns={"N": "Denominator"}, inplace=True)
                ylabel = "Mean Test Coverage (%)"

            base_long = long_df.copy()
            base_long["err_high"] = np.nan
            base_long["err_low"] = np.nan
            # no top/bottom groups in national mode
            base_long["TopGroup"] = np.nan
            base_long["TopVal"] = np.nan
            base_long["TopDen"] = np.nan
            base_long["BotGroup"] = np.nan
            base_long["BotVal"] = np.nan
            base_long["BotDen"] = np.nan

            # 🔧 FIX: set plot_df here
            plot_df = base_long.copy()

        else:
            # STRATIFIED MODE
            sizes = work.groupby(group_by, observed=False).size()
            if min_group_size > 1:
                keep = sizes[sizes >= min_group_size].index
                work = work[work[group_by].isin(keep)]
            if work.empty:
                raise ValueError("All strata dropped by min_group_size; relax the threshold.")

            if per_100k_population:
                if self.population_df is None:
                    raise ValueError("Population data required for per_100k_population mode.")
                tested = work.groupby([group_by, compare_col] if compare_col else [group_by],
                                    observed=False)[abx_cols_like].sum().reset_index()
                pop = self.population_df.copy()
                pop.columns = [c.strip() for c in pop.columns]
                if "Bundesland" not in pop.columns and "bundesland" in pop.columns:
                    pop["Bundesland"] = pop["bundesland"]
                if year is not None and "Year" in pop.columns:
                    pop = pop[pop["Year"] == year]
                elif "Year" in pop.columns:
                    latest = pop.groupby("Bundesland")["Year"].max().reset_index()
                    pop = pop.merge(latest, on=["Bundesland", "Year"], how="inner")
                group_pop = pop.groupby("Bundesland", observed=False)["total"].sum().rename("Population").reset_index()
                tested = tested.merge(group_pop.rename(columns={"Bundesland": group_by}),
                                    on=group_by, how="left")
                for c in abx_cols_like:
                    tested[c] = tested[c] / tested["Population"] * 100000
                ylabel = "Tests per 100k Population"
                base_long = tested.melt(
                    id_vars=[group_by, compare_col, "Population"] if compare_col else [group_by, "Population"],
                    value_vars=abx_cols_like, var_name="Antibiotic", value_name="Value"
                )
                base_long.rename(columns={"Population": "Denominator"}, inplace=True)
                if compare_col is None:
                    base_long[cmp_key] = "All"
            else:
                pct = work.groupby([group_by, compare_col] if compare_col else [group_by],
                                observed=False)[abx_cols_like].mean() * 100
                Ns = work.groupby([group_by, compare_col] if compare_col else [group_by],
                                observed=False).size().rename("N").reset_index()

                base_long = pct.reset_index().melt(
                    id_vars=[group_by, compare_col] if compare_col else [group_by],
                    value_vars=abx_cols_like, var_name="Antibiotic", value_name="Value"
                ).merge(Ns, on=[group_by, compare_col] if compare_col else [group_by], how="left")
                base_long.rename(columns={"N": "Denominator"}, inplace=True)
                if compare_col is None:
                    base_long[cmp_key] = "All"
                ylabel = "Coverage (%)"

            # aggregate across strata -> means + error bars

            merged = self.calculate_variation_stats(base_long, cmp_key=cmp_key, variation_mode=variation_mode)

            # top/bottom group_by per antibiotic & compare level
            tb = (base_long.groupby(["Antibiotic", cmp_key, group_by], observed=False)
                            .agg(Mean=("Value", "mean"), Den=("Denominator", "sum")).reset_index())
            tops = tb.loc[tb.groupby(["Antibiotic", cmp_key])["Mean"].idxmax()] \
                    .rename(columns={group_by: "TopGroup", "Mean": "TopVal", "Den": "TopDen"})
            bots = tb.loc[tb.groupby(["Antibiotic", cmp_key])["Mean"].idxmin()] \
                    .rename(columns={group_by: "BotGroup", "Mean": "BotVal", "Den": "BotDen"})

            plot_df = merged.merge(tops[["Antibiotic", cmp_key, "TopGroup", "TopVal", "TopDen"]],
                                on=["Antibiotic", cmp_key], how="left") \
                            .merge(bots[["Antibiotic", cmp_key, "BotGroup", "BotVal", "BotDen"]],
                                on=["Antibiotic", cmp_key], how="left") \
                            .rename(columns={"Mean": "Value"})

            # representative denominator for hover (sum across strata)
            den = base_long.groupby(["Antibiotic", cmp_key], observed=False)["Denominator"].sum().rename("Denominator").reset_index()
            plot_df = plot_df.merge(den, on=["Antibiotic", cmp_key], how="left")

        # ---------- normalize compare levels to strings ----------
        y_max = plot_df["Value"].max()
        if "err_high" in plot_df.columns and plot_df["err_high"].notna().any():
            y_max = max(y_max, (plot_df["Value"] + plot_df["err_high"]).max())
        if not per_100k_population:
            # For percentage data, cap at 100
            y_max = min(y_max, 100)
            y_range = [0, max(100, y_max)]  # Ensure at least 0-100 range
        else:
            # For population data, use natural max with some padding
            y_range = [0, y_max * 1.05]  # 5% padding
            
            
        plot_df[cmp_key] = plot_df[cmp_key].astype(str)
        if values is not None:
            values = [str(v) for v in values]
        if sort_group_value is not None:
            sort_group_value = str(sort_group_value)
        if color_map:
            color_map = {str(k): v for k, v in color_map.items()}

        # ---------- label mapping ----------
        if labels_are_clean:
            plot_df["XLabel"] = plot_df["Antibiotic"]
        else:
            plot_df["XLabel"] = plot_df["Antibiotic"].map(lambda c: self.get_label(c, lab_mode))

        # ---------- sorting ----------
        if compare_col is None:
            scores = plot_df.groupby(["Antibiotic", "XLabel"], observed=False)["Value"].mean()
        else:
            pivot = plot_df.pivot_table(index=["Antibiotic", "XLabel"], columns=cmp_key,
                                        values="Value", aggfunc="mean")
            pivot.columns = pivot.columns.astype(str)
            if sort_mode == "overall_mean":
                scores = pivot.mean(axis=1)
            elif sort_mode == "max":
                scores = pivot.max(axis=1)
            elif sort_mode == "by_group":
                if sort_group_value is None:
                    raise ValueError("sort_group_value is required when sort_mode='by_group'")
                if sort_group_value not in pivot.columns:
                    raise ValueError(f"{sort_group_value!r} not found among compare values.")
                scores = pivot[sort_group_value]
            else:
                raise ValueError("sort_mode must be one of: 'overall_mean', 'max', 'by_group'")
        order_index = scores.sort_values(ascending=ascending).index
        order_labels = [idx[1] if isinstance(idx, tuple) else idx for idx in order_index]
        plot_df["XLabel"] = pd.Categorical(plot_df["XLabel"], categories=order_labels, ordered=True)

        # ---------- colors ----------
        if compare_col is None:
            series_color = bar_color
        else:
            if not color_map:
                uniq = plot_df[cmp_key].unique().tolist()
                color_map = {v: OKABE_ITO[i % len(OKABE_ITO)] for i, v in enumerate(uniq)}

        # ---------- title ----------
        if title is None:
            title = ("Comparison by antibiotic" if compare_col is None
                    else f"{compare_col} comparison by antibiotic")
            if group_by:
                title += f" (aggregated across {group_by}, error: {variation_mode.upper()})"
            if per_100k_population:
                title += " — per 100k population"

        # ---------- plotting ----------
        customdata = np.stack([
            plot_df["Value"].values, plot_df["Denominator"].fillna(0).values, plot_df[cmp_key].values,
            plot_df.get("TopGroup", pd.Series([np.nan]*len(plot_df))).values,
            plot_df.get("TopVal",   pd.Series([np.nan]*len(plot_df))).values,
            plot_df.get("TopDen",   pd.Series([np.nan]*len(plot_df))).values,
            plot_df.get("BotGroup", pd.Series([np.nan]*len(plot_df))).values,
            plot_df.get("BotVal",   pd.Series([np.nan]*len(plot_df))).values,
            plot_df.get("BotDen",   pd.Series([np.nan]*len(plot_df))).values,
        ], axis=1)

        cmp_line = "" if compare_col is None else f"{cmp_key}: %{{customdata[2]}}<br>"
        tb_line  = "" if group_by is None else (
            "Top: %{customdata[3]} (%{customdata[4]:.2f}; N=%{customdata[5]:,})<br>"
            "Bottom: %{customdata[6]} (%{customdata[7]:.2f}; N=%{customdata[8]:,})<br>"
        )

        if vertical:
            fig = px.bar(
                plot_df,
                x="XLabel", y="Value",
                color=(cmp_key if compare_col is not None else None),
                barmode="group",
                labels={"XLabel": "Antibiotic", "Value": ylabel},
                title=title,
                color_discrete_map=(color_map if compare_col is not None else None)
            )
            if "err_high" in plot_df.columns and plot_df["err_high"].notna().any():
                fig.update_traces(error_y=dict(array=plot_df["err_high"], arrayminus=plot_df["err_low"]))
            if compare_col is None:
                fig.update_traces(marker_color=series_color, selector=dict(type="bar"))
            fig.update_xaxes(categoryorder="array", categoryarray=order_labels, tickangle=45,
                            showgrid=True, gridcolor="lightgrey")
            fig.update_yaxes(title=ylabel, showgrid=True, gridcolor="lightgrey", range=y_range)
            fig.update_layout(
                width=max(1100, len(order_labels) * 45),
                height=740,
                margin=dict(l=60, r=40, t=70, b=190),
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_x=0.5,
                showlegend=show_legend
            )
        else:
            fig = px.bar(
                plot_df,
                x="Value", y="XLabel",
                color=(cmp_key if compare_col is not None else None),
                barmode="group", orientation="h",
                labels={"XLabel": "Antibiotic", "Value": ylabel},
                title=title,
                color_discrete_map=(color_map if compare_col is not None else None)
            )
            if "err_high" in plot_df.columns and plot_df["err_high"].notna().any():
                fig.update_traces(error_x=dict(array=plot_df["err_high"], arrayminus=plot_df["err_low"]))
            if compare_col is None:
                fig.update_traces(marker_color=series_color, selector=dict(type="bar"))
            fig.update_yaxes(categoryorder="array", categoryarray=order_labels, automargin=True,
                            showgrid=True, gridcolor="lightgrey", range=y_range)
            fig.update_xaxes(title=ylabel, showgrid=True, gridcolor="lightgrey")
            fig.update_layout(
                width=1200,
                height=max(650, len(order_labels) * 28),
                margin=dict(l=260, r=40, t=70, b=70),
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_x=0.5,
                showlegend=show_legend
            )

        fig.update_traces(
            customdata=customdata,
            hovertemplate=(
                ("Antibiotic: %{x}<br>" if vertical else "Antibiotic: %{y}<br>")
                + cmp_line + tb_line
                + "Value: %{customdata[0]:.2f}<br>"
                "Denominator: %{customdata[1]:,}<extra></extra>"
            ),
            marker_line_width=0
        )

        # fig.show()

        if export_csv_path:
            out = plot_df.copy()
            out.rename(columns={"Value": ylabel}, inplace=True)
            out.to_csv(export_csv_path, index=False)

        if output_prefix:
            fig.write_image(f"{output_prefix}.pdf")
            fig.write_image(f"{output_prefix}.svg")
            fig.write_image(f"{output_prefix}.png", width=1600, height=950, scale=3)

        return fig