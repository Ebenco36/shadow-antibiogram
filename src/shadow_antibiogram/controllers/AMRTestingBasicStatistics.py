import pandas as pd
import numpy as np
import geopandas as gpd
import altair as alt
import json
import math
import os
import plotly.graph_objects as go
import plotly.express as px
import dash_bio
import plotly.io as pio
from shapely.geometry import Point, shape
from typing import Sequence, Union, Mapping, List, Optional, Any, Callable
from shadow_antibiogram.mappers.exempted_columns import columns_to_exempt

alt.data_transformers.enable('default')


class AMRTestingPatternAnalysis:
    def __init__(
        self,
        output_path="./datasets/output",
        population_file="./datasets/population-data-cleaned.csv",
        dataframe: pd.DataFrame = pd.DataFrame([]),
        suffix="Tested"
    ):
        """
        Initialize the AMRTestingPatternAnalysis with the dataset.
        """
        self.output_path = output_path
        self.population_file = population_file
        # setup directory structure for all files to be saved such as svg and tables
        _, self.charts_dir, self.tables_dir = self.setup_directories(
            self.output_path)
        self.data = dataframe
        # Load population data
        self.population_data = self.load_population_data()
        # Fetch testing variable from dataframe.
        self.antibiotic_columns = [
            col for col in self.data.columns if ' - ' in col if col not in columns_to_exempt]

        self.suffix = suffix

    def setup_directories(self, output_path, charts_dir_="charts", tables_dir_="tables"):
        """
        Creates necessary directories for storing charts and tables.

        Parameters:
            output_path (str): The base output directory.

        Returns:
            tuple: A dictionary containing paths to the created directories.
        """
        def ensure_directory(directory_path):
            """Creates a directory if it does not exist."""
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' is ready.")

        # Ensure the main output directory exists
        ensure_directory(output_path)

        # Create subdirectories for different types of files
        charts_dir = os.path.join(output_path, charts_dir_)
        ensure_directory(charts_dir)

        tables_dir = os.path.join(output_path, tables_dir_)
        ensure_directory(tables_dir)

        _, self.charts_dir, self.tables_dir = (
            output_path,
            charts_dir,
            tables_dir
        )

        return output_path, charts_dir, tables_dir

    def load_population_data(self):
        import pandas as pd
        df = pd.read_csv(self.population_file)
        df["Bundesland"] = df["bundesland"].str.strip()
        df["Year"] = df["Year"].astype(str)
        df["total"] = df["total"].astype(float)
        return df[["Bundesland", "Year", "total"]]

    def convert_to_tested_columns(self, antibiotics_list):
        """
        Convert a list of antibiotic names to their corresponding '_Tested' column names.

        Parameters:
            antibiotics_list (list): List of antibiotic names.

        Returns:
            list: List of antibiotic '_Tested' column names.
        """
        if not isinstance(antibiotics_list, list):
            raise ValueError("Input must be a list of antibiotic names.")

        return [f"{antibiotic}_{self.suffix}" for antibiotic in antibiotics_list]

    def calculate_testing_rates(self, df=None, group_by_cols=[], target_antibiotics=[], mode="per_1000"):
        """
        Calculate the rate of antibiotic testing, stratified by specified columns.

        Parameters:
            group_by_cols (list): List of columns to group by.
            target_antibiotics (str or list): Single antibiotic (str) or multiple antibiotics (list).

        Returns:
            pd.DataFrame: A DataFrame with testing rates.
        """

        data = df if isinstance(
            df, pd.DataFrame) and not df.empty else self.data
        testing_rates = None

        # Ensure target_antibiotics is a list
        if isinstance(target_antibiotics, str):
            target_antibiotics = [target_antibiotics]

        if len(target_antibiotics) == 1:
            # Handle the case where only one antibiotic is provided
            target_antibiotic = target_antibiotics[0]
            test_col = target_antibiotic if target_antibiotic.endswith(
                f"_{self.suffix}") else f"{target_antibiotic}_{self.suffix}"

            if test_col not in data.columns:
                raise KeyError(f"Antibiotic column not found: {test_col}")

            testing_rates = self.compute_antibiotic_testing_metrics(
                df=data, selected_fields=group_by_cols, mode=mode, test_columns=test_col
            )
        else:
            # Store all antibiotic testing rates together
            df_list = []

            for antibiotic in target_antibiotics:
                test_col = antibiotic if antibiotic.endswith(
                    f"_{self.suffix}") else f"{antibiotic}_{self.suffix}"

                if test_col not in data.columns:
                    raise KeyError(f"Antibiotic column not found: {test_col}")

                rates = self.compute_antibiotic_testing_metrics(
                    df=data, selected_fields=group_by_cols, mode=mode, test_columns=test_col
                )
                rates["Antibiotic"] = antibiotic
                rates = rates.rename(columns={test_col: "Testing Rate"})

                df_list.append(rates)

                # Convert to long format where each row has all antibiotics' testing rates
                testing_rates = pd.concat(df_list, ignore_index=True)

            testing_rates.to_csv(os.path.join(
                self.tables_dir, "testing_rates_tab.csv"), index=False)

        return testing_rates

    def sanitize_string(self, s):
        """Replace spaces and slashes with underscores in a string."""
        return s.replace(" ", "_").replace("/", "_").replace("\\", "_")

    def plot_and_save_barchart(
        self, df,
        x_field, y_field,
        color_field=None,
        color: str = "steelblue",  # fallback color for single charts
        title="", file_name="barchart",
        save_as_image=False, save_as_html=False,
        image_format="svg", sort_order="descending",
        order_by=None, orientation="vertical", log_scale=False,
        height=400, width=600, show_text=True,
        custom_sort_order: list = None,
        stack_colors: Optional[Union[dict, list]] = None,
        extra_dir_to_save=None
    ):
        if df.empty:
            raise ValueError(
                "The DataFrame is empty. Please provide a valid dataset.")

        # Handle case where both fields are numeric
        if pd.api.types.is_numeric_dtype(df[x_field]) and pd.api.types.is_numeric_dtype(df[y_field]):
            value_field = y_field if orientation == "vertical" else x_field
            category_field = x_field if orientation == "vertical" else y_field
        else:
            # Auto-detect numeric axis
            if pd.api.types.is_numeric_dtype(df[x_field]):
                value_field = x_field
                category_field = y_field
            elif pd.api.types.is_numeric_dtype(df[y_field]):
                value_field = y_field
                category_field = x_field
            else:
                raise ValueError(
                    "Neither x_field nor y_field is numeric. One must be numeric.")

        # Automatically convert stack_by to string if it's numeric
        if color_field and pd.api.types.is_numeric_dtype(df[color_field]):
            df[color_field] = df[color_field].astype(str)

        if log_scale:
            df = df[df[value_field] > 0]
            if df.empty:
                raise ValueError(
                    "No positive values in value field for log scale.")
            df[value_field] += 1e-6

        order_by = order_by or value_field

        # Apply custom sort to category axis
        if custom_sort_order:
            category_field = y_field if pd.api.types.is_numeric_dtype(
                df[x_field]) else x_field

            # Reverse order if needed
            effective_order = (
                list(reversed(custom_sort_order))
                if sort_order == "descending"
                else custom_sort_order
            )

            df[category_field] = pd.Categorical(
                df[category_field], categories=effective_order, ordered=True)
            df = df.sort_values(category_field)
            sort_order_vals = effective_order
        else:
            sort_order_vals = alt.EncodingSortField(
                field=order_by, order=sort_order)

        # Axis encodings
        if orientation == "horizontal":
            x_enc = alt.X(value_field, title=value_field,
                          scale=alt.Scale(type="log") if log_scale else alt.Scale())
            y_enc = alt.Y(category_field, title=category_field,
                          sort=sort_order_vals)
        else:  # vertical
            x_enc = alt.X(category_field, title=category_field,
                          sort=sort_order_vals)
            y_enc = alt.Y(value_field, title=value_field,
                          scale=alt.Scale(type="log") if log_scale else alt.Scale())

        # Determine bar mark with or without color stacking
        if color_field:
            if stack_colors:
                color_enc = alt.Color(
                    color_field,
                    scale=alt.Scale(
                        domain=list(stack_colors.keys()) if isinstance(
                            stack_colors, dict) else alt.Undefined,
                        range=list(stack_colors.values()) if isinstance(
                            stack_colors, dict) else stack_colors
                    ),
                    title=color_field
                )
            else:
                color_enc = alt.Color(color_field, title=color_field)
        else:
            color_enc = alt.value(color)

        bars = alt.Chart(df).mark_bar().encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            tooltip=[x_field, y_field] + ([color_field] if color_field else [])
        )

        if show_text:
            text = bars.mark_text(
                align='left' if orientation == "horizontal" else 'center',
                baseline='middle',
                dx=3 if orientation == "horizontal" else 0,
                dy=0 if orientation == "horizontal" else -10
            ).encode(
                text=alt.Text(f"{value_field}:Q", format=",.0f")
            )
            chart = (bars + text).properties(title=title,
                                             width=width, height=height)
        else:
            chart = bars.properties(title=title, width=width, height=height)

        chart = (
            chart
            .configure_axis(
                labelFontSize=18,
                titleFontSize=18,
                titleFontWeight="bold"
            )
            .configure_axisY(
                labelLimit=1200  # Prevent truncation of y labels
            )
            .configure_legend(
                labelFontSize=14,
                titleFontSize=18,
                titleFontWeight="bold"
            )
            .configure_title(
                fontSize=22,
                fontWeight="bold"
            )
            .configure_text(
                fontSize=14
            )
            .configure_view(
                stroke=None  # No border
            )
        )
        # Force x-axis slant when horizontal
        if orientation == "vertical":
            chart = chart.configure_axisX(labelAngle=45)


        os.makedirs(self.charts_dir + "/" + extra_dir_to_save, exist_ok=True)
        if save_as_html:
            chart.save(os.path.join(self.charts_dir + "/" +
                       extra_dir_to_save, f"{self.sanitize_string(file_name)}.html"))
        if save_as_image:
            if image_format == "svg":
                chart.save(os.path.join(self.charts_dir + "/" + extra_dir_to_save,
                           f"{self.sanitize_string(file_name)}.{image_format}"))
            else:
                chart.save(os.path.join(self.charts_dir + "/" + extra_dir_to_save,
                           f"{self.sanitize_string(file_name)}.{image_format}"), scale_factor=4)
        if save_as_image:
            chart.save(os.path.join(self.charts_dir + "/" + extra_dir_to_save,
                       f"{self.sanitize_string(file_name)}.pdf"), scale_factor=4)

        return chart

    def top_values_with_tests(
        self,
        field="Pathogen",
        stack_by=None,  # New: field to stack by (e.g., Sex)
        stack_colors=None,  # Optional: color override for stacked categories
        top_n=None,
        file_name="top_values_barchart",
        save_as_image=True,
        save_as_html=True,
        image_format="svg",
        sort_order="descending",
        order_by=None,
        orientation="vertical",
        log_scale=False,
        height=400,
        width=600,
        show_text=True,
        color: str = "steelblue",  # fallback color for single charts
        custom_sort_order: list = None,  # Now accepts a single list
        prefilter=None,
        chart_title_param=None,
        extra_dir_to_save=None
    ):
        """
        Identify the top values in a specified field and plot a bar chart with optional custom ordering.
        Also writes a CSV including a 'Percentage' column indicating each row's proportion (%)
        relative to the total number of records (before top_n truncation).
        """
        # Apply Data Filters
        data = self.data.copy()

        # `prefilter` if provided
        if callable(prefilter):
            try:
                data = prefilter(data)
            except Exception as e:
                print(f"[ERROR] Failed to apply prefilter: {e}")

        if field not in data.columns:
            raise KeyError(f"The dataset does not contain a '{field}' column.")
        if stack_by and stack_by not in data.columns:
            raise KeyError(
                f"The dataset does not contain a '{stack_by}' column."
            )

        # Prepare data
        if stack_by:
            # Count combinations
            df = data.groupby([field, stack_by]).size().reset_index(name="Count")

            # Total across ALL rows, used as denominator for percentages
            total_all = df["Count"].sum()

            # Totals by main field to find top_n keys
            totals = df.groupby(field)["Count"].sum().reset_index(name="Total")
            top_keys = totals.sort_values(
                "Total", ascending=(sort_order != "descending")
            )

            if top_n:
                top_keys = top_keys.head(top_n)

            df = df[df[field].isin(top_keys[field])]

            if custom_sort_order:
                df[field] = pd.Categorical(
                    df[field], categories=custom_sort_order, ordered=True
                )
                df = df.sort_values(field)

            # Add Percentage column: share of ALL observations, not just top_n
            df["Percentage"] = (df["Count"] / total_all) * 100

            csv_path = os.path.join(
                self.tables_dir, f"grouped_by_{field}_and_{stack_by}.csv"
            )
            chart_title = f"{field} Distribution Stacked by {stack_by}"
            color_field = stack_by
            show_text = False  # Disable text overlay for stacked charts

        else:
            # Simple counts by field
            counts = data[field].value_counts().reset_index()
            counts.columns = [field, "Count"]

            # Total across ALL rows before top_n
            total_all = counts["Count"].sum()

            df = counts

            if top_n:
                df = df.head(top_n)

            if custom_sort_order:
                df[field] = pd.Categorical(
                    df[field], categories=custom_sort_order, ordered=True
                )
                df = df.sort_values(field)

            # Add Percentage column: share of ALL observations, not just top_n
            df["Percentage"] = (df["Count"] / total_all) * 100

            csv_path = os.path.join(self.tables_dir, f"grouped_by_{field}.csv")
            chart_title = (
                f"Top {top_n} {field} by Count" if top_n else f"{field} Distribution by Count"
            )
            color_field = None  # No color grouping

        title = chart_title_param if chart_title_param else chart_title

        # Save the table for better insights (now includes Percentage)
        df.to_csv(csv_path, index=False)

        # Default to ordering by Count
        order_by = order_by or "Count"

        # Set axis fields for plot depending on orientation
        x_field = field if orientation == "vertical" else "Count"
        y_field = "Count" if orientation == "vertical" else field

        # Plot chart
        self.plot_and_save_barchart(
            df=df,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            title=title,
            file_name=file_name,
            save_as_image=save_as_image,
            save_as_html=save_as_html,
            image_format=image_format,
            sort_order=sort_order,
            order_by=order_by,
            orientation=orientation,
            log_scale=log_scale,
            height=height,
            width=width,
            show_text=show_text,
            custom_sort_order=custom_sort_order,
            color=color,
            stack_colors=stack_colors,
            extra_dir_to_save=extra_dir_to_save
        )

        return df


    def altair_categorical_viz(
        self,
        df: pd.DataFrame,
        features: List[str],
        chart_type: str = 'bar',
        orientation: str = 'vertical',
        show_values: bool = False,
        width: int = 600,
        height: int = 400,
        title: Optional[str] = None,
        *,
        agg: Union[str, Callable[[pd.Series], float]] = 'count',
        value: Optional[str] = None,
        color_scheme: Optional[Union[
            Sequence[str], Mapping[str, str], Mapping[str, Mapping[str, str]]
        ]] = None,
        filter_groups: Optional[Mapping[str, List]] = None,
        top_n: Optional[int] = None,
        top_n_per_group: Optional[int] = None,
        sort: Union[str, List[str], None] = None,
        axis_config: Optional[Mapping[str, Any]] = None,
        title_config: Optional[Mapping[str, Any]] = None,
        log_scale: bool = False,
        save_path: str = "chart.html"
    ) -> Union[alt.Chart, Any]:
        """
        Versatile categorical viz with Plotly treemap and HTML export.
        All bar types respect orientation ('vertical' or 'horizontal').
        Supports: bar, stacked_bar, dodged_bar, dot, lollipop, heatmap, pie, donut, treemap.
        """
        # Formatting
        default_axis = {'labelAngle': -45,
                        'labelFontSize': 12, 'titleFontSize': 14}
        default_title = {'fontSize': 16, 'anchor': 'start'}
        axis_cfg = {**default_axis, **(axis_config or {})}
        title_cfg = {**default_title, **(title_config or {})}

        # Validate
        types = ('bar', 'stacked_bar', 'dodged_bar', 'dot',
                 'lollipop', 'heatmap', 'pie', 'donut', 'treemap')
        if chart_type not in types:
            raise ValueError(f"Unsupported chart_type: {chart_type}")
        if chart_type in ('stacked_bar', 'dodged_bar', 'heatmap') and len(features) != 2:
            raise ValueError(f"{chart_type} requires 2 features")
        if chart_type in ('bar', 'dot', 'lollipop', 'pie', 'donut') and len(features) != 1:
            raise ValueError(f"{chart_type} requires 1 feature")
        if chart_type == 'treemap' and len(features) not in (1, 2):
            raise ValueError("treemap requires 1 or 2 features")
        if orientation not in ('vertical', 'horizontal'):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        # Features
        if len(features) == 1:
            col = features[0]
        else:
            col_x, col_color = features

        # Filter
        if filter_groups:
            for f, vals in filter_groups.items():
                if f not in df.columns:
                    raise KeyError(f"Column '{f}' not in DataFrame")
                df = df[df[f].isin(vals)]

        # Agg func
        if isinstance(agg, str):
            if agg == 'count':
                def func(s): return s.count()
            elif agg in ('sum', 'mean', 'median', 'min', 'max'):
                func = getattr(pd.Series, agg)
            else:
                raise ValueError(f"Unknown agg {agg}")
        elif callable(agg):
            func = agg
        else:
            raise ValueError("agg must be string or callable")

        # Aggregate
        if agg == 'count':
            measure = 'count'
            if len(features) == 1:
                data = df[[col]].assign(count=1).groupby(
                    col, as_index=False).sum()
            else:
                data = df[[col_x, col_color]].assign(count=1).groupby(
                    [col_x, col_color], as_index=False).sum()
        else:
            if not value or value not in df.columns:
                raise ValueError("value column required for agg")
            measure = agg
            if len(features) == 1:
                data = df[[col, value]].groupby(col)[value].apply(
                    func).reset_index(name=measure)
            else:
                data = df[[col_x, col_color, value]].groupby(
                    [col_x, col_color])[value].apply(func).reset_index(name=measure)

        # Top-N first
        first = features[0]
        has_map = isinstance(color_scheme, Mapping) and first in color_scheme and isinstance(
            color_scheme[first], Mapping)
        if not has_map and top_n is None:
            top_n = 10
        if top_n is not None:
            tops = data.groupby(first)[measure].sum().nlargest(
                top_n).index.tolist()
            data = data[data[first].isin(tops)]

        # Top-N per group
        if top_n_per_group and len(features) == 2:
            data['_r'] = data.groupby(col_x)[measure].rank(
                method='first', ascending=False)
            data = data[data['_r'] <= top_n_per_group].drop(columns=['_r'])

        # Log scale
        if log_scale:
            data = data[data[measure] > 0]
            mn, mx = data[measure].min(), data[measure].max()
            count_scale = alt.Scale(type='log', domain=[mn*0.1, mx])
            quant_axis = alt.Axis(format='d', domain=False)
        else:
            count_scale = alt.Scale()
            quant_axis = alt.Axis(format='d', tickMinStep=2)

        # Sort
        if sort == 'count':
            sort_spec = alt.SortField(measure, order='descending')
        elif isinstance(sort, list):
            sort_spec = sort
        else:
            sort_spec = None

        # Color
        def get_color_enc(fld):
            if isinstance(color_scheme, Mapping) and fld in color_scheme and isinstance(color_scheme[fld], Mapping):
                mapping = dict(color_scheme[fld])
            elif isinstance(color_scheme, Mapping):
                mapping = dict(color_scheme)
            else:
                mapping = None
            if mapping:
                default = '#4C78A8'
                cats = data[fld].unique().tolist()
                for c in cats:
                    if c not in mapping:
                        mapping[c] = default
                return alt.Color(f'{fld}:N', scale=alt.Scale(domain=list(mapping), range=list(mapping.values())), title=fld)
            elif isinstance(color_scheme, Sequence):
                return alt.Color(f'{fld}:N', scale=alt.Scale(range=list(color_scheme)), title=fld)
            else:
                return alt.Color(f'{fld}:N', title=fld)

        # Build
        if chart_type == 'treemap':
            import plotly.express as px
            args = {'data_frame': data, 'path': features, 'values': measure}
            last = features[-1]
            if isinstance(color_scheme, Mapping) and last in color_scheme:
                args.update(color=last, color_discrete_map=color_scheme[last])
            fig = px.treemap(**args)
            fig.update_layout(width=width, height=height,
                              title=title or ",".join(features))
            # fig.write_html(save_path)
            if save_path:
                file_name = "_".join(features)
                fig.write_html(os.path.join(self.charts_dir,
                               f"{self.sanitize_string(file_name)}.html"))
                fig.write_image(os.path.join(self.charts_dir,
                                f"{self.sanitize_string(file_name)}.svg"))
            return fig

        base = alt.Chart(data)
        # Bar
        if chart_type == 'bar':
            if orientation == 'vertical':
                enc = {'x': alt.X(f'{col}:N', title=col, sort=sort_spec), 'y': alt.Y(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale)}
            else:
                enc = {'y': alt.Y(f'{col}:N', title=col, sort=sort_spec), 'x': alt.X(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale)}
            enc['tooltip'] = [f'{measure}:Q', f'{col}:N']
            chart = base.mark_bar().encode(**enc)
        # Stacked
        elif chart_type == 'stacked_bar':
            if orientation == 'vertical':
                enc = {'x': alt.X(f'{col_x}:N', title=col_x, sort=sort_spec), 'y': alt.Y(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale)}
            else:
                enc = {'y': alt.Y(f'{col_x}:N', title=col_x, sort=sort_spec), 'x': alt.X(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale)}
            enc.update(color=get_color_enc(col_color), tooltip=[
                       f'{measure}:Q', f'{col_x}:N', f'{col_color}:N'])
            chart = base.mark_bar().encode(**enc)
        # Dodged
        elif chart_type == 'dodged_bar':
            if orientation == 'vertical':
                enc = {'x': alt.X(f'{col_x}:N', title=col_x, sort=sort_spec), 'y': alt.Y(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale), 'xOffset': alt.XOffset(f'{col_color}:N')}
            else:
                enc = {'y': alt.Y(f'{col_x}:N', title=col_x, sort=sort_spec), 'x': alt.X(
                    f'{measure}:Q', axis=quant_axis, scale=count_scale), 'yOffset': alt.YOffset(f'{col_color}:N')}
            enc.update(color=get_color_enc(col_color), tooltip=[
                       f'{measure}:Q', f'{col_x}:N', f'{col_color}:N'])
            chart = base.mark_bar().encode(**enc)
        # Dot
        elif chart_type == 'dot':
            chart = base.mark_circle(size=100).encode(y=alt.Y(f'{col}:N', sort=sort_spec), x=alt.X(
                f'{measure}:Q', axis=quant_axis, scale=count_scale), tooltip=[f'{measure}:Q', f'{col}:N'])
        # Lollipop
        elif chart_type == 'lollipop':
            stem = base.mark_rule().encode(x=alt.X(f'{col}:N', sort=sort_spec) if orientation == 'vertical' else alt.X(f'{measure}:Q', axis=quant_axis, scale=count_scale), y=alt.Y(
                f'{measure}:Q', axis=quant_axis, scale=count_scale) if orientation == 'vertical' else alt.Y(f'{col}:N', sort=sort_spec))
            pts = base.mark_circle(size=100).encode(x=alt.X(f'{col}:N', sort=sort_spec) if orientation == 'vertical' else alt.X(f'{measure}:Q', axis=quant_axis, scale=count_scale), y=alt.Y(
                f'{measure}:Q', axis=quant_axis, scale=count_scale) if orientation == 'vertical' else alt.Y(f'{col}:N', sort=sort_spec), tooltip=[f'{measure}:Q', f'{col}:N'])
            chart = alt.layer(stem, pts)
        # Heatmap
        elif chart_type == 'heatmap':
            chart = base.mark_rect().encode(x=alt.X(f'{col_x}:N', sort=sort_spec), y=alt.Y(
                f'{col_color}:N', sort=sort_spec), color=alt.Color(f'{measure}:Q', scale=count_scale), tooltip=[f'{measure}:Q'])
        # Pie/Donut
        elif chart_type in ('pie', 'donut'):
            inner = 50 if chart_type == 'donut' else 0
            chart = base.mark_arc(innerRadius=inner).encode(theta=alt.Theta(f'{measure}:Q'), color=alt.Color(f'{col}:N', scale=alt.Scale(
                range=list(color_scheme)) if isinstance(color_scheme, Sequence) else None), tooltip=[f'{measure}:Q', f('{col}:N')])
        else:
            raise ValueError(
                "Unsupported chart_type after orientation support")

        # Values
        if show_values and chart_type in ('bar', 'stacked_bar', 'dodged_bar'):
            text = chart.mark_text(dx=5, align='left', baseline='middle') if orientation == 'horizontal' else chart.mark_text(
                dy=-5, align='center')
            chart = alt.layer(chart, text.encode(text=f'{measure}:Q'))

        # Export
        # chart.save(save_path)
        if save_path:
            file_name = "_".join(features)
            chart.save(os.path.join(self.charts_dir,
                       f"{self.sanitize_string(file_name)}.html"))
            chart.save(os.path.join(self.charts_dir,
                       f"{self.sanitize_string(file_name)}.svg"))

        return chart.properties(width=width, height=height, title=title or ",".join(features)).configure_axis(**axis_cfg).configure_title(**title_cfg)

    # Not used anywhere, but kept for reference

    def plot_testing_heatmap_altair(self, testing_data, x_col, y_col, value_col, title, sort_order_x=None, sort_order_y=None, mode="per_1000"):
        """
        Plot an interactive heatmap of antibiotic testing rates using Altair.

        Parameters:
            testing_data (pd.DataFrame): DataFrame containing testing rates.
            x_col (str): Column for x-axis.
            y_col (str): Column for y-axis.
            value_col (str): Column containing heatmap values.
            title (str): Title of the heatmap.
        """
        if mode == "per_1000":
            chart_caption = "Testing Rate Per 1,000 Isolates"
        elif mode == "mean":
            chart_caption = "Mean Testing Rate"
        # Ensure there are no NaN values in the DataFrame
        testing_data = testing_data.dropna(subset=[x_col, y_col, value_col])

        # Ensure the DataFrame is in the correct format
        if not isinstance(testing_data, pd.DataFrame):
            raise TypeError("testing_data must be a pandas DataFrame")
        # Print the first few rows of the DataFrame to check the data

        x_sort = alt.EncodingSortField(
            "Testing Rate", op="sum", order="descending") if sort_order_x else None
        y_sort = alt.EncodingSortField(
            "Testing Rate", op="sum", order="descending") if sort_order_y else None

        heatmap = alt.Chart(testing_data).mark_rect().encode(
            x=alt.X(y_col, title=y_col, sort=x_sort),
            y=alt.Y(x_col, title=x_col, sort=y_sort),
            # color=alt.Color(value_col, title="Testing Rate"),
            color=alt.Color(
                value_col,
                scale=alt.Scale(
                    scheme="blues",
                    domain=[0, testing_data[value_col].max()],
                    type='log'
                ),
                title=chart_caption
            ),
            tooltip=[x_col, y_col, value_col]
        )

        # Text Labels (Ensuring they don't go out of the box)
        text = alt.Chart(testing_data).mark_text(size=10, color="black", fontWeight="bold").encode(
            x=alt.X("Antibiotic:N", sort=x_sort),
            y=alt.Y(f"{x_col}:N", sort=y_sort),
            text=alt.condition(
                # Only display text if Testing Rate is 0.0
                alt.datum["Testing Rate"] == 0.0,
                alt.Text("Testing Rate:Q", format=".1f"),
                alt.value("")  # Empty text for other values
            ),
            color=alt.condition(
                # If Testing Rate is 0.0, text should be black
                alt.datum["Testing Rate"] == 0.0,
                alt.value("black"),
                alt.value("white")  # Otherwise, text should be white
            )
        )

        # Combine both layers
        layered_heatmap = alt.layer(heatmap, text).properties(
            title=title
        )

        # Clean column names for filename safety (optional, if needed)
        safe_x_col = self.sanitize_string(x_col)
        safe_y_col = self.sanitize_string(y_col)

        # Save the heatmap
        layered_heatmap.save(os.path.join(
            self.charts_dir, f"{safe_x_col}_{safe_y_col}_testing_pattern_heatmap.svg"))
        testing_data.to_csv(os.path.join(
            self.tables_dir, f"{safe_x_col}_{safe_y_col}_testing_pattern_heatmap_tab.csv"), index=False)

        return layered_heatmap

    def temporal_analysis(self, target_antibiotics, time_col='Date', time_freq='Y', filter_dict=None, file_name="temporal_chart_information", mode="per_1000"):
        """
        Analyze temporal trends in antibiotic testing rates.

        Parameters:
            target_antibiotics (list or str): The antibiotic(s) to analyze.
            time_col (str): The column containing time data (default: 'Date').
            time_freq (str): The frequency for time aggregation (default: 'Y' for yearly, 'M' for monthly).
            filter_dict (dict): Dictionary of column-value pairs to filter the data.

        Returns:
            pd.DataFrame: Aggregated testing rates over time.
        """
        # Ensure target_antibiotics is a list
        if isinstance(target_antibiotics, str):
            target_antibiotics = [target_antibiotics]

        # Work on a copy of the data
        filtered_data = self.data.copy()

        # Apply filtering if provided
        if filter_dict:
            for col, values in filter_dict.items():
                filtered_data = filtered_data[filtered_data[col].isin(values)]

        # Ensure the time column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(filtered_data[time_col]):
            filtered_data[time_col] = pd.to_datetime(
                filtered_data[time_col], errors='coerce')

        # Create a new column for time aggregation
        if time_freq == 'Y':
            filtered_data[time_freq] = filtered_data[time_col].dt.year
        elif time_freq == 'M':
            filtered_data[time_freq] = filtered_data[time_col].dt.to_period(
                'M').astype(str)
        else:
            raise ValueError(
                "time_freq must be either 'Y' (yearly) or 'M' (monthly)")

        # Build the list of test indicator columns
        test_cols = [f'{ab}_{self.suffix}' for ab in target_antibiotics]

        # Ensure columns are numeric
        for col in test_cols:
            if col in filtered_data.columns:
                filtered_data[col] = pd.to_numeric(
                    filtered_data[col], errors='coerce')
            else:
                raise KeyError(f"Column '{col}' not found in data.")

        # Calculate testing rates over time
        temporal_testing = self.compute_antibiotic_testing_metrics(
            filtered_data, time_freq, mode=mode, test_columns=test_cols)

        # Reshape the data for plotting
        plot_df = temporal_testing.melt(
            id_vars=time_freq, var_name='Antibiotic', value_name='Testing Rate')
        plot_df['Antibiotic'] = plot_df['Antibiotic'].str.replace(
            f'_{self.suffix}', '')

        # Plot the trends
        chart = alt.Chart(plot_df).mark_line(point=True, opacity=0.3).encode(
            x=alt.X(time_freq, title='Time'),
            y=alt.Y('Testing Rate', title='Testing Rate'),
            color=alt.Color(
                'Antibiotic', legend=alt.Legend(title="Antibiotic")),
            tooltip=['Antibiotic', time_freq, 'Testing Rate']
        ).properties(
            title="Temporal Trends in Antibiotic Testing Rates"
        )

        chart.save(os.path.join(self.charts_dir,
                   f"{self.sanitize_string(file_name)}_time_pattern.svg"))

        temporal_testing.to_csv(os.path.join(
            self.tables_dir, "temporal_testing_tab.csv"), index=False)

        return temporal_testing

    def compute_antibiotic_testing_metrics(self, df, selected_fields, mode="mean", test_columns=None, population_agg_mode="mean"):
        """
        Computes antibiotic testing metrics with robust handling of edge cases.

        Modes:
        - "sum": Total tests per group
        - "mean": Proportion of isolates tested per antibiotic
        - "percentage": Percentage of isolates tested (mean * 100)
        - "coverage": Weighted test coverage percentage
        - "per_100k_population": Normalized per 100,000 pop (needs Bundesland)
        """
        if not selected_fields:
            raise ValueError("selected_fields must be non-empty")

        antibiotic_columns = self._get_antibiotic_columns(df, test_columns)
        df = df.copy()

        if mode == "per_100k_population" and "Date" in df.columns and "Year" not in df.columns:
            df = df.assign(Year=df["Date"].dt.year.astype(str))

        mode_handlers = {
            "sum": self._calculate_sum,
            "mean": self._calculate_mean,
            "percentage": self._calculate_percentage,
            "coverage": self._calculate_coverage,
            "per_100k_population": self._calculate_per_100k
        }

        handler = mode_handlers.get(mode)
        if not handler:
            raise ValueError(
                f"Invalid mode: '{mode}'. Choose from {list(mode_handlers.keys())}")

        result = handler(df, selected_fields,
                         antibiotic_columns, population_agg_mode)
        return result

    # ----------------- Helpers -----------------

    def _get_antibiotic_columns(self, df, test_columns):
        if test_columns:
            if not isinstance(test_columns, list):
                test_columns = [test_columns]
            return [col for col in test_columns if col in df.columns and self._is_antibiotic_column(col)]
        return [col for col in df.columns if self._is_antibiotic_column(col)]

    def _is_antibiotic_column(self, col_name):
        return ' - ' in col_name and f'_{self.suffix}' in col_name

    def _calculate_sum(self, df, group_fields, antibiotic_columns, population_agg_mode=None):
        return self._groupby_operation(df, group_fields, antibiotic_columns, "sum")

    def _calculate_mean(self, df, group_fields, antibiotic_columns, population_agg_mode=None):
        return self._groupby_operation(df, group_fields, antibiotic_columns, "mean")

    def _calculate_percentage(self, df, group_fields, antibiotic_columns, population_agg_mode=None):
        result = self._calculate_mean(df, group_fields, antibiotic_columns)
        result[antibiotic_columns] = result[antibiotic_columns] * 100
        return result

    def _calculate_coverage(self, df, group_fields, antibiotic_columns, population_agg_mode=None):
        isolate_counts = self._groupby_operation(
            df, group_fields, [], "size", "total_isolates")
        test_sums = self._groupby_operation(
            df, group_fields, antibiotic_columns, "sum")
        merged = test_sums.merge(isolate_counts, on=group_fields, how="left")
        merged["total_isolates"] = merged["total_isolates"].replace(0, 1)
        merged[antibiotic_columns] = merged[antibiotic_columns].div(
            merged["total_isolates"], axis=0) * 100
        return merged[group_fields + antibiotic_columns]

    def _calculate_per_100k(self, df, group_fields, antibiotic_columns, population_agg_mode):
        if "Bundesland" not in group_fields:
            raise ValueError(
                "Population normalization requires 'Bundesland' in group_fields")
        if not hasattr(self, "population_data") or self.population_data is None:
            raise ValueError("Population data not loaded or invalid")

        test_sums = self._groupby_operation(
            df, group_fields, antibiotic_columns, "sum")
        pop_df = self._prepare_population_data(
            group_fields, population_agg_mode)
        # pop_df.to_csv(os.path.join(self.tables_dir, "population_data.csv"), index=False)
        merge_keys = pop_df.columns.intersection(group_fields).tolist()
        if not merge_keys:
            raise ValueError(
                "No common keys found between population data and selected fields")
        for k in merge_keys:
            if k.lower() == "year":
                # force numeric Year on both sides
                test_sums[k] = pd.to_numeric(test_sums[k], errors="coerce").astype("Int64")
                pop_df[k]   = pd.to_numeric(pop_df[k],   errors="coerce").astype("Int64")
            else:
                # Make non-year keys comparable; use string (or category) consistently
                test_sums[k] = test_sums[k].astype(str)
                pop_df[k]    = pop_df[k].astype(str)
                
        merged = test_sums.merge(pop_df, on=merge_keys, how="left")
        # merged.to_csv(os.path.join(self.tables_dir, "merged_population_data_with_testing.csv"), index=False)
        population = merged["total"].fillna(
            merged["total"].mean()).replace(0, 1)
        merged[f"{antibiotic_columns}_original"] = merged[antibiotic_columns]
        merged[antibiotic_columns] = merged[antibiotic_columns].div(
            population, axis=0) * 100000
        return merged[group_fields + antibiotic_columns]

    def _groupby_operation(self, df, group_fields, value_columns, operation, result_name=None):
        if not value_columns:
            result = df.groupby(group_fields, observed=False).size()
        else:
            result = df.groupby(group_fields, observed=False)[
                value_columns].agg(operation)
        result = result.reset_index()
        if result_name:
            size_column = result.columns[-1]
            if size_column == 0:
                result = result.rename(columns={0: result_name})
            else:
                result = result.rename(columns={size_column: result_name})
        return result

    def _prepare_population_data(self, selected_fields, agg_mode):
        if not hasattr(self, 'population_data') or self.population_data is None:
            try:
                self.population_data = pd.read_csv(
                    "./datasets/population-data-cleaned.csv")
            except Exception as e:
                raise ValueError(
                    "Population data not loaded and default file not found") from e

        if "bundesland" in self.population_data.columns:
            self.population_data = self.population_data.rename(
                columns={"bundesland": "Bundesland"})

        pop_df = self.population_data.copy()

        group_fields = ["Bundesland"]
        if "Year" in selected_fields:
            group_fields.append("Year")

        if agg_mode == "mean":
            return pop_df.groupby(group_fields, observed=False)["total"].mean().reset_index()
        elif agg_mode == "sum":
            return pop_df.groupby(group_fields, observed=False)["total"].sum().reset_index()
        elif agg_mode == "latest":
            latest_years = pop_df.groupby("Bundesland")[
                "Year"].max().reset_index()
            # latest_years.to_csv(os.path.join(self.tables_dir, "latest_population_years.csv"), index=False)
            pop_df.merge(latest_years, on=["Bundesland", "Year"]).to_csv(os.path.join(
                self.tables_dir, "merged_latest_population_data.csv"), index=False)
            return pop_df.merge(latest_years, on=["Bundesland", "Year"])
        else:
            raise ValueError(f"Invalid population_agg_mode: {agg_mode}")

    def wrap_title(self, text, max_words_per_line=5):
        """Wraps text by inserting newlines after a certain number of words"""
        words = text.split()
        lines = [" ".join(words[i:i + max_words_per_line])
                 for i in range(0, len(words), max_words_per_line)]
        return "\n".join(lines)

    def create_heatmap(self, testing_freq_long, selected_fields, new_combination, heatmap_title, sort_enabled=True):
        """
        Creates a heatmap for antibiotic testing patterns with dynamic sorting.

        Parameters:
            testing_freq_long (pd.DataFrame): Long-format DataFrame for Altair visualization.
            selected_fields (list): Grouping fields used for visualization.
            new_combination (str): Combined label for hierarchical grouping.
            heatmap_title (str): Title of the heatmap.
            sort_enabled (bool): If True, sorts by "Testing Rate", otherwise retains original order.

        Returns:
            alt.Chart: Heatmap visualization.
        """

        # Define sorting logic dynamically
        x_sort = alt.EncodingSortField(
            "Testing Rate", op="sum", order="descending") if sort_enabled else None
        y_sort = alt.EncodingSortField(
            "Testing Rate", op="sum", order="descending") if sort_enabled else None

        # Heatmap
        heatmap = alt.Chart(testing_freq_long).mark_rect().encode(
            x=alt.X("Antibiotic:N", title="Antibiotics", sort=x_sort),
            y=alt.Y(f"{new_combination}:N",
                    title=f"{' → '.join(selected_fields)}", sort=y_sort),
            color=alt.Color("Testing Rate:Q",
                            scale=alt.Scale(scheme="blues", domain=[
                                            0, testing_freq_long["Testing Rate"].max()], type='log'),
                            title="Testing Frequency Per 1,000 Isolates"),
            tooltip=selected_fields +
            ["Antibiotic", alt.Tooltip("Testing Rate:Q", format=".2f")]
        )

        # Text Labels (Ensuring they don't go out of the box)
        text = alt.Chart(testing_freq_long).mark_text(size=10, color="black", fontWeight="bold").encode(
            x=alt.X("Antibiotic:N", sort=x_sort),
            y=alt.Y(f"{new_combination}:N", sort=y_sort),
            text=alt.condition(
                # Only display text if Testing Rate is 0.0
                alt.datum["Testing Rate"] == 0.0,
                alt.Text("Testing Rate:Q", format=".1f"),
                alt.value("")  # Empty text for other values
            ),
            color=alt.condition(
                # If Testing Rate is 0.0, text should be black
                alt.datum["Testing Rate"] == 0.0,
                alt.value("black"),
                alt.value("white")  # Otherwise, text should be white
            )
        )

        # Combine both layers
        layered_heatmap = alt.layer(heatmap, text).properties(
            title=heatmap_title
        )

        return layered_heatmap

    def visualize_antibiotic_testing_patterns(
        self,
        group_list=[],
        group2_list=[],
        selected_fields=["ARS_HospitalLevelManual"],
        min_threshold=0,
        file_name="bundesland_kind",
        mode="mean",  # New parameter: "mean" or "per_1000"
        sort_enabled=True,
        alpha_filter_key=None,
        alpha_filter_value=None
    ):
        """
        Visualizes antibiotic testing patterns while ensuring Bundesland is clearly demarcated (without disrupting ARS_HospitalLevelManual alignment).

        Parameters:
            bundesland_list (list): List of Bundesländer/groups to include.
            min_threshold (float): Minimum mean testing rate required to include an antibiotic.
            selected_fields (list): This cannot be more than 2. List items must be related and must exist in dataframe.

        Returns:
            Altair Charts: Heatmap with Bundesland Demarcation & Stacked Bar Chart.
        """

        filtered_data = self.data.copy()
        if alpha_filter_key and alpha_filter_value:
            filtered_data = filtered_data[filtered_data[alpha_filter_key]
                                          == alpha_filter_value]

        # check if list is more or less than 2, then throw error
        if not len(selected_fields) in [1, 2]:
            print(
                "selected_fields must be of length 1 or 2 and values specified must exist in dataframe.")
            return False

        # Filter dataset for selected Bundesländer/group
        # Filter data based on the presence of group2_list
        if len(selected_fields) == 2:
            # Apply filtering for the first group (Bundesland or equivalent)
            if group_list:
                filtered_data = filtered_data[filtered_data[selected_fields[0]].isin(
                    group_list)]

            # Apply filtering for the second group (ARS_HospitalLevelManual or equivalent)
            if group2_list:
                filtered_data = filtered_data[filtered_data[selected_fields[1]].isin(
                    group2_list)]

            # Ensure second selected field is not null
            filtered_data = filtered_data[filtered_data[selected_fields[1]].notna(
            )]

        else:
            filtered_data = self.data.copy()  # Default to all data

            # Apply filtering for the first group (if specified)
            if group_list:
                filtered_data = filtered_data[filtered_data[selected_fields[0]].isin(
                    group_list)]

        new_combination = self.sanitize_string('_'.join(selected_fields))

        # Select only antibiotic test columns
        antibiotic_columns = [
            col for col in filtered_data.columns if ' - ' in col and f'_{self.suffix}' in col]

        # Compute the testing frequency (mean per antibiotic)
        # testing_freq = filtered_data.groupby(selected_fields)[antibiotic_columns].mean().reset_index()

        # Compute testing frequency using the class method
        testing_freq = self.compute_antibiotic_testing_metrics(
            filtered_data, selected_fields, mode)

        # Remove antibiotics where mean testing rate is 0 or below the threshold
        antibiotics_to_keep = testing_freq[antibiotic_columns].mean(
        ).loc[lambda x: x >= min_threshold].index.tolist()

        # Get all antibiotics that were considered
        all_antibiotics = testing_freq[antibiotic_columns].columns.tolist()

        # Get the antibiotics that were removed (did not meet the threshold)
        # We need to do this to really confirm
        antibiotics_removed = list(
            set(all_antibiotics) - set(antibiotics_to_keep))

        removed_antibiotic_tests = testing_freq[selected_fields +
                                                antibiotics_removed]

        removed_antibiotic_tests.to_csv(os.path.join(
            self.tables_dir, f"{new_combination}_removed_antibiograms_tab.csv"), index=False)

        # If all antibiotics are filtered out, return an error message
        if not antibiotics_to_keep:
            raise ValueError(
                "No antibiotics meet the minimum threshold. Try lowering the min_threshold value.")

        # Keep only selected antibiotics
        testing_freq = testing_freq[selected_fields + antibiotics_to_keep]

        testing_freq.to_csv(os.path.join(
            self.tables_dir, f"{new_combination}_best_antibiograms_tab.csv"), index=False)
        # Convert to long format for Altair
        testing_freq_long = testing_freq.melt(
            id_vars=selected_fields, var_name="Antibiotic", value_name="Testing Rate")
        testing_freq_long["Antibiotic"] = testing_freq_long["Antibiotic"].str.replace(
            f"_{self.suffix}", "", regex=True)

        # **Fix Hierarchical Labeling**: Use Region → ARS_HospitalLevelManual
        # as a single label. I need to get alternative for this later.
        if len(selected_fields) == 2:
            # testing_freq_long[new_combination] = testing_freq_long[selected_fields[0]] + " → " + testing_freq_long[selected_fields[1]]
            testing_freq_long[new_combination] = testing_freq_long[selected_fields[1]]
        else:
            testing_freq_long[new_combination] = testing_freq_long[selected_fields[0]]

        # **Ensure Correct Sorting**
        column_data = selected_fields + ["Testing Rate"]
        # testing_freq_long = testing_freq_long.sort_values(by=column_data, ascending=[True, True, False])
        testing_freq_long = testing_freq_long.sort_values(
            by=column_data, ascending=[True] * len(selected_fields) + [False])

        long_title = f"Antibiotic Testing Patterns by {', '.join(selected_fields)}"
        heatmap_title = self.wrap_title(long_title, max_words_per_line=8)

        heatmap = self.create_heatmap(
            testing_freq_long, selected_fields,
            new_combination, heatmap_title,
            sort_enabled=sort_enabled
        )

        if len(selected_fields) == 2:
            # --- ADDING WHITE LINES TO SEPARATE ONLY BUNDESLAND ---
            group_separators = (
                alt.Chart(testing_freq_long)
                .mark_rule(strokeWidth=3, color="white")
                .encode(
                    # Regular title
                    y=alt.Y(f"{selected_fields[0]}:N",
                            title=f"{selected_fields[0]}"),
                    size=alt.value(4)
                )
            )

            # **Layer heatmap & bundesland separators properly**
            layered_heatmap = alt.layer(heatmap, group_separators).configure_axis(
                titleFontWeight="bold"
            )
        else:
            layered_heatmap = heatmap

        # --- BAR CHART ---
        bar_chart = alt.Chart(testing_freq_long).mark_bar().encode(
            x=alt.X("Testing Rate:Q", title="Proportion of Samples Tested"),
            y=alt.Y("Antibiotic:N", title="Antibiotics", sort="-x"),
            color=alt.Color(
                f"{selected_fields[0]}:N", title=f"{selected_fields[0]}"),
            # Ensures 4 Bundesländer per row
            facet=alt.Facet(
                f"{selected_fields[0]}:N", title=f"{selected_fields[0]}", columns=4),
            tooltip=selected_fields +
            ["Antibiotic", alt.Tooltip("Testing Rate:Q", format=".2f")]
        ).properties(
            title=f"Antibiotic Testing Distribution in {', '.join(group_list)} (Min Rate ≥ {min_threshold})",
            width=250,  # Adjust width to fit more antibiotics
            height=500
        )

        bar_chart.save(os.path.join(self.charts_dir,
                       f"{self.sanitize_string(file_name)}_bar_chart.svg"))
        layered_heatmap.save(os.path.join(
            self.charts_dir, f"{self.sanitize_string(file_name)}_heatmap.svg"))
        layered_heatmap.save(os.path.join(
            self.charts_dir, f"{self.sanitize_string(file_name)}_heatmap.png"))
        return layered_heatmap, bar_chart

    def load_geojson(self, url):
        # Load GeoJSON once and cache it
        global GEOJSON_CACHE, CENTROID_CACHE
        GEOJSON_CACHE = None
        CENTROID_CACHE = None
        if GEOJSON_CACHE is None:
            gdf = gpd.read_file(url)
            gdf['Bundesland'] = gdf['name'].str.strip()

            # Precompute centroids with special handling for Berlin and Brandenburg
            centroids = []
            for idx, row in gdf.iterrows():
                geom = row['geometry']
                centroid = geom.centroid
                state = row['Bundesland']

                # Adjust positions to prevent label overlap
                if state == "Berlin":
                    # Shift Berlin northeast
                    centroid = Point(centroid.x + 0.25, centroid.y + 0.15)
                if state == "Brandenburg":
                    # Shift Brandenburg slightly southwest
                    centroid = Point(centroid.x - 0.1, centroid.y - 0.5)

                centroids.append({
                    "Bundesland": state,
                    "longitude": centroid.x,
                    "latitude": centroid.y
                })
            CENTROID_CACHE = pd.DataFrame(centroids)
            GEOJSON_CACHE = gdf
        return GEOJSON_CACHE, CENTROID_CACHE

    def map_testing_rates(self, target_antibiotics, mode="per_1000", df=None, file_name_prefix="default", num_columns=6, group_by_cols=["Year"]):
        # Load cached GeoJSON data
        url = "./datasets/1_sehr_hoch.geo.json"
        base_gdf, centroids_df = self.load_geojson(url)

        # Calculate testing rates
        testing_rates = self.calculate_testing_rates(
            df=df,
            group_by_cols=list(set(["Bundesland"] + group_by_cols)),
            target_antibiotics=target_antibiotics,
            mode=mode
        )
        testing_rates['Bundesland'] = testing_rates['Bundesland'].str.strip()
        # testing_rates.to_csv("./testing_rates.csv", index=False)
        self.save_table(
            testing_rates, f"{self.sanitize_string(file_name_prefix)}_maps_data")
        # Precompute min/max for color scaling
        min_val = testing_rates["Testing Rate"].min()
        max_val = testing_rates["Testing Rate"].max()
        domain_values = [min_val + (max_val - min_val) * i / 5 for i in range(
            6)] if max_val - min_val >= 1e-6 else [min_val] * 6

        color_scale = alt.Scale(
            domain=domain_values,
            range=['#e3ecf9', '#c1d4f4', '#99bbea',
                   '#6ca2dd', '#4a86c5', '#2c68af']
        )

        # Create antibiotic feature dictionary
        features_dict = {}
        for antibiotic in testing_rates["Antibiotic"].unique():
            ab_df = testing_rates[testing_rates["Antibiotic"] == antibiotic]
            merged = base_gdf.merge(ab_df, on="Bundesland", how="left")
            merged['Testing Rate'] = merged['Testing Rate'].fillna(0)
            features_dict[antibiotic] = json.loads(
                merged.to_json())["features"]

        charts = []
        for antibiotic in sorted(testing_rates["Antibiotic"].unique()):
            antibiotic_features = features_dict.get(antibiotic, [])
            if not antibiotic_features:
                print(f"No data for antibiotic {antibiotic}")
                continue

            antibiotic_name = antibiotic.replace(f'_{self.suffix}', '')

            # Create centroids DF with values
            ab_centroids = centroids_df.merge(
                testing_rates[testing_rates["Antibiotic"] == antibiotic],
                on="Bundesland",
                how="left"
            ).fillna(0)
            ab_centroids["Value"] = ab_centroids["Testing Rate"]

            # Create base map without any configurations
            base_map = alt.Chart(alt.Data(values=antibiotic_features)).mark_geoshape(
                stroke="white",
                strokeWidth=0.5
            ).encode(
                color=alt.Color(
                    "properties.Testing Rate:Q",
                    scale=color_scale,
                    title="Testing Rate"
                ),
                tooltip=[
                    alt.Tooltip("properties.Bundesland:N", title="State"),
                    alt.Tooltip("properties.Testing Rate:Q",
                                title="Rate", format=".1f"),
                    alt.Tooltip("properties.Antibiotic:N", title="Antibiotic")
                ]
            ).properties(
                width=600,
                height=500,
                title=antibiotic_name
            )

            text_layer = alt.Chart(ab_centroids).mark_text(
                align='center',
                fontSize=10,
                dy=-5,
                fontWeight="bold",
                color='black'
            ).encode(
                longitude='longitude:Q',
                latitude='latitude:Q',
                text=alt.Text('Value:Q', format='.1f')
            )

            charts.append(base_map + text_layer)

        if not charts:
            print("No valid data to display.")
            return None

        num_rows = math.ceil(len(charts) / num_columns)
        rows = [
            alt.hconcat(*charts[i:i+num_columns],
                        spacing=0).resolve_scale(color='shared')
            for i in range(0, len(charts), num_columns)
        ]

        # Final vertically stacked chart with minimal spacing
        final_chart = alt.vconcat(*rows).properties(
            title={
                "text": f"{file_name_prefix} - Antibiotic Testing Rates ({mode})",
                "align": "center",
                "fontSize": 14
            }, spacing=0
        ).configure_title(
            fontSize=12,
            anchor='middle'
        ).configure_legend(
            gradientLength=200,
            gradientThickness=15
        )

        return final_chart

    # useful for other
    def plot_test_rates_grid(
        self, df,
        x_col='PathogenGenus',
        facet_col='Antibiotic',
        value_col='Testing Rate',
        time_col='Year',
        selected_x=None,
        num_columns=4,
        width=250,
        height=300,
        title=None
    ):
        """
        General Altair grid chart for test rate-like data.

        Parameters:
        - df: DataFrame
        - x_col: variable for x-axis (categorical, e.g., 'PathogenGenus', 'Bundesland')
        - facet_col: variable for faceting (e.g., 'Antibiotic')
        - value_col: variable for y-axis (e.g., 'Testing Rate')
        - time_col: optional time variable for colored lines (e.g., 'Year')
        - selected_x: list of x_col values to include (ordered), optional
        - num_columns: number of facet columns
        - width, height: per-facet dimensions
        - title: chart title (auto-generated if None)
        """
        required_cols = {x_col, facet_col, value_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain at least: {required_cols}")

        has_time = time_col in df.columns

        df = df.copy()

        # Filter x values
        if selected_x is not None:
            df = df[df[x_col].isin(selected_x)]
            x_order = selected_x
        else:
            # Use average value_col to order x_col
            x_order = (
                df.groupby(x_col)[value_col]
                .mean()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

        # Ensure ordered categorical x
        df[x_col] = pd.Categorical(df[x_col], categories=x_order, ordered=True)

        # Encodings
        encodings = {
            'x': alt.X(f'{x_col}:N', sort=x_order, title=x_col.replace('_', ' ')),
            'y': alt.Y(f'{value_col}:Q', title=value_col.replace('_', ' ')),
            'tooltip': [x_col, facet_col, value_col]
        }

        if has_time:
            years = sorted(df[time_col].dropna().astype(str).unique().tolist())
            df[time_col] = df[time_col].astype(str)
            encodings['color'] = alt.Color(f'{time_col}:N', title=time_col)
            encodings['strokeDash'] = alt.StrokeDash(
                f'{time_col}:N', title=time_col)
            encodings['shape'] = alt.Shape(f'{time_col}:N', title=time_col)
            encodings['tooltip'].insert(0, time_col)

        base = alt.Chart(df).encode(
            **encodings).properties(width=width, height=height)

        chart = (base.mark_line() + base.mark_point(filled=True, size=80)).facet(
            facet=alt.Facet(f'{facet_col}:N',
                            title=facet_col.replace('_', ' ')),
            columns=num_columns
        ).configure_facet(
            spacing=40
        ).properties(
            title=title or f'{value_col} by {x_col}' +
            (f' and {time_col}' if has_time else ''),
            bounds='flush'
        ).configure_title(
            fontSize=20,
            anchor='middle',
            offset=30
        ).configure_axisX(
            labelAngle=45
        ).configure_axis(
            labelFontSize=14,
            titleFontSize=16
        ).configure_legend(
            labelFontSize=13,
            titleFontSize=14
        ).configure_header(
            titleFontSize=16,
            labelFontSize=14
        ).configure_view(
            stroke=None
        )

        return chart

    # useful for map
    def plot_antibiotic_test_rates_grid(self, df, num_columns=4, width=250, height=300):
        """
        Faceted Altair chart showing antibiotic test rates across Bundesländer, colored by Year.
        States with similar (high or low) testing rates are clustered visually on the x-axis.
        """
        required_cols = {'Antibiotic', 'Bundesland', 'Testing Rate'}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain at least: {required_cols}")

        has_year = 'Year' in df.columns

        # 1. Compute overall average testing rate per Bundesland
        mean_by_state = (
            df.groupby('Bundesland')['Testing Rate']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        clustered_order = mean_by_state['Bundesland'].tolist()

        # 2. Add ordering as category to preserve order in plotting
        df['Bundesland'] = pd.Categorical(
            df['Bundesland'], categories=clustered_order, ordered=True)

        # 3. Encoding
        encodings = {
            'x': alt.X('Bundesland:N', sort=clustered_order, title='Bundesland'),
            'y': alt.Y('Testing Rate:Q', title='Testing Rate'),
            'tooltip': ['Bundesland', 'Antibiotic', 'Testing Rate']
        }

        if has_year:
            years = sorted(df['Year'].unique().tolist())
            encodings['color'] = alt.Color(
                'Year:N', title='Year', scale=alt.Scale(scheme='tableau10'))
            encodings['strokeDash'] = alt.StrokeDash('Year:N', title='Year')
            encodings['shape'] = alt.Shape('Year:N', title='Year',
                                           scale=alt.Scale(domain=years, range=['circle', 'square', 'triangle', 'cross', 'diamond', 'star']))
            encodings['tooltip'].insert(0, 'Year')

        base = alt.Chart(df).encode(
            **encodings).properties(width=width, height=height)

        chart = (base.mark_line() + base.mark_point(filled=True, size=80)).facet(
            facet=alt.Facet('Antibiotic:N', title='Antibiotic'),
            columns=num_columns
        ).configure_facet(
            spacing=40
        ).properties(
            title='Antibiotic Test Rates by Bundesland' +
            (' and Year' if has_year else ''),
            bounds='flush'
        ).configure_title(
            fontSize=20,
            anchor='middle',
            offset=30
        ).configure_axisX(
            labelAngle=45
        ).configure_axis(
            labelFontSize=14,
            titleFontSize=16
        ).configure_legend(
            labelFontSize=13,
            titleFontSize=14
        ).configure_header(
            titleFontSize=16,
            labelFontSize=14
        ).configure_view(
            stroke=None
        )

        return chart

    def visualize_antibiotic_testing_patterns_general(
        self,
        chart_type="clustergram",
        selected_fields=["ARS_HospitalLevelManual"],
        group_list=None,
        group2_list=None,
        min_threshold=0,
        file_name_prefix="antibiotic_visualization",
        save_as_image=True,
        save_as_html=True,
        image_format="svg",
        mode="mean",
        quantiles_x=None,
        filters=None,
        prefilter=None,
        specific_antibiotics=None,
        cluster=True,
        width=1900,
        height=1700,
        map_num_columns=6
    ):
        """
        Generalized function for visualizing antibiotic testing patterns.

        Parameters:
            chart_type (str): One of ['clustergram', 'polar', 'circus', 'parallel', 'sunburst'].
            selected_fields (list): One or two grouping fields (e.g., ["region", "ARS_HospitalLevelManual"]).
            quantiles_x (list): Optional quantiles for column grouping (antibiotics).
            filters (list of tuples): List of filtering conditions, e.g. [("Bundesland", ["Bavaria", "Berlin"], "!=")].
            specific_antibiotics (list): Specific antibiotics to include in the visualization.
        """

        # Apply Data Filters
        data = self.data.copy()

        # `prefilter` if provided
        if callable(prefilter):
            try:
                data = prefilter(data)
            except Exception as e:
                print(f"[ERROR] Failed to apply prefilter: {e}")

        if filters:
            for filter_tuple in filters:
                if len(filter_tuple) == 2:
                    key, value = filter_tuple
                    operator = "=="  # Default to equality
                elif len(filter_tuple) == 3:
                    key, value, operator = filter_tuple
                else:
                    print(
                        f"[WARNING] Invalid filter format: {filter_tuple}. Skipping.")
                    continue

                if key in data.columns:
                    if operator == "==":
                        if isinstance(value, list):
                            data = data[data[key].isin(value)]
                        else:
                            data = data[data[key] == value]
                    elif operator == "!=":
                        if isinstance(value, list):
                            data = data[~data[key].isin(value)]
                        else:
                            data = data[data[key] != value]
                    else:
                        print(
                            f"[WARNING] Unsupported operator '{operator}' in filter {filter_tuple}. Skipping.")
                else:
                    print(
                        f"[WARNING] Filter key '{key}' not found. Skipping filter.")

        for field in selected_fields:
            if field not in data.columns:
                raise KeyError(f"[ERROR] Selected field '{field}' not found.")

        if group_list:
            data = data[data[selected_fields[0]].isin(group_list)]
        if len(selected_fields) > 1 and group2_list:
            data = data[data[selected_fields[1]].isin(group2_list)]

        data = data.dropna(subset=selected_fields)

        # Identify Antibiotic Columns
        antibiotic_columns = [
            col for col in data.columns if ' - ' in col and f'_{self.suffix}' in col]

        if specific_antibiotics:
            antibiotic_columns = [
                col for col in antibiotic_columns if col in specific_antibiotics]

        # Compute Testing Frequency
        testing_freq = self.compute_antibiotic_testing_metrics(
            data, selected_fields, mode)
        # 1. Columns whose sum is <= 0
        low_sum_abxs = [
            abx
            for abx in antibiotic_columns
            if testing_freq[abx].sum() <= 0
        ]

        # Save those columns (and their data) to CSV
        # testing_freq[low_sum_abxs].to_csv('abx_sum_le_0.csv', index=False)
        testing_freq[low_sum_abxs].to_csv(os.path.join(
            self.tables_dir, f"{file_name_prefix}_low_sum_abxs.csv"), index=False)

        valid_abxs = [
            abx for abx in antibiotic_columns
            if testing_freq[abx].sum() > 0
        ]

        # Apply min_threshold Filtering
        antibiotic_means = testing_freq[valid_abxs].mean()

        antibiotics_to_keep = antibiotic_means[antibiotic_means >= min_threshold].index.tolist(
        )

        if not antibiotics_to_keep:
            raise ValueError(
                "[ERROR] No antibiotics meet the min_threshold. Lower the value.")

        testing_freq = testing_freq[selected_fields + antibiotics_to_keep]
        testing_freq.to_csv(os.path.join(
            self.tables_dir, f"{file_name_prefix}_best_antibiograms_tab.csv"), index=False)
        # Compute Quantiles for Categorization
        use_quantiles_x = quantiles_x is not None
        mean_values_x = testing_freq[antibiotics_to_keep].mean(
        ) if use_quantiles_x else None
        quantiles_x_values = mean_values_x.quantile(
            quantiles_x) if use_quantiles_x else None

        # Categorize Antibiotics by Testing Levels
        if use_quantiles_x:
            antibiotic_groups = {
                "Critically_Tested": mean_values_x[mean_values_x >= quantiles_x_values.iloc[3]].index.tolist(),
                "Highly_Tested": mean_values_x[(mean_values_x < quantiles_x_values.iloc[3]) & (mean_values_x >= quantiles_x_values.iloc[2])].index.tolist(),
                "Mediumly_Tested": mean_values_x[(mean_values_x < quantiles_x_values.iloc[2]) & (mean_values_x >= quantiles_x_values.iloc[1])].index.tolist(),
                "Lowly_Tested": mean_values_x[mean_values_x < quantiles_x_values.iloc[1]].index.tolist()
            }
        else:
            antibiotic_groups = {
                "Overall": antibiotics_to_keep
            }

        fig = None  # Ensure fig is always initialized
        # Load GeoJSON data for Germany
        url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/refs/heads/main/2_bundeslaender/1_sehr_hoch.geo.json"

        # Determine the global color scale range
        global_min = testing_freq[antibiotics_to_keep].values.min()
        global_max = testing_freq[antibiotics_to_keep].values.max()

        # Sort Antibiotics by Mean Testing Frequency (Highest to Lowest)
        antibiotic_means_sorted = testing_freq[antibiotics_to_keep].mean(
        ).sort_values(ascending=False)
        sorted_antibiotics = antibiotic_means_sorted.index.tolist()

        # Generate Visualization
        for group_name, antibiotics in antibiotic_groups.items():
            if not antibiotics:
                continue

            loc_group_data_filtered = testing_freq[selected_fields + antibiotics]
            # --- Clustergram ---
            if chart_type == "clustergram":
                # Handle hierarchical labeling for y-axis
                if len(selected_fields) == 2:
                    loc_group_data_filtered["combined_label"] = loc_group_data_filtered[selected_fields[0]
                                                                                        ] + " -> " + loc_group_data_filtered[selected_fields[1]]
                    y_labels = loc_group_data_filtered["combined_label"].values.tolist(
                    )
                else:
                    y_labels = loc_group_data_filtered[selected_fields[0]].values.tolist(
                    )

                if use_quantiles_x:
                    antibiotic_means_group_sorted = loc_group_data_filtered[antibiotics].mean(
                    ).sort_values(ascending=False)
                    sorted_group_antibiotics = antibiotic_means_group_sorted.index.tolist()
                    sorted_data = loc_group_data_filtered[sorted_group_antibiotics].values
                    column_label = sorted_group_antibiotics
                else:
                    sorted_data = loc_group_data_filtered[sorted_antibiotics].values
                    column_label = sorted_antibiotics

                fig = dash_bio.Clustergram(
                    data=sorted_data,
                    row_labels=y_labels,
                    column_labels=column_label,
                    color_map=[
                        [0.0, '#d5def0'],  # Lightest color
                        [0.25, '#aabce1'],
                        [0.5, '#7b9dcf'],
                        [0.75, '#4580c0'],
                        [1.0, '#2c68af']   # Darkest color
                    ],
                    line_width=2,
                    height=height,
                    width=width,
                    cluster=cluster,
                    tick_font={'size': 10},
                    color_threshold={'row': 0, 'col': 0},
                    color_list={'row': ['#08306b'], 'col': ['#08306b']},
                    center_values=False,            # Center data around zero
                    row_dist='euclidean',
                    col_dist='correlation',
                )
                fig.layout.update(
                    title=dict(
                        text=f"Antibiotic Testing Clustergram by {group_name}",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=20)
                    )
                )
                # fix colorbar
                heat = fig.data[-1]
                heat.zmin, heat.zmax = global_min, global_max
                # Handle hierarchical labeling for y-axis

            # --- Circus Plot ---
            elif chart_type == "circus":
                df_melted = loc_group_data_filtered.melt(
                    id_vars=selected_fields, value_vars=antibiotics)

                # Check for empty dataset
                if df_melted.empty:
                    print(
                        f"[WARNING] No valid data for {group_name}. Skipping circus plot.")
                    continue

                # Assign Angles to Categories
                unique_categories = df_melted[selected_fields[-1]].unique()
                df_melted["Angle"] = df_melted[selected_fields[-1]].apply(
                    lambda x: (np.where(unique_categories == x)[
                               0][0]) * (360 / len(unique_categories))
                )

                # Normalize Values
                df_melted["Normalized Rate"] = df_melted["value"] / \
                    df_melted["value"].max()

                # Create the Figure
                fig = go.Figure()
                for antibiotic in df_melted["variable"].unique():
                    df_subset = df_melted[df_melted["variable"] == antibiotic]

                    fig.add_trace(go.Barpolar(
                        r=df_subset["Normalized Rate"],
                        theta=df_subset["Angle"],
                        name=antibiotic,
                        marker=dict(color=np.random.choice(
                            px.colors.qualitative.Set3)),
                        opacity=0.7
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(showticklabels=True)),
                    title=f"Circus Plot - {group_name}",
                    showlegend=True
                )

            elif chart_type == "polar":
                try:
                    df_melted = loc_group_data_filtered.melt(
                        id_vars=selected_fields, value_vars=antibiotics)

                    # Check if DataFrame is empty
                    if df_melted.empty:
                        print(
                            f"[WARNING] No valid data for {group_name}. Skipping polar chart.")
                        continue  # Skip this group

                    df_melted["Normalized Value"] = df_melted["value"] / \
                        df_melted["value"].max()

                    fig = px.line_polar(
                        df_melted,
                        r="Normalized Value",
                        theta="variable",  # Category
                        color=selected_fields[-1],
                        line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        template="plotly_dark",
                        title=f"Polar Line Chart - {group_name}"
                    )
                except Exception as e:
                    print(e)
                    print("Problem encountered. Revisit Implementation later.")

            # --- Sunburst Chart ---
            elif chart_type == "sunburst":
                fig = px.sunburst(
                    loc_group_data_filtered.melt(
                        id_vars=selected_fields, value_vars=antibiotics),
                    path=selected_fields + ["variable"],
                    values="value",
                    title=f"Sunburst Chart - {group_name}",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

            # --- Parallel Coordinates ---
            elif chart_type == "parallel":
                fig = px.parallel_categories(
                    loc_group_data_filtered.melt(
                        id_vars=selected_fields, value_vars=antibiotics),
                    dimensions=selected_fields + ["variable"],
                    color="value",
                    color_continuous_scale=px.colors.sequential.Blues,
                    title=f"Parallel Coordinates - {group_name}"
                )

            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # Save Files
            if fig:
                fig.update_layout(
                    width=width,
                    height=height,
                    xaxis=dict(
                        tickangle=45,
                        # Reduce font size for better readability
                        tickfont=dict(size=8)
                    ),
                    coloraxis=dict(
                        cmin=global_min,
                        cmax=global_max,
                        colorbar=dict(
                            title="Testing Frequency",
                            orientation='h',
                            yanchor="bottom",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                            thicknessmode="pixels",
                            thickness=15,
                            lenmode="fraction",
                            len=0.6,
                            tickmode="array",
                            # Ensure tick marks align with colors
                            tickvals=[0, 0.25, 0.5, 0.75, 1],
                        )
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                os.makedirs("../datasets/output/charts", exist_ok=True)
                file_name = f"{file_name_prefix}_{group_name}"

                if save_as_html:
                    pio.write_html(fig, os.path.join(
                        self.charts_dir, f"{self.sanitize_string(file_name)}.html"))

                if save_as_image:
                    pio.write_image(fig, os.path.join(
                        self.charts_dir, f"{self.sanitize_string(file_name)}.{image_format}"), scale=3)

                print(
                    f"[INFO] {chart_type} saved as {file_name} (HTML + {image_format})")
                return fig

    def visualize_map(self, df=None, target_antibiotics=[], file_name="map_store", mode="per_1000", file_name_prefix="", num_columns=6, group_by_cols=["Year"]):
        """
        Display the interactive Altair maps for multiple antibiotics.

        Parameters:
            target_antibiotics (list): List of antibiotics to analyze.
        """
        chart = self.map_testing_rates(target_antibiotics=target_antibiotics, mode=mode, df=df,
                                       file_name_prefix=file_name_prefix, num_columns=num_columns, group_by_cols=group_by_cols)
        
        chart.save(os.path.join(self.charts_dir,
                   f"{self.sanitize_string(file_name)}_maps.svg"))
        chart.save(os.path.join(self.charts_dir,
                   f"{self.sanitize_string(file_name)}_maps.html"))

        return chart

    def save_table(self, df, file_name):
        safe_name = self.sanitize_string(file_name)
        path = os.path.join(self.tables_dir, f"{safe_name}.csv")
        df.to_csv(path, index=False)
        print(f"[INFO] Table saved: {path}")

    def calculate_tests_per_100k(self, amr_file, population_file, output_file=None, verbose=True):
        amr_df = pd.read_csv(amr_file)
        pop_df = pd.read_csv(population_file)

        tested_cols = [
            col for col in amr_df.columns if col.endswith('_Tested')]
        amr_df['total_tests_row'] = amr_df[tested_cols].sum(axis=1)
        amr_df = amr_df[amr_df['total_tests_row'] > 0].copy()

        grouped = amr_df.groupby(['Bundesland', 'Year'])[
            tested_cols].sum().reset_index()

        long_df = grouped.melt(
            id_vars=['Bundesland', 'Year'],
            var_name='Antibiotic',
            value_name='NumTests'
        )
        long_df['Antibiotic'] = long_df['Antibiotic'].str.replace(
            '_Tested', '', regex=False)

        merged = long_df.merge(
            pop_df,
            how='left',
            left_on=['Bundesland', 'Year'],
            right_on=['bundesland', 'Year']
        )
        merged['Tests_per_100k'] = (
            merged['NumTests'] / merged['total']) * 100000

        final_df = merged[['Bundesland', 'Year', 'Antibiotic',
                           'NumTests', 'total', 'Tests_per_100k']]

        if output_file:
            self.save_table(final_df, output_file)

        return final_df

    def calculate_tests_per_100k_by_sex(self, amr_file, population_file, output_file=None, verbose=True):
        amr_df = pd.read_csv(amr_file)
        pop_df = pd.read_csv(population_file)

        tested_cols = [
            col for col in amr_df.columns if col.endswith('_Tested')]
        amr_df['total_tests_row'] = amr_df[tested_cols].sum(axis=1)
        amr_df = amr_df[amr_df['total_tests_row'] > 0].copy()
        amr_df = amr_df[amr_df['Sex'].isin(['Man', 'Woman'])]

        grouped = amr_df.groupby(['Bundesland', 'Year', 'Sex'])[
            tested_cols].sum().reset_index()

        long_df = grouped.melt(
            id_vars=['Bundesland', 'Year', 'Sex'],
            var_name='Antibiotic',
            value_name='NumTests'
        )
        long_df['Antibiotic'] = long_df['Antibiotic'].str.replace(
            '_Tested', '', regex=False)

        sex_pop_col = {'Man': 'male_count', 'Woman': 'female_count'}
        long_df['PopColumn'] = long_df['Sex'].map(sex_pop_col)

        merged = long_df.merge(
            pop_df,
            how='left',
            left_on=['Bundesland', 'Year'],
            right_on=['bundesland', 'Year']
        )
        merged['Population'] = merged.apply(
            lambda row: row[row['PopColumn']], axis=1)
        merged['Tests_per_100k'] = (
            merged['NumTests'] / merged['Population']) * 100000

        final_df = merged[['Bundesland', 'Year', 'Sex',
                           'Antibiotic', 'NumTests', 'Population', 'Tests_per_100k']]

        if output_file:
            self.save_table(final_df, output_file)

        return final_df

    def visualize_chart_grid_data(
        self,
        mode="per_100k_population",
        target_antibiotics=[],
        filters=None,
        prefilter=None,
        file_name="map_grid",
        file_name_prefix="",
        num_columns=6,
        group_by_cols=["Year"],
        obj_type="map"
    ):
        # Apply Data Filters
        data = self.data.copy()

        # `prefilter` if provided
        if callable(prefilter):
            try:
                data = prefilter(data)
            except Exception as e:
                print(f"[ERROR] Failed to apply prefilter: {e}")

        if filters:
            for filter_tuple in filters:
                if len(filter_tuple) == 2:
                    key, value = filter_tuple
                    operator = "=="  # Default to equality
                elif len(filter_tuple) == 3:
                    key, value, operator = filter_tuple
                else:
                    print(
                        f"[WARNING] Invalid filter format: {filter_tuple}. Skipping.")
                    continue

                if key in data.columns:
                    if operator == "==":
                        if isinstance(value, list):
                            data = data[data[key].isin(value)]
                        else:
                            data = data[data[key] == value]
                    elif operator == "!=":
                        if isinstance(value, list):
                            data = data[~data[key].isin(value)]
                        else:
                            data = data[data[key] != value]
                    else:
                        print(
                            f"[WARNING] Unsupported operator '{operator}' in filter {filter_tuple}. Skipping.")
                else:
                    print(
                        f"[WARNING] Filter key '{key}' not found. Skipping filter.")

        # Identify Antibiotic Columns
        antibiotic_columns = [
            col for col in data.columns if ' - ' in col and f'_{self.suffix}' in col]

        if target_antibiotics and len(target_antibiotics) > 0:
            antibiotic_columns = [
                col for col in antibiotic_columns if col in target_antibiotics]

        if obj_type == "map":
            print("we are in map")
            chart = self.visualize_map(df=data, target_antibiotics=antibiotic_columns, file_name=file_name,
                                       mode=mode, file_name_prefix=file_name_prefix, num_columns=num_columns, group_by_cols=group_by_cols)
            return chart
        else:
            # 4. Compute aggregated test metrics
            rate = self.compute_antibiotic_testing_metrics(
                df=data,
                selected_fields=group_by_cols,
                mode=mode
            )

            # 5. Reshape to long format
            df_long = rate.melt(
                id_vars=group_by_cols,
                var_name="Antibiotic",
                value_name="Testing Rate"
            )
            df_long["Antibiotic"] = df_long["Antibiotic"].str.replace(
                "_Tested", "", regex=False)

            # 6. Clean and convert
            df_long = df_long.dropna(
                subset=["Testing Rate", "Antibiotic"] + group_by_cols)
            df_long["Testing Rate"] = pd.to_numeric(
                df_long["Testing Rate"], errors="coerce")
            df_long["Antibiotic"] = df_long["Antibiotic"].astype(str)

            for col in group_by_cols:
                df_long[col] = df_long[col].astype(str)
            return df_long

    def display_top_antibiotics(self, final_df, top_n=20, file_name="top_antibiotics"):
        top_antibiotics = (
            final_df.groupby('Antibiotic')['NumTests']
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )
        df_plot = final_df[final_df['Antibiotic'].isin(top_antibiotics)].copy()

        ranked = (
            df_plot
            .groupby(['Year', 'Bundesland', 'Antibiotic'])['Tests_per_100k']
            .mean()
            .reset_index()
            .sort_values(['Year', 'Bundesland', 'Tests_per_100k'], ascending=[True, True, False])
        )
        ranked['Rank'] = ranked.groupby(['Year', 'Bundesland']).cumcount() + 1

        df_plot = df_plot.merge(
            ranked[['Year', 'Bundesland', 'Antibiotic', 'Rank']],
            on=['Year', 'Bundesland', 'Antibiotic'],
            how='left'
        )

        df_plot = df_plot[df_plot['Bundesland'].str.lower() != 'unknown']
        df_plot['Tests_per_100k'] = df_plot['Tests_per_100k'].replace(0, 0.1)

        chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X('Antibiotic:N', axis=alt.Axis(labelAngle=90)),
            y=alt.Y('Tests_per_100k:Q',
                    title='Tests per 100K (log scale)',
                    scale=alt.Scale(type='log')),
            color=alt.Color('Sex:N', title='Gender'),
            tooltip=['Rank', 'Antibiotic', 'Tests_per_100k',
                     'Sex', 'Bundesland', 'Year']
        ).properties(
            width=200,
            height=200
        ).facet(
            row=alt.Row('Year:N', title='Year'),
            column=alt.Column('Bundesland:N', title='Bundesland')
        ).resolve_scale(
            y='shared'
        )

        # Save chart
        safe_name = self.sanitize_string(file_name)
        chart.save(os.path.join(self.charts_dir, f"{safe_name}_line.svg"))
        chart.save(os.path.join(self.charts_dir, f"{safe_name}_line.html"))

        return chart
