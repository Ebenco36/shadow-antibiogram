# pip install plotly pandas
# (Optional for PNG export) pip install -U kaleido

from __future__ import annotations
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import re, os

def _pretty(label: str) -> str:
    # Drop the trailing "_Tested" and any extra spaces
    return re.sub(r"\s*_Tested\s*$", "", label)

class TemporalAntibioticLineChartPlotly:
    """
    Build interactive monthly temporal line charts from antibiotic *_Tested columns.
    - Flexible: choose any subset of antibiotic columns (validated)
    - Robust: handles missing columns/months; sorts chronologically
    - Output: returns a Plotly Figure; can export to HTML/PNG/PDF and CSV
    """

    def __init__(
        self, df: pd.DataFrame,
        year_col: str = "Year",
        month_col: str = "Month",
        color_map=None
    ):
        if year_col not in df.columns:
            raise ValueError(f"Missing required column: {year_col}")
        if month_col not in df.columns:
            raise ValueError(f"Missing required column: {month_col}")

        self.df = df.copy()
        # Normalize year/month to integers and drop invalids
        self.df[year_col]  = pd.to_numeric(self.df[year_col], errors="coerce").astype("Int64")
        self.df[month_col] = pd.to_numeric(self.df[month_col], errors="coerce").astype("Int64")
        self.df = self.df.dropna(subset=[year_col, month_col])
        self.df[year_col]  = self.df[year_col].astype(int)
        self.df[month_col] = self.df[month_col].astype(int)

        # Create a monthly date column for proper Plotly time axis
        self.df["__date"] = pd.to_datetime(dict(
            year=self.df[year_col], month=self.df[month_col], day=1
        ))

        # Detect antibiotic columns
        self.antibiotic_cols = [c for c in self.df.columns if c.endswith("_Tested")]
        if not self.antibiotic_cols:
            raise ValueError("No antibiotic *_Tested columns found.")
        self.year_col = year_col
        self.month_col = month_col
        self.color_map = color_map or {}

    def _get_color(self, pretty, idx):
        if isinstance(self.color_map, dict):
            return self.color_map.get(pretty, None)
        elif isinstance(self.color_map, (list, tuple)):
            if idx < len(self.color_map):
                return self.color_map[idx]
        return None
    
    def validate_antibiotics(self, antibiotics: list[str]) -> list[str]:
        missing = [a for a in antibiotics if a not in self.antibiotic_cols]
        if missing:
            # Keep going but be explicit
            print(f"[WARN] Missing columns ignored: {missing}")
        valid = [a for a in antibiotics if a in self.antibiotic_cols]
        if not valid:
            pass
            # raise ValueError("None of the requested antibiotics exist in the dataset.")
        return valid

    def aggregate_monthly(self, antibiotics: list[str]) -> pd.DataFrame:
        """Monthly totals (sum of 0/1 flags) for selected antibiotics."""
        cols = self.validate_antibiotics(antibiotics)
        # Aggregate; fill gaps with 0 and ensure full monthly index
        g = (self.df
             .groupby("__date")[cols]
             .sum(min_count=1)
             .sort_index()
             .fillna(0)
             .astype(int))
        return g

    def plot_monthly(self, aggregated: pd.DataFrame,
                     title: str = "Antibiotic Testing Trends by Month"):
        import plotly.graph_objects as go

        fig = go.Figure()

        # Ensure chronological order
        aggregated = aggregated.sort_index()
        date_index = aggregated.index
        n_points = max(len(date_index), 1)

        # -------- Dynamic layout based on number of months --------
        # Keep width reasonable: scales with #points but no huge spaces
        min_width = 900
        max_width = 1600
        pixels_per_point = 15  # modest spacing per month
        dynamic_width = int(
            max(min_width, min(max_width, pixels_per_point * n_points))
        )

        # Dynamic tick label angle to mitigate overlap:
        # few points -> horizontal, more points -> more rotation
        if n_points <= 18:
            tick_angle = 0
        elif n_points <= 36:
            tick_angle = 45
        else:
            tick_angle = 60

        # Use multi-line labels; extra newlines give more vertical spacing
        tick_fmt = "%b\n\n\n\n%Y"

        # -------- Traces --------
        for (idx, col) in enumerate(aggregated.columns):
            pretty = _pretty(col)
            fig.add_trace(go.Scatter(
                x=aggregated.index,
                y=aggregated[col],
                mode="lines+markers",
                name=pretty,
                line=dict(color=self._get_color(pretty, idx)),
                hovertemplate=f"%{{x|%b %Y}}<br>%{{y}} tests<extra>{pretty}</extra>"
            ))

        # -------- Layout --------
        fig.update_layout(
            # title=title,
            title=dict(
                text=(title or ""),
                font=dict(size=28, family="Arial", color="black")
            ),
            showlegend=True,
            xaxis_title="Month",
            yaxis_title="Isolate Count",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=70, b=150),  # a bit more room at bottom
            plot_bgcolor="#fff",
            width=dynamic_width,
            height=600,
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.5, yanchor="top",
                itemsizing="constant",
                traceorder="normal",
                tracegroupgap=0
            )
        )

        # Keep monthly resolution but rotate labels when dense
        fig.update_xaxes(
            dtick="M3",
            tickformat=tick_fmt,
            tickangle=tick_angle,
            showgrid=True,
            ticks="outside",
            automargin=True,
            tickfont=dict(size=20),
            title_font=dict(size=24)
        )
        fig.update_yaxes(
            showgrid=True, zeroline=True,
            tickfont=dict(size=20),
            title_font=dict(size=24)
        )
        return fig

    
    def select_top_antibiotics(
        self,
        antibiotics: list[str],
        top_n: int = 12,
        start_year: int | None = None,
        end_year: int | None = None,
        min_presence_ratio: float = 0.0,  # filter out antibiotics barely present
    ) -> list[str]:
        """
        Return the top-N antibiotics by total tests (sum of 0/1 flags).
        - Optional year window via start_year / end_year.
        - Optional presence filter: keep columns present in at least X% of rows.
        """
        cols = self.validate_antibiotics(antibiotics)

        df = self.df
        if start_year is not None:
            df = df[df[self.year_col] >= start_year]
        if end_year is not None:
            df = df[df[self.year_col] <= end_year]

        # presence filtering (avoid super-sparse columns)
        if min_presence_ratio > 0:
            present = (df[cols].notna().sum() / len(df)).fillna(0.0)
            cols = [c for c in cols if present.get(c, 0.0) >= min_presence_ratio]
            if not cols:
                pass
                # raise ValueError("No antibiotics meet the presence filter.")

        totals = df[cols].sum(numeric_only=True).sort_values(ascending=False)
        return totals.head(top_n).index.tolist()


    @staticmethod
    def export(
        fig: go.Figure,
        data: pd.DataFrame | None = None,
        out_html: Path | None = None, 
        out_png: Path | None = None,
        out_pdf: Path | None = None,
        out_csv: Path | None = None
    ):
        """
        Export figure to HTML/PNG/PDF and optionally the underlying data to CSV.
        - If `data` is provided and `out_csv` is not None, a CSV is written with:
          index -> 'Month' (ISO date), columns -> antibiotics' monthly counts.
        """
        # --- Figures ---
        if out_html:
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out_html, include_plotlyjs="cdn")
        if out_png:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(out_png, scale=2)  # needs kaleido
        if out_pdf:
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(out_pdf)  # also uses kaleido

        # --- Data CSV ---
        if out_csv is not None and data is not None and not data.empty:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_to_save = data.copy().sort_index()
            df_to_save.index.name = "Month"
            df_to_save.reset_index().to_csv(out_csv, index=False)


from shadow_antibiogram.controllers.DataLoader import DataLoader

def run_temp_basic(data_loader: DataLoader, df: pd.DataFrame):
    OUTPUT_DIR_WHO = "./outputs/trend_analysis_for_antibiotic_group/AWaRe_classes/"
    OUTPUT_DIR_CLASSIFICATION = "./outputs/trend_analysis_for_antibiotic_group/Antibiotics_classifications/"
    os.makedirs(OUTPUT_DIR_WHO, exist_ok=True)
    os.makedirs(OUTPUT_DIR_CLASSIFICATION, exist_ok=True)

    color_map = [
        "#8dd3c7",
        "#e5c494",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5"
    ]

    # AWaRe classes
    for group in ["Access", "Watch", "Reserve", "Not Set"]:
        try:
            candidates = data_loader.get_abx_by_category([group])
            chart = TemporalAntibioticLineChartPlotly(df, color_map=color_map)
            top_n = 7
            top = chart.select_top_antibiotics(
                antibiotics=candidates,
                top_n=top_n,
                start_year=2020,
                end_year=None,
                min_presence_ratio=0.02
            )
            monthly = chart.aggregate_monthly(top)
            fig = chart.plot_monthly(
                monthly,
                title=f"Monthly Top {top_n} Antibiotic Testing Trends ({group})"
            )
            base = Path(f"{OUTPUT_DIR_WHO}/antibiotic_trends_{group}_monthly")
            chart.export(
                fig,
                data=monthly,
                out_html=base.with_suffix(".html"),
                out_png=base.with_suffix(".png"),
                out_pdf=base.with_suffix(".pdf"),
                out_csv=base.with_suffix(".csv")
            )
        except Exception as e:
            print(str(e))
            continue
        except ValueError as e:
            print(str(e))
            continue

    # Detailed antibiotic classes
    for group in data_loader.load_abx_classes():
        try:
            candidates = data_loader.get_abx_by_class([group])
            chart = TemporalAntibioticLineChartPlotly(df, color_map=color_map)
            top_n = 5
            top = chart.select_top_antibiotics(
                antibiotics=candidates,
                top_n=top_n,
                start_year=None,
                end_year=None,
                min_presence_ratio=0.02
            )
            monthly = chart.aggregate_monthly(top)
            fig = chart.plot_monthly(
                monthly,
                title=f"Monthly Top {top_n} Antibiotic Testing Trends ({group})"
            )
            base = Path(f"{OUTPUT_DIR_CLASSIFICATION}/antibiotic_trends_{group}_monthly")
            chart.export(
                fig,
                data=monthly,
                out_html=base.with_suffix(".html"),
                out_png=base.with_suffix(".png"),
                out_pdf=base.with_suffix(".pdf"),
                out_csv=base.with_suffix(".csv")
            )
        except Exception as e:
            print(str(e))
            continue
        except ValueError as e:
            print(str(e))
            continue
