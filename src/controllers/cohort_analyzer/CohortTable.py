from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================
@dataclass(frozen=True)
class CohortTableConfig:
    """Configuration for building cohort characteristics tables."""
    pathogen_col: str = "Pathogen"
    specimen_col: str = "TextMaterialgroupRkiL0"
    tested_suffix: str = "_Tested"
    top_n_pathogens_default: int = 4
    thousands_sep: str = ","
    completeness_digits: int = 1
    mean_digits: int = 1
    sd_digits: int = 1


# ============================================================================
# MAIN BUILDER CLASS
# ============================================================================
class CohortCharacteristicsTable:
    """
    Build a publication-ready cohort characteristics table from cohort DataFrames.

    Features:
      - Total isolates
      - Primary pathogens (top-N with percentages)
      - Mean antibiotics tested per isolate ± SD
      - Testing completeness (%)
      - Specimen type (top with %)
      - Save to CSV and LaTeX with automatic directory creation
    """

    def __init__(
        self,
        cohorts: Dict[str, pd.DataFrame],
        pretty_names: Optional[Dict[str, str]] = None,
        top_n_pathogens: Optional[Dict[str, int]] = None,
        config: CohortTableConfig = CohortTableConfig(),
    ) -> None:
        if not cohorts:
            raise ValueError("No cohorts provided.")
        self.cohorts = cohorts
        self.pretty_names = pretty_names or {}
        self.top_n_pathogens = top_n_pathogens or {}
        self.cfg = config

        if not any(self._abx_cols(df) for df in cohorts.values()):
            logger.warning("⚠️ No *_Tested columns detected in provided cohorts.")

    # ----------------------------------------------------------------------
    # CORE METHODS
    # ----------------------------------------------------------------------
    def build_dataframe(self) -> pd.DataFrame:
        """Return the characteristics table as a pandas DataFrame."""
        headers = [self.pretty_names.get(k, k) for k in self._cohort_keys_in_order()]
        rows = {
            "Characteristic": [
                "Total Isolates",
                "Primary Pathogens",
                "Mean Antibiotics Tested Per Isolate (±SD)",
                "Testing Completeness*",
                "Specimen Type",
            ]
        }

        for key in self._cohort_keys_in_order():
            df = self.cohorts[key]
            rows[self.pretty_names.get(key, key)] = [
                self._fmt_thousands(len(df)),
                self._primary_pathogens(df, self.top_n_pathogens.get(key, self.cfg.top_n_pathogens_default)),
                self._mean_abx_tested_per_isolate(df),
                self._testing_completeness(df),
                self._specimen_summary(df),
            ]

        table_df = pd.DataFrame(rows)
        table_df = table_df[["Characteristic"] + headers]
        self._last_dataframe = table_df.copy()
        return table_df

    # ----------------------------------------------------------------------
    # SAVE FUNCTIONS
    # ----------------------------------------------------------------------
    def save_csv(self, path: str | Path) -> Path:
        """Save table as CSV, creating directories if needed."""
        df = getattr(self, "_last_dataframe", None)
        if df is None:
            df = self.build_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"💾 CSV table saved to: {path.resolve()}")
        return path


    def save_latex(
        self,
        path: str | Path,
        caption: str,
        label: str,
        col_widths: Tuple[float, float, float],
        footnote: Optional[str] = None,
        booktabs: bool = False,
    ) -> Path:
        """Save LaTeX table, creating directories if needed."""
        latex = self.to_latex(caption, label, col_widths, footnote, booktabs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex, encoding="utf-8")
        logger.info(f"💾 LaTeX table saved to: {path.resolve()}")
        return path

    # ----------------------------------------------------------------------
    # FORMATTING / LATEX EXPORT
    # ----------------------------------------------------------------------
    def to_latex(
        self,
        caption: str,
        label: str,
        col_widths: Tuple[float, ...],
        footnote: Optional[str] = None,
        booktabs: bool = False,
    ) -> str:
        """Return the LaTeX representation of the characteristics table."""
        df = getattr(self, "_last_dataframe", None)
        if df is None:
            df = self.build_dataframe()

        widths = list(col_widths)
        if len(widths) < df.shape[1]:
            widths += [widths[-1]] * (df.shape[1] - len(widths))
        col_spec = "".join([f"p{{{w}cm}}" for w in widths])

        h_top, h_mid, h_bot = (
            ("\\toprule", "\\midrule", "\\bottomrule") if booktabs else ("\\hline", "\\hline", "\\hline")
        )

        latex_rows = [h_top]
        latex_rows.append(" & ".join([f"\\textbf{{{c}}}" for c in df.columns]) + " \\\\")
        latex_rows.append(h_mid)

        for _, row in df.iterrows():
            cells = [self._escape_latex(str(x)) for x in row.tolist()]
            latex_rows.append(" & ".join(cells) + " \\\\")

        latex_rows.append(h_bot)

        foot = (
            (
                "\\begin{flushleft}\n\\footnotesize "
                + (footnote or "")
                + "\n\\end{flushleft}"
            ) if footnote else ""
        )

        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{col_spec}}}\n"
            + "\n".join(latex_rows)
            + "\n\\end{tabular}\n"
            + foot
            + "\n\\end{table}"
        ).strip()

        return latex

    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------
    def _cohort_keys_in_order(self) -> List[str]:
        return list(self.cohorts.keys())

    def _abx_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.endswith(self.cfg.tested_suffix)]

    def _fmt_thousands(self, n: int) -> str:
        return f"{n:,}".replace(",", self.cfg.thousands_sep)

    def _primary_pathogens(self, df: pd.DataFrame, top_n: int) -> str:
        col = self.cfg.pathogen_col
        if col not in df.columns or df[col].dropna().empty:
            return "—"
        vc = df[col].value_counts(normalize=True, dropna=True)
        items = [f"{self._latex_species(name)} ({round(frac*100):.0f}\\%)" for name, frac in vc.head(top_n).items()]
        return ", ".join(items)

    def _specimen_summary(self, df: pd.DataFrame) -> str:
        col = self.cfg.specimen_col
        if col not in df.columns or df[col].dropna().empty:
            return "—"
        vc = df[col].value_counts(normalize=True, dropna=True)
        top_name, frac = vc.index[0], vc.iloc[0]
        return f"{top_name} ({round(frac*100):.0f}\\%)"

    def _mean_abx_tested_per_isolate(self, df: pd.DataFrame) -> str:
        cols = self._abx_cols(df)
        if not cols:
            return f"0.0 ± 0.0"
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        per_iso = X.sum(axis=1)
        mean = per_iso.mean()
        sd = per_iso.std(ddof=1)
        return f"{mean:.{self.cfg.mean_digits}f} ± {sd:.{self.cfg.sd_digits}f}"

    def _testing_completeness(self, df: pd.DataFrame) -> str:
        cols = self._abx_cols(df)
        if not cols:
            return "0.0\\%"
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        completeness = X.mean().mean() * 100
        return f"{completeness:.{self.cfg.completeness_digits}f}\\%"

    def _latex_species(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            return "Unknown"
        specials = {
            "Coagulase-negative Staphylococci": "Coagulase-negative \\textit{Staphylococci}",
        }
        if name in specials:
            return specials[name]
        parts = name.split()
        if len(parts) >= 2:
            genus, species = parts[0], " ".join(parts[1:])
            return f"\\textit{{{genus[0]}. {species}}}"
        return f"\\textit{{{name}}}"

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters except already valid commands."""
        replace_map = {
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
        }
        for old, new in replace_map.items():
            text = text.replace(old, new)
        return text


# ============================================================================
# CONVENIENCE CLASSMETHOD
# ============================================================================
    @classmethod
    def from_generator(
        cls,
        generator,
        cohort_keys: Iterable[str],
        pretty_names: Optional[Dict[str, str]] = None,
        top_n_pathogens: Optional[Dict[str, int]] = None,
        config: CohortTableConfig = CohortTableConfig(),
    ) -> "CohortCharacteristicsTable":
        """Build directly from a ProductionCohortGenerator instance."""
        missing = [k for k in cohort_keys if k not in getattr(generator, "cohorts", {})]
        if missing:
            raise KeyError(f"Cohorts not found in generator: {missing}")
        cohorts = {k: generator.cohorts[k] for k in cohort_keys}
        return cls(cohorts, pretty_names, top_n_pathogens, config)
