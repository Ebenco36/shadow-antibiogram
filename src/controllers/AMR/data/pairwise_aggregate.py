from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


PAIRWISE_CORE = {"ab_1", "ab_2", "a", "b", "c", "d"}

CONTEXT_COLUMNS = [
    "Year",
    "CareType",
    "ARS_WardType",
    "Sex",
    "BroadAgeGroup",
    "HighLevelAgeRange",
    "Hospital_Priority",
    "Care_Complexity",
]

KEY_COLUMNS = [
    "Pathogen",
    "PathogenGenus",
    "TextMaterialgroupRkiL0",
    *CONTEXT_COLUMNS,
    "ab_1",
    "ab_2",
]


@dataclass(frozen=True)
class PairwiseAggregateSummary:
    n_rows: int
    n_pairs: int
    n_antibiotics: int
    genera: List[str]
    pathogens: List[str]
    materials: List[str]
    layers: Dict[str, int]

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_pairs": self.n_pairs,
            "n_antibiotics": self.n_antibiotics,
            "genera": self.genera,
            "pathogens": self.pathogens,
            "materials": self.materials,
            "layers": self.layers,
        }


def is_pairwise_df(df: pd.DataFrame) -> bool:
    return PAIRWISE_CORE.issubset(set(df.columns))


def _as_context_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("-")


def add_pairwise_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `_pairwise_layer` describing which marginal stratum each row belongs to.

    Valid production layers are intentionally narrow:
      - all_context: all context columns are "-"
      - year: Year is set and all other context columns are "-"
      - care_type: CareType is set and all other context columns are "-"
      - ward_type: ARS_WardType is set and all other context columns are "-"
      - sex, broad_age, high_level_age, hospital_priority, care_complexity:
        reserved for future compatible aggregate files

    Any row with more than one non-empty context dimension is marked
    mixed_or_unsupported because aggregating it with other layers would
    double-count isolates.
    """
    if not is_pairwise_df(df):
        raise ValueError("Pairwise aggregate expected columns ab_1, ab_2, a, b, c, d.")

    out = df.copy()
    present_contexts = [c for c in CONTEXT_COLUMNS if c in out.columns]
    if not present_contexts:
        out["_pairwise_layer"] = "all_context"
        return out

    non_empty = pd.DataFrame(index=out.index)
    for col in present_contexts:
        non_empty[col] = ~_as_context_string(out[col]).eq("-")

    n_set = non_empty.sum(axis=1)
    layer = pd.Series("mixed_or_unsupported", index=out.index, dtype="object")
    layer.loc[n_set == 0] = "all_context"

    layer_names = {
        "Year": "year",
        "CareType": "care_type",
        "ARS_WardType": "ward_type",
        "Sex": "sex",
        "BroadAgeGroup": "broad_age",
        "HighLevelAgeRange": "high_level_age",
        "Hospital_Priority": "hospital_priority",
        "Care_Complexity": "care_complexity",
    }
    for col, name in layer_names.items():
        if col in non_empty.columns:
            layer.loc[(n_set == 1) & non_empty[col]] = name

    out["_pairwise_layer"] = layer
    return out


def select_material_level_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select one non-overlapping layer for material-level networks.

    Prefer true all-context rows if present. If absent, use year-only rows,
    because years are non-overlapping partitions and can be pooled by summing
    a/b/c/d counts. Never mix year/care/ward summaries.
    """
    layered = add_pairwise_layer(df)
    all_context = layered[layered["_pairwise_layer"].eq("all_context")].copy()
    if not all_context.empty:
        return all_context.drop(columns=["_pairwise_layer"])

    year_only = layered[layered["_pairwise_layer"].eq("year")].copy()
    if not year_only.empty:
        return year_only.drop(columns=["_pairwise_layer"])

    return layered.iloc[0:0].drop(columns=["_pairwise_layer"])


def aggregate_pairwise_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pool duplicate antibiotic pairs by summing counts before metric estimation.
    This is correct only after selecting a non-overlapping layer.
    """
    if not is_pairwise_df(df):
        raise ValueError("Cannot aggregate counts: input is not pairwise aggregate data.")
    work = df.copy()
    for col in ["a", "b", "c", "d"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
    return work.groupby(["ab_1", "ab_2"], as_index=False)[["a", "b", "c", "d"]].sum()


def summarize_pairwise_aggregate(df: pd.DataFrame) -> PairwiseAggregateSummary:
    if not is_pairwise_df(df):
        raise ValueError("Input is not pairwise aggregate data.")

    layered = add_pairwise_layer(df)
    abx = set(layered["ab_1"].astype(str)).union(set(layered["ab_2"].astype(str)))
    pairs = layered[["ab_1", "ab_2"]].drop_duplicates()

    def unique_values(col: str) -> List[str]:
        if col not in layered.columns:
            return []
        vals = layered[col].dropna().astype(str).unique().tolist()
        return sorted(vals)

    return PairwiseAggregateSummary(
        n_rows=int(len(layered)),
        n_pairs=int(len(pairs)),
        n_antibiotics=int(len(abx)),
        genera=unique_values("PathogenGenus"),
        pathogens=unique_values("Pathogen"),
        materials=unique_values("TextMaterialgroupRkiL0"),
        layers={str(k): int(v) for k, v in layered["_pairwise_layer"].value_counts().sort_index().items()},
    )


def validate_pairwise_aggregate(
    df: pd.DataFrame,
    *,
    expected_genera: Optional[Iterable[str]] = None,
    expected_materials: Optional[Iterable[str]] = None,
    strict_expected: bool = True,
) -> PairwiseAggregateSummary:
    """
    Validate the aggregate parquet as a production data product.

    The checks are intentionally fail-loud for structural errors that could
    alter network estimates.
    """
    if not is_pairwise_df(df):
        raise ValueError("Expected pairwise aggregate columns ab_1, ab_2, a, b, c, d.")

    missing_key = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_key:
        raise ValueError(f"Pairwise aggregate is missing required key columns: {missing_key}")

    bad_counts = []
    for col in ["a", "b", "c", "d"]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any():
            bad_counts.append(f"{col}: non-numeric values")
        if (numeric < 0).any():
            bad_counts.append(f"{col}: negative values")
    if bad_counts:
        raise ValueError("Invalid pairwise counts: " + "; ".join(bad_counts))

    if (df["ab_1"].astype(str) == df["ab_2"].astype(str)).any():
        raise ValueError("Pairwise aggregate contains self-pairs where ab_1 == ab_2.")

    duplicate_keys = df.duplicated(KEY_COLUMNS).sum()
    if duplicate_keys:
        raise ValueError(f"Pairwise aggregate contains {int(duplicate_keys)} duplicated stratum/pair keys.")

    layered = add_pairwise_layer(df)
    bad_layers = layered[layered["_pairwise_layer"].eq("mixed_or_unsupported")]
    if not bad_layers.empty:
        examples = bad_layers[KEY_COLUMNS].head(3).to_dict(orient="records")
        raise ValueError(
            "Pairwise aggregate contains mixed context rows that cannot be pooled safely. "
            f"Examples: {examples}"
        )

    summary = summarize_pairwise_aggregate(df)

    def check_expected(name: str, expected: Optional[Iterable[str]], observed: List[str]) -> None:
        if not expected:
            return
        missing = sorted(set(map(str, expected)) - set(map(str, observed)))
        if missing and strict_expected:
            raise ValueError(
                f"Configured {name} missing from aggregate data: {missing}. "
                f"Observed {name}: {observed}"
            )

    check_expected("genera", expected_genera, summary.genera)
    check_expected("materials", expected_materials, summary.materials)
    return summary


def parquet_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    if path.suffix == ".parquet":
        return [path]
    return []
