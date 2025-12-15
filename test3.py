# from src.controllers.DataLoader import DataLoader

# loader = DataLoader("./datasets/WHO_Aware_data")
# df = loader.get_combined()

# print(df.groupby("Pathogen").size().sort_values(ascending=False).head(15))
# print(df.groupby("PathogenGenus").size().sort_values(ascending=False).head(15))


import pandas as pd

# Load single-run results
df = pd.read_csv("./results/single_run_results.csv")

# Choose what to aggregate and how
agg = (
    df
    .groupby(["genus", "material", "metric", "tau", "gamma"], as_index=False)
    .agg(
        n_seeds       = ("seed", "nunique"),
        mean_n_clusters = ("n_clusters", "mean"),
        std_n_clusters  = ("n_clusters", "std"),
        mean_silhouette = ("silhouette", "mean"),
        std_silhouette  = ("silhouette", "std"),
        mean_ics        = ("ics_score", "mean"),
        std_ics         = ("ics_score", "std"),
        mean_stability  = ("stability", "mean"),
        std_stability   = ("stability", "std"),
        mean_hier       = ("hierarchical_score", "mean"),
        std_hier        = ("hierarchical_score", "std"),
        mean_ARI_broad  = ("external_ARI_broad", "mean"),
        mean_NMI_broad  = ("external_NMI_broad", "mean"),
        mean_ARI_fine   = ("external_ARI_fine", "mean"),
        mean_NMI_fine   = ("external_NMI_fine", "mean"),
        mean_ARI_who    = ("external_ARI_who", "mean"),
        mean_NMI_who    = ("external_NMI_who", "mean"),
    )
)

# Optional: round for readability
agg_rounded = agg.round({
    "mean_n_clusters": 1, "std_n_clusters": 1,
    "mean_silhouette": 3, "std_silhouette": 3,
    "mean_ics": 2, "std_ics": 2,
    "mean_stability": 2, "std_stability": 2,
    "mean_hier": 2, "std_hier": 2,
    "mean_ARI_broad": 3, "mean_NMI_broad": 3,
    "mean_ARI_fine": 3, "mean_NMI_fine": 3,
    "mean_ARI_who": 3, "mean_NMI_who": 3,
})

# Save a reviewer-friendly summary
agg_rounded.to_csv("./results/single_run_aggregated_for_review.csv", index=False)



################################################################################
# Summary table for AWaRe and Fluoroquinolone analysis
################################################################################

import pandas as pd
from pathlib import Path

# ---------- 1. Load CSVs ----------
base = Path("./datasets/output/Phase_I/basic_trend_analysis_for_group_antibiotics")

df_access = pd.read_csv(
    base / "AWaRe_classes/antibiotic_trends_Access_monthly.csv",
    parse_dates=["Month"],
)
df_watch = pd.read_csv(
    base / "AWaRe_classes/antibiotic_trends_Watch_monthly.csv",
    parse_dates=["Month"],
)
df_res = pd.read_csv(
    base / "AWaRe_classes/antibiotic_trends_Reserve_monthly.csv",
    parse_dates=["Month"],
)
df_fluoro = pd.read_csv(
    base / "Antibiotics_classifications/antibiotic_trends_Fluoroquinolone_monthly.csv",
    parse_dates=["Month"],
)

# ---------- 2. Clean column names ----------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "Month" if c == "Month"
        else c.split(" - ")[0].split("_")[0].strip()
        for c in out.columns
    ]
    return out

df_access = clean_cols(df_access).set_index("Month")
df_watch  = clean_cols(df_watch).set_index("Month")
df_res    = clean_cols(df_res).set_index("Month")
df_fluoro = clean_cols(df_fluoro).set_index("Month")

# ---------- 3. Combine & totals ----------
# Combine all unique drugs once; this prevents double counting
all_tests = pd.concat([df_access, df_watch, df_res, df_fluoro], axis=1)
all_tests = all_tests.loc[:, ~all_tests.columns.duplicated()].copy()

access_cols  = df_access.columns.tolist()
watch_cols   = sorted(set(df_watch.columns) | set(df_fluoro.columns))
reserve_cols = df_res.columns.tolist()

all_tests["Access_Total"]  = all_tests[access_cols].sum(axis=1)
all_tests["Watch_Total"]   = all_tests[watch_cols].sum(axis=1)
all_tests["Reserve_Total"] = all_tests[reserve_cols].sum(axis=1)
all_tests["Total_Tests"]   = (
    all_tests["Access_Total"] + all_tests["Watch_Total"] + all_tests["Reserve_Total"]
)

# ---------- 4. Fluoro individual drugs (absolute + proportion) ----------
fluoro_drugs = ["CIP", "LEV", "MOX"]  # ciprofloxacin, levofloxacin, moxifloxacin
for drug in fluoro_drugs:
    if drug not in all_tests.columns:
        raise KeyError(f"{drug} not found in combined columns.")
    all_tests[f"{drug}_Total"] = all_tests[drug]
    all_tests[f"{drug}_Prop"]  = all_tests[drug] / all_tests["Total_Tests"]

# ---------- 5. AWaRe proportions ----------
all_tests["Access_Prop"]  = all_tests["Access_Total"]  / all_tests["Total_Tests"]
all_tests["Watch_Prop"]   = all_tests["Watch_Total"]   / all_tests["Total_Tests"]
all_tests["Reserve_Prop"] = all_tests["Reserve_Total"] / all_tests["Total_Tests"]

# ---------- 6. Annual averages (absolute + %) ----------
all_tests["Year"] = all_tests.index.year
annual = all_tests.groupby("Year").mean()  # still in counts / proportions

records = []

for year, row in annual.iterrows():
    # AWaRe groups
    records.extend([
        {
            "Year": year,
            "Group": "Access",
            "Drug": "All_Access",
            "Mean_monthly_tests": row["Access_Total"],
            "Mean_pct_of_all_tests": row["Access_Prop"] * 100,
        },
        {
            "Year": year,
            "Group": "Watch",
            "Drug": "All_Watch",
            "Mean_monthly_tests": row["Watch_Total"],
            "Mean_pct_of_all_tests": row["Watch_Prop"] * 100,
        },
        {
            "Year": year,
            "Group": "Reserve",
            "Drug": "All_Reserve",
            "Mean_monthly_tests": row["Reserve_Total"],
            "Mean_pct_of_all_tests": row["Reserve_Prop"] * 100,
        },
    ])

    # Key fluoroquinolones
    for drug in fluoro_drugs:
        records.append(
            {
                "Year": year,
                "Group": "Watch_Fluoroquinolone",
                "Drug": drug,
                "Mean_monthly_tests": row[f"{drug}_Total"],
                "Mean_pct_of_all_tests": row[f"{drug}_Prop"] * 100,
            }
        )

summary = pd.DataFrame.from_records(records)
summary = summary.sort_values(["Year", "Group", "Drug"])

# Round for publication
summary["Mean_monthly_tests"] = summary["Mean_monthly_tests"].round(1)
summary["Mean_pct_of_all_tests"] = summary["Mean_pct_of_all_tests"].round(1)

# ---------- 7. Save comprehensive, reviewer-ready table ----------
out_path = base / "aware_fluoro_summary_annual_clean.csv"
summary.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
