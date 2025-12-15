import pandas as pd

# =========================
# Load data
# =========================
single = pd.read_csv("./results/single_run_results.csv")
ret = pd.read_csv("./results/retention/retention_summary.csv")

print("single_run_results columns:", single.columns.tolist())
print("retention_summary columns:", ret.columns.tolist())

# =========================
# STAGE 1: cross-grid means by metric
# =========================
# Columns in single_run_results:
# ['metric', 'tau', 'gamma', 'seed', 'n_clusters', 'silhouette',
#  'ics_score', 'stability', 'hierarchical_score', 'genus', 'material', ...]

metrics_of_interest = ["jaccard", "dice", "cosine", "phi"]

stage1_sub = single[single["metric"].isin(metrics_of_interest)].copy()

stage1_summary = (
    stage1_sub
    .groupby("metric", as_index=False)
    .agg(
        mean_ICS=("ics_score", "mean"),
        mean_ARI=("stability", "mean"),
        mean_clusters=("n_clusters", "mean"),
        mean_hierarchical=("hierarchical_score", "mean"),
        n_rows=("ics_score", "count"),
    )
)

print("\n=== Stage 1: cross-grid means by metric ===")
print(stage1_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# =========================
# STAGE 2: retention and testable edges
# =========================
# retention_summary columns:
# ['genus', 'material', 'metric', 'tau', 'gamma', 'n_nodes',
#  'n_edges_jaccard', 'n_edges_testable', 'n_edges_significant',
#  'retention_fraction', 'retention_percent', 'alpha_fdr',
#  'min_total', 'min_positive']

ret_renamed = ret.rename(columns={
    "retention_percent": "retention_pct",
    "n_edges_testable": "n_testable_edges",
})

# 1) Jaccard per-genus/specimen retention
jaccard_ret = ret_renamed[ret_renamed["metric"] == "jaccard"]

jaccard_min = jaccard_ret["retention_pct"].min()
jaccard_med = jaccard_ret["retention_pct"].median()
jaccard_max = jaccard_ret["retention_pct"].max()

print("\n=== Stage 2: Jaccard retention across genus–material combinations ===")
print(f"Min retention:    {jaccard_min:.1f}%")
print(f"Median retention: {jaccard_med:.1f}%")
print(f"Max retention:    {jaccard_max:.1f}%")

idx_min = jaccard_ret["retention_pct"].idxmin()
idx_max = jaccard_ret["retention_pct"].idxmax()
print("Lowest-retention combo:",
      jaccard_ret.loc[idx_min, ["genus", "material", "retention_pct"]].to_dict())
print("Highest-retention combo:",
      jaccard_ret.loc[idx_max, ["genus", "material", "retention_pct"]].to_dict())

# 2) Per-metric median retention
metric_ret_summary = (
    ret_renamed[ret_renamed["metric"].isin(metrics_of_interest)]
    .groupby("metric", as_index=False)
    .agg(
        median_retention_pct=("retention_pct", "median"),
        min_retention_pct=("retention_pct", "min"),
        max_retention_pct=("retention_pct", "max"),
    )
)

print("\n=== Stage 2: retention by metric (percent) ===")
print(metric_ret_summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# 3) Typical range of testable edges per network by metric
edges_summary = (
    ret_renamed[ret_renamed["metric"].isin(metrics_of_interest)]
    .groupby("metric", as_index=False)
    .agg(
        median_edges=("n_testable_edges", "median"),
        min_edges=("n_testable_edges", "min"),
        max_edges=("n_testable_edges", "max"),
    )
)

print("\n=== Stage 2: number of testable edges per network by metric ===")
print(edges_summary.to_string(index=False))

# =========================
# LaTeX‑ready summary lines
# =========================
print("\n=== Values for LaTeX (rounded) ===")

for _, row in stage1_summary.iterrows():
    m = row["metric"]
    print(
        f"{m}: mean ICS={row['mean_ICS']:.2f}, "
        f"mean ARI={row['mean_ARI']:.2f}, "
        f"mean clusters={row['mean_clusters']:.1f}, "
        f"mean hierarchical score={row['mean_hierarchical']:.2f}"
    )

print("\nJaccard retention (min/median/max): "
      f"{jaccard_min:.1f}% / {jaccard_med:.1f}% / {jaccard_max:.1f}%")

for _, row in metric_ret_summary.iterrows():
    print(
        f"{row['metric']}: median retention={row['median_retention_pct']:.1f}%, "
        f"range {row['min_retention_pct']:.1f}–{row['max_retention_pct']:.1f}%"
    )

for _, row in edges_summary.iterrows():
    print(
        f"{row['metric']}: testable edges median={row['median_edges']:.0f}, "
        f"range {row['min_edges']:.0f}–{row['max_edges']:.0f}"
    )
