from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results/qwen36_27b_full/publication")
OUT.mkdir(parents=True, exist_ok=True)

# =========================
# RQ1
# =========================
rq1 = pd.read_csv(
    "results/qwen36_27b_full/metrics/rq1_stability.csv"
)

rq1_metrics = [
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
    "score_stability",
]

rq1_summary = (
    rq1.groupby("method")[rq1_metrics]
       .agg(["mean", "std"])
       .reset_index()
)

rq1_summary.to_csv(
    OUT / "table_rq1_stability_summary.csv",
    index=False,
)

# =========================
# RQ2
# =========================
rq2 = pd.read_csv(
    "results/qwen36_27b_full/metrics/"
    "rq2_model_response_alignment.csv"
)

rq2_metrics = [
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_delta_probability",
    "mean_absolute_delta_probability",
]

# First collapse repeated runs at instance level
rq2_instance = (
    rq2.groupby(
        ["dataset", "category", "instance_id", "method"],
        as_index=False
    )[rq2_metrics]
    .mean()
)

rq2_summary = (
    rq2_instance.groupby("method")[rq2_metrics]
                .agg(["mean", "std"])
                .reset_index()
)

rq2_summary.to_csv(
    OUT / "table_rq2_alignment_summary.csv",
    index=False,
)

# =========================
# RQ3 instance-level
# =========================
rq3i = pd.read_csv(
    "results/qwen36_27b_full/metrics/"
    "rq3_instance_specificity.csv"
)

rq3_metrics = [
    "pairwise_jaccard_within_instance",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
    "separability_gap",
]

rq3_summary = (
    rq3i.groupby("method")[rq3_metrics]
        .agg(["mean", "std"])
        .reset_index()
)

rq3_summary.to_csv(
    OUT / "table_rq3_discriminativeness_summary.csv",
    index=False,
)

# =========================
# RQ3 group entropy
# =========================
rq3g = pd.read_csv(
    "results/qwen36_27b_full/metrics/"
    "rq3_discriminativeness.csv"
)

entropy_summary = (
    rq3g.groupby("method")["normalized_feature_entropy"]
        .agg(["mean", "std"])
        .reset_index()
)

entropy_summary.to_csv(
    OUT / "table_rq3_entropy_summary.csv",
    index=False,
)

# =========================
# RQ4 interpreted overall
# =========================
rq4 = pd.read_csv(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_primary_overall_interpreted.csv"
)

rq4.to_csv(
    OUT / "table_rq4_primary_overall.csv",
    index=False,
)

# =========================
# RQ4 outcome counts
# =========================
counts = (
    rq4["holm_interpretation"]
       .value_counts()
       .rename_axis("outcome")
       .reset_index(name="n")
)

counts["percent_of_48"] = counts["n"] / 48 * 100

directional = rq4[
    rq4["holm_interpretation"] != "not_directional"
].copy()

directional_counts = (
    directional["holm_interpretation"]
    .value_counts()
    .rename_axis("outcome")
    .reset_index(name="n")
)

directional_counts["percent_of_44"] = (
    directional_counts["n"] / 44 * 100
)

counts.to_csv(
    OUT / "table_rq4_outcome_counts_all48.csv",
    index=False,
)

directional_counts.to_csv(
    OUT / "table_rq4_outcome_counts_directional44.csv",
    index=False,
)

print("Publication tables written to:", OUT)
print()
print("RQ1 rows =", len(rq1))
print("RQ2 raw rows =", len(rq2))
print("RQ2 instance rows =", len(rq2_instance))
print("RQ3 instance rows =", len(rq3i))
print("RQ3 group rows =", len(rq3g))
print("RQ4 overall rows =", len(rq4))
print()

print("RQ4 all 48:")
print(counts.to_string(index=False))
print()

print("RQ4 directional 44:")
print(directional_counts.to_string(index=False))

assert len(rq1) == 640
assert len(rq2) == 3200
assert len(rq2_instance) == 640
assert len(rq3i) == 640
assert len(rq3g) == 32
assert len(rq4) == 48
assert len(directional) == 44

print()
print("QWEN3.6 PUBLICATION TABLE QA: PASS")
