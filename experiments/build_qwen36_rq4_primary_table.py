from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_paired_comparison_corrected.csv"
)

OUTPUT = Path(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_primary_overall_interpreted.csv"
)


HIGHER_IS_BETTER = {
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
    "score_stability",
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_absolute_delta_probability",
    "pairwise_jaccard_within_instance",
    "instance_idf_specificity",
    "separability_gap",
}

LOWER_IS_BETTER = {
    "pairwise_jaccard_between_instance",
}

NO_WINNER_DIRECTION = {
    "mean_delta_probability",
}


df = pd.read_csv(INPUT)
df = df[df["scope"] == "overall"].copy()

assert len(df) == 48


def metric_direction(metric):
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    if metric in NO_WINNER_DIRECTION:
        return "signed_no_winner"
    raise RuntimeError(f"Unknown metric direction: {metric}")


df["metric_direction"] = df["metric"].map(metric_direction)

df["improvement_signed"] = np.nan

higher = df["metric_direction"] == "higher_is_better"
lower = df["metric_direction"] == "lower_is_better"

df.loc[higher, "improvement_signed"] = \
    df.loc[higher, "mean_difference_llm_minus_traditional"]

df.loc[lower, "improvement_signed"] = \
    -df.loc[lower, "mean_difference_llm_minus_traditional"]


def classify(row):
    if row["metric_direction"] == "signed_no_winner":
        return "not_directional"

    if row["p_holm"] >= 0.05:
        return "no_significant_difference"

    if row["improvement_signed"] > 0:
        return "llm_informed_better"

    if row["improvement_signed"] < 0:
        return "traditional_better"

    return "tie"


df["holm_interpretation"] = df.apply(classify, axis=1)

cols = [
    "metric",
    "traditional_method",
    "llm_informed_method",
    "llm_model",
    "n_pairs",
    "traditional_mean",
    "llm_mean",
    "mean_difference_llm_minus_traditional",
    "metric_direction",
    "improvement_signed",
    "paired_bootstrap_ci_low",
    "paired_bootstrap_ci_high",
    "p_value",
    "p_holm",
    "p_bh_fdr",
    "rank_biserial_correlation",
    "holm_interpretation",
]

df = df[cols].sort_values(
    ["metric", "traditional_method"]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("output =", OUTPUT)
print("rows =", len(df))
print()

print("HOLM INTERPRETATION:")
print(df["holm_interpretation"].value_counts(dropna=False))
print()

print("BY METRIC:")
print(
    df.groupby(
        ["metric", "holm_interpretation"]
    ).size().to_string()
)

assert len(df) == 48
assert (df["n_pairs"] == 160).all()

print()
print("RQ4 PRIMARY INTERPRETATION QA: PASS")
