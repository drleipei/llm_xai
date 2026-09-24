from pathlib import Path
from collections import Counter
import math

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = PROJECT_ROOT / "results" / "qwen36_rq6_shards"
OUT_ROOT = PROJECT_ROOT / "results" / "qwen36_rq6_final"

METRICS_DIR = OUT_ROOT / "metrics"
STATS_DIR = OUT_ROOT / "statistics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    "full",
    "no_semantics",
    "no_grounding",
    "no_constraints",
    "no_instance_context",
]

ABLATIONS = [
    "no_semantics",
    "no_grounding",
    "no_constraints",
    "no_instance_context",
]


# Primary scientific outcomes.
#
# valid_interventions / invalid_interventions are operational counts,
# while structural_compliance_rate / execution_success_rate are
# operational QA outcomes. They are retained in the merged dataset
# and summarized separately, but are not included in the primary
# inferential family below.
PRIMARY_METRICS = [
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
    "score_stability",
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_delta_probability",
    "mean_absolute_delta_probability",
    "normalized_feature_entropy",
    "pairwise_jaccard_within_instance",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
    "separability_gap",
]

QA_METRICS = [
    "valid_interventions",
    "invalid_interventions",
    "structural_compliance_rate",
    "execution_success_rate",
]


def holm_adjust(pvalues):
    """
    Holm step-down family-wise error-rate adjustment.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def safe_wilcoxon(full, ablated):
    """
    Two-sided paired Wilcoxon.

    Returns statistic and p-value.
    Handles the all-zero paired-difference case explicitly.
    """
    x = np.asarray(full, dtype=float)
    y = np.asarray(ablated, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return float("nan"), float("nan"), 0

    diff = x - y

    if np.allclose(diff, 0.0):
        return 0.0, 1.0, len(x)

    stat, p = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        alternative="two-sided",
    )

    return float(stat), float(p), len(x)


def median_iqr(values):
    x = pd.Series(values, dtype=float).dropna()

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(x.median()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
    }


# ============================================================
# 1. Load all 32 shard-level metric files
# ============================================================

frames = []

paths = sorted(
    SOURCE_ROOT.glob(
        "[0-9][0-9]_*/metrics/rq6_prompt_ablation.csv"
    )
)

assert len(paths) == 32, (
    f"Expected 32 shard metric files, found {len(paths)}"
)

for path in paths:
    df = pd.read_csv(path)
    df["source_shard"] = path.parents[1].name
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

assert len(all_df) == 160, (
    f"Expected 160 metric rows, found {len(all_df)}"
)

assert set(all_df["prompt_variant"]) == set(VARIANTS)

assert all_df[
    [
        "dataset",
        "category",
        "method",
        "llm_model",
        "prompt_variant",
        "temperature",
        "top_p",
    ]
].duplicated().sum() == 0


# ============================================================
# 2. Strict block completeness
# ============================================================

BLOCK_KEYS = [
    "dataset",
    "category",
    "method",
]

block_counts = (
    all_df
    .groupby(BLOCK_KEYS)["prompt_variant"]
    .agg(list)
)

assert len(block_counts) == 32

for block, variants in block_counts.items():
    assert len(variants) == 5, (
        f"Incomplete block {block}: {variants}"
    )

    assert set(variants) == set(VARIANTS), (
        f"Wrong variants in block {block}: {variants}"
    )


# Save global merged metrics.
merged_path = (
    METRICS_DIR
    / "rq6_prompt_ablation_all_shards.csv"
)

all_df.to_csv(merged_path, index=False)


# ============================================================
# 3. Descriptive statistics — overall by prompt variant
# ============================================================

desc_rows = []

for metric in PRIMARY_METRICS + QA_METRICS:

    assert metric in all_df.columns, (
        f"Missing metric: {metric}"
    )

    for variant in VARIANTS:
        sub = all_df[
            all_df["prompt_variant"] == variant
        ]

        summary = median_iqr(sub[metric])

        desc_rows.append({
            "scope": "overall",
            "metric": metric,
            "prompt_variant": variant,
            **summary,
        })

desc_df = pd.DataFrame(desc_rows)

desc_path = (
    METRICS_DIR
    / "rq6_descriptive_by_variant.csv"
)

desc_df.to_csv(desc_path, index=False)


# ============================================================
# 4. Descriptive statistics by method
# ============================================================

method_rows = []

for method in sorted(all_df["method"].unique()):

    mdf = all_df[all_df["method"] == method]

    for metric in PRIMARY_METRICS + QA_METRICS:

        for variant in VARIANTS:

            sub = mdf[
                mdf["prompt_variant"] == variant
            ]

            summary = median_iqr(sub[metric])

            method_rows.append({
                "method": method,
                "metric": metric,
                "prompt_variant": variant,
                **summary,
            })

method_df = pd.DataFrame(method_rows)

method_path = (
    METRICS_DIR
    / "rq6_descriptive_by_method.csv"
)

method_df.to_csv(method_path, index=False)


# ============================================================
# 5. Descriptive statistics by dataset
# ============================================================

dataset_rows = []

for dataset in sorted(all_df["dataset"].unique()):

    ddf = all_df[all_df["dataset"] == dataset]

    for metric in PRIMARY_METRICS + QA_METRICS:

        for variant in VARIANTS:

            sub = ddf[
                ddf["prompt_variant"] == variant
            ]

            summary = median_iqr(sub[metric])

            dataset_rows.append({
                "dataset": dataset,
                "metric": metric,
                "prompt_variant": variant,
                **summary,
            })

dataset_df = pd.DataFrame(dataset_rows)

dataset_path = (
    METRICS_DIR
    / "rq6_descriptive_by_dataset.csv"
)

dataset_df.to_csv(dataset_path, index=False)


# ============================================================
# 6. Primary Friedman omnibus + Kendall's W
# ============================================================

friedman_rows = []

for metric in PRIMARY_METRICS:

    wide = all_df.pivot(
        index=BLOCK_KEYS,
        columns="prompt_variant",
        values=metric,
    )

    wide = wide[VARIANTS].dropna()

    assert len(wide) <= 32

    arrays = [
        wide[v].to_numpy(dtype=float)
        for v in VARIANTS
    ]

    n = len(wide)
    k = len(VARIANTS)

    if n == 0:
        chi2 = np.nan
        p = np.nan
        kendall_w = np.nan

    else:
        # scipy can warn or return NaN when every condition
        # is exactly identical. Treat that deterministic case
        # as no effect.
        matrix = np.column_stack(arrays)

        if np.allclose(
            matrix,
            matrix[:, [0]],
            equal_nan=False,
        ):
            chi2 = 0.0
            p = 1.0
            kendall_w = 0.0

        else:
            chi2, p = friedmanchisquare(*arrays)
            chi2 = float(chi2)
            p = float(p)

            kendall_w = (
                chi2 / (n * (k - 1))
                if n > 0 and k > 1
                else np.nan
            )

    friedman_rows.append({
        "metric": metric,
        "n_blocks": n,
        "k_conditions": k,
        "friedman_chi2": chi2,
        "friedman_p": p,
        "kendalls_w": kendall_w,
    })


friedman_df = pd.DataFrame(friedman_rows)

friedman_path = (
    STATS_DIR
    / "rq6_friedman_kendalls_w.csv"
)

friedman_df.to_csv(friedman_path, index=False)


# ============================================================
# 7. FULL vs each ablation: paired Wilcoxon + Holm
# ============================================================

pairwise_rows = []

for metric in PRIMARY_METRICS:

    wide = all_df.pivot(
        index=BLOCK_KEYS,
        columns="prompt_variant",
        values=metric,
    )

    family_rows = []

    for ablation in ABLATIONS:

        paired = (
            wide[["full", ablation]]
            .dropna()
        )

        full = paired["full"].to_numpy(dtype=float)
        ablated = paired[ablation].to_numpy(dtype=float)

        stat, p_raw, n = safe_wilcoxon(
            full,
            ablated,
        )

        diff = full - ablated

        row = {
            "metric": metric,
            "comparison": f"full_vs_{ablation}",
            "ablation": ablation,
            "n_pairs": int(n),
            "full_mean": float(np.mean(full)),
            "ablation_mean": float(np.mean(ablated)),
            "mean_difference_full_minus_ablation":
                float(np.mean(diff)),
            "median_difference_full_minus_ablation":
                float(np.median(diff)),
            "full_greater_n":
                int(np.sum(diff > 0)),
            "equal_n":
                int(np.sum(np.isclose(diff, 0.0))),
            "full_lower_n":
                int(np.sum(diff < 0)),
            "wilcoxon_statistic": stat,
            "p_raw": p_raw,
        }

        family_rows.append(row)

    adjusted = holm_adjust(
        [r["p_raw"] for r in family_rows]
    )

    for row, p_holm in zip(
        family_rows,
        adjusted,
    ):
        row["p_holm"] = float(p_holm)
        row["significant_holm_0_05"] = (
            bool(p_holm < 0.05)
        )

        pairwise_rows.append(row)


pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_path = (
    STATS_DIR
    / "rq6_full_vs_ablation_wilcoxon_holm.csv"
)

pairwise_df.to_csv(pairwise_path, index=False)


# ============================================================
# 8. Operational / QA summary
# ============================================================

qa_rows = []

for metric in QA_METRICS:
    for variant in VARIANTS:

        vals = all_df.loc[
            all_df["prompt_variant"] == variant,
            metric,
        ]

        qa_rows.append({
            "metric": metric,
            "prompt_variant": variant,
            **median_iqr(vals),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })

qa_df = pd.DataFrame(qa_rows)

qa_path = (
    METRICS_DIR
    / "rq6_operational_qa_summary.csv"
)

qa_df.to_csv(qa_path, index=False)


# ============================================================
# 9. Final integrity checks
# ============================================================

assert len(all_df) == 160
assert len(block_counts) == 32

assert len(friedman_df) == len(PRIMARY_METRICS)

assert len(pairwise_df) == (
    len(PRIMARY_METRICS) * len(ABLATIONS)
)

assert pairwise_df["n_pairs"].eq(32).all(), (
    pairwise_df[
        ~pairwise_df["n_pairs"].eq(32)
    ]
)

assert pairwise_df["p_holm"].between(
    0.0,
    1.0,
).all()


print()
print("========================================")
print("RQ6 FINAL STATISTICS: PASS")
print("========================================")

print("Shard metric files =", len(paths))
print("Merged rows =", len(all_df))
print("Paired blocks =", len(block_counts))
print("Primary metrics =", len(PRIMARY_METRICS))
print("Friedman rows =", len(friedman_df))
print("Pairwise rows =", len(pairwise_df))

print()
print("OUTPUTS:")
print(" ", merged_path)
print(" ", desc_path)
print(" ", method_path)
print(" ", dataset_path)
print(" ", friedman_path)
print(" ", pairwise_path)
print(" ", qa_path)

print()
print("HOLM-SIGNIFICANT PAIRWISE:")
print(
    pairwise_df[
        pairwise_df["significant_holm_0_05"]
    ][
        [
            "metric",
            "comparison",
            "n_pairs",
            "mean_difference_full_minus_ablation",
            "p_raw",
            "p_holm",
        ]
    ].to_string(index=False)
)
