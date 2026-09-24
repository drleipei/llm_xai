from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/qwen36_rq5_shards"
OUT = ROOT / "results/qwen36_rq5_final"
MOUT = OUT / "metrics"
SOUT = OUT / "statistics"

MOUT.mkdir(parents=True, exist_ok=True)
SOUT.mkdir(parents=True, exist_ok=True)

BLOCK = ["dataset", "category", "method"]

METRICS = [
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

QA = [
    "valid_interventions",
    "invalid_interventions",
    "structural_compliance_rate",
    "execution_success_rate",
]

EXPECTED = {
    (0.0, 1.0),
    (0.2, 1.0),
    (0.5, 1.0),
    (0.8, 1.0),
    (0.2, 0.9),
    (0.2, 0.8),
}

TEMP = [0.0, 0.2, 0.5, 0.8]
TOPP = [1.0, 0.9, 0.8]


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    out = np.empty(len(ps))
    running = 0.0

    for rank, idx in enumerate(order):
        val = (len(ps) - rank) * ps[idx]
        running = max(running, val)
        out[idx] = min(running, 1.0)

    return out


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    d = x - y

    if len(x) == 0:
        return np.nan, np.nan, 0

    if np.allclose(d, 0):
        return 0.0, 1.0, len(x)

    stat, p = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        alternative="two-sided",
    )

    return float(stat), float(p), len(x)


def summary(x):
    x = pd.Series(x, dtype=float).dropna()

    return {
        "n": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": x.median(),
        "q1": x.quantile(.25),
        "q3": x.quantile(.75),
    }


# --------------------------------------------------
# Load
# --------------------------------------------------

paths = sorted(
    SRC.glob(
        "[0-9][0-9]_*/metrics/"
        "rq5_decoding_sensitivity.csv"
    )
)

assert len(paths) == 32

frames = []

for p in paths:
    df = pd.read_csv(p)
    assert len(df) == 6
    df["source_shard"] = p.parents[1].name
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

assert len(df) == 192

actual = {
    (round(float(t), 8), round(float(p), 8))
    for t, p in zip(df.temperature, df.top_p)
}

assert actual == EXPECTED

blocks = list(df.groupby(BLOCK))
assert len(blocks) == 32

for key, g in blocks:
    cond = {
        (round(float(t), 8), round(float(p), 8))
        for t, p in zip(g.temperature, g.top_p)
    }
    assert cond == EXPECTED, (key, cond)

merged = MOUT / "rq5_decoding_sensitivity_all_shards.csv"
df.to_csv(merged, index=False)


# --------------------------------------------------
# Condition accessor
# --------------------------------------------------

def values(metric, t, p):
    sub = df[
        np.isclose(df.temperature, t)
        &
        np.isclose(df.top_p, p)
    ]

    s = (
        sub
        .set_index(BLOCK)[metric]
        .sort_index()
    )

    assert len(s) == 32
    return s


# --------------------------------------------------
# Descriptive
# --------------------------------------------------

desc = []

for metric in METRICS + QA:
    for t, p in sorted(EXPECTED):

        x = df.loc[
            np.isclose(df.temperature, t)
            &
            np.isclose(df.top_p, p),
            metric,
        ]

        desc.append({
            "metric": metric,
            "temperature": t,
            "top_p": p,
            **summary(x),
        })

desc_df = pd.DataFrame(desc)

desc_path = MOUT / "rq5_descriptive_by_decoding.csv"
desc_df.to_csv(desc_path, index=False)


# --------------------------------------------------
# Friedman + Kendall W
# --------------------------------------------------

omnibus = []

families = {
    "temperature": [
        (t, 1.0)
        for t in TEMP
    ],
    "top_p": [
        (0.2, p)
        for p in TOPP
    ],
}

for metric in METRICS:

    for family, conds in families.items():

        matrix = np.column_stack([
            values(metric, t, p).to_numpy(float)
            for t, p in conds
        ])

        matrix = matrix[
            np.isfinite(matrix).all(axis=1)
        ]

        n, k = matrix.shape

        if np.allclose(matrix, matrix[:, [0]]):
            chi2, pval, w = 0.0, 1.0, 0.0
        else:
            chi2, pval = friedmanchisquare(
                *[
                    matrix[:, i]
                    for i in range(k)
                ]
            )

            chi2 = float(chi2)
            pval = float(pval)
            w = chi2 / (n * (k - 1))

        omnibus.append({
            "family": family,
            "metric": metric,
            "n_blocks": n,
            "k_conditions": k,
            "friedman_chi2": chi2,
            "friedman_p": pval,
            "kendalls_w": w,
        })

omni_df = pd.DataFrame(omnibus)

omni_path = SOUT / "rq5_friedman_kendalls_w.csv"
omni_df.to_csv(omni_path, index=False)


# --------------------------------------------------
# Paired reference comparisons
# reference = T=.2, p=1.0
# --------------------------------------------------

pairs = []

for metric in METRICS:

    ref = values(metric, 0.2, 1.0)

    comparison_families = {
        "temperature": [
            (0.0, 1.0),
            (0.5, 1.0),
            (0.8, 1.0),
        ],
        "top_p": [
            (0.2, 0.9),
            (0.2, 0.8),
        ],
    }

    for family, alternatives in comparison_families.items():

        tmp = []

        for t, p in alternatives:

            alt = values(metric, t, p)

            z = pd.concat(
                [ref, alt],
                axis=1,
            ).dropna()

            z.columns = ["ref", "alt"]

            x = z.ref.to_numpy(float)
            y = z.alt.to_numpy(float)
            diff = x - y

            stat, p_raw, n = safe_wilcoxon(x, y)

            tmp.append({
                "family": family,
                "metric": metric,
                "comparison":
                    f"T0.2_p1.0_vs_T{t}_p{p}",
                "alternative_temperature": t,
                "alternative_top_p": p,
                "n_pairs": n,
                "reference_mean": np.mean(x),
                "alternative_mean": np.mean(y),
                "mean_difference_reference_minus_alternative":
                    np.mean(diff),
                "median_difference_reference_minus_alternative":
                    np.median(diff),
                "reference_greater_n":
                    np.sum(diff > 0),
                "equal_n":
                    np.sum(np.isclose(diff, 0)),
                "reference_lower_n":
                    np.sum(diff < 0),
                "wilcoxon_statistic": stat,
                "p_raw": p_raw,
            })

        adjusted = holm(
            [r["p_raw"] for r in tmp]
        )

        for r, ph in zip(tmp, adjusted):
            r["p_holm"] = float(ph)
            r["significant_holm_0_05"] = ph < .05
            pairs.append(r)

pair_df = pd.DataFrame(pairs)

pair_path = SOUT / "rq5_reference_wilcoxon_holm.csv"
pair_df.to_csv(pair_path, index=False)


# --------------------------------------------------
# QA
# --------------------------------------------------

qa_rows = []

for metric in QA:

    for t, p in sorted(EXPECTED):

        x = df.loc[
            np.isclose(df.temperature, t)
            &
            np.isclose(df.top_p, p),
            metric,
        ]

        qa_rows.append({
            "metric": metric,
            "temperature": t,
            "top_p": p,
            **summary(x),
            "min": x.min(),
            "max": x.max(),
        })

qa_df = pd.DataFrame(qa_rows)

qa_path = MOUT / "rq5_operational_qa_summary.csv"
qa_df.to_csv(qa_path, index=False)


# --------------------------------------------------
# Final gate
# --------------------------------------------------

assert len(df) == 192
assert len(blocks) == 32
assert len(omni_df) == 26
assert len(pair_df) == 65

assert omni_df.n_blocks.eq(32).all()
assert pair_df.n_pairs.eq(32).all()
assert pair_df.p_holm.between(0, 1).all()

print()
print("========================================")
print("RQ5 FINAL STATISTICS: PASS")
print("========================================")
print("Shard files =", len(paths))
print("Merged rows =", len(df))
print("Paired blocks =", len(blocks))
print("Primary metrics =", len(METRICS))
print("Omnibus rows =", len(omni_df))
print("Pairwise rows =", len(pair_df))

print()
print("TEMPERATURE OMNIBUS SIGNIFICANT:")
print(
    omni_df[
        (omni_df.family == "temperature")
        &
        (omni_df.friedman_p < .05)
    ][
        ["metric", "friedman_p", "kendalls_w"]
    ].to_string(index=False)
)

print()
print("TOP-P OMNIBUS SIGNIFICANT:")
print(
    omni_df[
        (omni_df.family == "top_p")
        &
        (omni_df.friedman_p < .05)
    ][
        ["metric", "friedman_p", "kendalls_w"]
    ].to_string(index=False)
)

print()
print("HOLM-SIGNIFICANT PAIRWISE:")
print(
    pair_df[
        pair_df.significant_holm_0_05
    ][
        [
            "family",
            "metric",
            "comparison",
            "mean_difference_reference_minus_alternative",
            "p_holm",
        ]
    ].to_string(index=False)
)

print()
print("OUTPUTS:")
print(" ", merged)
print(" ", desc_path)
print(" ", omni_path)
print(" ", pair_path)
print(" ", qa_path)
