from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_paired_comparison.csv"
)

OUTPUT = Path(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_paired_comparison_corrected.csv"
)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=float)

    running_max = 0.0
    for i, pv in enumerate(ranked):
        value = (m - i) * pv
        running_max = max(running_max, value)
        adjusted_ranked[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked

    return adjusted


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=float)

    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * m / rank
        running_min = min(running_min, value)
        adjusted_ranked[i] = min(running_min, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked

    return adjusted


df = pd.read_csv(INPUT)

assert len(df) == 432
assert set(df["scope"]) == {"overall", "dataset_category"}

df["p_holm"] = np.nan
df["p_bh_fdr"] = np.nan
df["multiplicity_family"] = ""

families = {
    "primary_overall_48": df["scope"] == "overall",
    "exploratory_dataset_category_384":
        df["scope"] == "dataset_category",
}

for family_name, mask in families.items():
    p = df.loc[mask, "p_value"].to_numpy(dtype=float)

    assert np.isfinite(p).all()

    df.loc[mask, "p_holm"] = holm_adjust(p)
    df.loc[mask, "p_bh_fdr"] = bh_adjust(p)
    df.loc[mask, "multiplicity_family"] = family_name

df["significant_raw_0_05"] = df["p_value"] < 0.05
df["significant_holm_0_05"] = df["p_holm"] < 0.05
df["significant_bh_fdr_0_05"] = df["p_bh_fdr"] < 0.05

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("output =", OUTPUT)
print("rows =", len(df))
print()

for family_name in families:
    sub = df[df["multiplicity_family"] == family_name]

    print("====", family_name, "====")
    print("n =", len(sub))
    print(
        "raw p<.05 =",
        int(sub["significant_raw_0_05"].sum()),
    )
    print(
        "Holm p<.05 =",
        int(sub["significant_holm_0_05"].sum()),
    )
    print(
        "BH-FDR p<.05 =",
        int(sub["significant_bh_fdr_0_05"].sum()),
    )
    print()

assert df["p_holm"].notna().all()
assert df["p_bh_fdr"].notna().all()
assert ((df["p_holm"] >= 0) & (df["p_holm"] <= 1)).all()
assert ((df["p_bh_fdr"] >= 0) & (df["p_bh_fdr"] <= 1)).all()

assert (
    df[df["scope"] == "overall"]["multiplicity_family"]
    == "primary_overall_48"
).all()

assert (
    df[df["scope"] == "dataset_category"]["multiplicity_family"]
    == "exploratory_dataset_category_384"
).all()

print("RQ4 MULTIPLE-TESTING CORRECTION QA: PASS")
