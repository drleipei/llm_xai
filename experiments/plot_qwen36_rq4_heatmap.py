from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT = Path(
    "results/rq4_qwen36_comparison/statistics/"
    "rq4_primary_overall_interpreted.csv"
)

OUTDIR = Path("results/qwen36_27b_full/publication")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

method_order = [
    "counterfactual",
    "kernelshap",
    "lime",
    "lofo",
]

method_labels = [
    "Counterfactual",
    "KernelSHAP",
    "LIME",
    "LOFO",
]

metric_order = [
    "direction_agreement_at_k",
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_absolute_delta_probability",
    "overlap_at_k",
    "rank_agreement_at_k",
    "score_stability",
    "pairwise_jaccard_within_instance",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
    "separability_gap",
    "mean_delta_probability",
]

metric_labels = [
    "Direction Agreement@K",
    "Direction Consistency",
    "Meaningful Effect",
    "Mean |Δp|",
    "Overlap@K",
    "Rank Agreement@K",
    "Score Stability",
    "Within-instance Jaccard",
    "Between-instance Jaccard",
    "Instance-IDF",
    "Separability Gap",
    "Mean Δp",
]

codes = {
    "traditional_better": -1,
    "no_significant_difference": 0,
    "llm_informed_better": 1,
    "not_directional": 2,
}

symbols = {
    -1: "T",
    0: "NS",
    1: "Q",
    2: "N/A",
}

matrix = np.full(
    (len(metric_order), len(method_order)),
    np.nan,
)

for i, metric in enumerate(metric_order):
    for j, method in enumerate(method_order):

        sub = df[
            (df["metric"] == metric)
            & (df["traditional_method"] == method)
        ]

        assert len(sub) == 1, (metric, method)

        outcome = sub.iloc[0]["holm_interpretation"]
        matrix[i, j] = codes[outcome]

fig, ax = plt.subplots(
    figsize=(8.5, 8.5)
)

im = ax.imshow(
    matrix,
    aspect="auto",
)

ax.set_xticks(range(len(method_labels)))
ax.set_xticklabels(method_labels, rotation=25, ha="right")

ax.set_yticks(range(len(metric_labels)))
ax.set_yticklabels(metric_labels)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        code = int(matrix[i, j])
        ax.text(
            j,
            i,
            symbols[code],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

ax.set_title(
    "Traditional vs Qwen3.6-informed XAI\n"
    "Holm-corrected overall paired comparisons"
)

ax.set_xlabel("Base explanation method")
ax.set_ylabel("Evaluation metric")

fig.tight_layout()

png = OUTDIR / "figure_rq4_outcome_heatmap.png"
pdf = OUTDIR / "figure_rq4_outcome_heatmap.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")

print("saved:", png)
print("saved:", pdf)
print()
print("Legend:")
print("T   = Traditional significantly better")
print("Q   = Qwen3.6-informed significantly better")
print("NS  = No significant difference")
print("N/A = Signed metric; directional winner not assigned")
