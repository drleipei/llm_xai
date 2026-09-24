from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PAIRWISE = (
    ROOT
    / "results"
    / "qwen36_rq6_final"
    / "statistics"
    / "rq6_full_vs_ablation_wilcoxon_holm.csv"
)

OUT = (
    ROOT
    / "results"
    / "qwen36_rq6_final"
    / "publication"
)

OUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(PAIRWISE)

rows = []

for ablation, sub in df.groupby("ablation"):

    sig = sub[
        sub["significant_holm_0_05"] == True
    ]

    positive = sig[
        sig["mean_difference_full_minus_ablation"] > 0
    ]

    negative = sig[
        sig["mean_difference_full_minus_ablation"] < 0
    ]

    rows.append({
        "ablation": ablation,
        "significant_metrics": len(sig),
        "full_higher_significant_metrics": len(positive),
        "ablation_higher_significant_metrics": len(negative),
        "mean_absolute_effect_across_metrics":
            sub[
                "mean_difference_full_minus_ablation"
            ].abs().mean(),
    })


out = (
    pd.DataFrame(rows)
    .sort_values(
        [
            "significant_metrics",
            "mean_absolute_effect_across_metrics",
        ],
        ascending=[False, False],
    )
)

path = OUT / "rq6_component_summary.csv"

out.to_csv(path, index=False)

print(out.to_string(index=False))
print()
print("WROTE:", path)
