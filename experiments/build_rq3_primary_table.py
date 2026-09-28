from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STATS = (
    ROOT
    / "results"
    / "qwen36_rq6_final"
    / "statistics"
)

OUT = (
    ROOT
    / "results"
    / "qwen36_rq6_final"
    / "publication"
)

OUT.mkdir(parents=True, exist_ok=True)


friedman = pd.read_csv(
    STATS / "rq6_friedman_kendalls_w.csv"
)

pairwise = pd.read_csv(
    STATS / "rq6_full_vs_ablation_wilcoxon_holm.csv"
)


ABLATION_LABELS = {
    "no_semantics": "No semantics",
    "no_grounding": "No grounding",
    "no_constraints": "No constraints",
    "no_instance_context": "No instance context",
}


rows = []

for _, frow in friedman.iterrows():

    metric = frow["metric"]

    row = {
        "metric": metric,
        "friedman_p": frow["friedman_p"],
        "kendalls_w": frow["kendalls_w"],
    }

    sub = pairwise[
        pairwise["metric"] == metric
    ]

    for ablation, label in ABLATION_LABELS.items():

        r = sub[
            sub["ablation"] == ablation
        ].iloc[0]

        prefix = ablation

        row[f"{prefix}_delta_full_minus_ablation"] = (
            r["mean_difference_full_minus_ablation"]
        )

        row[f"{prefix}_p_holm"] = r["p_holm"]

        row[f"{prefix}_significant"] = (
            r["significant_holm_0_05"]
        )

    rows.append(row)


df = pd.DataFrame(rows)

csv_path = OUT / "rq6_primary_table.csv"
df.to_csv(csv_path, index=False)


# Compact Markdown version for manuscript drafting.
def fmt_p(p):
    p = float(p)

    if p < 0.001:
        return "<.001"

    return f"{p:.3f}"


def fmt_delta(x):
    return f"{float(x):+.3f}"


md_rows = []

for _, r in df.iterrows():

    md_rows.append({
        "Metric": r["metric"],
        "Friedman p": fmt_p(r["friedman_p"]),
        "Kendall W": f'{r["kendalls_w"]:.3f}',
        "No semantics Δ": fmt_delta(
            r["no_semantics_delta_full_minus_ablation"]
        ),
        "No semantics Holm p": fmt_p(
            r["no_semantics_p_holm"]
        ),
        "No grounding Δ": fmt_delta(
            r["no_grounding_delta_full_minus_ablation"]
        ),
        "No grounding Holm p": fmt_p(
            r["no_grounding_p_holm"]
        ),
        "No constraints Δ": fmt_delta(
            r["no_constraints_delta_full_minus_ablation"]
        ),
        "No constraints Holm p": fmt_p(
            r["no_constraints_p_holm"]
        ),
        "No instance context Δ": fmt_delta(
            r["no_instance_context_delta_full_minus_ablation"]
        ),
        "No instance context Holm p": fmt_p(
            r["no_instance_context_p_holm"]
        ),
    })


md = pd.DataFrame(md_rows)

md_path = OUT / "rq6_primary_table.md"

with md_path.open("w") as f:
    headers = list(md.columns)

    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join("---" for _ in headers) + " |\n")

    for _, row in md.iterrows():
        vals = [
            str(row[h]).replace("|", "\\|")
            for h in headers
        ]
        f.write("| " + " | ".join(vals) + " |\n")


print("RQ6 PRIMARY TABLE: PASS")
print("Rows =", len(df))
print(csv_path)
print(md_path)

print()
print(md.to_string(index=False))
