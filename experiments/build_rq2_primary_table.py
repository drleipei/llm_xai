from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT
    / "results"
    / "qwen36_rq5_final"
    / "statistics"
    / "rq5_friedman_kendalls_w.csv"
)

OUT = (
    ROOT
    / "results"
    / "qwen36_rq5_final"
    / "publication"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

temp = (
    df[df.family == "temperature"]
    .set_index("metric")
)

top = (
    df[df.family == "top_p"]
    .set_index("metric")
)

assert set(temp.index) == set(top.index)
assert len(temp) == 13

rows = []

for metric in temp.index:

    rows.append({
        "metric": metric,
        "temperature_friedman_p":
            temp.loc[metric, "friedman_p"],
        "temperature_kendalls_w":
            temp.loc[metric, "kendalls_w"],
        "top_p_friedman_p":
            top.loc[metric, "friedman_p"],
        "top_p_kendalls_w":
            top.loc[metric, "kendalls_w"],
    })

out = pd.DataFrame(rows)

csv_path = OUT / "rq5_primary_table.csv"
out.to_csv(csv_path, index=False)


def fp(x):
    x = float(x)
    return "<.001" if x < .001 else f"{x:.3f}"


pretty = out.copy()

pretty["temperature_friedman_p"] = (
    pretty["temperature_friedman_p"].map(fp)
)

pretty["temperature_kendalls_w"] = (
    pretty["temperature_kendalls_w"]
    .map(lambda x: f"{float(x):.3f}")
)

pretty["top_p_friedman_p"] = (
    pretty["top_p_friedman_p"].map(fp)
)

pretty["top_p_kendalls_w"] = (
    pretty["top_p_kendalls_w"]
    .map(lambda x: f"{float(x):.3f}")
)

md_path = OUT / "rq5_primary_table.md"

headers = list(pretty.columns)

with md_path.open("w") as f:
    f.write(
        "| " + " | ".join(headers) + " |\n"
    )
    f.write(
        "| " + " | ".join(
            "---" for _ in headers
        ) + " |\n"
    )

    for _, row in pretty.iterrows():
        f.write(
            "| "
            + " | ".join(
                str(row[h])
                for h in headers
            )
            + " |\n"
        )

print("RQ5 PRIMARY TABLE: PASS")
print("Rows =", len(out))
print(csv_path)
print(md_path)
print()
print(pretty.to_string(index=False))
