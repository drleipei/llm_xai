from pathlib import Path
import csv
import json
from collections import Counter

ROOT = Path(
    "results/qwen36_rq5_shards"
)

MANIFEST = Path(
    "configs/rq5_shards/manifest.tsv"
)

EXPECTED_DECODINGS = {
    "t00_p100",
    "t02_p100",
    "t05_p100",
    "t08_p100",
    "t02_p090",
    "t02_p080",
}

with MANIFEST.open() as f:
    rows = list(
        csv.DictReader(
            f,
            delimiter="\t",
        )
    )

if len(rows) != 32:
    raise RuntimeError(
        f"Expected 32 RQ5 shards, found {len(rows)}"
    )

all_records = []
all_ids = set()

baseline_ids = set()

problems = []

for row in rows:

    name = row["shard_name"]

    expected_dataset = row["dataset"]
    expected_method = row["method"]
    expected_category = row["category"]

    shard = ROOT / name

    files = sorted(
        p
        for p in shard.rglob("rq5_*.jsonl")
        if p.parent.name == "explanations"
    )

    records = []

    for p in files:

        with p.open() as f:

            for line in f:

                if line.strip():
                    records.append(
                        json.loads(line)
                    )

    if len(records) != 600:
        problems.append(
            f"{name}: explanations="
            f"{len(records)} expected=600"
        )

    shard_ids = set()

    decoding_count = Counter()

    instance_ids = set()

    for r in records:

        if r.get("dataset") != expected_dataset:
            problems.append(
                f"{name}: wrong dataset"
            )

        if r.get("method") != expected_method:
            problems.append(
                f"{name}: wrong method"
            )

        if r.get("category") != expected_category:
            problems.append(
                f"{name}: wrong category"
            )

        did = r.get(
            "decoding_config_id"
        )

        decoding_count[did] += 1

        instance_ids.add(
            str(r["instance_id"])
        )

        identity = (
            str(r["dataset"]),
            str(r["category"]),
            str(r["instance_id"]),
            str(r["method"]),
            str(r["llm_model"]),
            int(r["run_id"]),
            str(did),
        )

        if identity in shard_ids:
            problems.append(
                f"{name}: duplicate identity "
                f"{identity}"
            )

        shard_ids.add(identity)

        if identity in all_ids:
            problems.append(
                "GLOBAL duplicate identity "
                f"{identity}"
            )

        all_ids.add(identity)

        if did == "t02_p100":

            try:
                t = float(
                    r["temperature"]
                )
                pval = float(
                    r["top_p"]
                )
            except Exception:
                problems.append(
                    f"{name}: invalid baseline "
                    "decoding fields"
                )
                continue

            if (
                abs(t - 0.2) > 1e-12
                or abs(pval - 1.0) > 1e-12
                or r.get("prompt_variant") != "full"
            ):
                problems.append(
                    f"{name}: invalid "
                    "t02_p100 baseline"
                )

            baseline_identity = (
                str(r["dataset"]),
                str(r["category"]),
                str(r["instance_id"]),
                str(r["method"]),
                str(r["llm_model"]),
                int(r["run_id"]),
            )

            if baseline_identity in baseline_ids:
                problems.append(
                    "GLOBAL duplicate baseline "
                    f"{baseline_identity}"
                )

            baseline_ids.add(
                baseline_identity
            )

    if len(instance_ids) != 20:
        problems.append(
            f"{name}: instances="
            f"{len(instance_ids)} expected=20"
        )

    if set(decoding_count) != EXPECTED_DECODINGS:
        problems.append(
            f"{name}: decoding set="
            f"{set(decoding_count)}"
        )

    for did in EXPECTED_DECODINGS:

        if decoding_count[did] != 100:
            problems.append(
                f"{name}: {did}="
                f"{decoding_count[did]} "
                "expected=100"
            )

    # Every instance × decoding must have runs 1..5.
    run_map = {}

    for r in records:

        key = (
            str(r["instance_id"]),
            str(
                r["decoding_config_id"]
            ),
        )

        run_map.setdefault(
            key,
            set(),
        ).add(
            int(r["run_id"])
        )

    for key, runs in run_map.items():

        if runs != {1, 2, 3, 4, 5}:

            problems.append(
                f"{name}: runs {key}="
                f"{sorted(runs)}"
            )

    all_records.extend(records)


print("===== RQ5 FINAL GATE =====")
print("shards =", len(rows))
print(
    "explanations =",
    len(all_records),
)
print(
    "unique scientific identities =",
    len(all_ids),
)
print(
    "RQ6 reusable t02_p100 =",
    len(baseline_ids),
)

if len(all_records) != 19200:

    problems.append(
        "GLOBAL explanations="
        f"{len(all_records)} "
        "expected=19200"
    )

if len(all_ids) != 19200:

    problems.append(
        "GLOBAL unique identities="
        f"{len(all_ids)} "
        "expected=19200"
    )

if len(baseline_ids) != 3200:

    problems.append(
        "RQ6 baseline="
        f"{len(baseline_ids)} "
        "expected=3200"
    )

if problems:

    print("\n===== FAILURES =====")

    for item in problems[:200]:
        print("FAIL:", item)

    print(
        "\nRQ5 FINAL GATE: FAIL"
    )

    raise SystemExit(1)

print()
print("RQ5 FINAL GATE: PASS")
print(
    "RQ6 baseline is ready."
)
