"""Run RQ3: discriminativeness from existing RQ1 explanations."""

from __future__ import annotations

import math
from typing import Any

from experiments._common import (
    PROJECT_ROOT,
    group_records,
    load_project_configs,
    read_jsonl,
    setup_project_logger,
    write_csv,
)
from src.metrics import (
    instance_idf_specificity,
    pairwise_jaccard,
    rq3_discriminativeness_summary,
)


def main() -> None:
    """Reuse RQ1 explanations and compute group plus per-instance specificity metrics."""
    config, _, _ = load_project_configs()
    logger = setup_project_logger(config)
    rq_cfg = config["rq3"]
    files = sorted(
        (PROJECT_ROOT / str(config["paths"]["explanations_dir"])).glob("rq1_*.jsonl")
    )
    records = [r for path in files for r in read_jsonl(path)]
    if not records:
        raise FileNotFoundError(
            "No RQ1 explanations found. Run experiments/run_rq1.py first."
        )
    top_k = int(rq_cfg["top_k"])
    feature_space = [str(v) for v in config["data_schema"]["predictor_features"]]
    group_keys = [
        "dataset",
        "category",
        "method",
        "llm_model",
        "prompt_variant",
        "temperature",
        "top_p",
    ]
    rows: list[dict[str, Any]] = []
    per_instance_rows: list[dict[str, Any]] = []
    for key, group in group_records(records, group_keys).items():
        rows.append(
            dict(zip(group_keys, key))
            | {
                "n_explanations": len(group),
                **rq3_discriminativeness_summary(group, feature_space, top_k),
            }
        )
        idf = instance_idf_specificity(group, top_k)
        by_instance = group_records(group, ["instance_id"])
        for instance_key, instance_records in by_instance.items():
            instance_id = str(instance_key[0])
            within_values = [
                pairwise_jaccard(instance_records[i], instance_records[j], top_k)
                for i in range(len(instance_records))
                for j in range(i + 1, len(instance_records))
            ]
            other_records = [
                r for r in group if str(r.get("instance_id")) != instance_id
            ]
            between_values = [
                pairwise_jaccard(a, b, top_k)
                for a in instance_records
                for b in other_records
            ]
            within = (
                sum(within_values) / len(within_values)
                if within_values
                else float("nan")
            )
            between = (
                sum(between_values) / len(between_values)
                if between_values
                else float("nan")
            )
            gap = (
                within - between
                if not math.isnan(within) and not math.isnan(between)
                else float("nan")
            )
            per_instance_rows.append(
                dict(zip(group_keys, key))
                | {
                    "instance_id": instance_id,
                    "pairwise_jaccard_within_instance": within,
                    "pairwise_jaccard_between_instance": between,
                    "instance_idf_specificity": idf["per_instance"].get(
                        instance_id, float("nan")
                    ),
                    "separability_gap": gap,
                }
            )
    metric_dir = PROJECT_ROOT / str(config["paths"]["metrics_dir"])
    write_csv(metric_dir / "rq3_discriminativeness.csv", rows)
    write_csv(metric_dir / "rq3_instance_specificity.csv", per_instance_rows)
    logger.info(
        "RQ3 complete: %s group rows, %s instance rows",
        len(rows),
        len(per_instance_rows),
    )


if __name__ == "__main__":
    main()
