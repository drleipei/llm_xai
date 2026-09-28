"""Run RQ2: independently evaluate existing RQ1 explanations against model response.

This runner never constructs an LLMClient and never regenerates explanations.
"""

from __future__ import annotations

from typing import Any

from experiments._common import (
    PROJECT_ROOT,
    append_jsonl,
    group_records,
    load_project_configs,
    read_jsonl,
    setup_project_logger,
    write_csv,
)
from src.data import load_raw_dataset
from src.metrics import rq2_alignment_summary
from src.perturbation import (
    build_independent_evaluation_probe,
    evaluate_independent_probe,
    evaluation_audit_record,
    generation_probe_from_record,
    load_reference_statistics,
)
from src.predictor import load_frozen_predictor


def _evaluation_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(k)
        for k in (
            "dataset",
            "category",
            "instance_id",
            "method",
            "llm_model",
            "run_id",
            "feature_rank",
        )
    )


def main() -> None:
    """Read RQ1 JSONL, perform independent Top-5 interventions, and aggregate RQ2."""
    config, _, perturbation_config = load_project_configs()
    logger = setup_project_logger(config)
    rq_cfg = config["rq2"]
    if not bool(rq_cfg["do_not_call_llm_during_evaluation"]):
        raise RuntimeError("RQ2 config must prohibit LLM calls during evaluation")
    if not bool(rq_cfg["generation_probe_and_evaluation_must_be_separate"]):
        raise RuntimeError("RQ2 separation contract is disabled")

    explanation_files = sorted(
        (PROJECT_ROOT / str(config["paths"]["explanations_dir"])).glob("rq1_*.jsonl")
    )
    explanations = [record for path in explanation_files for record in read_jsonl(path)]
    if not explanations:
        raise FileNotFoundError(
            "No RQ1 explanations found. Run experiments/run_rq1.py first."
        )

    eval_path = PROJECT_ROOT / str(rq_cfg["evaluation_detail_output"])
    completed = {_evaluation_identity(record) for record in read_jsonl(eval_path)}
    features = [str(v) for v in config["data_schema"]["predictor_features"]]
    instance_col = str(config["data_schema"]["instance_id_column"])
    dataset_cache: dict[str, Any] = {}

    for explanation in explanations:
        dataset = str(explanation["dataset"])
        if dataset not in dataset_cache:
            frame = load_raw_dataset(config, dataset, PROJECT_ROOT)
            index = {str(row[instance_col]): row for _, row in frame.iterrows()}
            dataset_cache[dataset] = (
                index,
                load_frozen_predictor(config, dataset, PROJECT_ROOT),
                load_reference_statistics(
                    dataset_id=dataset,
                    perturbation_config=perturbation_config,
                    project_root=PROJECT_ROOT,
                ),
            )
        index, predictor, stats = dataset_cache[dataset]
        instance_id = str(explanation["instance_id"])
        if instance_id not in index:
            raise RuntimeError(
                f"Explanation instance not found in raw dataset: {dataset}/{instance_id}"
            )
        instance = index[instance_id]
        for feature_rank, feature_record in enumerate(
            explanation.get("top_features", [])[: int(rq_cfg["top_k"])], 1
        ):
            identity_stub = {
                "dataset": dataset,
                "category": explanation["category"],
                "instance_id": instance_id,
                "method": explanation["method"],
                "llm_model": explanation.get("llm_model"),
                "run_id": explanation["run_id"],
                "feature_rank": feature_rank,
            }
            if _evaluation_identity(identity_stub) in completed:
                continue
            feature = str(feature_record["feature"])
            # The independent constructor has no generation-probe argument by design.
            probe = build_independent_evaluation_probe(
                instance, feature, stats, perturbation_config, features
            )
            result = evaluate_independent_probe(
                predictor,
                instance,
                probe,
                perturbation_config,
                config,
                claimed_direction=str(feature_record["direction"]),
            )
            generation_probe = generation_probe_from_record(
                feature_record,
                original_probability=explanation.get("predicted_response_probability"),
            )
            audit = evaluation_audit_record(
                result,
                schema_version=str(perturbation_config["schema_version"]),
                dataset=dataset,
                category=str(explanation["category"]),
                instance_id=instance_id,
                method=str(explanation["method"]),
                llm_model=explanation.get("llm_model"),
                run_id=int(explanation["run_id"]),
                feature_rank=feature_rank,
                generation_probe=generation_probe,
            )
            audit.update(
                {
                    "rq": "rq2",
                    "seed": explanation.get("seed"),
                    "prompt_variant": explanation.get("prompt_variant"),
                    "temperature": explanation.get("temperature"),
                    "top_p": explanation.get("top_p"),
                }
            )
            append_jsonl(eval_path, audit)
            completed.add(_evaluation_identity(audit))
            logger.info(
                "RQ2 saved %s/%s/%s feature=%s",
                dataset,
                instance_id,
                explanation["method"],
                feature,
            )

    evaluations = read_jsonl(eval_path)
    threshold = float(perturbation_config["meaningful_effect"]["primary_threshold"])
    keys = [
        "dataset",
        "category",
        "instance_id",
        "method",
        "llm_model",
        "prompt_variant",
        "temperature",
        "top_p",
        "run_id",
    ]
    rows: list[dict[str, Any]] = []
    for key, records in group_records(evaluations, keys).items():
        rows.append(dict(zip(keys, key)) | rq2_alignment_summary(records, threshold))
    write_csv(PROJECT_ROOT / str(rq_cfg["aggregate_output"]), rows)
    logger.info(
        "RQ2 complete: %s per-instance/run metric rows; no LLM calls made", len(rows)
    )


if __name__ == "__main__":
    main()
