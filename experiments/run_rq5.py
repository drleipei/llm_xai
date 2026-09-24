"""Run RQ5: sensitivity to the six configured temperature/top-p conditions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments._common import (
    PROJECT_ROOT,
    append_error_record,
    append_jsonl,
    build_context,
    explanation_conditions,
    explanation_path,
    generate_one_explanation,
    group_records,
    identity_tuple,
    load_dataset_state,
    load_generation_reference_data,
    load_project_configs,
    read_jsonl,
    run_seed,
    setup_project_logger,
    write_csv,
)
from src.llm import LLMClient
from experiments.rq56_helpers import structural_validation
from src.metrics import (
    pairwise_stability_summary,
    rq2_alignment_summary,
    rq3_discriminativeness_summary,
)
from src.perturbation import (
    build_independent_evaluation_probe,
    evaluate_independent_probe,
    evaluation_audit_record,
    generation_probe_from_record,
    load_reference_statistics,
)


def _eval_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(k)
        for k in (
            "dataset",
            "category",
            "instance_id",
            "method",
            "llm_model",
            "run_id",
            "prompt_variant",
            "temperature",
            "top_p",
            "feature_rank",
        )
    )


def main() -> None:
    """Generate six decoding conditions, resume safely, and compute RQ1--RQ3 outcomes."""
    config, prompts_config, perturbation_config = load_project_configs()
    logger = setup_project_logger(config)
    rq = "rq5"
    rq_cfg = config[rq]
    if len(rq_cfg["configurations"]) != 6:
        raise RuntimeError("RQ5 requires exactly six unique decoding configurations")
    category_col = str(config["data_schema"]["derived_columns"]["prediction_category"])
    instance_col = str(config["data_schema"]["instance_id_column"])
    top_k = int(config["explanation"]["top_k"])
    feature_space = [str(v) for v in config["data_schema"]["predictor_features"]]
    eval_path = (
        PROJECT_ROOT / str(config["paths"]["evaluations_dir"]) / "rq5_evaluations.jsonl"
    )
    eval_done = {_eval_identity(r) for r in read_jsonl(eval_path)}
    output_paths: set[Path] = set()

    for dataset in config["datasets"]:
        _, sample, predictor = load_dataset_state(config, str(dataset), rq)
        reference_data = load_generation_reference_data(config, str(dataset))
        stats = load_reference_statistics(
            dataset_id=str(dataset),
            perturbation_config=perturbation_config,
            project_root=PROJECT_ROOT,
        )
        for (
            method,
            model,
            variant,
            temperature,
            top_p,
            decoding_id,
        ) in explanation_conditions(config, prompts_config, rq=rq):
            assert model is not None
            path = explanation_path(
                config,
                rq=rq,
                dataset=str(dataset),
                method=method,
                llm_model=model,
                prompt_variant=variant,
            )
            output_paths.add(path)
            existing_records = read_jsonl(path)
            record_map = {
                identity_tuple(record, config): record for record in existing_records
            }
            done = set(record_map)
            client = LLMClient(model, config, PROJECT_ROOT)
            for _, instance in sample.iterrows():
                category, instance_id = (
                    str(instance[category_col]),
                    str(instance[instance_col]),
                )
                for run_id in range(
                    int(config["runs"]["run_id_start"]),
                    int(config["runs"]["run_id_start"]) + int(rq_cfg["repeated_runs"]),
                ):
                    seed = run_seed(
                        config,
                        rq=rq,
                        dataset=str(dataset),
                        category=category,
                        instance_id=instance_id,
                        method=method,
                        run_id=run_id,
                        llm_model=model,
                        prompt_variant=variant,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    context = build_context(
                        dataset=str(dataset),
                        category=category,
                        instance_id=instance_id,
                        run_id=run_id,
                        seed=seed,
                        llm_model=model,
                        prompt_variant=variant,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    stub = {
                        "rq": rq,
                        "dataset": str(dataset),
                        "category": category,
                        "instance_id": instance_id,
                        "method": method,
                        "llm_model": model,
                        "run_id": run_id,
                        "prompt_variant": variant,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    if identity_tuple(stub, config) not in done:
                        try:
                            explanation = generate_one_explanation(
                                config=config,
                                prompts_config=prompts_config,
                                predictor=predictor,
                                method=method,
                                instance=instance,
                                context=context,
                                reference_data=reference_data,
                                reference_stats=stats,
                                llm_client=client,
                            )
                        except Exception as exc:
                            append_error_record(
                                config,
                                rq=rq,
                                stage="explanation_generation",
                                metadata=stub
                                | {"seed": seed, "decoding_config_id": decoding_id},
                                exc=exc,
                            )
                            logger.exception(
                                "RQ5 failed %s/%s/%s/%s run=%s decoding=%s; failure remains retryable",
                                dataset,
                                category,
                                method,
                                model,
                                run_id,
                                decoding_id,
                            )
                            continue
                        json_audit = getattr(
                            client,
                            "last_json_audit",
                            {},
                        )

                        structure = structural_validation(
                            explanation,
                            feature_space=feature_space,
                            top_k=top_k,
                        )

                        explanation.update(
                            {
                                "rq": rq,
                                "decoding_config_id": decoding_id,
                                "json_attempt_count":
                                    json_audit.get("attempt_count"),
                                "first_attempt_direct_json_object":
                                    json_audit.get(
                                        "first_attempt_direct_json_object"
                                    ),
                                "first_attempt_json_valid":
                                    json_audit.get(
                                        "first_attempt_json_valid"
                                    ),
                                "final_json_valid":
                                    json_audit.get("final_json_valid"),
                                "recovered_after_retry":
                                    json_audit.get(
                                        "recovered_after_retry"
                                    ),
                                **structure,
                                "structural_compliance":
                                    structure[
                                        "final_structural_compliance"
                                    ],
                                "execution_success": True,
                            }
                        )
                        append_jsonl(path, explanation)
                        record_map[identity_tuple(explanation, config)] = explanation
                        done.add(identity_tuple(explanation, config))
                    # Fetch exact saved record from the in-memory resume map.
                    explanation = record_map.get(identity_tuple(stub, config))
                    if explanation is None:
                        continue
                    for rank, feature_record in enumerate(
                        explanation.get("top_features", [])[:top_k], 1
                    ):
                        e_stub = dict(stub) | {"feature_rank": rank}
                        if _eval_identity(e_stub) in eval_done:
                            continue
                        probe = build_independent_evaluation_probe(
                            instance,
                            str(feature_record["feature"]),
                            stats,
                            perturbation_config,
                            feature_space,
                        )
                        result = evaluate_independent_probe(
                            predictor,
                            instance,
                            probe,
                            perturbation_config,
                            config,
                            claimed_direction=str(feature_record["direction"]),
                        )
                        audit = evaluation_audit_record(
                            result,
                            schema_version=str(perturbation_config["schema_version"]),
                            dataset=str(dataset),
                            category=category,
                            instance_id=instance_id,
                            method=method,
                            llm_model=model,
                            run_id=run_id,
                            feature_rank=rank,
                            generation_probe=generation_probe_from_record(
                                feature_record,
                                original_probability=explanation.get(
                                    "predicted_response_probability"
                                ),
                            ),
                        )
                        audit.update(
                            {
                                "rq": rq,
                                "seed": seed,
                                "prompt_variant": variant,
                                "temperature": temperature,
                                "top_p": top_p,
                                "decoding_config_id": decoding_id,
                            }
                        )
                        append_jsonl(eval_path, audit)
                        eval_done.add(_eval_identity(audit))

    explanations = [r for p in sorted(output_paths) for r in read_jsonl(p)]
    evaluations = read_jsonl(eval_path)
    group_keys = [
        "dataset",
        "category",
        "method",
        "llm_model",
        "prompt_variant",
        "temperature",
        "top_p",
        "decoding_config_id",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in group_records(explanations, group_keys).items():
        row = dict(zip(group_keys, key))
        # Stability: compute per instance then average across instances.
        per_instance = group_records(group, ["instance_id"])
        stable = [
            pairwise_stability_summary(v, top_k)
            for v in per_instance.values()
            if len(v) >= 2
        ]
        for metric in (
            "overlap_at_k",
            "rank_agreement_at_k",
            "direction_agreement_at_k",
            "score_stability",
        ):
            vals = [float(s[metric]) for s in stable]
            valid = [v for v in vals if not math.isnan(v)]
            row[metric] = sum(valid) / len(valid) if valid else float("nan")
        eval_group = [
            e
            for e in evaluations
            if all(
                e.get(k) == row.get(k)
                for k in (
                    "dataset",
                    "category",
                    "method",
                    "llm_model",
                    "prompt_variant",
                    "temperature",
                    "top_p",
                    "decoding_config_id",
                )
            )
        ]
        row.update(
            rq2_alignment_summary(
                eval_group,
                float(perturbation_config["meaningful_effect"]["primary_threshold"]),
            )
        )
        row.update(rq3_discriminativeness_summary(group, feature_space, top_k))
        expected = int(rq_cfg["sample_per_category_per_dataset"]) * int(
            rq_cfg["repeated_runs"]
        )
        row["structural_compliance_rate"] = (
            sum(bool(r.get("structural_compliance")) for r in group) / expected
        )
        row["execution_success_rate"] = len(group) / expected
        rows.append(row)
    write_csv(
        PROJECT_ROOT
        / str(config["paths"]["metrics_dir"])
        / "rq5_decoding_sensitivity.csv",
        rows,
    )
    logger.info("RQ5 complete: six configured decoding conditions evaluated")


if __name__ == "__main__":
    main()
