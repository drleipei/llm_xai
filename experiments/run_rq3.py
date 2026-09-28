"""Run RQ6: one-component-at-a-time prompt ablation (FULL + four variants)."""

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
    """Generate/reuse FULL plus four ablations on the exact RQ5 sample and aggregate outcomes."""
    config, prompts_config, perturbation_config = load_project_configs()
    logger = setup_project_logger(config)
    rq, rq_cfg = "rq6", config["rq6"]
    variants = [str(v) for v in rq_cfg["variants"]]
    expected_variants = [
        "full",
        "no_semantics",
        "no_grounding",
        "no_constraints",
        "no_instance_context",
    ]
    if variants != expected_variants:
        raise RuntimeError(
            f"RQ6 must use FULL + four fixed ablations: {expected_variants}"
        )
    category_col = str(config["data_schema"]["derived_columns"]["prediction_category"])
    instance_col = str(config["data_schema"]["instance_id_column"])
    top_k = int(config["explanation"]["top_k"])
    feature_space = [str(v) for v in config["data_schema"]["predictor_features"]]
    eval_path = (
        PROJECT_ROOT / str(config["paths"]["evaluations_dir"]) / "rq6_evaluations.jsonl"
    )
    eval_done = {_eval_identity(r) for r in read_jsonl(eval_path)}

    # RQ6 FULL baseline is reused from the completed RQ5 t02_p100
    # decoding condition. Load the configured source once and build
    # an exact scientific-identity lookup.
    rq5_full_lookup = {}

    if bool(rq_cfg.get("reuse_full_variant_when_already_generated", False)):
        source_root = PROJECT_ROOT / str(
            rq_cfg["reuse_full_variant_source"]
        )

        if not source_root.exists():
            raise FileNotFoundError(
                f"RQ6 FULL reuse source does not exist: {source_root}"
            )

        source_files = (
            [source_root]
            if source_root.is_file()
            else sorted(
                p
                for p in source_root.rglob("rq5_*.jsonl")
                if p.parent.name == "explanations"
            )
        )

        for source_file in source_files:
            for record in read_jsonl(source_file):
                if record.get("rq") != "rq5":
                    continue

                if record.get("prompt_variant") != "full":
                    continue

                if record.get("decoding_config_id") != "t02_p100":
                    continue

                try:
                    record_temperature = float(record.get("temperature"))
                    record_top_p = float(record.get("top_p"))
                except (TypeError, ValueError):
                    continue

                if (
                    abs(record_temperature - 0.2) > 1e-12
                    or abs(record_top_p - 1.0) > 1e-12
                ):
                    continue

                key = (
                    str(record.get("dataset")),
                    str(record.get("category")),
                    str(record.get("instance_id")),
                    str(record.get("method")),
                    str(record.get("llm_model")),
                    int(record.get("run_id")),
                )

                if key in rq5_full_lookup:
                    raise RuntimeError(
                        "Duplicate RQ5 t02_p100 baseline identity: "
                        f"{key}"
                    )

                rq5_full_lookup[key] = record

        logger.info(
            "Loaded %d RQ5 t02_p100 FULL baseline records from %s",
            len(rq5_full_lookup),
            source_root,
        )
    output_paths: set[Path] = set()

    for dataset in config["datasets"]:
        # Explicitly use rq5 sampling settings to guarantee the same 20/category subset.
        _, sample, predictor = load_dataset_state(config, str(dataset), "rq5")
        reference_data = load_generation_reference_data(config, str(dataset))
        stats = load_reference_statistics(
            dataset_id=str(dataset),
            perturbation_config=perturbation_config,
            project_root=PROJECT_ROOT,
        )
        for method, model, variant, temperature, top_p, _ in explanation_conditions(
            config, prompts_config, rq=rq
        ):
            assert model is not None and variant is not None
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
            client: LLMClient | None = None
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
                        # FULL is reused from the exact RQ5 t02_p100 baseline.
                        reused = None

                        if variant == "full" and bool(
                            rq_cfg["reuse_full_variant_when_already_generated"]
                        ):
                            baseline_key = (
                                str(dataset),
                                str(category),
                                str(instance_id),
                                str(method),
                                str(model),
                                int(run_id),
                            )

                            source_record = rq5_full_lookup.get(baseline_key)

                            if source_record is None:
                                raise RuntimeError(
                                    "Missing required RQ5 t02_p100 FULL baseline "
                                    f"identity: {baseline_key}"
                                )

                            reused = dict(source_record)
                        if reused is not None:
                            explanation = reused
                            explanation.update(
                                {
                                    "rq": rq,
                                    "prompt_variant": variant,
                                    "reused_from_rq5": True,
                                    "source_rq": "rq5",
                                }
                            )
                        else:
                            if client is None:
                                client = LLMClient(model, config, PROJECT_ROOT)
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
                                    metadata=stub | {"seed": seed},
                                    exc=exc,
                                )
                                logger.exception(
                                    "RQ6 failed %s/%s/%s/%s run=%s variant=%s; failure remains retryable",
                                    dataset,
                                    category,
                                    method,
                                    model,
                                    run_id,
                                    variant,
                                )
                                continue
                            json_audit = getattr(
                                client,
                                "last_json_audit",
                                {},
                            )

                            explanation.update(
                                {
                                    "rq": rq,
                                    "reused_from_rq5": False,
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
                                        json_audit.get(
                                            "final_json_valid"
                                        ),
                                    "recovered_after_retry":
                                        json_audit.get(
                                            "recovered_after_retry"
                                        ),
                                }
                            )

                        structure = structural_validation(
                            explanation,
                            feature_space=feature_space,
                            top_k=top_k,
                        )

                        explanation.update(
                            {
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
    ]
    rows: list[dict[str, Any]] = []
    for key, group in group_records(explanations, group_keys).items():
        row = dict(zip(group_keys, key))
        stable = [
            pairwise_stability_summary(v, top_k)
            for v in group_records(group, ["instance_id"]).values()
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
        PROJECT_ROOT / str(config["paths"]["metrics_dir"]) / "rq6_prompt_ablation.csv",
        rows,
    )
    logger.info("RQ6 complete: FULL + four one-component ablations evaluated")


if __name__ == "__main__":
    main()
