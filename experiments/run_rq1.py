"""Run RQ1: explanation stability over five repeated runs per sampled instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import (
    PROJECT_ROOT,
    append_error_record,
    append_jsonl,
    build_context,
    completed_identities,
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
from src.llm import LLMClient, LLMError
from src.metrics import pairwise_stability_summary
from src.perturbation import load_reference_statistics


def main() -> None:
    """Generate/reuse RQ1 explanations and compute per-instance stability metrics."""
    config, prompts_config, perturbation_config = load_project_configs()
    logger = setup_project_logger(config)
    rq = "rq1"
    rq_cfg = config[rq]
    top_k = int(rq_cfg["top_k"])
    repeated_runs = int(rq_cfg["repeated_runs"])
    instance_col = str(config["data_schema"]["instance_id_column"])
    category_col = str(config["data_schema"]["derived_columns"]["prediction_category"])

    output_paths: set[Path] = set()
    all_new_or_existing: list[dict[str, Any]] = []
    for dataset in rq_cfg["datasets"]:
        dataset = str(dataset)
        _, sample, predictor = load_dataset_state(config, dataset, rq)
        reference_data = load_generation_reference_data(config, dataset)
        reference_stats = load_reference_statistics(
            dataset_id=dataset,
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
            path = explanation_path(
                config,
                rq=rq,
                dataset=dataset,
                method=method,
                llm_model=model,
                prompt_variant=variant,
            )
            output_paths.add(path)
            done = completed_identities([path], config)
            client = (
                LLMClient(model, config, PROJECT_ROOT) if model is not None else None
            )
            for _, instance in sample.iterrows():
                category = str(instance[category_col])
                instance_id = str(instance[instance_col])
                for run_id in range(
                    int(config["runs"]["run_id_start"]),
                    int(config["runs"]["run_id_start"]) + repeated_runs,
                ):
                    seed = run_seed(
                        config,
                        rq=rq,
                        dataset=dataset,
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
                        dataset=dataset,
                        category=category,
                        instance_id=instance_id,
                        run_id=run_id,
                        seed=seed,
                        llm_model=model,
                        prompt_variant=variant,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    identity_record = {
                        "rq": rq,
                        "dataset": dataset,
                        "category": category,
                        "instance_id": instance_id,
                        "method": method,
                        "llm_model": model,
                        "run_id": run_id,
                        "prompt_variant": variant,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    if identity_tuple(identity_record, config) in done:
                        continue
                    try:
                        explanation = generate_one_explanation(
                            config=config,
                            prompts_config=prompts_config,
                            predictor=predictor,
                            method=method,
                            instance=instance,
                            context=context,
                            reference_data=reference_data,
                            reference_stats=reference_stats,
                            llm_client=client,
                        )
                    except Exception as exc:
                        append_error_record(
                            config,
                            rq=rq,
                            stage="explanation_generation",
                            metadata=identity_record
                            | {"seed": seed, "decoding_config_id": decoding_id},
                            exc=exc,
                        )
                        if (
                            isinstance(exc, LLMError)
                            and "budget exhausted" in str(exc).lower()
                        ):
                            logger.critical(
                                "RQ1 aborting: shared LLM budget exhausted at "
                                "%s/%s/%s/%s run=%s",
                                dataset,
                                category,
                                method,
                                model or "none",
                                run_id,
                            )
                            raise

                        logger.exception(
                            "RQ1 failed %s/%s/%s/%s run=%s; failure remains retryable",
                            dataset,
                            category,
                            method,
                            model or "none",
                            run_id,
                        )
                        continue
                    explanation["rq"] = rq
                    explanation["decoding_config_id"] = decoding_id
                    append_jsonl(path, explanation)
                    done.add(identity_tuple(explanation, config))
                    logger.info(
                        "RQ1 saved %s/%s/%s/%s run=%s",
                        dataset,
                        category,
                        method,
                        model or "none",
                        run_id,
                    )

    for path in sorted(output_paths):
        all_new_or_existing.extend(read_jsonl(path))
    metric_rows: list[dict[str, Any]] = []
    keys = [
        "dataset",
        "category",
        "instance_id",
        "method",
        "llm_model",
        "prompt_variant",
        "temperature",
        "top_p",
    ]
    for key, records in group_records(all_new_or_existing, keys).items():
        if len(records) < 2:
            continue
        values = pairwise_stability_summary(records, top_k)
        metric_rows.append(dict(zip(keys, key)) | {"n_runs": len(records), **values})
    metric_path = PROJECT_ROOT / str(config["outputs"]["metric_file_template"]).format(
        rq=rq, metric_name="stability"
    )
    write_csv(metric_path, metric_rows)
    logger.info("RQ1 complete: %s per-instance metric rows", len(metric_rows))


if __name__ == "__main__":
    main()
