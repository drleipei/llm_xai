"""Private orchestration helpers shared by RQ1--RQ6 runners.

This module does not define experimental algorithms; it only centralises config
loading, resumable JSONL I/O, deterministic run metadata, reference-artifact
loading, and the already-fixed explainer call interfaces.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    categorise_with_predictor,
    load_raw_dataset,
    stratified_sample_by_category,
)
from src.explainers import ExplanationContext, create_explainer
from src.llm import LLMClient
from src.predictor import FrozenPredictor, load_frozen_predictor
from src.utils import (
    derive_seed,
    ensure_directory,
    get_global_seed,
    load_yaml_config,
    resolve_project_path,
    setup_logging,
)


def load_project_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load project configurations, allowing an experiment-config override."""
    experiment_config = os.environ.get(
        "LLM_XAI_EXPERIMENT_CONFIG",
        "configs/experiment.yaml",
    )
    experiment_path = Path(experiment_config)
    if not experiment_path.is_absolute():
        experiment_path = PROJECT_ROOT / experiment_path

    return (
        load_yaml_config(experiment_path),
        load_yaml_config(PROJECT_ROOT / "configs" / "prompts.yaml"),
        load_yaml_config(PROJECT_ROOT / "configs" / "perturbation.yaml"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid JSON objects from JSONL; absent files yield an empty list."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL record at {path}:{line_no} must be an object")
            records.append(payload)
    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append and fsync one record immediately for crash-safe resume."""
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def error_path(config: Mapping[str, Any], rq: str) -> Path:
    """Resolve the append-only error JSONL path for one research question."""
    template = str(config["outputs"]["error_file_template"])
    return resolve_project_path(PROJECT_ROOT, template.format(rq=rq))


def append_error_record(
    config: Mapping[str, Any],
    *,
    rq: str,
    stage: str,
    metadata: Mapping[str, Any],
    exc: Exception,
) -> None:
    """Persist a recoverable failure without marking the experimental record complete.

    Failed calls remain retryable on the next run because only successful explanation/evaluation
    JSONL records participate in completion identities.
    """
    record = {
        "rq": rq,
        "stage": stage,
        **dict(metadata),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    append_jsonl(error_path(config, rq), record)


def validate_explanation_record(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Validate the unified explanation schema used by all eight explainers."""
    explanation_cfg = config["explanation"]
    required = [str(v) for v in explanation_cfg["required_record_fields"]]
    missing = [field for field in required if field not in record]
    if missing:
        raise RuntimeError(f"Explanation record missing required field(s): {missing}")
    allowed_methods = {str(v) for v in config["methods"]["all"]}
    if str(record["method"]) not in allowed_methods:
        raise RuntimeError(f"Unknown explanation method: {record['method']!r}")
    allowed_features = {str(v) for v in config["data_schema"]["predictor_features"]}
    allowed_directions = {str(v) for v in explanation_cfg["allowed_directions"]}
    seen: set[str] = set()
    top_features = record.get("top_features")
    if not isinstance(top_features, list):
        raise TypeError("Explanation top_features must be a list")
    if len(top_features) > int(explanation_cfg["top_k"]):
        raise RuntimeError("Explanation contains more than configured top_k features")
    for index, item in enumerate(top_features, 1):
        if not isinstance(item, Mapping):
            raise TypeError(f"top_features[{index}] must be an object")
        missing_item = [
            str(v)
            for v in explanation_cfg["top_feature_required_fields"]
            if v not in item
        ]
        if missing_item:
            raise RuntimeError(
                f"top_features[{index}] missing field(s): {missing_item}"
            )
        feature = str(item["feature"])
        if feature not in allowed_features:
            raise RuntimeError(
                f"top_features[{index}] contains unknown feature {feature!r}"
            )
        if feature in seen:
            raise RuntimeError(f"Duplicate feature in explanation: {feature!r}")
        seen.add(feature)
        if str(item["direction"]) not in allowed_directions:
            raise RuntimeError(
                f"Invalid direction for {feature!r}: {item['direction']!r}"
            )
        score = float(item["score"])
        if not math.isfinite(score) or score < 0:
            raise RuntimeError(f"Invalid explanation score for {feature!r}: {score!r}")
    probability = float(record["predicted_response_probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            "predicted_response_probability must be finite and in [0, 1]"
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a CSV aggregate file."""
    ensure_directory(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def identity_tuple(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Return the configured resume identity tuple for one explanation record."""
    fields = [str(v) for v in config["resume"]["record_identity_fields"]]
    return tuple(record.get(field) for field in fields)


def completed_identities(
    paths: Iterable[Path], config: Mapping[str, Any]
) -> set[tuple[Any, ...]]:
    """Collect completed record identities from existing JSONL outputs."""
    done: set[tuple[Any, ...]] = set()
    for path in paths:
        for record in read_jsonl(path):
            done.add(identity_tuple(record, config))
    return done


def explanation_path(
    config: Mapping[str, Any],
    *,
    rq: str,
    dataset: str,
    method: str,
    llm_model: str | None,
    prompt_variant: str | None,
) -> Path:
    """Resolve the configured per-condition explanation JSONL path."""
    template = str(config["outputs"]["explanation_file_template"])
    relative = template.format(
        rq=rq,
        dataset=dataset,
        method=method,
        llm_model_or_none=llm_model or "none",
        prompt_variant_or_none=prompt_variant or "none",
    )
    return resolve_project_path(PROJECT_ROOT, relative)


def load_dataset_state(
    config: Mapping[str, Any],
    dataset: str,
    rq_for_sampling: str,
) -> tuple[pd.DataFrame, pd.DataFrame, FrozenPredictor]:
    """Load raw data, frozen predictor, categorise TP/TN/FP/FN, and sample strata."""
    predictor = load_frozen_predictor(config, dataset, PROJECT_ROOT)
    raw = load_raw_dataset(config, dataset, PROJECT_ROOT)
    categorised = categorise_with_predictor(raw, predictor, config)
    sampled = stratified_sample_by_category(
        categorised, config, rq_for_sampling, dataset
    )
    return categorised, sampled, predictor


def load_generation_reference_data(
    config: Mapping[str, Any], dataset: str
) -> pd.DataFrame:
    """Load the pre-created generation reference partition.

    The filename is deliberately separate from raw/evaluation samples. Runners
    never construct this partition from the evaluation pool. Users should place
    a predictor-feature-compatible training/reference partition at
    ``<processed_dir>/generation_reference.csv`` before generation experiments.
    """
    processed = resolve_project_path(
        PROJECT_ROOT, str(config["datasets"][dataset]["processed_dir"])
    )
    path = processed / "generation_reference.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Generation reference data not found: {path}. Create it from the training/predefined "
            "reference partition; the runner will not fall back to the evaluation pool."
        )
    frame = pd.read_csv(path)
    features = [str(v) for v in config["data_schema"]["predictor_features"]]
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Generation reference data missing predictor features: {missing}"
        )
    return frame.loc[:, features].copy()


def reference_values_from_stats(
    stats: Mapping[str, Mapping[str, float]], features: Sequence[str]
) -> dict[str, float]:
    """Build deterministic LOFO replacement values from training-derived medians."""
    result: dict[str, float] = {}
    for feature in features:
        values = stats.get(feature)
        if values and "median" in values:
            result[feature] = float(values["median"])
    return result


def candidate_values_from_stats(
    stats: Mapping[str, Mapping[str, float]], features: Sequence[str]
) -> dict[str, list[float]]:
    """Build deterministic counterfactual candidates from configured reference statistics."""
    result: dict[str, list[float]] = {}
    for feature in features:
        values = stats.get(feature)
        if not values:
            continue
        ordered = [values.get(name) for name in ("q25", "q75", "min", "max", "median")]
        result[feature] = [
            float(v)
            for i, v in enumerate(ordered)
            if v is not None
            and float(v) not in [float(x) for x in ordered[:i] if x is not None]
        ]
    return result


def run_seed(
    config: Mapping[str, Any],
    *,
    rq: str,
    dataset: str,
    category: str,
    instance_id: str,
    method: str,
    run_id: int,
    llm_model: str | None,
    prompt_variant: str | None,
    temperature: float | None,
    top_p: float | None,
) -> int:
    """Derive the reproducible seed from all configured condition dimensions."""
    return derive_seed(
        get_global_seed(config),
        rq,
        dataset,
        category,
        method,
        instance_id,
        run_id,
        llm_model or "none",
        prompt_variant or "none",
        temperature,
        top_p,
    )


def build_context(
    *,
    dataset: str,
    category: str,
    instance_id: str,
    run_id: int,
    seed: int,
    llm_model: str | None,
    prompt_variant: str | None,
    temperature: float | None,
    top_p: float | None,
) -> ExplanationContext:
    """Construct the fixed explanation metadata object."""
    return ExplanationContext(
        dataset=dataset,
        category=category,
        instance_id=instance_id,
        run_id=run_id,
        seed=seed,
        llm_model=llm_model,
        prompt_variant=prompt_variant,
        temperature=temperature,
        top_p=top_p,
    )


def generate_one_explanation(
    *,
    config: Mapping[str, Any],
    prompts_config: Mapping[str, Any],
    predictor: FrozenPredictor,
    method: str,
    instance: pd.Series,
    context: ExplanationContext,
    reference_data: pd.DataFrame,
    reference_stats: Mapping[str, Mapping[str, float]],
    llm_client: LLMClient | None,
) -> dict[str, Any]:
    """Call one of the eight already-fixed explainer interfaces."""
    explainer = create_explainer(
        method,
        predictor,
        config,
        prompts_config=prompts_config if method.startswith("llm_") else None,
        llm_client=llm_client,
        project_root=PROJECT_ROOT if method.startswith("llm_") else None,
    )
    features = [str(v) for v in config["data_schema"]["predictor_features"]]
    summary: Mapping[str, Any] = reference_stats
    if method in {"lime", "kernelshap"}:
        result = explainer.explain(instance, context, reference_data=reference_data)
    elif method == "lofo":
        result = explainer.explain(
            instance,
            context,
            reference_values=reference_values_from_stats(reference_stats, features),
        )
    elif method == "counterfactual":
        result = explainer.explain(
            instance,
            context,
            candidate_values=candidate_values_from_stats(reference_stats, features),
        )
    elif method == "llm_kernelshap":
        result = explainer.explain(
            instance, context, reference_data=reference_data, reference_summary=summary
        )
    else:
        result = explainer.explain(instance, context, reference_summary=summary)
    validate_explanation_record(result, config)
    return result


def explanation_conditions(
    config: Mapping[str, Any],
    prompts_config: Mapping[str, Any],
    *,
    rq: str,
) -> list[tuple[str, str | None, str | None, float | None, float | None, str | None]]:
    """Enumerate method/model/prompt/decoding conditions for RQ1, RQ5, or RQ6.

    Returns tuples ``(method, model, variant, temperature, top_p, decoding_id)``.
    """
    rq_cfg = config[rq]

    model_spec = rq_cfg.get("llm_models", "all")
    if model_spec == "all":
        llm_models = [str(v) for v in config["llm_models"]]
    else:
        if not isinstance(model_spec, list):
            raise ValueError(
                f"{rq}.llm_models must be a list or 'all', got {model_spec!r}"
            )
        llm_models = [str(v) for v in model_spec]

    unknown_models = sorted(set(llm_models) - set(config["llm_models"]))
    if unknown_models:
        raise ValueError(
            f"{rq}.llm_models contains unknown models: {unknown_models}"
        )

    default_variant = str(prompts_config["default_variant"])
    default_decoding = config["llm_default_decoding"]
    conditions: list[
        tuple[str, str | None, str | None, float | None, float | None, str | None]
    ] = []
    if rq == "rq1":
        method_spec = rq_cfg.get("methods", "all")
        if method_spec == "all":
            methods = [str(v) for v in config["methods"]["all"]]
        else:
            if not isinstance(method_spec, list):
                raise ValueError(
                    f"{rq}.methods must be a list or 'all', got {method_spec!r}"
                )
            methods = [str(v) for v in method_spec]

        unknown_methods = sorted(
            set(methods) - set(config["methods"]["all"])
        )
        if unknown_methods:
            raise ValueError(
                f"{rq}.methods contains unknown methods: {unknown_methods}"
            )

        for method in methods:
            if method.startswith("llm_"):
                for model in llm_models:
                    conditions.append(
                        (
                            method,
                            str(model),
                            default_variant,
                            None
                            if default_decoding.get("temperature") is None
                            else float(default_decoding["temperature"]),
                            None
                            if default_decoding.get("top_p") is None
                            else float(default_decoding["top_p"]),
                            None,
                        )
                    )
            else:
                conditions.append((method, None, None, None, None, None))
    elif rq == "rq5":
        for method in rq_cfg["methods"]:
            for model in llm_models:
                for decoding in rq_cfg["configurations"]:
                    conditions.append(
                        (
                            str(method),
                            str(model),
                            default_variant,
                            float(decoding["temperature"]),
                            float(decoding["top_p"]),
                            str(decoding["id"]),
                        )
                    )
    elif rq == "rq6":
        rq6_decoding = rq_cfg.get(
            "decoding",
            default_decoding,
        )
        temperature = (
            None
            if rq6_decoding.get("temperature") is None
            else float(rq6_decoding["temperature"])
        )
        top_p = (
            None
            if rq6_decoding.get("top_p") is None
            else float(rq6_decoding["top_p"])
        )
        for method in rq_cfg["methods"]:
            for model in llm_models:
                for variant in rq_cfg["variants"]:
                    conditions.append(
                        (
                            str(method),
                            str(model),
                            str(variant),
                            temperature,
                            top_p,
                            None,
                        )
                    )
    else:
        raise ValueError(f"Unsupported generation RQ: {rq}")
    return conditions


def finite_or_none(value: Any) -> float | None:
    """Return finite float or None for CSV-friendly aggregates."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def group_records(
    records: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group records by an ordered key tuple."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(k) for k in keys)].append(dict(record))
    return groups


def setup_project_logger(config: Mapping[str, Any]):
    """Create the configured project logger."""
    return setup_logging(config, PROJECT_ROOT)
