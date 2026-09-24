"""Generation probes and independent RQ2 evaluation perturbations.

The two perturbation mechanisms in this module are deliberately separated.
Generation probes may originate from an explainer or LLM planner and are used
only during explanation construction/grounding.  Independent evaluation probes
are constructed solely from training/reference statistics, feature type, and
``configs/perturbation.yaml``.  The independent constructor intentionally has no
parameter through which a generation-probe value can be supplied.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # package-style import
    from .predictor import FrozenPredictor
    from .utils import ConfigError, resolve_project_path
except ImportError:  # pragma: no cover
    from predictor import FrozenPredictor
    from utils import ConfigError, resolve_project_path


class PerturbationError(RuntimeError):
    """Raised when a perturbation cannot be constructed or evaluated safely."""


@dataclass(frozen=True)
class GenerationProbe:
    """Audit representation of a generation-time probe.

    A generation probe is never used by :func:`build_independent_evaluation_probe`
    and therefore cannot determine the RQ2 evaluation value.
    """

    feature: str
    original_value: float | int
    probe_value: float | int
    original_probability: float | None = None
    probe_probability: float | None = None
    delta_probability: float | None = None


@dataclass(frozen=True)
class IndependentEvaluationProbe:
    """A method-independent, single-feature RQ2 evaluation intervention."""

    feature: str
    original_value: float | int
    evaluation_value: float | int | None
    strategy: str
    valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Observed frozen-predictor response to one independent intervention."""

    feature: str
    original_value: float | int
    evaluation_value: float | int | None
    original_predicted_label: Any
    original_predicted_response: str
    original_probability: float
    evaluation_probability: float | None
    delta_probability: float | None
    absolute_delta_probability: float | None
    claimed_direction: str | None
    observed_direction: str | None
    direction_consistent: bool | None
    meaningful_effect: bool | None
    perturbation_valid: bool
    invalid_reason: str | None


def generation_probe_from_record(
    feature_record: Mapping[str, Any],
    *,
    original_probability: float | None = None,
) -> GenerationProbe | None:
    """Extract an optional generation probe from an explanation feature record.

    This helper exists for audit/reporting only.  It does not participate in
    independent evaluation-value construction.
    """
    if "generation_probe_value" not in feature_record:
        return None
    return GenerationProbe(
        feature=str(feature_record["feature"]),
        original_value=_finite_number(feature_record["value"], "original_value"),
        probe_value=_finite_number(
            feature_record["generation_probe_value"], "generation_probe_value"
        ),
        original_probability=(
            None
            if original_probability is None
            else _finite_probability(original_probability)
        ),
        delta_probability=(
            None
            if "generation_delta_probability" not in feature_record
            else _finite_number(
                feature_record["generation_delta_probability"],
                "generation_delta_probability",
            )
        ),
    )


def compute_reference_statistics(
    training_features: pd.DataFrame,
    perturbation_config: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute configured numeric reference statistics from training data only.

    Args:
        training_features: Training/reference partition.  Evaluation instances
            must not be mixed into this frame.
        perturbation_config: Parsed ``configs/perturbation.yaml``.
        feature_names: Exact frozen-predictor feature list/order.

    Returns:
        Mapping ``feature -> {q25, median, q75, min, max}`` for every
        configured predictor feature, including binary features.
    """
    feature_types = _feature_type_config(perturbation_config)
    required_stats = list(
        perturbation_config["reference_distribution"][
            "statistics_required_for_numeric_features"
        ]
    )
    allowed = {"q25", "median", "q75", "min", "max"}
    unknown = [name for name in required_stats if name not in allowed]
    if unknown:
        raise ConfigError(f"Unsupported configured reference statistic(s): {unknown}")

    missing = [name for name in feature_names if name not in training_features.columns]
    if missing:
        raise PerturbationError(
            f"Training reference data missing predictor feature(s): {missing}"
        )

    stats: dict[str, dict[str, float]] = {}
    for feature in feature_names:
        cfg = feature_types.get(feature)
        if cfg is None:
            raise ConfigError(
                f"Missing perturbation feature_types entry for {feature!r}"
            )
        series = pd.to_numeric(training_features[feature], errors="coerce")
        values = series[np.isfinite(series.to_numpy(dtype=float))]
        if values.empty:
            raise PerturbationError(
                f"No finite training values for feature {feature!r}"
            )
        all_stats = {
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        stats[feature] = {name: all_stats[name] for name in required_stats}
    return stats


def save_reference_statistics(
    statistics: Mapping[str, Mapping[str, float]],
    *,
    dataset_id: str,
    perturbation_config: Mapping[str, Any],
    project_root: str | Path,
) -> Path:
    """Persist training-derived perturbation statistics at the configured path."""
    template = str(
        perturbation_config["reference_distribution"][
            "reference_statistics_path_template"
        ]
    )
    path = resolve_project_path(project_root, template.format(dataset=dataset_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {feature: dict(values) for feature, values in statistics.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_reference_statistics(
    *,
    dataset_id: str,
    perturbation_config: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, dict[str, float]]:
    """Load previously persisted training-derived perturbation statistics."""
    template = str(
        perturbation_config["reference_distribution"][
            "reference_statistics_path_template"
        ]
    )
    path = resolve_project_path(project_root, template.format(dataset=dataset_id))
    if not path.is_file():
        raise FileNotFoundError(f"Perturbation reference statistics not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PerturbationError(f"Invalid reference-statistics file: {path}")
    return {
        str(feature): {str(k): float(v) for k, v in values.items()}
        for feature, values in data.items()
    }


def build_independent_evaluation_probe(
    instance: pd.Series | Mapping[str, Any],
    feature: str,
    reference_statistics: Mapping[str, Mapping[str, float]],
    perturbation_config: Mapping[str, Any],
    feature_names: Sequence[str],
) -> IndependentEvaluationProbe:
    """Construct an RQ2 evaluation probe without generation-probe information.

    The evaluation value may depend only on the original instance value,
    training/reference statistics, the configured feature type, and the
    deterministic strategy in ``perturbation.yaml``.  Its signature intentionally
    has no generation-probe parameter.
    """
    _validate_separation_contract(perturbation_config)
    names = tuple(str(name) for name in feature_names)
    if feature not in names:
        return IndependentEvaluationProbe(
            feature,
            math.nan,
            None,
            "unknown",
            False,
            "feature_not_in_predictor_features",
        )

    row = pd.Series(instance)
    if feature not in row.index:
        return IndependentEvaluationProbe(
            feature, math.nan, None, "unknown", False, "feature_missing_from_instance"
        )
    try:
        original = _finite_number(row[feature], "original_value")
    except PerturbationError:
        return IndependentEvaluationProbe(
            feature, math.nan, None, "unknown", False, "original_value_not_finite"
        )

    feature_cfg = _feature_type_config(perturbation_config).get(feature)
    if not isinstance(feature_cfg, Mapping):
        raise ConfigError(f"Missing perturbation feature_types entry for {feature!r}")
    strategy = str(feature_cfg.get("evaluation_strategy", ""))

    try:
        if strategy == "flip":
            candidate = _binary_flip(original, feature_cfg, perturbation_config)
        elif strategy == "opposite_quartile":
            if feature not in reference_statistics:
                raise PerturbationError("missing_reference_statistics")
            candidate = _opposite_quartile(
                original,
                reference_statistics[feature],
                feature_cfg,
                perturbation_config,
            )
        else:
            raise ConfigError(
                f"Unsupported evaluation strategy {strategy!r} for {feature!r}"
            )
        candidate = _postprocess_value(
            candidate,
            feature_cfg,
            reference_statistics.get(feature),
            perturbation_config,
        )
    except PerturbationError as exc:
        return IndependentEvaluationProbe(
            feature, original, None, strategy, False, str(exc)
        )

    require_distinct = bool(
        perturbation_config["validation"][
            "require_evaluation_value_different_from_original"
        ]
    )
    if require_distinct and _values_equal(original, candidate):
        return IndependentEvaluationProbe(
            feature, original, None, strategy, False, "evaluation_value_equals_original"
        )
    return IndependentEvaluationProbe(
        feature, original, candidate, strategy, True, None
    )


def apply_single_feature_probe(
    instance: pd.Series | Mapping[str, Any],
    probe: IndependentEvaluationProbe,
    feature_names: Sequence[str],
    perturbation_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply one valid evaluation intervention while holding every other feature fixed."""
    if not probe.valid or probe.evaluation_value is None:
        raise PerturbationError(
            f"Cannot apply invalid perturbation: {probe.invalid_reason}"
        )
    names = tuple(str(name) for name in feature_names)
    row = pd.Series(instance)
    missing = [name for name in names if name not in row.index]
    if missing:
        raise PerturbationError(f"Instance missing predictor feature(s): {missing}")
    original = pd.DataFrame([[row[name] for name in names]], columns=names)
    changed = original.copy()
    changed.at[0, probe.feature] = probe.evaluation_value
    changed_features = [
        name
        for name in names
        if not _values_equal(original.at[0, name], changed.at[0, name])
    ]
    if bool(
        perturbation_config["validation"]["require_exactly_one_changed_feature"]
    ) and changed_features != [probe.feature]:
        raise PerturbationError(
            f"Independent evaluation must change exactly one feature; changed={changed_features}"
        )
    return changed


def evaluate_independent_probe(
    predictor: FrozenPredictor,
    instance: pd.Series | Mapping[str, Any],
    probe: IndependentEvaluationProbe,
    perturbation_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    *,
    claimed_direction: str | None = None,
) -> EvaluationResult:
    """Evaluate one independent RQ2 intervention with the frozen predictor.

    Both probabilities refer to the *same originally predicted class*.  The
    perturbed class label, even if it flips, never changes the probability target.
    """
    names = tuple(
        str(v) for v in experiment_config["data_schema"]["predictor_features"]
    )
    row = pd.Series(instance)
    X = pd.DataFrame([[row[name] for name in names]], columns=names)
    original_label = predictor.predict(X)[0]
    original_probability = float(
        predictor.probability_for_class(X, [original_label])[0]
    )
    response = _response_name(original_label, experiment_config)

    if claimed_direction is not None:
        allowed = {
            str(v)
            for v in perturbation_config["direction"]["explanation_direction_values"]
        }
        if claimed_direction not in allowed:
            raise PerturbationError(f"Invalid claimed direction: {claimed_direction!r}")

    if not probe.valid:
        return EvaluationResult(
            feature=probe.feature,
            original_value=probe.original_value,
            evaluation_value=probe.evaluation_value,
            original_predicted_label=original_label,
            original_predicted_response=response,
            original_probability=original_probability,
            evaluation_probability=None,
            delta_probability=None,
            absolute_delta_probability=None,
            claimed_direction=claimed_direction,
            observed_direction=None,
            direction_consistent=None,
            meaningful_effect=None,
            perturbation_valid=False,
            invalid_reason=probe.invalid_reason,
        )

    try:
        X_eval = apply_single_feature_probe(instance, probe, names, perturbation_config)
        evaluation_probability = float(
            predictor.probability_for_class(X_eval, [original_label])[0]
        )
        _finite_probability(evaluation_probability)
    except Exception as exc:  # noqa: BLE001 - convert predictor failure into invalid evaluation
        return EvaluationResult(
            feature=probe.feature,
            original_value=probe.original_value,
            evaluation_value=probe.evaluation_value,
            original_predicted_label=original_label,
            original_predicted_response=response,
            original_probability=original_probability,
            evaluation_probability=None,
            delta_probability=None,
            absolute_delta_probability=None,
            claimed_direction=claimed_direction,
            observed_direction=None,
            direction_consistent=None,
            meaningful_effect=None,
            perturbation_valid=False,
            invalid_reason=f"predictor_evaluation_failed:{type(exc).__name__}",
        )

    delta = evaluation_probability - original_probability
    neutral_threshold = float(perturbation_config["direction"]["neutral_threshold"])
    observed = direction_from_delta(delta, neutral_threshold)
    meaningful_threshold = float(
        perturbation_config["meaningful_effect"]["primary_threshold"]
    )
    comparison = str(perturbation_config["meaningful_effect"]["comparison"])
    if comparison != "absolute_delta_probability >= threshold":
        raise ConfigError(f"Unsupported meaningful-effect comparison: {comparison!r}")
    meaningful = abs(delta) >= meaningful_threshold
    consistent = None if claimed_direction is None else claimed_direction == observed

    return EvaluationResult(
        feature=probe.feature,
        original_value=probe.original_value,
        evaluation_value=probe.evaluation_value,
        original_predicted_label=original_label,
        original_predicted_response=response,
        original_probability=original_probability,
        evaluation_probability=evaluation_probability,
        delta_probability=delta,
        absolute_delta_probability=abs(delta),
        claimed_direction=claimed_direction,
        observed_direction=observed,
        direction_consistent=consistent,
        meaningful_effect=meaningful,
        perturbation_valid=True,
        invalid_reason=None,
    )


def direction_from_delta(delta_probability: float, neutral_threshold: float) -> str:
    """Map signed Δp to support/oppose/neutral using the configured convention.

    ``Δp = p_evaluation - p_original`` for the original predicted response.
    A negative Δp means moving the feature away from its observed value reduced
    support for that response, so the observed value is classified as ``support``.
    """
    delta = _finite_number(delta_probability, "delta_probability")
    threshold = float(neutral_threshold)
    if threshold < 0 or not math.isfinite(threshold):
        raise PerturbationError("neutral_threshold must be finite and non-negative")
    if delta < -threshold:
        return "support"
    if delta > threshold:
        return "oppose"
    return "neutral"


def evaluation_audit_record(
    result: EvaluationResult,
    *,
    schema_version: str,
    dataset: str,
    category: str,
    instance_id: str,
    method: str,
    llm_model: str | None,
    run_id: int,
    feature_rank: int,
    generation_probe: GenerationProbe | None,
) -> dict[str, Any]:
    """Build the configured RQ2 audit record without influencing evaluation."""
    generation_value = (
        None if generation_probe is None else generation_probe.probe_value
    )
    equality = (
        None
        if generation_value is None or result.evaluation_value is None
        else _values_equal(generation_value, result.evaluation_value)
    )
    return {
        "schema_version": schema_version,
        "dataset": dataset,
        "category": category,
        "instance_id": instance_id,
        "method": method,
        "llm_model": llm_model,
        "run_id": int(run_id),
        "feature_rank": int(feature_rank),
        "feature": result.feature,
        "original_value": result.original_value,
        "generation_probe_value": generation_value,
        "evaluation_value": result.evaluation_value,
        "generation_probe_equals_evaluation_value": equality,
        "original_predicted_response": result.original_predicted_response,
        "original_probability": result.original_probability,
        "evaluation_probability": result.evaluation_probability,
        "delta_probability": result.delta_probability,
        "absolute_delta_probability": result.absolute_delta_probability,
        "claimed_direction": result.claimed_direction,
        "observed_direction": result.observed_direction,
        "direction_consistent": result.direction_consistent,
        "meaningful_effect": result.meaningful_effect,
        "perturbation_valid": result.perturbation_valid,
        "invalid_reason": result.invalid_reason,
    }


def _opposite_quartile(
    original: float,
    stats: Mapping[str, float],
    feature_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    strategy = config["independent_evaluation_strategy"]["opposite_quartile"]
    for required in ("q25", "median", "q75"):
        if required not in stats:
            raise PerturbationError(f"missing_reference_statistic:{required}")
    median = _finite_number(stats["median"], "median")
    primary_key = (
        str(strategy["rule_if_original_greater_than_or_equal_to_median"]).removeprefix(
            "use_"
        )
        if float(original) >= median
        else str(strategy["rule_if_original_less_than_median"]).removeprefix("use_")
    )
    if primary_key not in stats:
        raise PerturbationError(f"missing_reference_statistic:{primary_key}")
    primary = _finite_number(stats[primary_key], primary_key)
    if not _values_equal(original, primary):
        return primary

    fallback = config["independent_evaluation_strategy"]["same_value_fallback"]
    if not bool(fallback["enabled"]):
        raise PerturbationError("evaluation_value_equals_original")
    if (
        str(fallback["policy"])
        != "use_farthest_valid_reference_quantile_without_generation_probe_information"
    ):
        raise ConfigError(
            f"Unsupported same-value fallback policy: {fallback['policy']!r}"
        )

    candidates: list[float] = []
    for key in fallback["ordered_candidates"]:
        if key in stats:
            value = _finite_number(stats[key], key)
            if not _values_equal(value, original):
                candidates.append(value)
    if not candidates:
        raise PerturbationError("no_distinct_valid_reference_value")
    return max(candidates, key=lambda value: abs(float(value) - float(original)))


def _binary_flip(
    original: float,
    feature_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    valid_values = list(feature_cfg.get("valid_values", []))
    if original not in valid_values:
        raise PerturbationError("binary_original_outside_valid_domain")
    binary_cfg = config["independent_evaluation_strategy"]["binary_flip"]
    if original == 0 and bool(binary_cfg["zero_to_one"]):
        return 1
    if original == 1 and bool(binary_cfg["one_to_zero"]):
        return 0
    raise PerturbationError("binary_flip_not_configured_for_original_value")


def _postprocess_value(
    value: float,
    feature_cfg: Mapping[str, Any],
    stats: Mapping[str, float] | None,
    config: Mapping[str, Any],
) -> float | int:
    post = config["independent_evaluation_strategy"]["postprocessing"]
    numeric = _finite_number(value, "evaluation_value")
    if (
        bool(post["clamp_to_reference_min_max"])
        and stats is not None
        and "min" in stats
        and "max" in stats
    ):
        numeric = min(
            max(numeric, float(stats["min"])),
            float(stats["max"]),
        )
    minimum = feature_cfg.get("minimum")
    if minimum is not None and numeric < float(minimum):
        if bool(post["enforce_feature_domain"]):
            numeric = float(minimum)
        else:
            raise PerturbationError("evaluation_value_below_feature_minimum")
    valid_values = feature_cfg.get("valid_values")
    if valid_values is not None and numeric not in valid_values:
        raise PerturbationError("evaluation_value_outside_valid_values")
    if bool(feature_cfg.get("integer", False)):
        if str(post["integer_features_rounding"]) != "nearest_integer":
            raise ConfigError("Unsupported integer rounding policy")
        numeric = round(numeric)
    if bool(post["forbid_nan"]) and isinstance(numeric, float) and math.isnan(numeric):
        raise PerturbationError("evaluation_value_is_nan")
    if bool(post["forbid_infinity"]) and not math.isfinite(float(numeric)):
        raise PerturbationError("evaluation_value_is_infinite")
    return numeric


def _validate_separation_contract(config: Mapping[str, Any]) -> None:
    sep = config["separation_contract"]
    independent = sep["independent_evaluation_perturbation"]
    prohibition = sep["prohibition"]
    if bool(independent["may_depend_on_generation_probe_value"]):
        raise ConfigError(
            "Independent evaluation is configured to depend on generation probes"
        )
    if not bool(
        independent["constructor_must_not_accept_generation_probe_value_argument"]
    ):
        raise ConfigError(
            "Independent evaluation constructor separation is not enforced"
        )
    if not bool(prohibition["reuse_generation_probe_as_evaluation_value"]):
        raise ConfigError("Config must prohibit direct generation-probe reuse")


def _feature_type_config(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    value = config.get("feature_types")
    if not isinstance(value, Mapping):
        raise ConfigError("Missing or invalid perturbation.feature_types")
    return value  # type: ignore[return-value]


def _response_name(label: Any, experiment_config: Mapping[str, Any]) -> str:
    positive = experiment_config["data_schema"]["positive_target_value"]
    prediction = experiment_config["prediction"]
    return str(
        prediction["positive_response_name"]
        if label == positive
        else prediction["negative_response_name"]
    )


def _finite_number(value: Any, name: str) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PerturbationError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise PerturbationError(f"{name} must be finite")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return number


def _finite_probability(value: Any) -> float:
    probability = float(_finite_number(value, "probability"))
    if not 0.0 <= probability <= 1.0:
        raise PerturbationError(f"Probability outside [0,1]: {probability}")
    return probability


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(
            np.isclose(float(left), float(right), rtol=0.0, atol=0.0, equal_nan=True)
        )
    except (TypeError, ValueError):
        return left == right
