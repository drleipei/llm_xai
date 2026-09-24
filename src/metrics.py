"""Evaluation metrics for RQ1 stability, RQ2 alignment, and RQ3 discriminativeness.

The functions are intentionally independent of experiment orchestration.  RQ
scripts should read ``configs/experiment.yaml`` / ``configs/perturbation.yaml``
and pass configured ``k`` and thresholds into these functions.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from statistics import mean
from typing import Any


class MetricError(ValueError):
    """Raised when explanation/evaluation records violate a metric contract."""


# -----------------------------------------------------------------------------
# Shared explanation validation
# -----------------------------------------------------------------------------


def top_features(
    explanation: Mapping[str, Any],
    k: int,
    *,
    require_unique: bool = True,
) -> list[Mapping[str, Any]]:
    """Return the first Top-k feature records after structural validation.

    Duplicate features are an invalid explanation when ``require_unique=True``;
    they are never silently counted twice.  This matches the project's common
    explanation-schema contract.
    """
    if k <= 0:
        raise MetricError("k must be positive")
    raw = explanation.get("top_features")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MetricError("Explanation top_features must be a sequence")
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in raw[:k]:
        if not isinstance(item, Mapping) or "feature" not in item:
            raise MetricError("Each top_features item must be a mapping with 'feature'")
        feature = str(item["feature"])
        if feature in seen:
            if require_unique:
                raise MetricError(f"Duplicate feature in Top-{k}: {feature!r}")
            continue
        seen.add(feature)
        selected.append(item)
    return selected


def _feature_names(explanation: Mapping[str, Any], k: int) -> list[str]:
    return [str(item["feature"]) for item in top_features(explanation, k)]


# -----------------------------------------------------------------------------
# RQ1: Stability
# -----------------------------------------------------------------------------


def overlap_at_k(first: Mapping[str, Any], second: Mapping[str, Any], k: int) -> float:
    """Compute Top-k feature-selection overlap: ``|A ∩ B| / k``.

    The denominator is the configured k, so incomplete explanations are not
    rewarded as if they had returned a full Top-k list.
    """
    a, b = set(_feature_names(first, k)), set(_feature_names(second, k))
    return len(a & b) / float(k)


def rank_agreement_at_k(
    first: Mapping[str, Any], second: Mapping[str, Any], k: int
) -> float:
    """Measure positional agreement of features shared by two Top-k lists.

    For each shared feature f with 1-based ranks r1(f), r2(f), positional
    agreement is ``1 - |r1-r2|/(k-1)``.  Rank Agreement@k is the mean across
    shared features.  If there is no shared feature it is 0.  For k=1, a shared
    singleton has agreement 1.
    """
    names_a, names_b = _feature_names(first, k), _feature_names(second, k)
    rank_a = {feature: index + 1 for index, feature in enumerate(names_a)}
    rank_b = {feature: index + 1 for index, feature in enumerate(names_b)}
    shared = set(rank_a) & set(rank_b)
    if not shared:
        return 0.0
    if k == 1:
        return 1.0
    return float(mean(1.0 - abs(rank_a[f] - rank_b[f]) / float(k - 1) for f in shared))


def direction_agreement_at_k(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    k: int,
) -> float:
    """Return sign/direction agreement among features shared by two Top-k lists.

    ``support``, ``oppose``, and ``neutral`` are compared exactly.  When no
    feature is shared, direction agreement is undefined and ``NaN`` is returned;
    selection disagreement is already represented by Overlap@k.
    """
    items_a = top_features(first, k)
    items_b = top_features(second, k)
    dir_a = {str(item["feature"]): str(item["direction"]) for item in items_a}
    dir_b = {str(item["feature"]): str(item["direction"]) for item in items_b}
    shared = set(dir_a) & set(dir_b)
    if not shared:
        return float("nan")
    return sum(dir_a[f] == dir_b[f] for f in shared) / float(len(shared))


def score_stability(
    first: Mapping[str, Any], second: Mapping[str, Any], k: int
) -> float:
    """Compare Top-k importance-score profiles on a common [0,1] stability scale.

    Each explanation's non-negative Top-k scores are L1-normalised. Missing
    features receive score 0. The metric is ``1 - 0.5 * L1_distance`` over the
    union of selected features (total-variation similarity), yielding 1 for
    identical score allocations and 0 for disjoint allocations.  If both score
    sums are zero, stability is 1 only when the selected feature sets match.
    """
    a = top_features(first, k)
    b = top_features(second, k)
    scores_a = _score_map(a)
    scores_b = _score_map(b)
    set_a, set_b = set(scores_a), set(scores_b)
    sum_a, sum_b = sum(scores_a.values()), sum(scores_b.values())
    if sum_a == 0.0 and sum_b == 0.0:
        return 1.0 if set_a == set_b else 0.0
    norm_a = {f: (v / sum_a if sum_a > 0 else 0.0) for f, v in scores_a.items()}
    norm_b = {f: (v / sum_b if sum_b > 0 else 0.0) for f, v in scores_b.items()}
    union = set_a | set_b
    l1 = sum(abs(norm_a.get(f, 0.0) - norm_b.get(f, 0.0)) for f in union)
    return float(max(0.0, min(1.0, 1.0 - 0.5 * l1)))


def pairwise_stability_summary(
    explanations: Sequence[Mapping[str, Any]],
    k: int,
) -> dict[str, float]:
    """Average RQ1 metrics across all unordered repeated-run pairs."""
    if len(explanations) < 2:
        raise MetricError("At least two repeated explanations are required")
    pairs = list(combinations(explanations, 2))
    overlaps = [overlap_at_k(a, b, k) for a, b in pairs]
    ranks = [rank_agreement_at_k(a, b, k) for a, b in pairs]
    directions = [direction_agreement_at_k(a, b, k) for a, b in pairs]
    scores = [score_stability(a, b, k) for a, b in pairs]
    return {
        "overlap_at_k": float(mean(overlaps)),
        "rank_agreement_at_k": float(mean(ranks)),
        "direction_agreement_at_k": _nanmean(directions),
        "score_stability": float(mean(scores)),
    }


# -----------------------------------------------------------------------------
# RQ2: Independent model-response alignment
# -----------------------------------------------------------------------------


def direction_from_delta(delta_probability: float, neutral_threshold: float) -> str:
    """Map ``Δp = p_evaluation - p_original`` to support/oppose/neutral."""
    delta = _finite_float(delta_probability, "delta_probability")
    threshold = _finite_float(neutral_threshold, "neutral_threshold")
    if threshold < 0:
        raise MetricError("neutral_threshold must be non-negative")
    if delta < -threshold:
        return "support"
    if delta > threshold:
        return "oppose"
    return "neutral"


def direction_consistency_rate(evaluations: Sequence[Mapping[str, Any]]) -> float:
    """Compute DCR over valid interventions with comparable directions.

    Invalid interventions and records lacking either claimed or observed
    direction are excluded from the denominator, as required by the configured
    invalid-record policy.
    """
    values: list[bool] = []
    for record in evaluations:
        if not bool(record.get("perturbation_valid", False)):
            continue
        claimed, observed = (
            record.get("claimed_direction"),
            record.get("observed_direction"),
        )
        if claimed is None or observed is None:
            continue
        values.append(str(claimed) == str(observed))
    return _rate_or_nan(values)


def meaningful_effect_rate(
    evaluations: Sequence[Mapping[str, Any]],
    threshold: float,
) -> float:
    """Compute MER = proportion of valid interventions with ``|Δp| >= τ``."""
    tau = _finite_float(threshold, "threshold")
    if tau < 0:
        raise MetricError("Meaningful-effect threshold must be non-negative")
    values: list[bool] = []
    for record in evaluations:
        if not bool(record.get("perturbation_valid", False)):
            continue
        delta = record.get("delta_probability")
        if delta is None:
            continue
        values.append(abs(_finite_float(delta, "delta_probability")) >= tau)
    return _rate_or_nan(values)


def response_probability_changes(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return valid signed Δp and |Δp| values plus aggregate means.

    Invalid interventions are retained only in the reported count and excluded
    from numerical denominators.
    """
    signed: list[float] = []
    absolute: list[float] = []
    invalid = 0
    for record in evaluations:
        if not bool(record.get("perturbation_valid", False)):
            invalid += 1
            continue
        delta = record.get("delta_probability")
        if delta is None:
            invalid += 1
            continue
        value = _finite_float(delta, "delta_probability")
        signed.append(value)
        absolute.append(abs(value))
    return {
        "delta_probability": signed,
        "absolute_delta_probability": absolute,
        "mean_delta_probability": float(mean(signed)) if signed else float("nan"),
        "mean_absolute_delta_probability": float(mean(absolute))
        if absolute
        else float("nan"),
        "valid_interventions": len(signed),
        "invalid_interventions": invalid,
    }


def rq2_alignment_summary(
    evaluations: Sequence[Mapping[str, Any]],
    meaningful_threshold: float,
) -> dict[str, float | int]:
    """Aggregate the currently specified RQ2 metrics for a record group."""
    changes = response_probability_changes(evaluations)
    return {
        "direction_consistency_rate": direction_consistency_rate(evaluations),
        "meaningful_effect_rate": meaningful_effect_rate(
            evaluations, meaningful_threshold
        ),
        "mean_delta_probability": changes["mean_delta_probability"],
        "mean_absolute_delta_probability": changes["mean_absolute_delta_probability"],
        "valid_interventions": int(changes["valid_interventions"]),
        "invalid_interventions": int(changes["invalid_interventions"]),
    }


# -----------------------------------------------------------------------------
# RQ3: Discriminativeness
# -----------------------------------------------------------------------------


def normalized_feature_entropy(
    explanations: Sequence[Mapping[str, Any]],
    feature_space: Sequence[str],
    k: int,
) -> float:
    """Measure dispersion of Top-k feature selections over the feature space.

    If p_f is the proportion of all selected slots occupied by feature f, then
    ``H = -Σ p_f log(p_f)`` and ``H_norm = H/log(|F|)``.  Values approach 0 when
    selections concentrate on a few features and 1 when they are broadly and
    uniformly distributed. Duplicate features within one explanation are invalid.
    """
    features = tuple(str(f) for f in feature_space)
    if not features:
        raise MetricError("feature_space must not be empty")
    allowed = set(features)
    counts: Counter[str] = Counter()
    total = 0
    for explanation in explanations:
        for feature in _feature_names(explanation, k):
            if feature not in allowed:
                raise MetricError(
                    f"Explanation contains feature outside feature_space: {feature!r}"
                )
            counts[feature] += 1
            total += 1
    if total == 0 or len(features) == 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return float(entropy / math.log(len(features)))


def pairwise_jaccard(
    first: Mapping[str, Any], second: Mapping[str, Any], k: int
) -> float:
    """Compute Jaccard similarity ``|A∩B| / |A∪B|`` for Top-k feature sets."""
    a, b = set(_feature_names(first, k)), set(_feature_names(second, k))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def pairwise_jaccard_by_instance(
    explanations: Sequence[Mapping[str, Any]],
    k: int,
    *,
    instance_key: str = "instance_id",
) -> dict[str, Any]:
    """Compute within-instance and between-instance pairwise Jaccard similarities.

    Repeated runs of the same instance contribute to ``within_instance``;
    explanation records from different instances contribute to ``between_instance``.
    """
    within: list[float] = []
    between: list[float] = []
    for first, second in combinations(explanations, 2):
        if instance_key not in first or instance_key not in second:
            raise MetricError(f"Missing {instance_key!r} in explanation record")
        value = pairwise_jaccard(first, second, k)
        if str(first[instance_key]) == str(second[instance_key]):
            within.append(value)
        else:
            between.append(value)
    return {
        "within_instance": within,
        "between_instance": between,
        "mean_within_instance": float(mean(within)) if within else float("nan"),
        "mean_between_instance": float(mean(between)) if between else float("nan"),
    }


def separability_gap(
    explanations: Sequence[Mapping[str, Any]],
    k: int,
    *,
    instance_key: str = "instance_id",
) -> float:
    """Compute ``mean within-instance Jaccard - mean between-instance Jaccard``."""
    values = pairwise_jaccard_by_instance(explanations, k, instance_key=instance_key)
    within = float(values["mean_within_instance"])
    between = float(values["mean_between_instance"])
    if math.isnan(within) or math.isnan(between):
        return float("nan")
    return within - between


def instance_idf_specificity(
    explanations: Sequence[Mapping[str, Any]],
    k: int,
    *,
    instance_key: str = "instance_id",
) -> dict[str, Any]:
    """Compute study-defined instance-level IDF specificity.

    Repeated runs are first collapsed per instance by feature union. For N unique
    instances, ``IDF(f)=log(N/df_f)`` where ``df_f`` is the number of instances
    whose explanations contain f.  For N>1, IDF is divided by ``log(N)`` so a
    feature unique to one instance scores 1 and a feature present in every
    instance scores 0. Instance specificity is the mean normalised IDF of the
    features selected for that instance.
    """
    per_instance: dict[str, set[str]] = defaultdict(set)
    for explanation in explanations:
        if instance_key not in explanation:
            raise MetricError(f"Missing {instance_key!r} in explanation record")
        per_instance[str(explanation[instance_key])].update(
            _feature_names(explanation, k)
        )
    n = len(per_instance)
    if n == 0:
        return {
            "per_instance": {},
            "mean_instance_idf_specificity": float("nan"),
            "feature_idf": {},
        }

    df: Counter[str] = Counter()
    for features in per_instance.values():
        for feature in features:
            df[feature] += 1
    if n == 1:
        feature_idf = {feature: 0.0 for feature in df}
    else:
        denominator = math.log(n)
        feature_idf = {
            feature: math.log(n / count) / denominator for feature, count in df.items()
        }

    specificity: dict[str, float] = {}
    for instance_id, features in per_instance.items():
        specificity[instance_id] = (
            float(mean(feature_idf[f] for f in features)) if features else float("nan")
        )
    return {
        "per_instance": specificity,
        "mean_instance_idf_specificity": _nanmean(list(specificity.values())),
        "feature_idf": feature_idf,
    }


def rq3_discriminativeness_summary(
    explanations: Sequence[Mapping[str, Any]],
    feature_space: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Aggregate the currently specified RQ3 discriminativeness metrics."""
    jaccard = pairwise_jaccard_by_instance(explanations, k)
    idf = instance_idf_specificity(explanations, k)
    gap = separability_gap(explanations, k)
    return {
        "normalized_feature_entropy": normalized_feature_entropy(
            explanations, feature_space, k
        ),
        "pairwise_jaccard_within_instance": float(jaccard["mean_within_instance"]),
        "pairwise_jaccard_between_instance": float(jaccard["mean_between_instance"]),
        "instance_idf_specificity": float(idf["mean_instance_idf_specificity"]),
        "separability_gap": gap,
    }


def _score_map(items: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in items:
        if "score" not in item:
            raise MetricError("Top feature missing score")
        value = _finite_float(item["score"], "score")
        if value < 0:
            raise MetricError("Explanation score must be non-negative")
        scores[str(item["feature"])] = value
    return scores


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise MetricError(f"{name} must be finite")
    return number


def _rate_or_nan(values: Sequence[bool]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _nanmean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return float(mean(finite)) if finite else float("nan")
