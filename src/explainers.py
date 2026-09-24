"""Traditional and LLM-informed explainers with one common output schema.

All predictor probabilities and numerical effects are computed by Python using
:class:`src.predictor.FrozenPredictor`. LLM-informed methods may only propose
candidate features, reference choices, or single-feature probe values. Any
probability-like field supplied by an LLM is rejected rather than trusted.

The module contains no RQ orchestration. Experiment scripts added later should
construct these explainers, pass run metadata, and persist returned records.
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

try:
    from .llm import LLMClient
    from .predictor import FrozenPredictor
    from .utils import ConfigError, resolve_project_path
except ImportError:  # pragma: no cover
    from llm import LLMClient
    from predictor import FrozenPredictor
    from utils import ConfigError, resolve_project_path


class ExplainerError(RuntimeError):
    """Raised when an explanation cannot satisfy the configured contract."""


@dataclass(frozen=True)
class ExplanationContext:
    """Run metadata required by the unified explanation schema."""

    dataset: str
    category: str
    instance_id: str
    run_id: int
    seed: int
    llm_model: str | None = None
    prompt_variant: str | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Numerical result of one Python-executed single-feature probe."""

    feature: str
    original_value: float | int
    probe_value: float | int
    original_probability: float
    probe_probability: float
    delta_probability: float

    @property
    def direction(self) -> str:
        """Return support/oppose/neutral using the sign of the model response."""
        if self.delta_probability < 0.0:
            return "support"
        if self.delta_probability > 0.0:
            return "oppose"
        return "neutral"

    @property
    def score(self) -> float:
        """Return absolute predictor-probability change as probe strength."""
        return abs(self.delta_probability)


class BaseExplainer:
    """Base class implementing schema validation and predictor utilities."""

    method_id: str

    def __init__(
        self,
        predictor: FrozenPredictor,
        experiment_config: Mapping[str, Any],
    ) -> None:
        self.predictor = predictor
        self.config = experiment_config
        explanation_cfg = experiment_config.get("explanation")
        if not isinstance(explanation_cfg, Mapping):
            raise ConfigError("Missing or invalid explanation configuration")
        self.explanation_cfg = explanation_cfg
        self.feature_names = tuple(
            str(v) for v in experiment_config["data_schema"]["predictor_features"]
        )
        self.top_k = int(explanation_cfg["top_k"])

    def _prepare_instance(
        self, instance: pd.Series | Mapping[str, Any]
    ) -> pd.DataFrame:
        row = pd.Series(instance)
        missing = [
            feature for feature in self.feature_names if feature not in row.index
        ]
        if missing:
            raise ExplainerError(
                f"Instance missing configured predictor feature(s): {missing}"
            )
        return pd.DataFrame(
            [[row[f] for f in self.feature_names]], columns=self.feature_names
        )

    def _prediction_state(self, X: pd.DataFrame) -> tuple[Any, str, float]:
        label = self.predictor.predict(X)[0]
        probability = float(self.predictor.probability_for_class(X, [label])[0])
        prediction_cfg = self.config["prediction"]
        positive_label = self.config["data_schema"]["positive_target_value"]
        response = (
            str(prediction_cfg["positive_response_name"])
            if label == positive_label
            else str(prediction_cfg["negative_response_name"])
        )
        return label, response, probability

    def _execute_probe(
        self,
        X: pd.DataFrame,
        feature: str,
        probe_value: float,
        original_label: Any,
        original_probability: float,
    ) -> ProbeResult:
        if feature not in self.feature_names:
            raise ExplainerError(
                f"Probe references unknown predictor feature: {feature!r}"
            )
        perturbed = X.copy()
        original_value = perturbed.at[0, feature]
        perturbed[feature] = perturbed[feature].astype(float)
        perturbed.at[0, feature] = float(probe_value)
        changed = [
            name
            for name in self.feature_names
            if not _values_equal(X.at[0, name], perturbed.at[0, name])
        ]
        if changed != [feature]:
            raise ExplainerError(
                f"Probe must change exactly one feature; requested={feature!r}, changed={changed}"
            )
        probe_probability = float(
            self.predictor.probability_for_class(perturbed, [original_label])[0]
        )
        return ProbeResult(
            feature=feature,
            original_value=_json_number(original_value),
            probe_value=_json_number(probe_value),
            original_probability=original_probability,
            probe_probability=probe_probability,
            delta_probability=probe_probability - original_probability,
        )

    def _record(
        self,
        context: ExplanationContext,
        predicted_response: str,
        predicted_probability: float,
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in features:
            feature = str(raw["feature"])
            if feature in seen or feature not in self.feature_names:
                continue
            direction = str(raw["direction"])
            if direction not in self.explanation_cfg["allowed_directions"]:
                raise ExplainerError(
                    f"Invalid direction {direction!r} for feature {feature!r}"
                )
            score = float(raw["score"])
            if not math.isfinite(score) or score < 0:
                raise ExplainerError(f"Invalid score for feature {feature!r}: {score}")
            item = {
                "feature": feature,
                "value": _json_number(raw["value"]),
                "direction": direction,
                "score": score,
            }
            # Audit-only generation fields are permitted beyond required schema fields.
            for optional in ("generation_probe_value", "generation_delta_probability"):
                if optional in raw:
                    item[optional] = _json_number(raw[optional])
            unique.append(item)
            seen.add(feature)

        if bool(self.explanation_cfg["sort_top_features_by_score_descending"]):
            unique.sort(key=lambda item: float(item["score"]), reverse=True)
        unique = unique[: self.top_k]

        return {
            "schema_version": str(self.explanation_cfg["schema_version"]),
            "instance_id": context.instance_id,
            "dataset": context.dataset,
            "category": context.category,
            "method": self.method_id,
            "llm_model": context.llm_model,
            "run_id": int(context.run_id),
            "seed": int(context.seed),
            "prompt_variant": context.prompt_variant,
            "temperature": context.temperature,
            "top_p": context.top_p,
            "predicted_response": predicted_response,
            "predicted_response_probability": float(predicted_probability),
            "top_features": unique,
        }


class LimeExplainer(BaseExplainer):
    """Traditional LIME using predictor probabilities as the numerical oracle."""

    method_id = "lime"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Explain one instance with tabular LIME and return the common schema."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:  # pragma: no cover
            raise ExplainerError("lime package is required for LimeExplainer") from exc

        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        background = _feature_frame(reference_data, self.feature_names)
        class_index = _class_index(self.predictor, label)

        lime = LimeTabularExplainer(
            background.to_numpy(dtype=float),
            feature_names=list(self.feature_names),
            class_names=[str(v) for v in self.predictor.classes_.tolist()],
            mode="classification",
            random_state=context.seed,
            discretize_continuous=False,
        )
        explanation = lime.explain_instance(
            X.iloc[0].to_numpy(dtype=float),
            lambda values: self.predictor.predict_proba(
                pd.DataFrame(values, columns=self.feature_names)
            ),
            labels=(class_index,),
            num_features=min(self.top_k, len(self.feature_names)),
        )
        weights = dict(explanation.as_map()[class_index])
        features = []
        for feature_index, weight in weights.items():
            feature = self.feature_names[int(feature_index)]
            numeric_weight = float(weight)
            features.append(
                {
                    "feature": feature,
                    "value": X.at[0, feature],
                    "direction": _weight_direction(numeric_weight),
                    "score": abs(numeric_weight),
                }
            )
        return self._record(context, response, original_probability, features)


class KernelSHAPExplainer(BaseExplainer):
    """Traditional KernelSHAP over the configured predictor feature space."""

    method_id = "kernelshap"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Explain one instance with KernelSHAP and return the common schema."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise ExplainerError(
                "shap package is required for KernelSHAPExplainer"
            ) from exc

        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        background = _kernelshap_background(
            reference_data,
            self.feature_names,
            self.config,
        )
        class_index = _class_index(self.predictor, label)

        def predict_target(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(np.asarray(values), columns=self.feature_names)
            return self.predictor.predict_proba(frame)[:, class_index]

        explainer = shap.KernelExplainer(
            predict_target, background.to_numpy(dtype=float)
        )
        shap_values = np.asarray(
            explainer.shap_values(X.to_numpy(dtype=float), silent=True)
        )
        vector = _single_shap_vector(shap_values, len(self.feature_names))
        features = [
            {
                "feature": feature,
                "value": X.at[0, feature],
                "direction": _weight_direction(float(vector[idx])),
                "score": abs(float(vector[idx])),
            }
            for idx, feature in enumerate(self.feature_names)
        ]
        return self._record(context, response, original_probability, features)


class LOFOExplainer(BaseExplainer):
    """Traditional local leave-one-feature-out explanation.

    Feature removal is represented by a caller-provided method-independent
    reference value per feature; Python executes the replacement and measures the
    frozen predictor response.
    """

    method_id = "lofo"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_values: Mapping[str, float | int],
    ) -> dict[str, Any]:
        """Replace each feature by its supplied reference and rank model effects."""
        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        results: list[dict[str, Any]] = []
        for feature in self.feature_names:
            if feature not in reference_values:
                raise ExplainerError(f"Missing LOFO reference value for {feature!r}")
            if _values_equal(X.at[0, feature], reference_values[feature]):
                continue
            probe = self._execute_probe(
                X, feature, reference_values[feature], label, original_probability
            )
            results.append(_probe_to_feature(X, probe))
        return self._record(context, response, original_probability, results)


class CounterfactualExplainer(BaseExplainer):
    """Traditional deterministic single-feature counterfactual search."""

    method_id = "counterfactual"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        candidate_values: Mapping[str, Sequence[float | int]],
    ) -> dict[str, Any]:
        """Evaluate supplied deterministic candidate values and keep strongest per feature."""
        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        best: list[dict[str, Any]] = []
        for feature in self.feature_names:
            candidates = candidate_values.get(feature, ())
            probes = [
                self._execute_probe(X, feature, value, label, original_probability)
                for value in candidates
                if not _values_equal(X.at[0, feature], value)
            ]
            if probes:
                strongest = max(probes, key=lambda item: item.score)
                best.append(_probe_to_feature(X, strongest))
        return self._record(context, response, original_probability, best)


class BaseLLMExplainer(BaseExplainer):
    """Base class for LLM planning followed by Python predictor execution."""

    planner_kind: str

    def __init__(
        self,
        predictor: FrozenPredictor,
        experiment_config: Mapping[str, Any],
        prompts_config: Mapping[str, Any],
        llm_client: LLMClient,
        project_root: str | Path,
    ) -> None:
        super().__init__(predictor, experiment_config)
        self.prompts_config = prompts_config
        self.llm_client = llm_client
        self.project_root = Path(project_root)
        if llm_client.llm_id not in [str(v) for v in experiment_config["llm_models"]]:
            raise ConfigError(f"LLM client ID not configured: {llm_client.llm_id!r}")

    def _load_prompt(self, variant: str) -> str:
        explainers = self.prompts_config.get("explainers")
        if not isinstance(explainers, Mapping) or self.method_id not in explainers:
            raise ConfigError(f"Missing prompt mapping for {self.method_id!r}")
        files = explainers[self.method_id].get("files", {})
        if variant not in files:
            raise ConfigError(
                f"Unknown prompt variant {variant!r} for {self.method_id!r}"
            )
        path = resolve_project_path(self.project_root, str(files[variant]))
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _render_prompt(
        self,
        template: str,
        X: pd.DataFrame,
        *,
        predicted_response: str,
        context: ExplanationContext,
        reference_summary: Mapping[str, Any] | None,
    ) -> str:
        payloads = {
            "<<FEATURE_NAMES_JSON>>": json.dumps(
                list(self.feature_names), ensure_ascii=False
            ),
            "<<INSTANCE_VALUES_JSON>>": json.dumps(
                {
                    feature: _json_number(X.at[0, feature])
                    for feature in self.feature_names
                },
                ensure_ascii=False,
            ),
            "<<PREDICTED_RESPONSE_JSON>>": json.dumps(predicted_response),
            "<<TOP_K_JSON>>": json.dumps(self.top_k),
            "<<REFERENCE_SUMMARY_JSON>>": json.dumps(
                reference_summary or {}, ensure_ascii=False
            ),
            "<<DATASET_JSON>>": json.dumps(context.dataset),
            "<<CATEGORY_JSON>>": json.dumps(context.category),
        }
        rendered = template
        for token, value in payloads.items():
            rendered = rendered.replace(token, value)
        return rendered

    def _plan(
        self,
        X: pd.DataFrame,
        context: ExplanationContext,
        *,
        predicted_response: str,
        reference_summary: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if context.llm_model != self.llm_client.llm_id:
            raise ExplainerError(
                f"context.llm_model={context.llm_model!r} does not match client "
                f"{self.llm_client.llm_id!r}"
            )
        if context.prompt_variant is None:
            raise ExplainerError("LLM explanation context requires prompt_variant")
        template = self._load_prompt(context.prompt_variant)
        prompt = self._render_prompt(
            template,
            X,
            predicted_response=predicted_response,
            context=context,
            reference_summary=reference_summary,
        )
        payload = self.llm_client.complete_json(
            prompt,
            temperature=context.temperature,
            top_p=context.top_p,
            system_prompt="You are participating in a software defect prediction XAI experiment.",
        )
        _reject_probability_fields(payload)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ExplainerError("LLM planner JSON must contain a 'candidates' list")
        return _validate_planner_candidates(candidates, self.feature_names)


class LLMLimeExplainer(BaseLLMExplainer):
    """LLM-informed local perturbation planning with Python response scoring."""

    method_id = "llm_lime"
    planner_kind = "local_perturbation_candidate_planning"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask the LLM for local single-feature probes, then score them in Python."""
        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        candidates = self._plan(
            X, context, predicted_response=response, reference_summary=reference_summary
        )
        features = _evaluate_planned_probes(
            self, X, candidates, label, original_probability, require_probe_value=True
        )
        return self._record(context, response, original_probability, features)


class LLMKernelSHAPExplainer(BaseLLMExplainer):
    """LLM-informed coalition/reference planning with Python KernelSHAP scoring."""

    method_id = "llm_kernelshap"
    planner_kind = "coalition_background_and_candidate_planning"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_data: pd.DataFrame,
        reference_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use LLM-selected candidate features, but compute SHAP values in Python."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise ExplainerError(
                "shap package is required for LLMKernelSHAPExplainer"
            ) from exc

        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        candidates = self._plan(
            X, context, predicted_response=response, reference_summary=reference_summary
        )
        selected = _ordered_unique_features(candidates, self.feature_names)
        if not selected:
            raise ExplainerError(
                "LLM-KernelSHAP planner returned no valid candidate features"
            )

        background = _kernelshap_background(
            reference_data,
            self.feature_names,
            self.config,
        )
        planned_rows: list[pd.DataFrame] = []
        for candidate in candidates:
            feature = str(candidate["feature"])
            probe_value = candidate.get("probe_value")
            if (
                isinstance(probe_value, (int, float))
                and not isinstance(probe_value, bool)
                and math.isfinite(float(probe_value))
                and not _values_equal(X.at[0, feature], probe_value)
            ):
                planned = X.copy()
                planned[feature] = planned[feature].astype(float)
                planned.at[0, feature] = float(probe_value)
                planned_rows.append(planned)
        if planned_rows:
            background = pd.concat([background, *planned_rows], ignore_index=True)
        class_index = _class_index(self.predictor, label)

        # KernelSHAP receives the full feature vector. LLM-proposed values may
        # augment the background with auditable one-feature planning rows, while
        # every SHAP attribution and predictor probability is still computed by
        # Python. Only LLM-selected features are eligible for final ranking.
        def predict_target(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(np.asarray(values), columns=self.feature_names)
            return self.predictor.predict_proba(frame)[:, class_index]

        explainer = shap.KernelExplainer(
            predict_target, background.to_numpy(dtype=float)
        )
        values = np.asarray(explainer.shap_values(X.to_numpy(dtype=float), silent=True))
        vector = _single_shap_vector(values, len(self.feature_names))
        features = []
        for feature in selected:
            idx = self.feature_names.index(feature)
            contribution = float(vector[idx])
            features.append(
                {
                    "feature": feature,
                    "value": X.at[0, feature],
                    "direction": _weight_direction(contribution),
                    "score": abs(contribution),
                }
            )
        return self._record(context, response, original_probability, features)


class LLMLOFOExplainer(BaseLLMExplainer):
    """LLM-informed feature replacement planning with Python LOFO scoring."""

    method_id = "llm_lofo"
    planner_kind = "feature_removal_or_replacement_candidate_planning"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate LLM-proposed one-feature replacement values via the predictor."""
        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        candidates = self._plan(
            X, context, predicted_response=response, reference_summary=reference_summary
        )
        features = _evaluate_planned_probes(
            self, X, candidates, label, original_probability, require_probe_value=True
        )
        return self._record(context, response, original_probability, features)


class LLMCounterfactualExplainer(BaseLLMExplainer):
    """LLM-planned single-feature counterfactuals scored only by frozen predictor."""

    method_id = "llm_counterfactual"
    planner_kind = "single_feature_counterfactual_intervention_planning"

    def explain(
        self,
        instance: pd.Series | Mapping[str, Any],
        context: ExplanationContext,
        *,
        reference_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate counterfactual probe proposals and rank absolute response changes."""
        X = self._prepare_instance(instance)
        label, response, original_probability = self._prediction_state(X)
        candidates = self._plan(
            X, context, predicted_response=response, reference_summary=reference_summary
        )
        features = _evaluate_planned_probes(
            self, X, candidates, label, original_probability, require_probe_value=True
        )
        return self._record(context, response, original_probability, features)


EXPLAINER_CLASSES: dict[str, type[BaseExplainer]] = {
    "lime": LimeExplainer,
    "kernelshap": KernelSHAPExplainer,
    "lofo": LOFOExplainer,
    "counterfactual": CounterfactualExplainer,
    "llm_lime": LLMLimeExplainer,
    "llm_kernelshap": LLMKernelSHAPExplainer,
    "llm_lofo": LLMLOFOExplainer,
    "llm_counterfactual": LLMCounterfactualExplainer,
}


def create_explainer(
    method_id: str,
    predictor: FrozenPredictor,
    experiment_config: Mapping[str, Any],
    *,
    prompts_config: Mapping[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    project_root: str | Path | None = None,
) -> BaseExplainer:
    """Factory for all eight method IDs already declared in ``experiment.yaml``."""
    configured = [str(v) for v in experiment_config["methods"]["all"]]
    if method_id not in configured or method_id not in EXPLAINER_CLASSES:
        raise ConfigError(f"Unknown or unimplemented method_id: {method_id!r}")
    cls = EXPLAINER_CLASSES[method_id]
    if method_id.startswith("llm_"):
        if prompts_config is None or llm_client is None or project_root is None:
            raise ConfigError(
                f"{method_id} requires prompts_config, llm_client, and project_root"
            )
        return cls(  # type: ignore[call-arg]
            predictor, experiment_config, prompts_config, llm_client, project_root
        )
    return cls(predictor, experiment_config)  # type: ignore[call-arg]


def _evaluate_planned_probes(
    explainer: BaseExplainer,
    X: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    original_label: Any,
    original_probability: float,
    *,
    require_probe_value: bool,
) -> list[dict[str, Any]]:
    best_by_feature: dict[str, ProbeResult] = {}
    for candidate in candidates:
        feature = str(candidate["feature"])
        if require_probe_value and "probe_value" not in candidate:
            raise ExplainerError(f"Planner candidate for {feature!r} lacks probe_value")
        probe_value = candidate.get("probe_value")
        if not isinstance(probe_value, (int, float)) or isinstance(probe_value, bool):
            raise ExplainerError(f"probe_value for {feature!r} must be numeric")
        if not math.isfinite(float(probe_value)):
            raise ExplainerError(f"probe_value for {feature!r} must be finite")
        if _values_equal(X.at[0, feature], probe_value):
            continue
        probe = explainer._execute_probe(
            X, feature, probe_value, original_label, original_probability
        )
        current = best_by_feature.get(feature)
        if current is None or probe.score > current.score:
            best_by_feature[feature] = probe
    return [_probe_to_feature(X, item) for item in best_by_feature.values()]


def _probe_to_feature(X: pd.DataFrame, probe: ProbeResult) -> dict[str, Any]:
    return {
        "feature": probe.feature,
        "value": X.at[0, probe.feature],
        "direction": probe.direction,
        "score": probe.score,
        "generation_probe_value": probe.probe_value,
        "generation_delta_probability": probe.delta_probability,
    }


def _validate_planner_candidates(
    candidates: Sequence[Any], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(feature_names)
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ExplainerError(f"Planner candidate #{index} must be an object")
        if "feature" not in raw:
            raise ExplainerError(f"Planner candidate #{index} is missing 'feature'")
        feature = str(raw["feature"])
        if feature not in allowed:
            raise ExplainerError(
                f"Planner proposed forbidden/unknown feature: {feature!r}"
            )
        item = dict(raw)
        item["feature"] = feature
        validated.append(item)
    return validated


def _reject_probability_fields(payload: Any, path: str = "$") -> None:
    """Reject probability/confidence fields anywhere in an LLM planner payload."""
    forbidden_tokens = (
        "probability",
        "prob",
        "confidence",
        "delta_probability",
        "score",
    )
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(
                token == normalized or normalized.endswith(f"_{token}")
                for token in forbidden_tokens
            ):
                raise ExplainerError(
                    f"LLM planner must not provide predictor numerical evidence; forbidden field "
                    f"at {path}.{key}"
                )
            _reject_probability_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_probability_fields(value, f"{path}[{index}]")


def _ordered_unique_features(
    candidates: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> list[str]:
    allowed = set(feature_names)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        feature = str(item["feature"])
        if feature in allowed and feature not in seen:
            ordered.append(feature)
            seen.add(feature)
    return ordered


def _kernelshap_background(
    reference_data: pd.DataFrame,
    feature_names: Sequence[str],
    experiment_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build a deterministic bounded KernelSHAP background dataset."""
    background = _feature_frame(reference_data, feature_names)

    config = experiment_config.get("kernelshap", {})
    sample_size = int(config.get("background_sample_size", 100))
    random_state = int(config.get("background_random_state", 42))

    if sample_size <= 0:
        raise ExplainerError(
            "kernelshap.background_sample_size must be greater than zero"
        )

    if len(background) <= sample_size:
        return background.reset_index(drop=True)

    return background.sample(
        n=sample_size,
        replace=False,
        random_state=random_state,
    ).reset_index(drop=True)


def _feature_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ExplainerError(f"Reference data missing predictor feature(s): {missing}")
    result = frame.loc[:, list(feature_names)].copy()
    if result.empty:
        raise ExplainerError("Reference data must contain at least one row")
    return result


def _single_shap_vector(values: np.ndarray, feature_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1 and array.shape[0] == feature_count:
        return array
    if array.ndim == 2 and array.shape == (1, feature_count):
        return array[0]
    raise ExplainerError(f"Unexpected KernelSHAP output shape: {array.shape}")


def _class_index(predictor: FrozenPredictor, label: Any) -> int:
    matches = np.flatnonzero(predictor.classes_ == label)
    if len(matches) != 1:
        raise ExplainerError(
            f"Predicted class {label!r} not uniquely present in predictor.classes_"
        )
    return int(matches[0])


def _weight_direction(weight: float) -> str:
    if weight > 0:
        return "support"
    if weight < 0:
        return "oppose"
    return "neutral"


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0))
    except (TypeError, ValueError):
        return left == right


def _json_number(value: Any) -> float | int | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return float(value)
