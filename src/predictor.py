"""Read-only interface for dataset-specific frozen defect predictors.

The wrapper deliberately exposes inference only. It does not expose ``fit``,
``partial_fit``, or any training/update operation, which helps enforce the
experiment's frozen-predictor contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:  # Supports both package-style and direct-module imports.
    from .utils import ConfigError, require_config_keys, resolve_project_path
except ImportError:  # pragma: no cover - convenience for running src on PYTHONPATH
    from utils import ConfigError, require_config_keys, resolve_project_path


class PredictorError(RuntimeError):
    """Raised when a frozen predictor cannot be loaded or used safely."""


@dataclass(frozen=True)
class FrozenPredictor:
    """Read-only adapter around an already fitted binary classifier.

    Attributes:
        estimator: Deserialized fitted estimator or pipeline.
        feature_names: Exact predictor feature order from ``experiment.yaml``.
        positive_label: Configured positive target value.
        negative_label: Configured negative target value.
        model_path: Source path used to load the estimator.

    Notes:
        This class is frozen at the Python object level and intentionally defines
        only inference methods. The underlying estimator is treated as immutable
        by project code. Experiments must never call training methods on it.
    """

    estimator: Any
    feature_names: tuple[str, ...]
    positive_label: int | str
    negative_label: int | str
    model_path: Path

    def _prepare_features(
        self, X: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Validate and order model inputs without changing their values."""
        if isinstance(X, pd.DataFrame):
            missing = [
                feature for feature in self.feature_names if feature not in X.columns
            ]
            if missing:
                raise PredictorError(
                    f"Predictor input is missing configured feature(s): {missing}"
                )
            extras = [
                column for column in X.columns if column not in self.feature_names
            ]
            if extras:
                raise PredictorError(
                    "Predictor input contains non-feature column(s). Pass only configured "
                    f"predictor_features: {extras}"
                )
            return X.loc[:, list(self.feature_names)]

        array = np.asarray(X)
        if array.ndim != 2:
            raise PredictorError(
                f"Predictor input must be 2-D; got shape {array.shape}"
            )
        if array.shape[1] != len(self.feature_names):
            raise PredictorError(
                f"Expected {len(self.feature_names)} predictor features, got {array.shape[1]}"
            )
        return array

    @property
    def classes_(self) -> np.ndarray:
        """Return fitted class labels from the underlying estimator."""
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            raise PredictorError("Frozen predictor does not expose classes_")
        return np.asarray(classes)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels using the unchanged frozen estimator."""
        prepared = self._prepare_features(X)
        predictions = np.asarray(self.estimator.predict(prepared))
        if predictions.ndim != 1:
            predictions = predictions.reshape(-1)
        return predictions

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities using the unchanged frozen estimator."""
        if not hasattr(self.estimator, "predict_proba"):
            raise PredictorError("Frozen predictor does not implement predict_proba")
        prepared = self._prepare_features(X)
        probabilities = np.asarray(self.estimator.predict_proba(prepared), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes_):
            raise PredictorError(
                "predict_proba returned an unexpected shape: "
                f"{probabilities.shape}; classes={self.classes_.tolist()}"
            )
        if not np.isfinite(probabilities).all():
            raise PredictorError("predict_proba returned NaN or infinite values")
        return probabilities

    def probability_for_class(
        self,
        X: pd.DataFrame | np.ndarray,
        class_labels: Sequence[int | str] | np.ndarray,
    ) -> np.ndarray:
        """Return each row's probability for its requested class label.

        This helper is useful later for RQ2, where evaluation tracks the
        probability of the *originally predicted class* even after a perturbation.
        """
        probabilities = self.predict_proba(X)
        labels = np.asarray(class_labels)
        if labels.ndim != 1 or labels.shape[0] != probabilities.shape[0]:
            raise PredictorError(
                "class_labels must be one-dimensional and match the number of input rows"
            )

        class_to_index = {
            label: index for index, label in enumerate(self.classes_.tolist())
        }
        try:
            indices = np.asarray([class_to_index[label] for label in labels], dtype=int)
        except KeyError as exc:
            raise PredictorError(
                f"Requested unknown class label: {exc.args[0]!r}"
            ) from exc
        return probabilities[np.arange(probabilities.shape[0]), indices]

    def positive_class_probability(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability of the positive class configured in ``experiment.yaml``."""
        row_count = len(X)
        labels = np.repeat(self.positive_label, row_count)
        return self.probability_for_class(X, labels)


def load_frozen_predictor(
    experiment_config: Mapping[str, Any],
    dataset_id: str,
    project_root: str | Path,
) -> FrozenPredictor:
    """Load and validate the configured fitted predictor for one dataset.

    Args:
        experiment_config: Parsed ``configs/experiment.yaml``.
        dataset_id: Configured dataset identifier such as ``openstack`` or ``qt``.
        project_root: Project root for resolving the configured model path.

    Returns:
        A read-only :class:`FrozenPredictor` adapter.

    Raises:
        ConfigError: If the relevant configuration is incomplete.
        FileNotFoundError: If the configured model file is absent.
        PredictorError: If the estimator is unfitted or lacks required inference
            capabilities.
    """
    prediction_cfg = experiment_config.get("prediction")
    if not isinstance(prediction_cfg, Mapping):
        raise ConfigError("Missing or invalid prediction configuration")
    require_config_keys(
        prediction_cfg,
        [
            "frozen_predictor_required",
            "prohibit_retraining_during_experiments",
            "model_serializer",
        ],
        "prediction",
    )
    if not bool(prediction_cfg["frozen_predictor_required"]):
        raise ConfigError(
            "This project stage requires prediction.frozen_predictor_required=true"
        )
    if not bool(prediction_cfg["prohibit_retraining_during_experiments"]):
        raise ConfigError(
            "This project stage requires prediction.prohibit_retraining_during_experiments=true"
        )

    datasets_cfg = experiment_config.get("datasets")
    if not isinstance(datasets_cfg, Mapping) or dataset_id not in datasets_cfg:
        raise ConfigError(f"Unknown dataset_id in experiment config: {dataset_id!r}")
    dataset_cfg = datasets_cfg[dataset_id]
    if not isinstance(dataset_cfg, Mapping):
        raise ConfigError(f"Invalid datasets.{dataset_id} configuration")
    require_config_keys(dataset_cfg, ["frozen_predictor"], f"datasets.{dataset_id}")

    schema_cfg = experiment_config.get("data_schema")
    if not isinstance(schema_cfg, Mapping):
        raise ConfigError("Missing or invalid data_schema configuration")
    require_config_keys(
        schema_cfg,
        ["predictor_features", "positive_target_value", "negative_target_value"],
        "data_schema",
    )

    model_path = resolve_project_path(
        project_root, str(dataset_cfg["frozen_predictor"])
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Configured frozen predictor not found for {dataset_id!r}: {model_path}"
        )

    serializer = str(prediction_cfg["model_serializer"]).lower()
    if serializer != "joblib":
        raise ConfigError(
            f"Unsupported configured prediction.model_serializer: {serializer!r}. "
            "This stage currently implements the serializer selected by the project config."
        )

    try:
        estimator = joblib.load(model_path)
    except Exception as exc:  # Deserialization errors need model/path context.
        raise PredictorError(
            f"Failed to load frozen predictor from {model_path}: {exc}"
        ) from exc

    if not hasattr(estimator, "predict"):
        raise PredictorError("Loaded predictor does not implement predict")
    if not hasattr(estimator, "predict_proba"):
        raise PredictorError("Loaded predictor does not implement predict_proba")

    try:
        check_is_fitted(estimator)
    except Exception as exc:
        raise PredictorError(
            f"Configured predictor does not appear to be fitted: {model_path}"
        ) from exc

    feature_names = tuple(str(name) for name in schema_cfg["predictor_features"])
    if not feature_names:
        raise ConfigError("data_schema.predictor_features must not be empty")

    model_feature_names = getattr(estimator, "feature_names_in_", None)
    if model_feature_names is not None and list(map(str, model_feature_names)) != list(
        feature_names
    ):
        raise PredictorError(
            "Frozen predictor feature_names_in_ does not match configured predictor_features "
            f"order. model={list(map(str, model_feature_names))}, config={list(feature_names)}"
        )

    return FrozenPredictor(
        estimator=estimator,
        feature_names=feature_names,
        positive_label=schema_cfg["positive_target_value"],
        negative_label=schema_cfg["negative_target_value"],
        model_path=model_path,
    )
