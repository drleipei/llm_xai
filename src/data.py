"""Dataset loading, validation, prediction-category assignment, and sampling.

All dataset paths, column names, class semantics, categories, sample sizes, and
seeds are read from the parsed experiment configuration rather than embedded in
this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:  # Supports both package-style and direct-module imports.
    from .utils import (
        ConfigError,
        derive_seed,
        get_global_seed,
        require_config_keys,
        resolve_project_path,
    )
except ImportError:  # pragma: no cover
    from utils import (
        ConfigError,
        derive_seed,
        get_global_seed,
        require_config_keys,
        resolve_project_path,
    )


class DataValidationError(ValueError):
    """Raised when loaded data violates the configured data contract."""


class PredictorProtocol(Protocol):
    """Minimal inference interface needed for TP/TN/FP/FN categorisation."""

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return one predicted class label per input row."""
        ...


def _schema_config(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    schema_cfg = experiment_config.get("data_schema")
    if not isinstance(schema_cfg, Mapping):
        raise ConfigError("Missing or invalid data_schema configuration")
    return schema_cfg


def _dataset_config(
    experiment_config: Mapping[str, Any], dataset_id: str
) -> Mapping[str, Any]:
    datasets_cfg = experiment_config.get("datasets")
    if not isinstance(datasets_cfg, Mapping) or dataset_id not in datasets_cfg:
        raise ConfigError(f"Unknown dataset_id in experiment config: {dataset_id!r}")
    dataset_cfg = datasets_cfg[dataset_id]
    if not isinstance(dataset_cfg, Mapping):
        raise ConfigError(f"Invalid datasets.{dataset_id} configuration")
    return dataset_cfg


def validate_dataset_frame(
    frame: pd.DataFrame,
    experiment_config: Mapping[str, Any],
    dataset_id: str,
) -> None:
    """Validate a dataset against the schema and dataset-specific configuration.

    Validation covers required columns, configured dataset identity, target
    values, predictor feature order/availability, finite numeric predictor values,
    and optional duplicate-instance checks.
    """
    schema_cfg = _schema_config(experiment_config)
    dataset_cfg = _dataset_config(experiment_config, dataset_id)
    require_config_keys(
        schema_cfg,
        [
            "expected_columns",
            "target_column",
            "positive_target_value",
            "negative_target_value",
            "predictor_features",
            "forbidden_predictor_features",
            "instance_id_column",
            "fail_on_missing_columns",
            "fail_on_extra_predictor_features",
            "fail_on_duplicate_commit_id_within_dataset",
        ],
        "data_schema",
    )
    require_config_keys(
        dataset_cfg, ["expected_dataset_value"], f"datasets.{dataset_id}"
    )

    expected_columns = [str(column) for column in schema_cfg["expected_columns"]]
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing and bool(schema_cfg["fail_on_missing_columns"]):
        raise DataValidationError(
            f"Dataset {dataset_id!r} is missing required column(s): {missing}"
        )

    predictor_features = [str(feature) for feature in schema_cfg["predictor_features"]]
    forbidden = set(map(str, schema_cfg["forbidden_predictor_features"]))
    overlap = [feature for feature in predictor_features if feature in forbidden]
    if overlap:
        raise ConfigError(
            "Configured predictor_features overlap forbidden_predictor_features: "
            f"{overlap}"
        )

    # 'Extra predictor features' means the configured predictor feature contract
    # itself is exact; metadata/target columns may of course coexist in the CSV.
    if bool(schema_cfg["fail_on_extra_predictor_features"]):
        allowed_non_features = set(expected_columns) - set(predictor_features)
        unexpected_columns = [
            column
            for column in frame.columns
            if column not in predictor_features and column not in allowed_non_features
        ]
        if unexpected_columns:
            raise DataValidationError(
                f"Dataset {dataset_id!r} contains unexpected column(s): {unexpected_columns}"
            )

    require_config_keys(schema_cfg, ["dataset_column"], "data_schema")
    dataset_column = str(schema_cfg["dataset_column"])
    if dataset_column in expected_columns and dataset_column in frame.columns:
        expected_value = dataset_cfg["expected_dataset_value"]
        observed_values = set(
            frame[dataset_column].dropna().astype(str).unique().tolist()
        )
        if observed_values != {str(expected_value)}:
            raise DataValidationError(
                f"Dataset column mismatch for {dataset_id!r}: expected only "
                f"{expected_value!r}, observed {sorted(observed_values)}"
            )

    target_column = str(schema_cfg["target_column"])
    allowed_targets = {
        schema_cfg["negative_target_value"],
        schema_cfg["positive_target_value"],
    }
    invalid_targets = (
        set(frame[target_column].dropna().unique().tolist()) - allowed_targets
    )
    if frame[target_column].isna().any() or invalid_targets:
        raise DataValidationError(
            f"Target column {target_column!r} must contain only {allowed_targets}; "
            f"invalid={invalid_targets}, missing={int(frame[target_column].isna().sum())}"
        )

    if frame[predictor_features].isna().any().any():
        bad = (
            frame[predictor_features]
            .columns[frame[predictor_features].isna().any()]
            .tolist()
        )
        raise DataValidationError(f"Predictor feature(s) contain missing values: {bad}")

    non_numeric = [
        feature
        for feature in predictor_features
        if not pd.api.types.is_numeric_dtype(frame[feature])
    ]
    if non_numeric:
        raise DataValidationError(
            f"Predictor feature(s) must be numeric: {non_numeric}"
        )

    values = frame[predictor_features].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise DataValidationError(
            "Predictor features contain NaN or infinite numeric values"
        )

    instance_id_column = str(schema_cfg["instance_id_column"])
    if frame[instance_id_column].isna().any():
        raise DataValidationError(
            f"Instance ID column {instance_id_column!r} contains missing values"
        )
    if bool(schema_cfg["fail_on_duplicate_commit_id_within_dataset"]):
        duplicate_mask = frame[instance_id_column].duplicated(keep=False)
        if duplicate_mask.any():
            examples = (
                frame.loc[duplicate_mask, instance_id_column]
                .astype(str)
                .head()
                .tolist()
            )
            raise DataValidationError(
                f"Duplicate instance IDs found in {dataset_id!r}; examples={examples}"
            )


def load_raw_dataset(
    experiment_config: Mapping[str, Any],
    dataset_id: str,
    project_root: str | Path,
) -> pd.DataFrame:
    """Load and validate one raw CSV using its config-provided path."""
    dataset_cfg = _dataset_config(experiment_config, dataset_id)
    require_config_keys(dataset_cfg, ["raw_csv"], f"datasets.{dataset_id}")
    csv_path = resolve_project_path(project_root, str(dataset_cfg["raw_csv"]))
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Configured raw dataset not found for {dataset_id!r}: {csv_path}"
        )

    frame = pd.read_csv(csv_path)
    validate_dataset_frame(frame, experiment_config, dataset_id)
    return frame


def predictor_frame(
    frame: pd.DataFrame, experiment_config: Mapping[str, Any]
) -> pd.DataFrame:
    """Return predictor features in the exact configured order."""
    schema_cfg = _schema_config(experiment_config)
    require_config_keys(schema_cfg, ["predictor_features"], "data_schema")
    features = [str(feature) for feature in schema_cfg["predictor_features"]]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Cannot build predictor frame; missing feature(s): {missing}"
        )
    return frame.loc[:, features].copy()


def target_series(
    frame: pd.DataFrame, experiment_config: Mapping[str, Any]
) -> pd.Series:
    """Return the configured target column."""
    schema_cfg = _schema_config(experiment_config)
    require_config_keys(schema_cfg, ["target_column"], "data_schema")
    target_column = str(schema_cfg["target_column"])
    if target_column not in frame.columns:
        raise DataValidationError(f"Target column not found: {target_column!r}")
    return frame[target_column].copy()


def assign_prediction_categories(
    frame: pd.DataFrame,
    predictions: Sequence[Any] | np.ndarray | pd.Series,
    experiment_config: Mapping[str, Any],
    *,
    output_column: str | None = None,
    predicted_label_column: str | None = None,
) -> pd.DataFrame:
    """Attach predicted labels and configured TP/TN/FP/FN categories.

    The mapping is read from ``category_definitions`` rather than implemented as
    hardcoded Boolean formulas, so later code uses the experiment contract as the
    single source of truth.
    """
    schema_cfg = _schema_config(experiment_config)
    require_config_keys(schema_cfg, ["target_column"], "data_schema")
    target_column = str(schema_cfg["target_column"])
    derived_columns = schema_cfg.get("derived_columns")
    if not isinstance(derived_columns, Mapping):
        raise ConfigError(
            "Missing or invalid data_schema.derived_columns configuration"
        )
    require_config_keys(
        derived_columns,
        ["predicted_label", "prediction_category"],
        "data_schema.derived_columns",
    )
    predicted_label_column = predicted_label_column or str(
        derived_columns["predicted_label"]
    )
    output_column = output_column or str(derived_columns["prediction_category"])

    categories = experiment_config.get("prediction_categories")
    definitions = experiment_config.get("category_definitions")
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise ConfigError("prediction_categories must be a sequence")
    if not isinstance(definitions, Mapping):
        raise ConfigError("category_definitions must be a mapping")

    prediction_array = np.asarray(predictions)
    if prediction_array.ndim != 1:
        prediction_array = prediction_array.reshape(-1)
    if len(prediction_array) != len(frame):
        raise DataValidationError(
            f"Prediction count {len(prediction_array)} does not match row count {len(frame)}"
        )

    result = frame.copy()
    result[predicted_label_column] = prediction_array
    result[output_column] = pd.Series(index=result.index, dtype="object")

    for category in categories:
        category_name = str(category)
        definition = definitions.get(category_name)
        if not isinstance(definition, Mapping):
            raise ConfigError(f"Missing category_definitions.{category_name}")
        require_config_keys(
            definition,
            ["true_label", "predicted_label"],
            f"category_definitions.{category_name}",
        )
        mask = (result[target_column] == definition["true_label"]) & (
            result[predicted_label_column] == definition["predicted_label"]
        )
        result.loc[mask, output_column] = category_name

    if result[output_column].isna().any():
        unresolved = result.loc[
            result[output_column].isna(), [target_column, predicted_label_column]
        ].drop_duplicates()
        raise DataValidationError(
            "Some rows do not match any configured prediction category: "
            f"{unresolved.to_dict(orient='records')}"
        )
    return result


def categorise_with_predictor(
    frame: pd.DataFrame,
    predictor: PredictorProtocol,
    experiment_config: Mapping[str, Any],
    *,
    output_column: str | None = None,
    predicted_label_column: str | None = None,
) -> pd.DataFrame:
    """Run frozen-predictor inference and attach TP/TN/FP/FN categories."""
    X = predictor_frame(frame, experiment_config)
    predictions = predictor.predict(X)
    return assign_prediction_categories(
        frame,
        predictions,
        experiment_config,
        output_column=output_column,
        predicted_label_column=predicted_label_column,
    )


def stratified_sample_by_category(
    categorised_frame: pd.DataFrame,
    experiment_config: Mapping[str, Any],
    rq_key: str,
    dataset_id: str,
    *,
    category_column: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw an equal-size deterministic sample from each configured RQ category.

    Sample size, categories, and insufficient-category policy come from the RQ
    section of ``experiment.yaml``. When ``seed`` is omitted, a stable dataset/RQ
    seed is derived from the configured global seed.

    Args:
        categorised_frame: DataFrame containing a category column.
        experiment_config: Parsed experiment configuration.
        rq_key: RQ configuration key (for example ``rq1`` or ``rq5``) whose
            sampling settings should be used.
        dataset_id: Dataset identifier included in deterministic seed derivation.
        category_column: Column containing configured category labels.
        seed: Optional caller-provided seed. If omitted, config-derived seed is used.
    """
    rq_cfg = experiment_config.get(rq_key)
    if not isinstance(rq_cfg, Mapping):
        raise ConfigError(f"Missing or invalid RQ configuration: {rq_key}")
    require_config_keys(
        rq_cfg, ["sample_per_category_per_dataset", "categories"], rq_key
    )

    schema_cfg = _schema_config(experiment_config)
    derived_columns = schema_cfg.get("derived_columns")
    if not isinstance(derived_columns, Mapping):
        raise ConfigError(
            "Missing or invalid data_schema.derived_columns configuration"
        )
    require_config_keys(
        derived_columns, ["prediction_category"], "data_schema.derived_columns"
    )
    category_column = category_column or str(derived_columns["prediction_category"])
    if category_column not in categorised_frame.columns:
        raise DataValidationError(f"Category column not found: {category_column!r}")

    sample_size = int(rq_cfg["sample_per_category_per_dataset"])
    if sample_size <= 0:
        raise ConfigError(f"{rq_key}.sample_per_category_per_dataset must be positive")

    categories = [str(category) for category in rq_cfg["categories"]]
    require_config_keys(rq_cfg, ["insufficient_category_policy"], rq_key)
    policy = str(rq_cfg["insufficient_category_policy"])
    if policy != "fail_with_clear_error":
        raise ConfigError(
            f"Unsupported configured insufficient-category policy for {rq_key}: {policy!r}"
        )

    base_seed = get_global_seed(experiment_config) if seed is None else int(seed)
    sampled_parts: list[pd.DataFrame] = []
    for category in categories:
        candidates = categorised_frame[categorised_frame[category_column] == category]
        if len(candidates) < sample_size:
            raise DataValidationError(
                f"Insufficient {category} instances for {dataset_id}/{rq_key}: "
                f"required={sample_size}, available={len(candidates)}"
            )
        category_seed = derive_seed(base_seed, rq_key, dataset_id, category)
        sampled_parts.append(
            candidates.sample(n=sample_size, replace=False, random_state=category_seed)
        )

    sampled = pd.concat(sampled_parts, axis=0)
    preserve_order = bool(
        experiment_config.get("reproducibility", {}).get(
            "preserve_input_order_after_sampling", False
        )
    )
    if preserve_order:
        return sampled.sort_index().copy()

    shuffle_seed = derive_seed(base_seed, rq_key, dataset_id, "combined_strata")
    return sampled.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)


def category_counts(
    categorised_frame: pd.DataFrame,
    experiment_config: Mapping[str, Any],
    *,
    category_column: str | None = None,
) -> pd.Series:
    """Return category counts ordered according to ``prediction_categories``."""
    schema_cfg = _schema_config(experiment_config)
    derived_columns = schema_cfg.get("derived_columns")
    if not isinstance(derived_columns, Mapping):
        raise ConfigError(
            "Missing or invalid data_schema.derived_columns configuration"
        )
    require_config_keys(
        derived_columns, ["prediction_category"], "data_schema.derived_columns"
    )
    category_column = category_column or str(derived_columns["prediction_category"])
    if category_column not in categorised_frame.columns:
        raise DataValidationError(f"Category column not found: {category_column!r}")
    categories = [
        str(category) for category in experiment_config.get("prediction_categories", [])
    ]
    if not categories:
        raise ConfigError("prediction_categories must be configured and non-empty")
    counts = categorised_frame[category_column].value_counts()
    return counts.reindex(categories, fill_value=0).astype(int)


def processed_dataset_directory(
    experiment_config: Mapping[str, Any],
    dataset_id: str,
    project_root: str | Path,
) -> Path:
    """Resolve and create the configured processed-data directory for a dataset."""
    dataset_cfg = _dataset_config(experiment_config, dataset_id)
    require_config_keys(dataset_cfg, ["processed_dir"], f"datasets.{dataset_id}")
    path = resolve_project_path(project_root, str(dataset_cfg["processed_dir"]))
    path.mkdir(parents=True, exist_ok=True)
    return path
