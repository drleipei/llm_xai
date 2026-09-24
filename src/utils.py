"""Shared configuration, path, reproducibility, and I/O utilities.

This module intentionally contains no experiment-specific constants. Values such as
paths, seeds, logging behaviour, dataset names, and output locations are expected
to come from configuration files under ``configs/`` and be passed into helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML mapping. An empty YAML document becomes an empty dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigError: If the YAML document is not a mapping.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")
    return loaded


def require_config_keys(
    mapping: Mapping[str, Any], keys: Sequence[str], context: str
) -> None:
    """Validate that a mapping contains all required keys.

    Args:
        mapping: Configuration mapping to validate.
        keys: Required keys.
        context: Human-readable configuration location for error messages.

    Raises:
        ConfigError: If one or more required keys are absent.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing required config key(s) in {context}: {missing}")


def resolve_project_path(project_root: str | Path, configured_path: str | Path) -> Path:
    """Resolve a config-provided path relative to a project root.

    Absolute config paths are preserved. Relative paths are interpreted relative
    to ``project_root``. No experiment path is embedded in Python code.
    """
    root = Path(project_root).expanduser().resolve()
    candidate = Path(configured_path).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (including parents) if needed and return its path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def derive_seed(base_seed: int, *components: object) -> int:
    """Derive a stable 32-bit seed from a configured base seed and identifiers.

    Python's built-in ``hash`` is intentionally avoided because it is process
    salted. This helper gives deterministic seeds across processes and machines.
    """
    payload = "\x1f".join(
        [str(base_seed), *(str(component) for component in components)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def set_random_seed(seed: int) -> None:
    """Seed Python and NumPy random number generators for reproducible utilities."""
    random.seed(seed)
    np.random.seed(seed)


def get_global_seed(experiment_config: Mapping[str, Any]) -> int:
    """Read the global seed from the experiment configuration.

    The function first uses ``reproducibility.global_seed``. If that field is
    absent, it uses the explicitly configured ``project_seed``. It never inserts
    an experiment-specific seed in Python.
    """
    reproducibility = experiment_config.get("reproducibility", {})
    if isinstance(reproducibility, Mapping) and "global_seed" in reproducibility:
        return int(reproducibility["global_seed"])
    if "project_seed" in experiment_config:
        return int(experiment_config["project_seed"])
    raise ConfigError(
        "No configured seed found at reproducibility.global_seed or project_seed"
    )


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write JSON by replacing a temporary file in the same directory."""
    destination = Path(path)
    ensure_directory(destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, ensure_ascii=False, indent=indent, sort_keys=True
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def setup_logging(
    experiment_config: Mapping[str, Any], project_root: str | Path
) -> logging.Logger:
    """Configure project logging strictly from ``experiment.yaml`` settings.

    Args:
        experiment_config: Parsed experiment configuration.
        project_root: Project root used to resolve the configured log directory.

    Returns:
        Project logger.

    Raises:
        ConfigError: If the logging section is absent or incomplete.
    """
    logging_cfg = experiment_config.get("logging")
    if not isinstance(logging_cfg, Mapping):
        raise ConfigError("Missing or invalid logging configuration")
    require_config_keys(logging_cfg, ["level", "console", "file", "log_dir"], "logging")

    level_name = str(logging_cfg["level"]).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ConfigError(f"Invalid logging.level: {logging_cfg['level']}")

    project_id = str(experiment_config.get("project_id", __name__))
    logger = logging.getLogger(project_id)
    logger.setLevel(level)
    logger.propagate = False

    # Make repeated setup calls idempotent.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    require_config_keys(logging_cfg, ["include_timestamp"], "logging")
    include_timestamp = bool(logging_cfg["include_timestamp"])
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if include_timestamp
        else "%(levelname)s | %(name)s | %(message)s"
    )
    formatter = logging.Formatter(fmt)

    if bool(logging_cfg["console"]):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if bool(logging_cfg["file"]):
        log_dir = ensure_directory(
            resolve_project_path(project_root, str(logging_cfg["log_dir"]))
        )
        log_file = log_dir / f"{project_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def deep_get(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Retrieve a nested config value and fail clearly when a path is missing."""
    current: Any = mapping
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigError(f"Missing config path: {'.'.join(traversed)}")
        current = current[key]
    return current
