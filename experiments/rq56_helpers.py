"""Shared helpers for RQ5 decoding sensitivity and RQ6 prompt ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


VALID_DIRECTIONS = {
    "support",
    "oppose",
    "neutral",
}


def structural_validation(
    explanation: Mapping[str, Any],
    *,
    feature_space: Sequence[str],
    top_k: int,
) -> dict[str, bool]:
    raw = explanation.get("top_features")

    sequence_valid = (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
    )

    if not sequence_valid:
        return {
            "top_k_compliant": False,
            "unique_features_valid": False,
            "feature_names_valid": False,
            "direction_labels_valid": False,
            "final_structural_compliance": False,
        }

    features = []
    directions = []

    valid_item_structure = True

    for item in raw:
        if not isinstance(item, Mapping):
            valid_item_structure = False
            continue

        if "feature" not in item:
            valid_item_structure = False
        else:
            features.append(str(item["feature"]))

        if "direction" not in item:
            valid_item_structure = False
        else:
            directions.append(str(item["direction"]))

    top_k_ok = 1 <= len(raw) <= top_k

    unique_ok = (
        valid_item_structure
        and len(features) == len(set(features))
    )

    allowed = set(str(x) for x in feature_space)

    feature_ok = (
        valid_item_structure
        and len(features) == len(raw)
        and all(x in allowed for x in features)
    )

    direction_ok = (
        valid_item_structure
        and len(directions) == len(raw)
        and all(x in VALID_DIRECTIONS for x in directions)
    )

    final_ok = (
        top_k_ok
        and unique_ok
        and feature_ok
        and direction_ok
    )

    return {
        "top_k_compliant": bool(top_k_ok),
        "unique_features_valid": bool(unique_ok),
        "feature_names_valid": bool(feature_ok),
        "direction_labels_valid": bool(direction_ok),
        "final_structural_compliance": bool(final_ok),
    }
