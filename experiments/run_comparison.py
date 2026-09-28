"""Run RQ4: paired traditional vs LLM-informed comparisons using RQ1--RQ3 metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments._common import (
    PROJECT_ROOT,
    load_project_configs,
    setup_project_logger,
    write_csv,
)
from src.utils import derive_seed, get_global_seed


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.isfinite(differences) & (differences != 0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def _bootstrap_ci(
    diff: np.ndarray, iterations: int, confidence: float, seed: int
) -> tuple[float, float]:
    values = diff[np.isfinite(diff)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def _paired_rows(
    frame: pd.DataFrame,
    metric: str,
    pair: dict[str, str],
    model: str,
    rq4: dict[str, Any],
    seed: int,
    *,
    dataset: str | None = None,
    category: str | None = None,
) -> dict[str, Any] | None:
    traditional, llm_method = pair["traditional"], pair["llm_informed"]
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if category is not None:
        frame = frame[frame["category"] == category]
    join_cols = ["dataset", "category", "instance_id"]
    trad = (
        frame[frame["method"] == traditional][join_cols + [metric]]
        .groupby(join_cols, as_index=False)[metric]
        .mean()
    )
    llm = (
        frame[(frame["method"] == llm_method) & (frame["llm_model"] == model)][
            join_cols + [metric]
        ]
        .groupby(join_cols, as_index=False)[metric]
        .mean()
    )
    merged = trad.merge(llm, on=join_cols, suffixes=("_traditional", "_llm"))
    if merged.empty:
        return None
    a = merged[f"{metric}_traditional"].to_numpy(dtype=float)
    b = merged[f"{metric}_llm"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return None
    diff = b - a
    try:
        stat, p = (
            wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
            if np.any(diff != 0)
            else (0.0, 1.0)
        )
    except ValueError:
        stat, p = float("nan"), float("nan")
    primary = rq4["primary_analysis"]
    low, high = _bootstrap_ci(
        diff,
        int(primary["bootstrap_iterations"]),
        float(primary["confidence_level"]),
        seed,
    )
    return {
        "scope": "overall"
        if dataset is None and category is None
        else "dataset_category",
        "dataset": dataset,
        "category": category,
        "metric": metric,
        "traditional_method": traditional,
        "llm_informed_method": llm_method,
        "llm_model": model,
        "n_pairs": len(diff),
        "traditional_mean": float(a.mean()),
        "llm_mean": float(b.mean()),
        "mean_difference_llm_minus_traditional": float(diff.mean()),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p),
        "rank_biserial_correlation": _rank_biserial(diff),
        "paired_bootstrap_ci_low": low,
        "paired_bootstrap_ci_high": high,
        "confidence_level": float(primary["confidence_level"]),
    }


def main() -> None:
    """Load already-generated metric files; do not regenerate explanations or evaluations."""
    config, _, _ = load_project_configs()
    logger = setup_project_logger(config)
    rq4 = config["rq4"]
    metric_dir = PROJECT_ROOT / str(config["paths"]["metrics_dir"])
    rq1_path = metric_dir / "rq1_stability.csv"
    rq2_path = PROJECT_ROOT / str(config["rq2"]["aggregate_output"])
    rq3_path = metric_dir / "rq3_instance_specificity.csv"
    missing = [p for p in (rq1_path, rq2_path, rq3_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"RQ4 requires completed RQ1--RQ3 metrics; missing: {missing}"
        )
    frames = [pd.read_csv(rq1_path), pd.read_csv(rq2_path), pd.read_csv(rq3_path)]
    metric_sets = [
        [
            "overlap_at_k",
            "rank_agreement_at_k",
            "direction_agreement_at_k",
            "score_stability",
        ],
        [
            "direction_consistency_rate",
            "meaningful_effect_rate",
            "mean_delta_probability",
            "mean_absolute_delta_probability",
        ],
        [
            "pairwise_jaccard_within_instance",
            "pairwise_jaccard_between_instance",
            "instance_idf_specificity",
            "separability_gap",
        ],
    ]
    rows: list[dict[str, Any]] = []
    for frame, metrics in zip(frames, metric_sets):
        for metric in metrics:
            if metric not in frame.columns:
                continue
            for pair in config["method_pairs"]:
                for model in config["llm_models"]:
                    seed = derive_seed(
                        get_global_seed(config),
                        "rq4",
                        metric,
                        pair["traditional"],
                        model,
                    )
                    row = _paired_rows(frame, metric, pair, str(model), rq4, seed)
                    if row is not None:
                        rows.append(row)
                    if bool(rq4.get("category_interaction_reporting", False)):
                        for dataset in config["datasets"]:
                            for category in config["prediction_categories"]:
                                scoped_seed = derive_seed(seed, dataset, category)
                                scoped = _paired_rows(
                                    frame,
                                    metric,
                                    pair,
                                    str(model),
                                    rq4,
                                    scoped_seed,
                                    dataset=str(dataset),
                                    category=str(category),
                                )
                                if scoped is not None:
                                    rows.append(scoped)
    out = (
        PROJECT_ROOT
        / str(config["paths"]["statistics_dir"])
        / "rq4_paired_comparison.csv"
    )
    write_csv(out, rows)
    logger.info(
        "RQ4 complete: %s paired-comparison rows; noninferiority remains disabled",
        len(rows),
    )


if __name__ == "__main__":
    main()
