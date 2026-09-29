from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# RQ3 publication figures
#
# Place this script in:
#   results/publication_master/rq3/
#
# Required CSV files in the same directory:
#   rq3_primary_table.csv
#   rq3_component_summary.csv
#
# Output:
#   Figure7_RQ3_Prompt_Ablation_Effects.png
#   Figure8_RQ3_Component_Impact_Summary.png
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

PRIMARY_CSV = BASE_DIR / "rq3_primary_table.csv"
COMPONENT_CSV = BASE_DIR / "rq3_component_summary.csv"

DPI = 400


# ============================================================
# Unified publication palette
# ============================================================

COLOR_TEXT = "#202124"
COLOR_GRID = "#D9DEE5"
COLOR_SPINE = "#4B5563"

COLOR_BLUE = "#2E6DAA"
COLOR_ZERO = "#F7F7F7"
COLOR_ORANGE = "#C46A23"

COLOR_BAR_1 = "#214F7A"
COLOR_BAR_2 = "#D97706"


# ============================================================
# Metric labels
# ============================================================

METRIC_LABELS = {
    "overlap_at_k": "Overlap@K",
    "rank_agreement_at_k": "Rank Agreement@K",
    "direction_agreement_at_k": "Direction Agreement@K",
    "score_stability": "Score Stability",
    "direction_consistency_rate": "Direction Consistency",
    "meaningful_effect_rate": "Meaningful Effect",
    "mean_delta_probability": "Mean signed Δp",
    "mean_absolute_delta_probability": "Mean |Δp|",
    "normalized_feature_entropy": "Normalized Feature Entropy",
    "pairwise_jaccard_within_instance": "Within-instance Jaccard",
    "pairwise_jaccard_between_instance": "Between-instance Jaccard",
    "instance_idf_specificity": "Instance-IDF Specificity",
    "separability_gap": "Separability Gap",
}


METRIC_ORDER = [
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
    "score_stability",
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_delta_probability",
    "mean_absolute_delta_probability",
    "normalized_feature_entropy",
    "pairwise_jaccard_within_instance",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
    "separability_gap",
]


ABLATIONS = [
    "no_semantics",
    "no_grounding",
    "no_constraints",
    "no_instance_context",
]


ABLATION_LABELS = {
    "no_semantics": "No semantics",
    "no_grounding": "No grounding",
    "no_constraints": "No constraints",
    "no_instance_context": "No instance context",
}


# ============================================================
# Global style
# ============================================================

def set_global_style():

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,

        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,

        "axes.edgecolor": COLOR_SPINE,
        "axes.linewidth": 0.9,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,

        "legend.fontsize": 10,
        "legend.frameon": False,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ============================================================
# Load data
# ============================================================

def load_data():

    if not PRIMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {PRIMARY_CSV}"
        )

    if not COMPONENT_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {COMPONENT_CSV}"
        )

    primary = pd.read_csv(PRIMARY_CSV)
    component = pd.read_csv(COMPONENT_CSV)

    required_primary = {
        "metric",
        "friedman_p",
        "kendalls_w",
    }

    for ablation in ABLATIONS:
        required_primary.add(
            f"{ablation}_delta_full_minus_ablation"
        )
        required_primary.add(
            f"{ablation}_p_holm"
        )
        required_primary.add(
            f"{ablation}_significant"
        )

    missing = required_primary - set(primary.columns)

    if missing:
        raise ValueError(
            "rq3_primary_table.csv missing columns: "
            f"{sorted(missing)}"
        )

    required_component = {
        "ablation",
        "significant_metrics",
        "full_higher_significant_metrics",
        "ablation_higher_significant_metrics",
        "mean_absolute_effect_across_metrics",
    }

    missing = required_component - set(component.columns)

    if missing:
        raise ValueError(
            "rq3_component_summary.csv missing columns: "
            f"{sorted(missing)}"
        )

    return primary, component


# ============================================================
# Helper
# ============================================================

def p_to_star(p):

    if pd.isna(p):
        return ""

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return ""


# ============================================================
# Figure 7
# Prompt ablation effect heatmap
# ============================================================

def figure7_prompt_ablation_effects(primary):

    d = primary.copy()

    d["Metric"] = d["metric"].map(METRIC_LABELS)

    metric_labels = [
        METRIC_LABELS[m]
        for m in METRIC_ORDER
        if m in d["metric"].values
    ]

    d = (
        d.set_index("Metric")
        .reindex(metric_labels)
        .reset_index()
    )

    # --------------------------------------------------------
    # Build effect matrix:
    # Full - Ablation
    #
    # Positive:
    #   Full has larger numerical value
    #
    # Negative:
    #   Ablation has larger numerical value
    #
    # IMPORTANT:
    # This is NOT direction-corrected superiority.
    # --------------------------------------------------------

    effect_matrix = np.column_stack([
        d[
            f"{ablation}_delta_full_minus_ablation"
        ].to_numpy(dtype=float)
        for ablation in ABLATIONS
    ])

    p_matrix = np.column_stack([
        d[
            f"{ablation}_p_holm"
        ].to_numpy(dtype=float)
        for ablation in ABLATIONS
    ])

    # Symmetric color range around 0
    vmax = np.nanmax(np.abs(effect_matrix))

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    cmap = LinearSegmentedColormap.from_list(
        "rq3_ablation",
        [
            COLOR_BLUE,
            COLOR_ZERO,
            COLOR_ORANGE,
        ],
        N=256,
    )

    fig, ax = plt.subplots(
        figsize=(10.2, 8.2)
    )

    im = ax.imshow(
        effect_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(len(ABLATIONS))
    )

    ax.set_xticklabels(
        [ABLATION_LABELS[a] for a in ABLATIONS],
        rotation=18,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(metric_labels))
    )

    ax.set_yticklabels(
        metric_labels
    )

    # --------------------------------------------------------
    # Cell borders
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, len(ABLATIONS), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(metric_labels), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Cell labels
    # --------------------------------------------------------

    for i in range(effect_matrix.shape[0]):

        for j in range(effect_matrix.shape[1]):

            value = effect_matrix[i, j]
            pval = p_matrix[i, j]

            stars = p_to_star(pval)

            text_color = (
                "white"
                if abs(value) > 0.60 * vmax
                else COLOR_TEXT
            )

            ax.text(
                j,
                i,
                f"{value:+.3f}{stars}",
                ha="center",
                va="center",
                fontsize=9.3,
                color=text_color,
                fontweight=(
                    "bold"
                    if stars
                    else "normal"
                ),
            )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "RQ3: Prompt-Component Ablation Effects",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )

    cbar.set_label(
        "Mean difference\n(Full − Ablation)",
        fontsize=11,
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.30,
        right=0.90,
        bottom=0.12,
        top=0.92,
    )

    output = (
        BASE_DIR
        / "Figure7_RQ3_Prompt_Ablation_Effects.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ============================================================
# Figure 8
# Component impact summary
# ============================================================

def figure8_component_impact_summary(component):

    d = (
        component
        .set_index("ablation")
        .reindex(ABLATIONS)
        .reset_index()
    )

    labels = [
        ABLATION_LABELS[a]
        for a in d["ablation"]
    ]

    significant = (
        d["significant_metrics"]
        .to_numpy(dtype=float)
    )

    mean_abs_effect = (
        d["mean_absolute_effect_across_metrics"]
        .to_numpy(dtype=float)
    )

    y = np.arange(len(labels))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.4),
        gridspec_kw={"wspace": 0.32},
    )

    ax1, ax2 = axes

    # --------------------------------------------------------
    # Panel A: significant metrics
    # --------------------------------------------------------

    bars1 = ax1.barh(
        y,
        significant,
        color=COLOR_BAR_1,
        height=0.58,
    )

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)

    ax1.invert_yaxis()

    ax1.set_xlabel(
        "Significant metrics"
    )

    ax1.set_xlim(
        0,
        max(13, significant.max() + 1)
    )

    ax1.set_title(
        "(a) Number of significant changes",
        fontsize=13,
        pad=10,
    )

    ax1.grid(
        axis="x",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.5,
    )

    ax1.set_axisbelow(True)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar, value in zip(bars1, significant):
        ax1.text(
            value + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Panel B: mean absolute effect
    # --------------------------------------------------------

    bars2 = ax2.barh(
        y,
        mean_abs_effect,
        color=COLOR_BAR_2,
        height=0.58,
    )

    ax2.set_yticks(y)
    ax2.set_yticklabels([])

    ax2.invert_yaxis()

    ax2.set_xlabel(
        "Mean absolute effect"
    )

    ax2.set_xlim(
        0,
        mean_abs_effect.max() * 1.25,
    )

    ax2.set_title(
        "(b) Overall change magnitude",
        fontsize=13,
        pad=10,
    )

    ax2.grid(
        axis="x",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.5,
    )

    ax2.set_axisbelow(True)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, value in zip(bars2, mean_abs_effect):
        ax2.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "RQ3: Prompt-Component Impact Summary",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.14,
        top=0.82,
        wspace=0.30,
    )

    output = (
        BASE_DIR
        / "Figure8_RQ3_Component_Impact_Summary.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ============================================================
# Main
# ============================================================

def main():

    set_global_style()

    primary, component = load_data()

    outputs = [
        figure7_prompt_ablation_effects(
            primary
        ),
        figure8_component_impact_summary(
            component
        ),
    ]

    print()
    print("RQ3 publication figures generated:")
    print()

    for output in outputs:
        print(f"  {output}")

    print()


if __name__ == "__main__":
    main()