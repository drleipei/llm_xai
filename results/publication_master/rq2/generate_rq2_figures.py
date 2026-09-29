from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ============================================================
# RQ2 publication figures
#
# Place this script in:
#   results/publication_master/rq2/
#
# Required CSV files in the same directory:
#   rq2_primary_table.csv
#   rq2_friedman_kendalls_w.csv
#
# Also required from:
#   results/rq2/metrics/rq2_descriptive_by_decoding.csv
#
# Output:
#   Figure5_RQ2_Decoding_Sensitivity.png
#   Figure6_RQ2_Overall_Effect_Strength.png
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

PRIMARY_CSV = BASE_DIR / "rq2_primary_table.csv"
FRIEDMAN_CSV = BASE_DIR / "rq2_friedman_kendalls_w.csv"

# publication_master/rq2 -> results/rq2/metrics
DESCRIPTIVE_CSV = BASE_DIR / "rq2_descriptive_by_decoding.csv"

DPI = 400


# ============================================================
# Unified palette
# ============================================================

COLOR_BLUE = "#214F7A"
COLOR_ORANGE = "#D97706"
COLOR_TEAL = "#2A7F9E"
COLOR_RED = "#B85C38"

COLOR_GRID = "#D9DEE5"
COLOR_TEXT = "#202124"
COLOR_SPINE = "#4B5563"
COLOR_HIGHLIGHT = "#F6F2E8"

TEMP_COLORS = {
    0.0: "#214F7A",
    0.2: "#2A7F9E",
    0.5: "#D97706",
    0.8: "#B85C38",
}


# ============================================================
# Metric mappings
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
    "instance_idf_specificity": "Instance-IDF",
    "separability_gap": "Separability Gap",
}


STABILITY_METRICS = [
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
    "score_stability",
]

DISCRIM_METRICS = [
    "pairwise_jaccard_within_instance",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
    "separability_gap",
]

ALL_METRICS = list(METRIC_LABELS.keys())


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
        raise FileNotFoundError(f"Missing: {PRIMARY_CSV}")

    if not FRIEDMAN_CSV.exists():
        raise FileNotFoundError(f"Missing: {FRIEDMAN_CSV}")

    if not DESCRIPTIVE_CSV.exists():
        raise FileNotFoundError(
            f"Missing descriptive file: {DESCRIPTIVE_CSV}"
        )

    primary = pd.read_csv(PRIMARY_CSV)
    friedman = pd.read_csv(FRIEDMAN_CSV)
    descriptive = pd.read_csv(DESCRIPTIVE_CSV)

    return primary, friedman, descriptive


# ============================================================
# Utility
# ============================================================

def clean_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    ax.grid(
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.5,
    )

    ax.set_axisbelow(True)


# ============================================================
# Figure 5
# Temperature sensitivity
# ============================================================

def figure5_temperature_sensitivity(descriptive):

    # Only temperature analysis:
    # top_p fixed at 1.0
    temp_df = descriptive[
        descriptive["top_p"].eq(1.0)
        & descriptive["temperature"].isin([0.0, 0.2, 0.5, 0.8])
    ].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.8),
        gridspec_kw={"wspace": 0.25},
    )

    ax1, ax2 = axes

    temperatures = [0.0, 0.2, 0.5, 0.8]

    # --------------------------------------------------------
    # Panel A: Stability
    # --------------------------------------------------------

    for metric in STABILITY_METRICS:

        d = (
            temp_df[temp_df["metric"].eq(metric)]
            .set_index("temperature")
            .reindex(temperatures)
        )

        ax1.plot(
            temperatures,
            d["mean"],
            marker="o",
            linewidth=2.0,
            markersize=6.5,
            label=METRIC_LABELS[metric],
        )

    ax1.axvline(
        0.2,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_SPINE,
        alpha=0.6,
    )

    ax1.text(
        0.205,
        ax1.get_ylim()[1] if ax1.get_ylim()[1] else 1,
        "",
    )

    ax1.set_title(
        "(a) Repeat-run stability",
        fontsize=13,
        pad=10,
    )

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Mean metric value")

    ax1.set_xticks(temperatures)

    clean_axis(ax1)

    ax1.legend(
        loc="best",
        fontsize=9,
    )

    # --------------------------------------------------------
    # Panel B: Discriminativeness
    # --------------------------------------------------------

    for metric in DISCRIM_METRICS:

        d = (
            temp_df[temp_df["metric"].eq(metric)]
            .set_index("temperature")
            .reindex(temperatures)
        )

        ax2.plot(
            temperatures,
            d["mean"],
            marker="o",
            linewidth=2.0,
            markersize=6.5,
            label=METRIC_LABELS[metric],
        )

    ax2.axvline(
        0.2,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_SPINE,
        alpha=0.6,
    )

    ax2.set_title(
        "(b) Instance discriminativeness",
        fontsize=13,
        pad=10,
    )

    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Mean metric value")

    ax2.set_xticks(temperatures)

    clean_axis(ax2)

    ax2.legend(
        loc="best",
        fontsize=9,
    )

    fig.suptitle(
        "RQ2: Temperature Sensitivity",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.03,
        "Dashed line marks the reference temperature (T=0.2; top_p=1.0).",
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.15,
        top=0.84,
        wspace=0.25,
    )

    output = BASE_DIR / "Figure5_RQ2_Decoding_Sensitivity.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ============================================================
# Figure 6
# Overall effect strength heatmap
# ============================================================

def figure6_overall_effect_strength(primary):

    d = primary.copy()

    d["Metric"] = d["metric"].map(METRIC_LABELS)

    ordered_labels = [
        METRIC_LABELS[m]
        for m in ALL_METRICS
        if m in d["metric"].values
    ]

    d = (
        d.set_index("Metric")
        .reindex(ordered_labels)
        .reset_index()
    )

    values = np.column_stack([
        d["temperature_kendalls_w"].to_numpy(dtype=float),
        d["top_p_kendalls_w"].to_numpy(dtype=float),
    ])

    # blue -> pale -> orange
    cmap = LinearSegmentedColormap.from_list(
        "rq2_effect_strength",
        [
            "#F7F7F7",
            "#9BB9D3",
            "#2E6DAA",
        ],
        N=256,
    )

    norm = Normalize(
        vmin=0,
        vmax=1,
    )

    fig, ax = plt.subplots(
        figsize=(7.8, 8.0)
    )

    im = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels([
        "Temperature",
        "Top-p",
    ])

    ax.set_yticks(
        np.arange(len(d))
    )

    ax.set_yticklabels(
        d["Metric"]
    )

    # cell borders
    ax.set_xticks(
        np.arange(-0.5, 2, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(d), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1.0,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # significance markers
    for i in range(len(d)):

        temp_p = float(
            d.loc[i, "temperature_friedman_p"]
        )

        top_p_p = float(
            d.loc[i, "top_p_friedman_p"]
        )

        ps = [temp_p, top_p_p]

        for j in range(2):

            value = values[i, j]

            stars = (
                "***" if ps[j] < 0.001
                else "**" if ps[j] < 0.01
                else "*" if ps[j] < 0.05
                else ""
            )

            color = (
                "white"
                if value > 0.60
                else COLOR_TEXT
            )

            ax.text(
                j,
                i,
                f"{value:.3f}{stars}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight=(
                    "bold"
                    if stars
                    else "normal"
                ),
                color=color,
            )

    ax.set_title(
        "RQ2: Overall Decoding-Parameter Effects",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.04,
    )

    cbar.set_label(
        "Kendall's W",
        fontsize=11,
    )

    fig.text(
        0.5,
        0.02,
        "Stars indicate Friedman significance: "
        "* p<0.05, ** p<0.01, *** p<0.001.",
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.30,
        right=0.90,
        bottom=0.08,
        top=0.92,
    )

    output = (
        BASE_DIR
        / "Figure6_RQ2_Overall_Effect_Strength.png"
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

    primary, friedman, descriptive = load_data()

    outputs = [
        figure5_temperature_sensitivity(
            descriptive
        ),
        figure6_overall_effect_strength(
            primary
        ),
    ]

    print()
    print("RQ2 publication figures generated:")
    print()

    for output in outputs:
        print(f"  {output}")

    print()


if __name__ == "__main__":
    main()