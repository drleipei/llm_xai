from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# ============================================================
# RQ3 publication figures
#
# Final design:
#   3 dimensions × 3 metrics = 9 primary metrics
#
# Place this script in:
#   results/rq3/
#
# Required CSV:
#   rq3_primary_table.csv
#
# Output:
#   Figure7_RQ3_Prompt_Ablation_Mean_Differences.png
#   Figure8_RQ3_Component_Impact_Summary.png
#   rq3_component_summary_9metrics.csv
#
# MAMS:
#   Mean Absolute Metric Shift
#
#   MAMS =
#       mean over the final 9 metrics of
#       |Full - Ablation|
#
# Interpretation:
#   MAMS is a descriptive summary of the magnitude
#   of numerical changes across the 9 primary metrics.
#
#   It is NOT:
#       - a standardized effect size;
#       - a unified explanation-quality score;
#       - a direction-corrected superiority measure.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

PRIMARY_CSV = (
    BASE_DIR
    / "rq3_primary_table.csv"
)

DPI = 400


# ============================================================
# Publication palette
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
# Final RQ3 metric design
# 3 dimensions × 3 metrics = 9 primary metrics
# ============================================================

METRIC_LABELS = {

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    "overlap_at_k":
        "Overlap@K",

    "rank_agreement_at_k":
        "Rank Agreement@K",

    "direction_agreement_at_k":
        "Direction Agreement@K",

    # --------------------------------------------------------
    # Model-Response Alignment
    # --------------------------------------------------------

    "direction_consistency_rate":
        "Direction Consistency Rate",

    "meaningful_effect_rate":
        "Meaningful-Effect Rate",

    "mean_absolute_delta_probability":
        "Mean |Δp|",

    # --------------------------------------------------------
    # Discriminativeness
    # --------------------------------------------------------

    "normalized_feature_entropy":
        "Normalized Feature Entropy",

    "pairwise_jaccard_between_instance":
        "Between-instance Jaccard",

    "instance_idf_specificity":
        "Instance-IDF Specificity",
}


STABILITY_METRICS = [
    "overlap_at_k",
    "rank_agreement_at_k",
    "direction_agreement_at_k",
]


ALIGNMENT_METRICS = [
    "direction_consistency_rate",
    "meaningful_effect_rate",
    "mean_absolute_delta_probability",
]


DISCRIM_METRICS = [
    "normalized_feature_entropy",
    "pairwise_jaccard_between_instance",
    "instance_idf_specificity",
]


METRIC_ORDER = (
    STABILITY_METRICS
    + ALIGNMENT_METRICS
    + DISCRIM_METRICS
)


# ============================================================
# Prompt-component ablations
#
# Order is aligned with:
#   - manuscript discussion
#   - Table 10
#   - component-impact ranking
# ============================================================

ABLATIONS = [
    "no_instance_context",
    "no_semantics",
    "no_constraints",
    "no_grounding",
]


ABLATION_LABELS = {

    "no_instance_context":
        "No instance context",

    "no_semantics":
        "No semantics",

    "no_constraints":
        "No constraints",

    "no_grounding":
        "No grounding",
}


# ============================================================
# Global style
# ============================================================

def set_global_style():

    plt.rcParams.update({

        "font.family":
            "DejaVu Sans",

        "font.size":
            11,

        "axes.titlesize":
            13,

        "axes.titleweight":
            "bold",

        "axes.labelsize":
            12,

        "axes.edgecolor":
            COLOR_SPINE,

        "axes.linewidth":
            0.9,

        "xtick.labelsize":
            10,

        "ytick.labelsize":
            10,

        "text.color":
            COLOR_TEXT,

        "axes.labelcolor":
            COLOR_TEXT,

        "xtick.color":
            COLOR_TEXT,

        "ytick.color":
            COLOR_TEXT,

        "legend.fontsize":
            10,

        "legend.frameon":
            False,

        "figure.facecolor":
            "white",

        "axes.facecolor":
            "white",

        "savefig.facecolor":
            "white",
    })


# ============================================================
# Load data
# ============================================================

def load_data():

    if not PRIMARY_CSV.exists():

        raise FileNotFoundError(
            f"Missing file:\n"
            f"{PRIMARY_CSV}"
        )

    primary = pd.read_csv(
        PRIMARY_CSV
    )

    required = {
        "metric",
        "friedman_p",
        "kendalls_w",
    }

    for ablation in ABLATIONS:

        required.add(
            f"{ablation}_"
            "delta_full_minus_ablation"
        )

        required.add(
            f"{ablation}_p_holm"
        )

        required.add(
            f"{ablation}_significant"
        )

    missing = (
        required
        - set(
            primary.columns
        )
    )

    if missing:

        raise ValueError(
            "rq3_primary_table.csv "
            "missing columns:\n"
            f"{sorted(missing)}"
        )

    return primary


# ============================================================
# Helpers
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


def validate_final_design(
    primary
):

    print()

    print(
        "Validating final RQ3 design:"
    )

    print(
        "3 dimensions x 3 metrics "
        "= 9 primary metrics"
    )

    print()

    available = set(
        primary[
            "metric"
        ]
        .dropna()
        .astype(str)
    )

    for metric in METRIC_ORDER:

        if metric not in available:

            raise ValueError(
                "Missing final RQ3 metric: "
                f"{metric}"
            )

        print(
            f"[OK] "
            f"{METRIC_LABELS[metric]}"
        )

    print()

    print(
        "Metrics excluded from final "
        "primary RQ3 analysis:"
    )

    print(
        "  Score Stability"
    )

    print(
        "  Mean signed Δp"
    )

    print(
        "  Within-instance Jaccard"
    )

    print(
        "  Separability Gap"
    )

    print()

    print(
        "Final metric interpretation:"
    )

    print()

    print(
        "  Higher indicates a stronger value "
        "under the predefined construct:"
    )

    print(
        "    Overlap@K"
    )

    print(
        "    Rank Agreement@K"
    )

    print(
        "    Direction Agreement@K"
    )

    print(
        "    Direction Consistency Rate"
    )

    print(
        "    Meaningful-Effect Rate"
    )

    print(
        "    Mean |Δp|"
    )

    print(
        "    Normalized Feature Entropy"
    )

    print(
        "    Instance-IDF Specificity"
    )

    print()

    print(
        "  Lower indicates less "
        "cross-instance explanation overlap:"
    )

    print(
        "    Between-instance Jaccard"
    )

    print()

    print(
        "Note:"
    )

    print(
        "  Normalized Feature Entropy "
        "describes feature-selection dispersion "
        "under the study's operationalized construct."
    )

    print(
        "  It is not treated as a universal "
        "overall explanation-quality score."
    )

    print()


# ============================================================
# Build component summary
# using ONLY the final 9 primary metrics
# ============================================================

def build_final_component_summary(
    primary
):

    d = (
        primary[
            primary[
                "metric"
            ].isin(
                METRIC_ORDER
            )
        ]
        .copy()
    )

    rows = []

    for ablation in ABLATIONS:

        delta_col = (
            f"{ablation}_"
            "delta_full_minus_ablation"
        )

        p_col = (
            f"{ablation}_p_holm"
        )

        # ----------------------------------------------------
        # Holm-adjusted p is the authoritative
        # significance criterion.
        # ----------------------------------------------------

        significant = (
            d[
                p_col
            ]
            .astype(float)
            .lt(
                0.05
            )
        )

        # ----------------------------------------------------
        # Raw numerical difference:
        #
        #   Full - Ablation
        #
        # Positive:
        #   Full has a larger raw numerical value.
        #
        # Negative:
        #   Ablation has a larger raw numerical value.
        #
        # This sign is NOT a unified quality direction.
        # ----------------------------------------------------

        differences = (
            d[
                delta_col
            ]
            .astype(float)
        )

        significant_count = int(
            significant.sum()
        )

        full_higher_count = int(
            (
                significant
                &
                differences.gt(
                    0
                )
            )
            .sum()
        )

        ablation_higher_count = int(
            (
                significant
                &
                differences.lt(
                    0
                )
            )
            .sum()
        )

        # ----------------------------------------------------
        # MAMS
        #
        # Mean Absolute Metric Shift
        #
        # MAMS =
        #
        #   (1 / 9) *
        #   sum_m |Full_m - Ablation_m|
        #
        # MAMS is descriptive only.
        #
        # It is NOT:
        #   - a standardized effect size;
        #   - a unified quality score;
        #   - a direction-corrected superiority measure.
        # ----------------------------------------------------

        mams = float(
            differences
            .abs()
            .mean()
        )

        rows.append({

            "ablation":
                ablation,

            "significant_metrics":
                significant_count,

            "full_higher_significant_metrics":
                full_higher_count,

            "ablation_higher_significant_metrics":
                ablation_higher_count,

            "mams":
                mams,
        })

    summary = pd.DataFrame(
        rows
    )

    return summary


# ============================================================
# Figure 7
# Prompt-component ablation mean differences
#
# Exactly:
#   9 primary metrics × 4 ablation conditions
#
# Cell value:
#   Full - Ablation
#
# IMPORTANT:
#   These are RAW mean differences.
#
# They are NOT:
#   - standardized effect sizes;
#   - direction-corrected superiority scores;
#   - unified explanation-quality changes.
# ============================================================

def figure7_prompt_ablation_mean_differences(
    primary
):

    d = (
        primary[
            primary[
                "metric"
            ].isin(
                METRIC_ORDER
            )
        ]
        .copy()
    )

    d[
        "Metric"
    ] = (
        d[
            "metric"
        ]
        .map(
            METRIC_LABELS
        )
    )

    metric_labels = [
        METRIC_LABELS[
            m
        ]
        for m
        in METRIC_ORDER
    ]

    d = (
        d
        .set_index(
            "Metric"
        )
        .reindex(
            metric_labels
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Raw Full - Ablation mean differences
    # --------------------------------------------------------

    difference_matrix = np.column_stack(
        [
            d[
                f"{ablation}_"
                "delta_full_minus_ablation"
            ]
            .to_numpy(
                dtype=float
            )
            for ablation
            in ABLATIONS
        ]
    )

    p_matrix = np.column_stack(
        [
            d[
                f"{ablation}_"
                "p_holm"
            ]
            .to_numpy(
                dtype=float
            )
            for ablation
            in ABLATIONS
        ]
    )

    if (
        np.isnan(
            difference_matrix
        )
        .any()
        or
        np.isnan(
            p_matrix
        )
        .any()
    ):

        raise ValueError(
            "Missing values in final "
            "9-metric RQ3 mean-difference heatmap."
        )

    vmax = np.nanmax(
        np.abs(
            difference_matrix
        )
    )

    if vmax == 0:

        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    cmap = (
        LinearSegmentedColormap
        .from_list(
            "rq3_mean_differences",
            [
                COLOR_BLUE,
                COLOR_ZERO,
                COLOR_ORANGE,
            ],
            N=256,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            6.8,
        )
    )

    im = ax.imshow(
        difference_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            len(
                ABLATIONS
            )
        )
    )

    ax.set_xticklabels(
        [
            ABLATION_LABELS[
                a
            ]
            for a
            in ABLATIONS
        ],
        rotation=18,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(
                metric_labels
            )
        )
    )

    ax.set_yticklabels(
        metric_labels
    )

    # --------------------------------------------------------
    # Cell borders
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(
                ABLATIONS
            ),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(
                metric_labels
            ),
            1,
        ),
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
    # Dimension separators
    #
    # Rows 0--2:
    #   Stability
    #
    # Rows 3--5:
    #   Model-Response Alignment
    #
    # Rows 6--8:
    #   Discriminativeness
    # --------------------------------------------------------

    ax.axhline(
        2.5,
        color=COLOR_SPINE,
        linewidth=1.3,
        alpha=0.8,
    )

    ax.axhline(
        5.5,
        color=COLOR_SPINE,
        linewidth=1.3,
        alpha=0.8,
    )

    # --------------------------------------------------------
    # Cell labels
    # --------------------------------------------------------

    for i in range(
        difference_matrix.shape[
            0
        ]
    ):

        for j in range(
            difference_matrix.shape[
                1
            ]
        ):

            value = (
                difference_matrix[
                    i,
                    j,
                ]
            )

            pval = (
                p_matrix[
                    i,
                    j,
                ]
            )

            stars = (
                p_to_star(
                    pval
                )
            )

            text_color = (
                "white"
                if (
                    abs(
                        value
                    )
                    > 0.60 * vmax
                )
                else COLOR_TEXT
            )

            ax.text(
                j,
                i,
                f"{value:+.3f}"
                f"{stars}",
                ha="center",
                va="center",
                fontsize=9.4,
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
        "RQ3: Prompt-Component Ablation Mean Differences",
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
        "Mean difference\n"
        "(Full − Ablation)",
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.32,
        right=0.90,
        bottom=0.14,
        top=0.91,
    )

    output = (
        BASE_DIR
        / "Figure7_RQ3_Prompt_Ablation_Mean_Differences.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output


# ============================================================
# Figure 8
# Prompt-component impact summary
#
# Panel A:
#   Number of primary metrics significantly different
#   from Full after Holm correction.
#
# Panel B:
#   MAMS = Mean Absolute Metric Shift
#
# Both panels use ONLY the final 9 primary metrics.
# ============================================================

def figure8_component_impact_summary(
    component
):

    d = (
        component
        .set_index(
            "ablation"
        )
        .reindex(
            ABLATIONS
        )
        .reset_index()
    )

    labels = [
        ABLATION_LABELS[
            a
        ]
        for a
        in d[
            "ablation"
        ]
    ]

    significant = (
        d[
            "significant_metrics"
        ]
        .to_numpy(
            dtype=float
        )
    )

    mams = (
        d[
            "mams"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = np.arange(
        len(
            labels
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.8,
            5.4,
        ),
        gridspec_kw={
            "wspace":
                0.32
        },
    )

    ax1, ax2 = axes

    # --------------------------------------------------------
    # Panel A
    # Number of significant metric changes
    # --------------------------------------------------------

    bars1 = ax1.barh(
        y,
        significant,
        color=COLOR_BAR_1,
        height=0.58,
    )

    ax1.set_yticks(
        y
    )

    ax1.set_yticklabels(
        labels
    )

    ax1.invert_yaxis()

    ax1.set_xlabel(
        "Significant metrics"
    )

    # Final primary metric count = 9
    ax1.set_xlim(
        0,
        9.8,
    )

    ax1.set_xticks(
        np.arange(
            0,
            10,
            1,
        )
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

    ax1.set_axisbelow(
        True
    )

    ax1.spines[
        "top"
    ].set_visible(
        False
    )

    ax1.spines[
        "right"
    ].set_visible(
        False
    )

    for bar, value in zip(
        bars1,
        significant,
    ):

        ax1.text(
            value + 0.15,
            (
                bar.get_y()
                + bar.get_height() / 2
            ),
            f"{int(value)}",
            va="center",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Panel B
    # MAMS = Mean Absolute Metric Shift
    # --------------------------------------------------------

    bars2 = ax2.barh(
        y,
        mams,
        color=COLOR_BAR_2,
        height=0.58,
    )

    ax2.set_yticks(
        y
    )

    ax2.set_yticklabels(
        []
    )

    ax2.invert_yaxis()

    ax2.set_xlabel(
        "MAMS"
    )

    max_mams = float(
        np.nanmax(
            mams
        )
    )

    ax2.set_xlim(
        0,
        (
            max_mams * 1.25
            if max_mams > 0
            else 1.0
        ),
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

    ax2.set_axisbelow(
        True
    )

    ax2.spines[
        "top"
    ].set_visible(
        False
    )

    ax2.spines[
        "right"
    ].set_visible(
        False
    )

    text_offset = (
        max_mams * 0.015
        if max_mams > 0
        else 0.001
    )

    for bar, value in zip(
        bars2,
        mams,
    ):

        ax2.text(
            value + text_offset,
            (
                bar.get_y()
                + bar.get_height() / 2
            ),
            f"{value:.4f}",
            va="center",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "RQ3: Prompt-Component Impact Summary",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    fig.subplots_adjust(
        left=0.22,
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

    plt.close(
        fig
    )

    return output


# ============================================================
# Print final 9-metric statistics
# ============================================================

def print_final_summary(
    primary,
    component,
):

    d = (
        primary[
            primary[
                "metric"
            ].isin(
                METRIC_ORDER
            )
        ]
        .copy()
    )

    print()

    print(
        "Final 9-metric RQ3 "
        "Friedman/Kendall summary:"
    )

    print()

    for metric in METRIC_ORDER:

        row = d[
            d[
                "metric"
            ].eq(
                metric
            )
        ].iloc[
            0
        ]

        print(
            f"{METRIC_LABELS[metric]:34s} "
            f"p="
            f"{float(row['friedman_p']):.6g}  "
            f"W="
            f"{float(row['kendalls_w']):.4f}"
        )

    print()

    print(
        "Final 9-metric "
        "prompt-component summary:"
    )

    print()

    for _, row in (
        component
        .set_index(
            "ablation"
        )
        .reindex(
            ABLATIONS
        )
        .reset_index()
        .iterrows()
    ):

        print(
            f"{ABLATION_LABELS[row['ablation']]:22s} "
            f"sig="
            f"{int(row['significant_metrics'])}  "
            f"Full higher="
            f"{int(row['full_higher_significant_metrics'])}  "
            f"Ablation higher="
            f"{int(row['ablation_higher_significant_metrics'])}  "
            f"MAMS="
            f"{float(row['mams']):.4f}"
        )

    print()


# ============================================================
# Main
# ============================================================

def main():

    set_global_style()

    primary = load_data()

    validate_final_design(
        primary
    )

    component = (
        build_final_component_summary(
            primary
        )
    )

    # --------------------------------------------------------
    # Save recalculated final summary
    #
    # Final columns:
    #   ablation
    #   significant_metrics
    #   full_higher_significant_metrics
    #   ablation_higher_significant_metrics
    #   mams
    # --------------------------------------------------------

    summary_csv = (
        BASE_DIR
        / "rq3_component_summary_9metrics.csv"
    )

    component.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Generate publication figures
    # --------------------------------------------------------

    outputs = [

        figure7_prompt_ablation_mean_differences(
            primary
        ),

        figure8_component_impact_summary(
            component
        ),
    ]

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_final_summary(
        primary,
        component,
    )

    print(
        "RQ3 publication figures "
        "generated successfully:"
    )

    print()

    for output in outputs:

        print(
            f"  {output}"
        )

    print()

    print(
        "Recalculated 9-metric summary:"
    )

    print(
        f"  {summary_csv}"
    )

    print()


if __name__ == "__main__":
    main()