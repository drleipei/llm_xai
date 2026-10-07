from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ============================================================
# RQ2 publication figures
#
# Final design:
#   3 dimensions × 3 metrics = 9 primary metrics
#
# Place this script in:
#   results/rq2/
#
# Required CSV files in the same directory:
#   rq2_primary_table.csv
#   rq2_friedman_kendalls_w.csv
#   rq2_descriptive_by_decoding.csv
#
# Output:
#   Figure5_RQ2_Decoding_Sensitivity.png
#   Figure6_RQ2_Overall_Effect_Strength.png
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

PRIMARY_CSV = BASE_DIR / "rq2_primary_table.csv"
FRIEDMAN_CSV = BASE_DIR / "rq2_friedman_kendalls_w.csv"
DESCRIPTIVE_CSV = BASE_DIR / "rq2_descriptive_by_decoding.csv"

DPI = 400


# ============================================================
# Unified palette
# ============================================================

COLOR_BLUE = "#214F7A"
COLOR_ORANGE = "#D97706"
COLOR_TEAL = "#2A7F9E"

COLOR_GRID = "#D9DEE5"
COLOR_TEXT = "#202124"
COLOR_SPINE = "#4B5563"


# ============================================================
# Final RQ2 metric design
# 3 dimensions × 3 metrics = 9 metrics
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
    # Model-response alignment
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


ALL_METRICS = (
    STABILITY_METRICS
    + ALIGNMENT_METRICS
    + DISCRIM_METRICS
)


# ============================================================
# Plot styling by metric
# ============================================================

METRIC_COLORS = {

    # Stability
    "overlap_at_k":
        COLOR_BLUE,

    "rank_agreement_at_k":
        COLOR_TEAL,

    "direction_agreement_at_k":
        COLOR_ORANGE,

    # Model-response alignment
    "direction_consistency_rate":
        COLOR_BLUE,

    "meaningful_effect_rate":
        COLOR_TEAL,

    "mean_absolute_delta_probability":
        COLOR_ORANGE,

    # Discriminativeness
    "normalized_feature_entropy":
        COLOR_BLUE,

    "pairwise_jaccard_between_instance":
        COLOR_ORANGE,

    "instance_idf_specificity":
        COLOR_TEAL,
}


METRIC_MARKERS = {

    STABILITY_METRICS[0]:
        "o",

    STABILITY_METRICS[1]:
        "s",

    STABILITY_METRICS[2]:
        "^",

    ALIGNMENT_METRICS[0]:
        "o",

    ALIGNMENT_METRICS[1]:
        "s",

    ALIGNMENT_METRICS[2]:
        "^",

    DISCRIM_METRICS[0]:
        "o",

    DISCRIM_METRICS[1]:
        "s",

    DISCRIM_METRICS[2]:
        "^",
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
            11,

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
            9,

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

    required_files = [
        PRIMARY_CSV,
        FRIEDMAN_CSV,
        DESCRIPTIVE_CSV,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    primary = pd.read_csv(
        PRIMARY_CSV
    )

    friedman = pd.read_csv(
        FRIEDMAN_CSV
    )

    descriptive = pd.read_csv(
        DESCRIPTIVE_CSV
    )

    return (
        primary,
        friedman,
        descriptive,
    )


# ============================================================
# Utility
# ============================================================

def clean_axis(ax):

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_color(
        COLOR_SPINE
    )

    ax.spines[
        "bottom"
    ].set_color(
        COLOR_SPINE
    )

    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.55,
    )

    ax.set_axisbelow(
        True
    )


def validate_final_design(
    primary,
    descriptive,
):

    print()

    print(
        "Validating final RQ2 design:"
    )

    print(
        "3 dimensions x 3 metrics "
        "= 9 primary metrics"
    )

    print()

    descriptive_metrics = set(
        descriptive[
            "metric"
        ]
        .dropna()
        .astype(str)
    )

    primary_metrics = set(
        primary[
            "metric"
        ]
        .dropna()
        .astype(str)
    )

    for metric in ALL_METRICS:

        if (
            metric
            not in descriptive_metrics
        ):

            raise ValueError(
                "Metric missing from "
                "rq2_descriptive_by_decoding.csv: "
                f"{metric}"
            )

        if (
            metric
            not in primary_metrics
        ):

            raise ValueError(
                "Metric missing from "
                "rq2_primary_table.csv: "
                f"{metric}"
            )

        print(
            f"[OK] "
            f"{METRIC_LABELS[metric]}"
        )

    print()

    print(
        "Metrics excluded from final "
        "primary RQ2 analysis:"
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


def get_temperature_metric(
    temp_df,
    metric,
    temperatures,
):

    d = (
        temp_df[
            temp_df[
                "metric"
            ].eq(
                metric
            )
        ]
        .set_index(
            "temperature"
        )
        .reindex(
            temperatures
        )
    )

    if (
        d[
            "mean"
        ]
        .isna()
        .any()
    ):

        missing = [
            t
            for t
            in temperatures
            if t
            not in
            temp_df.loc[
                temp_df[
                    "metric"
                ].eq(
                    metric
                ),
                "temperature",
            ].tolist()
        ]

        raise ValueError(
            "Missing temperature values "
            f"for {metric}: "
            f"{missing}"
        )

    return d


def draw_temperature_panel(
    ax,
    temp_df,
    metrics,
    temperatures,
    title,
    panel_label,
    legend_loc="best",
    legend_bbox=None,
):

    for metric in metrics:

        d = get_temperature_metric(
            temp_df,
            metric,
            temperatures,
        )

        ax.plot(
            temperatures,
            d[
                "mean"
            ].astype(float),
            marker=(
                METRIC_MARKERS[
                    metric
                ]
            ),
            linewidth=2.0,
            markersize=6.5,
            color=(
                METRIC_COLORS[
                    metric
                ]
            ),
            label=(
                METRIC_LABELS[
                    metric
                ]
            ),
        )

    # Reference decoding setting
    ax.axvline(
        0.2,
        linestyle="--",
        linewidth=1.1,
        color=COLOR_SPINE,
        alpha=0.65,
    )

    ax.set_title(
        f"{panel_label} "
        f"{title}",
        fontsize=12.5,
        pad=10,
    )

    ax.set_xlabel(
        "temperature"
    )

    ax.set_xticks(
        temperatures
    )

    clean_axis(
        ax
    )

    legend_kwargs = {
        "loc":
            legend_loc,

        "fontsize":
            8.7,
    }

    if legend_bbox is not None:

        legend_kwargs[
            "bbox_to_anchor"
        ] = legend_bbox

    ax.legend(
        **legend_kwargs
    )


# ============================================================
# Figure 5
# Temperature sensitivity
#
# 3 panels:
#   (a) Repeat-Run Stability
#   (b) Model-Response Alignment
#   (c) Discriminativeness
# ============================================================

def figure5_temperature_sensitivity(
    descriptive
):

    # --------------------------------------------------------
    # temperature analysis:
    # top_p fixed at 1.0
    # --------------------------------------------------------

    temp_df = descriptive[
        (
            descriptive[
                "top_p"
            ].eq(
                1.0
            )
        )
        &
        (
            descriptive[
                "temperature"
            ].isin(
                [
                    0.0,
                    0.2,
                    0.5,
                    0.8,
                ]
            )
        )
    ].copy()

    temperatures = [
        0.0,
        0.2,
        0.5,
        0.8,
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14.2,
            5.4,
        ),
        gridspec_kw={
            "wspace":
                0.28
        },
    )

    # --------------------------------------------------------
    # Panel A — Stability
    # --------------------------------------------------------

    draw_temperature_panel(
        axes[
            0
        ],
        temp_df,
        STABILITY_METRICS,
        temperatures,
        "Repeat-Run Stability",
        "(a)",
        legend_loc="lower left",
    )

    axes[
        0
    ].set_ylabel(
        "Mean metric value"
    )

    # --------------------------------------------------------
    # Panel B — Model-response alignment
    # --------------------------------------------------------

    draw_temperature_panel(
        axes[
            1
        ],
        temp_df,
        ALIGNMENT_METRICS,
        temperatures,
        "Model-Response Alignment",
        "(b)",
        legend_loc="center right",
    )

    axes[
        1
    ].set_ylabel(
        "Mean metric value"
    )

    # --------------------------------------------------------
    # Panel C — Discriminativeness
    #
    # Legend deliberately positioned in the empty region
    # between the entropy and Jaccard curves so that it
    # does not overlap any plotted line.
    # --------------------------------------------------------

    draw_temperature_panel(
        axes[
            2
        ],
        temp_df,
        DISCRIM_METRICS,
        temperatures,
        "Discriminativeness",
        "(c)",
        legend_loc="center right",
        legend_bbox=(
            0.98,
            0.68,
        ),
    )

    axes[
        2
    ].set_ylabel(
        "Mean metric value"
    )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "RQ2: Temperature Sensitivity",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        "Dashed line marks the reference setting "
        "(T=0.2; top_p=1.0).",
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.17,
        top=0.84,
        wspace=0.28,
    )

    output = (
        BASE_DIR
        / "Figure5_RQ2_Decoding_Sensitivity.png"
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
# Figure 6
# Overall decoding-parameter effect strength
#
# Exactly:
#   9 primary metrics × 2 decoding parameters
#
# Cell values:
#   Kendall's W
#
# Stars:
#   Friedman test significance
# ============================================================

def figure6_overall_effect_strength(
    primary
):

    d = primary[
        primary[
            "metric"
        ].isin(
            ALL_METRICS
        )
    ].copy()

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

    ordered_labels = [
        METRIC_LABELS[
            m
        ]
        for m
        in ALL_METRICS
    ]

    d = (
        d
        .set_index(
            "Metric"
        )
        .reindex(
            ordered_labels
        )
        .reset_index()
    )

    required_cols = [
        "temperature_kendalls_w",
        "top_p_kendalls_w",
        "temperature_friedman_p",
        "top_p_friedman_p",
    ]

    for col in required_cols:

        if col not in d.columns:

            raise ValueError(
                "Missing required column "
                "in rq2_primary_table.csv: "
                f"{col}"
            )

    if (
        d[
            required_cols
        ]
        .isna()
        .any()
        .any()
    ):

        print()

        print(
            "Figure 6 input table:"
        )

        print(
            d[
                [
                    "Metric"
                ]
                + required_cols
            ]
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Missing values in final "
            "9-metric Figure 6 input."
        )

    values = np.column_stack(
        [
            d[
                "temperature_kendalls_w"
            ]
            .to_numpy(
                dtype=float
            ),

            d[
                "top_p_kendalls_w"
            ]
            .to_numpy(
                dtype=float
            ),
        ]
    )

    # --------------------------------------------------------
    # Heatmap palette
    # --------------------------------------------------------

    cmap = (
        LinearSegmentedColormap
        .from_list(
            "rq2_effect_strength",
            [
                "#F7F7F7",
                "#9BB9D3",
                "#2E6DAA",
            ],
            N=256,
        )
    )

    norm = Normalize(
        vmin=0,
        vmax=1,
    )

    fig, ax = plt.subplots(
        figsize=(
            8.4,
            6.8,
        )
    )

    im = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "temperature",
            "top-p",
        ]
    )

    ax.set_yticks(
        np.arange(
            len(
                d
            )
        )
    )

    ax.set_yticklabels(
        d[
            "Metric"
        ]
    )

    # --------------------------------------------------------
    # Cell borders
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            2,
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(
                d
            ),
            1,
        ),
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

    # --------------------------------------------------------
    # Cell labels + Friedman significance
    # --------------------------------------------------------

    for i in range(
        len(
            d
        )
    ):

        temp_p = float(
            d.loc[
                i,
                "temperature_friedman_p",
            ]
        )

        top_p_p = float(
            d.loc[
                i,
                "top_p_friedman_p",
            ]
        )

        ps = [
            temp_p,
            top_p_p,
        ]

        for j in range(
            2
        ):

            value = values[
                i,
                j,
            ]

            p = ps[
                j
            ]

            stars = (
                "***"
                if p < 0.001
                else "**"
                if p < 0.01
                else "*"
                if p < 0.05
                else ""
            )

            text_color = (
                "white"
                if value > 0.60
                else COLOR_TEXT
            )

            ax.text(
                j,
                i,
                f"{value:.3f}"
                f"{stars}",
                ha="center",
                va="center",
                fontsize=9.4,
                fontweight=(
                    "bold"
                    if stars
                    else "normal"
                ),
                color=text_color,
            )

    # --------------------------------------------------------
    # Dimension separators
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
    # Title / colorbar
    # --------------------------------------------------------

    ax.set_title(
        "RQ2: Overall Decoding-Parameter Effect Strength",
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
        left=0.36,
        right=0.90,
        bottom=0.09,
        top=0.91,
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

    plt.close(
        fig
    )

    return output


# ============================================================
# Main
# ============================================================

def main():

    set_global_style()

    (
        primary,
        friedman,
        descriptive,
    ) = load_data()

    validate_final_design(
        primary,
        descriptive,
    )

    outputs = [

        figure5_temperature_sensitivity(
            descriptive
        ),

        figure6_overall_effect_strength(
            primary
        ),
    ]

    print()

    print(
        "RQ2 publication figures "
        "generated successfully:"
    )

    print()

    for output in outputs:

        print(
            f"  {output}"
        )

    print()


if __name__ == "__main__":
    main()
