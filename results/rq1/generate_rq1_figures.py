from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COMPACT_CSV = BASE_DIR / "table_rq1_compact.csv"
PRIMARY_CSV = BASE_DIR / "table_rq1_primary_overall.csv"

TRAD_DISCRIM_CSV = (
    BASE_DIR
    / "rq1_raw"
    / "traditional"
    / "metrics"
    / "discriminativeness.csv"
)

LLM_DISCRIM_CSV = (
    BASE_DIR
    / "rq1_raw"
    / "qwen36"
    / "metrics"
    / "discriminativeness.csv"
)

DPI = 400


# ============================================================
# Publication palette
# ============================================================

COLOR_TRAD = "#214F7A"
COLOR_LLM = "#D97706"

COLOR_GRID = "#D9DEE5"
COLOR_TEXT = "#202124"
COLOR_SPINE = "#4B5563"

HEAT_NEG = "#2E6DAA"
HEAT_ZERO = "#F7F7F7"
HEAT_POS = "#C46A23"


# ============================================================
# Methods
# ============================================================

METHOD_ORDER = [
    "Counterfactual",
    "KernelSHAP",
    "LIME",
    "LOFO",
]

METHOD_LABELS = {
    "counterfactual": "Counterfactual",
    "kernelshap": "KernelSHAP",
    "lime": "LIME",
    "lofo": "LOFO",
    "llm_counterfactual": "Counterfactual",
    "llm_kernelshap": "KernelSHAP",
    "llm_lime": "LIME",
    "llm_lofo": "LOFO",
}


# ============================================================
# Final 3 × 3 RQ1 metric design
# ============================================================

STABILITY_METRICS = [
    "Overlap@K",
    "Rank Agreement@K",
    "Direction Agreement@K",
]

ALIGNMENT_METRICS = [
    "Direction Consistency Rate",
    "Meaningful-Effect Rate",
    "Mean |Δp|",
]

DISCRIM_METRICS = [
    "Normalized Feature Entropy",
    "Between-instance Jaccard",
    "Instance-IDF Specificity",
]


# ============================================================
# Eight metrics with instance-level paired comparison
#
# Normalized Feature Entropy is directional
# under the operationalized construct used in this study,
# but it is computed at dataset-category-method level,
# not at the 160-instance paired level.
# ============================================================

PAIRED_METRICS = [
    "Overlap@K",
    "Rank Agreement@K",
    "Direction Agreement@K",
    "Direction Consistency Rate",
    "Meaningful-Effect Rate",
    "Mean |Δp|",
    "Between-instance Jaccard",
    "Instance-IDF Specificity",
]


# ============================================================
# Metric aliases in compact table
#
# Canonical names are the final manuscript names.
# Older labels are retained only for backward compatibility
# with historical CSV files.
# ============================================================

METRIC_ALIASES = {

    "Overlap@K": [
        "Overlap@K",
    ],

    "Rank Agreement@K": [
        "Rank Agreement@K",
    ],

    "Direction Agreement@K": [
        "Direction Agreement@K",
    ],

    "Direction Consistency Rate": [
        "Direction Consistency Rate",
        "Direction Consistency",
    ],

    "Meaningful-Effect Rate": [
        "Meaningful-Effect Rate",
        "Meaningful Effect Rate",
        "Meaningful Effect",
    ],

    "Mean |Δp|": [
        "Mean |Δp|",
        "Mean |Delta p|",
    ],

    "Between-instance Jaccard": [
        "Between-instance Jaccard",
    ],

    "Instance-IDF Specificity": [
        "Instance-IDF Specificity",
        "Instance-IDF",
    ],
}


# ============================================================
# Mapping used in primary paired-comparison table
# ============================================================

PRIMARY_METRIC_LABELS = {

    "overlap_at_k":
        "Overlap@K",

    "rank_agreement_at_k":
        "Rank Agreement@K",

    "direction_agreement_at_k":
        "Direction Agreement@K",

    "direction_consistency_rate":
        "Direction Consistency Rate",

    "meaningful_effect_rate":
        "Meaningful-Effect Rate",

    "mean_absolute_delta_probability":
        "Mean |Δp|",

    "pairwise_jaccard_between_instance":
        "Between-instance Jaccard",

    "instance_idf_specificity":
        "Instance-IDF Specificity",
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

    required = [
        COMPACT_CSV,
        PRIMARY_CSV,
        TRAD_DISCRIM_CSV,
        LLM_DISCRIM_CSV,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    compact = pd.read_csv(
        COMPACT_CSV
    )

    primary = pd.read_csv(
        PRIMARY_CSV
    )

    trad_discrim = pd.read_csv(
        TRAD_DISCRIM_CSV
    )

    llm_discrim = pd.read_csv(
        LLM_DISCRIM_CSV
    )

    return (
        compact,
        primary,
        trad_discrim,
        llm_discrim,
    )


# ============================================================
# Helpers
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


def resolve_metric_rows(
    compact,
    canonical_metric,
):

    aliases = (
        METRIC_ALIASES[
            canonical_metric
        ]
    )

    rows = compact[
        compact[
            "Metric"
        ].isin(
            aliases
        )
    ].copy()

    if rows.empty:

        available = sorted(
            compact[
                "Metric"
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            f"\nCannot find metric: "
            f"{canonical_metric}\n"
            f"Accepted aliases: "
            f"{aliases}\n"
            f"Available metrics: "
            f"{available}\n"
        )

    return rows


def compact_metric_values(
    compact,
    metric,
):

    rows = resolve_metric_rows(
        compact,
        metric,
    )

    d = (
        rows
        .drop_duplicates(
            subset=[
                "Method"
            ]
        )
        .set_index(
            "Method"
        )
        .reindex(
            METHOD_ORDER
        )
    )

    cols = [
        "Traditional Mean",
        "Qwen3.6 Mean",
    ]

    if (
        d[cols]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            f"Missing values for metric: "
            f"{metric}"
        )

    trad = (
        d[
            "Traditional Mean"
        ]
        .astype(float)
        .to_numpy()
    )

    llm = (
        d[
            "Qwen3.6 Mean"
        ]
        .astype(float)
        .to_numpy()
    )

    return trad, llm


def weighted_method_mean(
    df,
    value_col,
):

    d = df.copy()

    d[
        "Method"
    ] = (
        d[
            "method"
        ]
        .map(
            METHOD_LABELS
        )
    )

    d = d[
        d[
            "Method"
        ].isin(
            METHOD_ORDER
        )
    ].copy()

    result = {}

    for method in METHOD_ORDER:

        g = d[
            d[
                "Method"
            ].eq(
                method
            )
        ].copy()

        if g.empty:

            raise ValueError(
                f"No data found for "
                f"{method}"
            )

        values = (
            g[
                value_col
            ]
            .astype(float)
            .to_numpy()
        )

        if (
            "n_explanations"
            in g.columns
            and
            g[
                "n_explanations"
            ]
            .notna()
            .all()
        ):

            weights = (
                g[
                    "n_explanations"
                ]
                .astype(float)
                .to_numpy()
            )

            mean_value = np.average(
                values,
                weights=weights,
            )

        else:

            mean_value = np.mean(
                values
            )

        result[
            method
        ] = mean_value

    return np.array(
        [
            result[m]
            for m in METHOD_ORDER
        ],
        dtype=float,
    )


def entropy_values(
    trad_discrim,
    llm_discrim,
):

    trad = weighted_method_mean(
        trad_discrim,
        "normalized_feature_entropy",
    )

    llm = weighted_method_mean(
        llm_discrim,
        "normalized_feature_entropy",
    )

    return trad, llm


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


def find_p_column(df):

    candidates = [
        "holm_p_corrected",
        "holm_p",
        "p_holm",
        "adjusted_p",
        "holm_adjusted_p",
        "p_value_corrected",
        "corrected_p",
    ]

    for col in candidates:

        if col in df.columns:

            return col

    return None


# ============================================================
# Generic grouped-bar drawing
# ============================================================

def draw_grouped_bars(
    ax,
    trad,
    llm,
    title,
    panel_label,
    ylim=None,
):

    x = np.arange(
        len(
            METHOD_ORDER
        )
    )

    width = 0.34

    ax.bar(
        x - width / 2,
        trad,
        width,
        color=COLOR_TRAD,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.bar(
        x + width / 2,
        llm,
        width,
        color=COLOR_LLM,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=8,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        METHOD_ORDER,
        rotation=22,
        ha="right",
    )

    if ylim is not None:

        ax.set_ylim(
            *ylim
        )

    clean_axis(
        ax
    )

    ax.text(
        -0.12,
        1.07,
        panel_label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="top",
    )


# ============================================================
# Figure 2
# Repeat-run stability
# ============================================================

def figure2_repeat_run_stability(
    compact
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.7),
        sharey=True,
    )

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
    ]

    for (
        ax,
        metric,
        panel,
    ) in zip(
        axes,
        STABILITY_METRICS,
        panel_labels,
    ):

        trad, llm = (
            compact_metric_values(
                compact,
                metric,
            )
        )

        draw_grouped_bars(
            ax,
            trad,
            llm,
            metric,
            panel,
            ylim=(
                0.80,
                1.015,
            ),
        )

    legend_items = [

        Line2D(
            [0],
            [0],
            color=COLOR_TRAD,
            lw=8,
            label="Traditional",
        ),

        Line2D(
            [0],
            [0],
            color=COLOR_LLM,
            lw=8,
            label="LLM-informed",
        ),
    ]

    fig.suptitle(
        "RQ1: Repeat-Run Stability",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.90,
        ),
        ncol=2,
        columnspacing=2.0,
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.03,
            0.99,
            0.82,
        ]
    )

    output = (
        BASE_DIR
        / "Figure2_RQ1_Repeat_Run_Stability.png"
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
# Figure 3
# Model-response alignment + discriminativeness
# ============================================================

def figure3_alignment_and_discriminativeness(
    compact,
    trad_discrim,
    llm_discrim,
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.6, 8.1),
    )

    panels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
    ]

    # --------------------------------------------------------
    # Row 1:
    # Model-response alignment
    # --------------------------------------------------------

    alignment_limits = {

        "Direction Consistency Rate":
            (
                0.55,
                1.00,
            ),

        "Meaningful-Effect Rate":
            (
                0.60,
                0.90,
            ),

        "Mean |Δp|":
            (
                0.08,
                0.14,
            ),
    }

    for j, metric in enumerate(
        ALIGNMENT_METRICS
    ):

        trad, llm = (
            compact_metric_values(
                compact,
                metric,
            )
        )

        draw_grouped_bars(
            axes[
                0,
                j,
            ],
            trad,
            llm,
            metric,
            panels[
                j
            ],
            ylim=(
                alignment_limits[
                    metric
                ]
            ),
        )

    # --------------------------------------------------------
    # Row 2A:
    # Normalized Feature Entropy
    # --------------------------------------------------------

    (
        entropy_trad,
        entropy_llm,
    ) = entropy_values(
        trad_discrim,
        llm_discrim,
    )

    draw_grouped_bars(
        axes[
            1,
            0,
        ],
        entropy_trad,
        entropy_llm,
        "Normalized Feature Entropy",
        panels[
            3
        ],
        ylim=(
            0.0,
            1.0,
        ),
    )

    # --------------------------------------------------------
    # Row 2B:
    # Between-instance Jaccard
    # --------------------------------------------------------

    trad, llm = (
        compact_metric_values(
            compact,
            "Between-instance Jaccard",
        )
    )

    draw_grouped_bars(
        axes[
            1,
            1,
        ],
        trad,
        llm,
        "Between-instance Jaccard",
        panels[
            4
        ],
        ylim=(
            0.0,
            0.90,
        ),
    )

    # --------------------------------------------------------
    # Row 2C:
    # Instance-IDF Specificity
    # --------------------------------------------------------

    trad, llm = (
        compact_metric_values(
            compact,
            "Instance-IDF Specificity",
        )
    )

    draw_grouped_bars(
        axes[
            1,
            2,
        ],
        trad,
        llm,
        "Instance-IDF Specificity",
        panels[
            5
        ],
        ylim=(
            0.0,
            0.35,
        ),
    )

    legend_items = [

        Line2D(
            [0],
            [0],
            color=COLOR_TRAD,
            lw=8,
            label="Traditional",
        ),

        Line2D(
            [0],
            [0],
            color=COLOR_LLM,
            lw=8,
            label="LLM-informed",
        ),
    ]

    fig.suptitle(
        "RQ1: Model-Response Alignment and Discriminativeness",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.935,
        ),
        ncol=2,
        columnspacing=2.0,
    )

    fig.tight_layout(
        rect=[
            0.03,
            0.03,
            0.99,
            0.89,
        ],
        h_pad=2.5,
        w_pad=1.3,
    )

    output = (
        BASE_DIR
        / "Figure3_RQ1_Alignment_and_Discriminativeness.png"
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
# Figure 4
# Direction-corrected mean differences
#
# IMPORTANT:
#
# The plotted values come from "improvement_signed".
# They represent direction-corrected paired mean differences,
# NOT rank-biserial correlations or standardized effect sizes.
#
# Positive:
#   LLM-informed version is numerically more favorable
#   under the predefined metric direction.
#
# Negative:
#   Traditional version is numerically more favorable.
#
# Between-instance Jaccard is direction-reversed because
# lower values represent less cross-instance overlap.
# ============================================================

def figure4_direction_corrected_mean_differences(
    primary
):

    d = primary.copy()

    d[
        "Metric"
    ] = (
        d[
            "metric"
        ]
        .map(
            PRIMARY_METRIC_LABELS
        )
    )

    d[
        "Method"
    ] = (
        d[
            "traditional_method"
        ]
        .map(
            METHOD_LABELS
        )
    )

    d = d.loc[
        (
            d[
                "Metric"
            ].isin(
                PAIRED_METRICS
            )
        )
        &
        (
            d[
                "Method"
            ].isin(
                METHOD_ORDER
            )
        )
    ].copy()

    matrix = (
        d.pivot(
            index="Metric",
            columns="Method",
            values="improvement_signed",
        )
        .reindex(
            index=PAIRED_METRICS,
            columns=METHOD_ORDER,
        )
    )

    if (
        matrix
        .isna()
        .any()
        .any()
    ):

        print(
            "\nFigure 4 "
            "direction-corrected "
            "mean-difference matrix:"
        )

        print(
            matrix
        )

        raise ValueError(
            "\nMissing direction-corrected "
            "mean-difference values.\n"
            "Check "
            "table_rq1_primary_overall.csv."
        )

    p_col = find_p_column(
        d
    )

    p_matrix = None

    if p_col is not None:

        p_matrix = (
            d.pivot(
                index="Metric",
                columns="Method",
                values=p_col,
            )
            .reindex(
                index=PAIRED_METRICS,
                columns=METHOD_ORDER,
            )
        )

    values = matrix.to_numpy(
        dtype=float
    )

    vmax = np.nanmax(
        np.abs(
            values
        )
    )

    if vmax == 0:

        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    cmap = (
        LinearSegmentedColormap
        .from_list(
            "rq1_mean_differences",
            [
                HEAT_NEG,
                HEAT_ZERO,
                HEAT_POS,
            ],
            N=256,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9.6,
            6.3,
        )
    )

    im = ax.imshow(
        values,
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
                METHOD_ORDER
            )
        )
    )

    ax.set_xticklabels(
        METHOD_ORDER,
        rotation=18,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(
                PAIRED_METRICS
            )
        )
    )

    ax.set_yticklabels(
        PAIRED_METRICS
    )

    # --------------------------------------------------------
    # Cell borders
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(
                METHOD_ORDER
            ),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(
                PAIRED_METRICS
            ),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Cell labels
    # --------------------------------------------------------

    for i in range(
        values.shape[
            0
        ]
    ):

        for j in range(
            values.shape[
                1
            ]
        ):

            value = values[
                i,
                j,
            ]

            stars = ""

            if p_matrix is not None:

                p = p_matrix.iloc[
                    i,
                    j,
                ]

                stars = (
                    p_to_star(
                        p
                    )
                )

            label = (
                f"{value:+.3f}"
                f"{stars}"
            )

            text_color = (
                "white"
                if (
                    abs(
                        value
                    )
                    >= 0.18
                )
                else "#111111"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
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
        "RQ1: Direction-Corrected Mean Differences",
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
        "Direction-corrected mean difference\n"
        "(+ LLM-informed, − traditional)",
        fontsize=11,
    )

    fig.tight_layout()

    output = (
        BASE_DIR
        / "Figure4_RQ1_Direction_Corrected_Mean_Differences.png"
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
# Validation
# ============================================================

def validate_final_design(
    compact,
    trad_discrim,
    llm_discrim,
):

    print()

    print(
        "Validating final RQ1 design:"
    )

    print(
        "3 dimensions x 3 metrics "
        "= 9 primary metrics"
    )

    print()

    # --------------------------------------------------------
    # Stability + model-response alignment
    # --------------------------------------------------------

    for metric in (
        STABILITY_METRICS
        + ALIGNMENT_METRICS
    ):

        rows = resolve_metric_rows(
            compact,
            metric,
        )

        print(
            f"[OK] {metric}: "
            f"{len(rows)} row(s)"
        )

    # --------------------------------------------------------
    # Normalized Feature Entropy
    # --------------------------------------------------------

    for name, df in [

        (
            "Traditional",
            trad_discrim,
        ),

        (
            "LLM-informed",
            llm_discrim,
        ),
    ]:

        if (
            "normalized_feature_entropy"
            not in df.columns
        ):

            raise ValueError(
                f"{name} discriminativeness "
                "file lacks "
                "'normalized_feature_entropy'."
            )

    print(
        "[OK] Normalized Feature Entropy: "
        "loaded from raw "
        "discriminativeness metrics"
    )

    # --------------------------------------------------------
    # Remaining discriminativeness metrics
    # --------------------------------------------------------

    for metric in [
        "Between-instance Jaccard",
        "Instance-IDF Specificity",
    ]:

        rows = resolve_metric_rows(
            compact,
            metric,
        )

        print(
            f"[OK] {metric}: "
            f"{len(rows)} row(s)"
        )

    print()

    print(
        "Final metric directions:"
    )

    print()

    print(
        "  Higher indicates stronger "
        "value under the predefined construct:"
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
        "cross-instance overlap:"
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
# Main
# ============================================================

def main():

    set_global_style()

    (
        compact,
        primary,
        trad_discrim,
        llm_discrim,
    ) = load_data()

    validate_final_design(
        compact,
        trad_discrim,
        llm_discrim,
    )

    outputs = [

        figure2_repeat_run_stability(
            compact
        ),

        figure3_alignment_and_discriminativeness(
            compact,
            trad_discrim,
            llm_discrim,
        ),

        figure4_direction_corrected_mean_differences(
            primary
        ),
    ]

    print()

    print(
        "RQ1 publication figures "
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