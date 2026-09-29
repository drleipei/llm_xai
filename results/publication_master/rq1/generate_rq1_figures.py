from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# RQ1 publication figures
# ============================================================
#
# Place this script in:
#
#   results/publication_master/rq1/
#
# Required CSV files:
#
#   table_rq1_compact.csv
#   table_rq1_primary_overall.csv
#
# Output:
#
#   Figure2_RQ1_Repeat_Run_Stability.png
#   Figure3_RQ1_Stability_Discriminativeness_Tradeoff.png
#   Figure4_RQ1_Paired_Traditional_vs_LLM_Effects.png
#
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

COMPACT_CSV = BASE_DIR / "table_rq1_compact.csv"
PRIMARY_CSV = BASE_DIR / "table_rq1_primary_overall.csv"

DPI = 400


# ============================================================
# Unified publication palette
# ============================================================

COLOR_TRAD = "#214F7A"          # deep navy blue
COLOR_LLM = "#D97706"           # restrained amber/orange
COLOR_CONNECTOR = "#9CA3AF"     # neutral grey
COLOR_GRID = "#D9DEE5"
COLOR_TEXT = "#202124"
COLOR_SPINE = "#4B5563"

COLOR_LIME_HIGHLIGHT = "#F7F4EC"

HEAT_NEG = "#2E6DAA"
HEAT_ZERO = "#F7F7F7"
HEAT_POS = "#C46A23"


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
}


STABILITY_METRICS = [
    "Overlap@K",
    "Rank Agreement@K",
    "Direction Agreement@K",
    "Score Stability",
]


DIRECTIONAL_METRICS = [
    "Overlap@K",
    "Rank Agreement@K",
    "Direction Agreement@K",
    "Score Stability",
    "Direction Consistency",
    "Meaningful Effect",
    "Mean |Δp|",
    "Within-instance Jaccard",
    "Between-instance Jaccard",
    "Instance-IDF",
    "Separability Gap",
]


METRIC_LABELS = {
    "overlap_at_k": "Overlap@K",
    "rank_agreement_at_k": "Rank Agreement@K",
    "direction_agreement_at_k": "Direction Agreement@K",
    "score_stability": "Score Stability",
    "direction_consistency_rate": "Direction Consistency",
    "meaningful_effect_rate": "Meaningful Effect",
    "mean_absolute_delta_probability": "Mean |Δp|",
    "pairwise_jaccard_within_instance": "Within-instance Jaccard",
    "pairwise_jaccard_between_instance": "Between-instance Jaccard",
    "instance_idf_specificity": "Instance-IDF",
    "separability_gap": "Separability Gap",
}


# ============================================================
# Global plotting style
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

        "legend.fontsize": 11,
        "legend.frameon": False,

        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "savefig.facecolor": "white",

        "savefig.bbox": "tight",
    })


# ============================================================
# Data loading
# ============================================================

def load_data():

    if not COMPACT_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {COMPACT_CSV}"
        )

    if not PRIMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {PRIMARY_CSV}"
        )

    compact = pd.read_csv(COMPACT_CSV)
    primary = pd.read_csv(PRIMARY_CSV)

    compact_required = {
        "Metric",
        "Method",
        "Traditional Mean",
        "Qwen3.6 Mean",
    }

    missing = compact_required - set(compact.columns)

    if missing:
        raise ValueError(
            "table_rq1_compact.csv missing columns: "
            f"{sorted(missing)}"
        )

    primary_required = {
        "metric",
        "traditional_method",
        "metric_direction",
        "improvement_signed",
    }

    missing = primary_required - set(primary.columns)

    if missing:
        raise ValueError(
            "table_rq1_primary_overall.csv missing columns: "
            f"{sorted(missing)}"
        )

    return compact, primary


# ============================================================
# Helper functions
# ============================================================

def clean_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.55,
    )

    ax.set_axisbelow(True)


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
# Figure 2
# Repeat-run stability
# ============================================================

def figure2_repeat_run_stability(compact):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.8),
        sharey=True,
    )

    axes = axes.flatten()

    x = np.arange(len(METHOD_ORDER))

    width = 0.34

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, metric) in enumerate(
        zip(axes, STABILITY_METRICS)
    ):

        d = (
            compact[
                compact["Metric"].eq(metric)
            ]
            .set_index("Method")
            .reindex(METHOD_ORDER)
        )

        trad = (
            d["Traditional Mean"]
            .astype(float)
            .to_numpy()
        )

        llm = (
            d["Qwen3.6 Mean"]
            .astype(float)
            .to_numpy()
        )

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
            metric,
            pad=8,
            fontsize=13,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            METHOD_ORDER,
            rotation=18,
            ha="right",
        )

        ax.set_ylim(
            0.72,
            1.02,
        )

        ax.set_yticks(
            np.arange(
                0.75,
                1.01,
                0.05,
            )
        )

        clean_axis(ax)

        ax.text(
            -0.14,
            1.07,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
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
        y=0.985,
    )

    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        columnspacing=2.0,
    )

    fig.tight_layout(
        rect=[
            0.03,
            0.04,
            0.99,
            0.89,
        ]
    )

    output = (
        BASE_DIR /
        "Figure2_RQ1_Repeat_Run_Stability.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
    )

    plt.close(fig)

    return output


# ============================================================
# Figure 3
# NEW DESIGN:
# Two-panel paired dumbbell chart
# ============================================================

def figure3_stability_discriminativeness_tradeoff(compact):
    """
    Figure 3:
    Stability–Discriminativeness Trade-off

    Panel (a): mean composite of four repeat-run stability metrics
    Panel (b): Separability Gap

    Traditional = blue circle
    LLM-informed = orange square
    Delta = LLM-informed - Traditional
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ========================================================
    # Prepare data
    # ========================================================

    records = []

    for method in METHOD_ORDER:

        method_rows = compact[
            compact["Method"].eq(method)
        ]

        stability_rows = method_rows[
            method_rows["Metric"].isin(STABILITY_METRICS)
        ]

        if len(stability_rows) != 4:
            raise ValueError(
                f"Expected 4 stability metrics for {method}, "
                f"but found {len(stability_rows)}."
            )

        stability_trad = (
            stability_rows["Traditional Mean"]
            .astype(float)
            .mean()
        )

        stability_llm = (
            stability_rows["Qwen3.6 Mean"]
            .astype(float)
            .mean()
        )

        sep_row = method_rows[
            method_rows["Metric"].eq("Separability Gap")
        ]

        if len(sep_row) != 1:
            raise ValueError(
                f"Expected one Separability Gap row for {method}."
            )

        sep_trad = float(
            sep_row["Traditional Mean"].iloc[0]
        )

        sep_llm = float(
            sep_row["Qwen3.6 Mean"].iloc[0]
        )

        records.append(
            {
                "Method": method,
                "Stability Traditional": stability_trad,
                "Stability LLM": stability_llm,
                "Separability Traditional": sep_trad,
                "Separability LLM": sep_llm,
            }
        )

    df = pd.DataFrame(records)

    # Counterfactual at top
    y = np.arange(len(METHOD_ORDER))[::-1]

    # ========================================================
    # Figure
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.8, 6.8)
    )

    # IMPORTANT:
    # Reserve a large dedicated title/legend band.
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.12,
        top=0.73,
        wspace=0.23,
    )

    # ========================================================
    # LIME highlight
    # ========================================================

    lime_idx = METHOD_ORDER.index("LIME")
    lime_y = y[lime_idx]

    for ax in (ax1, ax2):

        ax.axhspan(
            lime_y - 0.40,
            lime_y + 0.40,
            color=COLOR_LIME_HIGHLIGHT,
            alpha=0.35,
            zorder=0,
        )

    # ========================================================
    # Panel A — Stability
    # ========================================================

    stability_trad = (
        df["Stability Traditional"]
        .to_numpy(dtype=float)
    )

    stability_llm = (
        df["Stability LLM"]
        .to_numpy(dtype=float)
    )

    for i in range(len(df)):

        # connector
        ax1.plot(
            [stability_llm[i], stability_trad[i]],
            [y[i], y[i]],
            color=COLOR_CONNECTOR,
            linewidth=2.0,
            zorder=1,
        )

        delta = (
            stability_llm[i]
            - stability_trad[i]
        )

        mid_x = (
            stability_llm[i]
            + stability_trad[i]
        ) / 2

        # Put delta close to the connector.
        # Top row is placed BELOW the line to avoid title collision.
        if i == 0:
            delta_y = y[i] - 0.18
            va = "top"
        else:
            delta_y = y[i] + 0.15
            va = "bottom"

        ax1.text(
            mid_x,
            delta_y,
            rf"$\Delta$={delta:+.3f}",
            ha="center",
            va=va,
            fontsize=9.5,
            color=COLOR_TEXT,
            fontweight=(
                "bold"
                if METHOD_ORDER[i] == "LIME"
                else "normal"
            ),
            zorder=4,
        )

    # points
    ax1.scatter(
        stability_trad,
        y,
        s=135,
        marker="o",
        color=COLOR_TRAD,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    ax1.scatter(
        stability_llm,
        y,
        s=135,
        marker="s",
        color=COLOR_LLM,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    ax1.set_title(
        "(a) Repeat-run stability",
        fontsize=13,
        fontweight="bold",
        pad=7,
    )

    ax1.set_xlabel(
        "Stability composite",
        fontsize=12,
        labelpad=8,
    )

    ax1.set_xlim(
        0.83,
        1.025,
    )

    ax1.set_yticks(y)

    ax1.set_yticklabels(
        METHOD_ORDER,
        fontsize=11,
    )

    ax1.tick_params(
        axis="x",
        labelsize=10,
    )

    ax1.grid(
        axis="x",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.55,
    )

    ax1.grid(
        axis="y",
        visible=False,
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ========================================================
    # Panel B — Separability
    # ========================================================

    sep_trad = (
        df["Separability Traditional"]
        .to_numpy(dtype=float)
    )

    sep_llm = (
        df["Separability LLM"]
        .to_numpy(dtype=float)
    )

    for i in range(len(df)):

        ax2.plot(
            [sep_trad[i], sep_llm[i]],
            [y[i], y[i]],
            color=COLOR_CONNECTOR,
            linewidth=2.0,
            zorder=1,
        )

        delta = (
            sep_llm[i]
            - sep_trad[i]
        )

        mid_x = (
            sep_trad[i]
            + sep_llm[i]
        ) / 2

        # Again: keep top-row annotation away from title.
        if i == 0:
            delta_y = y[i] - 0.18
            va = "top"
        else:
            delta_y = y[i] + 0.15
            va = "bottom"

        ax2.text(
            mid_x,
            delta_y,
            rf"$\Delta$={delta:+.3f}",
            ha="center",
            va=va,
            fontsize=9.5,
            color=COLOR_TEXT,
            fontweight=(
                "bold"
                if METHOD_ORDER[i] == "LIME"
                else "normal"
            ),
            zorder=4,
        )

    ax2.scatter(
        sep_trad,
        y,
        s=135,
        marker="o",
        color=COLOR_TRAD,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    ax2.scatter(
        sep_llm,
        y,
        s=135,
        marker="s",
        color=COLOR_LLM,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    ax2.set_title(
        "(b) Instance discriminativeness",
        fontsize=13,
        fontweight="bold",
        pad=7,
    )

    ax2.set_xlabel(
        "Separability Gap",
        fontsize=12,
        labelpad=8,
    )

    ax2.set_xlim(
        0.0,
        0.74,
    )

    ax2.set_yticks(y)

    # Do not repeat method names
    ax2.set_yticklabels([])

    ax2.tick_params(
        axis="y",
        length=0,
    )

    ax2.tick_params(
        axis="x",
        labelsize=10,
    )

    ax2.grid(
        axis="x",
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.55,
    )

    ax2.grid(
        axis="y",
        visible=False,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ========================================================
    # Legend
    # ========================================================

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLOR_TRAD,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=10,
            label="Traditional",
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=COLOR_LLM,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=10,
            label="LLM-informed",
        ),

    ]

    # ========================================================
    # Dedicated title hierarchy
    # ========================================================

    # Main title — highest row
    fig.suptitle(
        "RQ1: Stability–Discriminativeness Trade-off",
        fontsize=17,
        fontweight="bold",
        x=0.5,
        y=0.975,
    )

    # Legend — dedicated second row
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handletextpad=0.6,
        fontsize=11,
    )

    # IMPORTANT:
    # Do NOT call tight_layout() here.

    # ========================================================
    # Save
    # ========================================================

    output = (
        BASE_DIR
        / "Figure3_RQ1_Stability_Discriminativeness_Tradeoff.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return output


# ============================================================
# Figure 4
# Direction-corrected paired effect heatmap
# ============================================================

def figure4_paired_effects(primary):

    d = primary.copy()

    d["Metric"] = (
        d["metric"]
        .map(METRIC_LABELS)
    )

    d["Method"] = (
        d["traditional_method"]
        .map(METHOD_LABELS)
    )


    d = d.loc[

        d["Metric"].isin(
            DIRECTIONAL_METRICS
        )

        & d["Method"].isin(
            METHOD_ORDER
        )

        & d[
            "metric_direction"
        ].ne("descriptive")

    ].copy()


    matrix = (

        d.pivot(
            index="Metric",
            columns="Method",
            values="improvement_signed",
        )

        .reindex(
            index=DIRECTIONAL_METRICS,
            columns=METHOD_ORDER,
        )

    )


    if matrix.isna().any().any():

        raise ValueError(
            "Missing values detected in "
            "paired-effect heatmap matrix."
        )


    p_col = find_p_column(d)

    p_matrix = None

    if p_col is not None:

        p_matrix = (

            d.pivot(
                index="Metric",
                columns="Method",
                values=p_col,
            )

            .reindex(
                index=DIRECTIONAL_METRICS,
                columns=METHOD_ORDER,
            )

        )


    values = matrix.to_numpy(
        dtype=float
    )


    vmax = np.nanmax(
        np.abs(values)
    )


    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )


    cmap = LinearSegmentedColormap.from_list(
        "rq1_effects",
        [
            HEAT_NEG,
            HEAT_ZERO,
            HEAT_POS,
        ],
        N=256,
    )


    fig, ax = plt.subplots(
        figsize=(9.5, 7.8)
    )


    im = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(METHOD_ORDER)
        )
    )


    ax.set_xticklabels(
        METHOD_ORDER,
        rotation=18,
        ha="right",
    )


    ax.set_yticks(
        np.arange(
            len(DIRECTIONAL_METRICS)
        )
    )


    ax.set_yticklabels(
        DIRECTIONAL_METRICS,
    )


    # Cell borders
    ax.set_xticks(
        np.arange(
            -0.5,
            len(METHOD_ORDER),
            1,
        ),
        minor=True,
    )


    ax.set_yticks(
        np.arange(
            -0.5,
            len(DIRECTIONAL_METRICS),
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


    # Cell values
    for i in range(
        values.shape[0]
    ):

        for j in range(
            values.shape[1]
        ):

            value = values[i, j]

            stars = ""

            if p_matrix is not None:

                p = p_matrix.iloc[
                    i,
                    j,
                ]

                stars = p_to_star(p)


            label = (
                f"{value:+.3f}"
                f"{stars}"
            )


            # contrast-aware text
            if abs(value) >= 0.18:
                text_color = "white"
            else:
                text_color = "#111111"


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


    ax.set_title(
        "RQ1: Direction-Corrected Paired Effects",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )


    cbar.set_label(
        "Direction-corrected effect\n"
        "(+ LLM-informed, − traditional)",
        fontsize=11,
    )


    fig.tight_layout(
        rect=[
            0.02,
            0.02,
            0.98,
            0.98,
        ]
    )


    output = (
        BASE_DIR /
        "Figure4_RQ1_Paired_Traditional_vs_LLM_Effects.png"
    )


    fig.savefig(
        output,
        dpi=DPI,
    )


    plt.close(fig)

    return output


# ============================================================
# Main
# ============================================================

def main():

    set_global_style()

    compact, primary = load_data()

    outputs = [

        figure2_repeat_run_stability(
            compact
        ),

        figure3_stability_discriminativeness_tradeoff(
            compact
        ),

        figure4_paired_effects(
            primary
        ),

    ]


    print()
    print("RQ1 publication figures generated successfully:")
    print()

    for output in outputs:
        print(f"  {output}")

    print()


if __name__ == "__main__":
    main()