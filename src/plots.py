"""Static fairness figures for the README, rendered in a light and a dark variant.

Colors follow the entity: Female is always categorical slot 1 (blue) and Male slot 2
(orange), in both figures. Both pairs were run through the palette validator against
their own chart surface (CVD delta-E ~25, normal-vision ~32, contrast >= 3:1).
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in CI or a container

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from src.data_preparation import load_raw_data  # noqa: E402
from src.fairness_audit import (  # noqa: E402
    gender_gap_by_role,
    gender_rate_bimodality,
    group_selection_rates,
)
from src.utils import logger  # noqa: E402

OUTPUT_DIR = Path("docs/images")
DPI = 200

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "female": "#2a78d6",
        "male": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "female": "#3987e5",
        "male": "#d95926",
    },
}


def _style_axes(ax: Axes, t: dict, grid_axis: Literal["x", "y"]) -> None:
    """Recessive chrome: solid hairline grid on one axis, a single baseline, no box."""
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)


def _heading(fig: Figure, t: dict, title: str, subtitle: str, top: float) -> None:
    offset = 0.3 / fig.get_figheight()  # fixed 0.3in between title and subtitle
    fig.text(0.06, top, title, color=t["ink"], fontsize=13, fontweight="bold", ha="left")
    fig.text(0.06, top - offset, subtitle, color=t["ink2"], fontsize=9.5, ha="left")


def _legend(
    ax: Axes,
    t: dict,
    loc: Literal["upper center", "lower center"],
    anchor: tuple[float, float] | None = None,
) -> None:
    legend = ax.legend(
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(t["ink2"])  # text wears ink, the marker beside it carries identity


def bimodality_figure(df: pd.DataFrame, theme: str) -> Figure:
    """Histogram of the 102 (role, gender) group rates, stacked by gender."""
    t = THEMES[theme]
    rates = group_selection_rates(df).unstack("Gender")
    summary = gender_rate_bimodality(df)

    bins = np.linspace(0, 1, 11)
    female, _ = np.histogram(rates["Female"], bins=bins)
    male, _ = np.histogram(rates["Male"], bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=DPI, facecolor=t["surface"])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.14)
    _style_axes(ax, t, grid_axis="y")

    # edgecolor = surface gives the 2px gap between adjacent bars and stacked segments
    gap: dict[str, Any] = {"edgecolor": t["surface"], "linewidth": 1.0}
    ax.bar(centers, female, width=0.085, color=t["female"], label="Female", **gap)
    ax.bar(centers, male, width=0.085, bottom=female, color=t["male"], label="Male", **gap)

    ax.set_xlim(0, 1)
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(
        "Best Match rate within a (job role, gender) group", color=t["ink2"], fontsize=9.5
    )
    ax.set_ylabel("Number of groups", color=t["ink2"], fontsize=9.5)

    peak = float((female + male).max())
    ax.set_ylim(0, peak * 1.3)
    ax.set_yticks(range(0, int(peak) + 1, 5))  # stop the grid below the direct labels
    label: dict[str, Any] = {"color": t["ink2"], "fontsize": 9.5, "ha": "center", "va": "bottom"}
    ax.text(0.1, peak * 1.04, f"{summary['n_near_zero']} groups\nat or below 20%", **label)
    ax.text(0.9, peak * 1.04, f"{summary['n_near_one']} groups\nat or above 80%", **label)
    ax.text(
        0.5,
        peak * 0.08,
        f"{summary['n_middle']} groups between 20% and 80%",
        color=t["ink2"],
        fontsize=9.5,
        ha="center",
        va="bottom",
    )
    _legend(ax, t, loc="upper center")

    _heading(
        fig,
        t,
        "Best Match is either rare or near-certain, never in between",
        f"Selection rate of all {summary['n_groups']} (job role, gender) groups",
        top=0.95,
    )
    return fig


def gap_by_role_figure(df: pd.DataFrame, theme: str) -> Figure:
    """Dumbbell of the female and male rate for every job role, sorted by gap."""
    t = THEMES[theme]
    gaps = gender_gap_by_role(df).sort_values("gap", ascending=False).reset_index(drop=True)
    n = len(gaps)
    y = np.arange(n)[::-1]  # most male-favored role on top

    fig, ax = plt.subplots(figsize=(8, 11), dpi=DPI, facecolor=t["surface"])
    fig.subplots_adjust(left=0.29, right=0.95, top=0.89, bottom=0.05)
    _style_axes(ax, t, grid_axis="x")

    ax.hlines(y, gaps["female_rate"], gaps["male_rate"], color=t["axis"], linewidth=2, zorder=2)
    marker: dict[str, Any] = {"s": 60, "edgecolors": t["surface"], "linewidths": 1.2, "zorder": 3}
    ax.scatter(gaps["female_rate"], y, color=t["female"], label="Female", **marker)
    ax.scatter(gaps["male_rate"], y, color=t["male"], label="Male", **marker)

    ax.set_yticks(y)
    ax.set_yticklabels(gaps["Job Roles"], color=t["ink2"], fontsize=8)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Best Match rate", color=t["ink2"], fontsize=9.5)

    # Selective direct labels: only the two extremes, never a number on every point.
    top, bottom = gaps.iloc[0], gaps.iloc[-1]
    note: dict[str, Any] = {"color": t["ink2"], "fontsize": 8.5, "va": "center"}
    ax.text(top["female_rate"] - 0.025, y[0], f"{top['female_rate']:.0%}", ha="right", **note)
    ax.text(top["male_rate"] + 0.025, y[0], f"{top['male_rate']:.0%}", ha="left", **note)
    ax.text(bottom["male_rate"] - 0.025, y[-1], f"{bottom['male_rate']:.0%}", ha="right", **note)
    ax.text(bottom["female_rate"] + 0.025, y[-1], f"{bottom['female_rate']:.0%}", ha="left", **note)
    _legend(ax, t, loc="lower center", anchor=(0.5, 1.0))

    _heading(
        fig,
        t,
        "Which gender gets the match depends on the job",
        f"Best Match rate by job role, {n} roles sorted from most male- to most female-favored",
        top=0.975,
    )
    return fig


def render_all(df: pd.DataFrame, out_dir: Path = OUTPUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {"fairness-bimodality": bimodality_figure, "fairness-gap-by-role": gap_by_role_figure}
    written = []
    for name, build in figures.items():
        for theme in THEMES:
            fig = build(df, theme)
            path = out_dir / f"{name}-{theme}.png"
            fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
            plt.close(fig)
            written.append(path)
            logger.info("Figure saved to: %s", path)
    return written


if __name__ == "__main__":
    render_all(load_raw_data())
