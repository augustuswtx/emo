#!/usr/bin/env python3
"""Generate F1 method overview in English and Chinese using matplotlib only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "figures"
DATA_FILE = FIG_DIR / "data" / "f1_method_overview.json"

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 112

COLORS = {
    "ink": "#24323D",
    "muted": "#60717E",
    "line": "#536874",
    "main_fill": "#EAF4F7",
    "main_edge": "#2F6F7E",
    "text_fill": "#E8F1EC",
    "text_edge": "#3D7462",
    "vision_fill": "#ECEAF7",
    "vision_edge": "#665B9A",
    "audio_fill": "#FFF1D8",
    "audio_edge": "#A56B1F",
    "reliability_fill": "#F0ECF8",
    "reliability_edge": "#705B9D",
    "aux_fill": "#FFF4DE",
    "aux_edge": "#A16B21",
    "neutral_fill": "#F3F5F6",
    "neutral_edge": "#7A8992",
    "white": "#FFFFFF",
}


def configure(locale: str) -> None:
    fonts = (
        ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
        if locale == "zh"
        else ["Arial", "Helvetica", "DejaVu Sans"]
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
            "font.size": 7.0,
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fc,
    ec,
    fontsize=6.5,
    lw=1.0,
    dashed=False,
    weight="normal",
    zorder=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.28,rounding_size=0.7",
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        facecolor=fc,
        edgecolor=ec,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.2,
    dashed=False,
    rad=0.0,
    zorder=2,
    mutation=9,
):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def tag(ax, x, y, text, *, fc, ec, color=None, fontsize=5.7):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or ec,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec=ec, lw=0.7),
        zorder=8,
    )


def draw(locale: str, labels: dict[str, str]) -> plt.Figure:
    configure(locale)
    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 61)
    ax.axis("off")

    # Main clean task path.
    ax.add_patch(
        FancyBboxPatch(
            (0.8, 34.0),
            98.4,
            25.5,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor="#F8FBFC",
            edgecolor=COLORS["main_edge"],
            linewidth=1.2,
            zorder=0,
        )
    )
    ax.text(
        2.0,
        57.6,
        labels["main_band"],
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["main_edge"],
    )

    modality_specs = [
        (labels["text"], labels["text_encoder"], 50.2, COLORS["text_fill"], COLORS["text_edge"]),
        (labels["vision"], labels["vision_encoder"], 43.2, COLORS["vision_fill"], COLORS["vision_edge"]),
        (labels["audio"], labels["audio_encoder"], 36.2, COLORS["audio_fill"], COLORS["audio_edge"]),
    ]
    for input_label, encoder_label, y, fill, edge in modality_specs:
        box(ax, 2.2, y, 8.2, 5.2, input_label, fc=fill, ec=edge, fontsize=6.2, weight="bold")
        box(ax, 13.7, y, 12.6, 5.2, encoder_label, fc=fill, ec=edge, fontsize=6.1)
        arrow(ax, (10.4, y + 2.6), (13.7, y + 2.6), color=edge, lw=1.25)

    box(
        ax,
        30.0,
        41.7,
        12.2,
        8.1,
        labels["embeddings"],
        fc=COLORS["main_fill"],
        ec=COLORS["main_edge"],
        fontsize=6.5,
        weight="bold",
    )
    for y in [52.8, 45.8, 38.8]:
        arrow(ax, (26.3, y), (30.0, 46.0), color=COLORS["main_edge"], lw=1.15)

    box(
        ax,
        46.0,
        41.7,
        13.5,
        8.1,
        labels["fusion"],
        fc=COLORS["main_fill"],
        ec=COLORS["main_edge"],
        fontsize=6.5,
        weight="bold",
    )
    box(
        ax,
        63.2,
        41.7,
        9.4,
        8.1,
        labels["prediction"],
        fc=COLORS["white"],
        ec=COLORS["main_edge"],
        fontsize=6.6,
        weight="bold",
    )
    box(
        ax,
        76.1,
        41.7,
        9.8,
        8.1,
        labels["task_loss"],
        fc=COLORS["neutral_fill"],
        ec=COLORS["neutral_edge"],
        fontsize=6.6,
    )
    arrow(ax, (42.2, 45.8), (46.0, 45.8), color=COLORS["main_edge"], lw=1.8)
    arrow(ax, (59.5, 45.8), (63.2, 45.8), color=COLORS["main_edge"], lw=1.8)
    arrow(ax, (72.6, 45.8), (76.1, 45.8), color=COLORS["main_edge"], lw=1.2, dashed=True)
    tag(
        ax,
        34.8,
        53.9,
        labels["clean"],
        fc="#FFFFFF",
        ec=COLORS["main_edge"],
        fontsize=5.4,
    )

    # Explicit inference boundary statement.
    box(
        ax,
        88.0,
        39.8,
        10.0,
        12.0,
        labels["no_gate"],
        fc="#FFFFFF",
        ec=COLORS["neutral_edge"],
        fontsize=5.15,
        dashed=True,
        weight="bold",
    )
    ax.plot([87.2, 87.2], [38.0, 54.0], color=COLORS["neutral_edge"], lw=0.9, linestyle=(0, (2, 2)))

    # Training-only band.
    ax.add_patch(
        FancyBboxPatch(
            (0.8, 1.2),
            98.4,
            30.5,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor="#FCFBFD",
            edgecolor=COLORS["reliability_edge"],
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            zorder=0,
        )
    )
    ax.text(
        2.0,
        29.8,
        labels["train_band"],
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["reliability_edge"],
    )
    tag(
        ax,
        95.2,
        29.7,
        labels["training_only"],
        fc=COLORS["reliability_fill"],
        ec=COLORS["reliability_edge"],
        fontsize=5.4,
    )

    # Reliability supervision row.
    box(
        ax,
        4.2,
        18.6,
        7.2,
        5.2,
        labels["clean_copy"],
        fc="#FFFFFF",
        ec=COLORS["neutral_edge"],
        fontsize=5.45,
        dashed=True,
    )
    arrow(ax, (10.4, 45.8), (7.8, 23.8), color=COLORS["vision_edge"], lw=0.95, dashed=True, rad=0.08)
    arrow(ax, (10.4, 38.8), (8.3, 23.8), color=COLORS["audio_edge"], lw=0.95, dashed=True, rad=-0.08)
    box(
        ax,
        15.3,
        17.0,
        15.0,
        8.3,
        labels["triplets"],
        fc=COLORS["reliability_fill"],
        ec=COLORS["reliability_edge"],
        fontsize=6.3,
    )
    box(
        ax,
        34.0,
        17.0,
        13.0,
        8.3,
        labels["heads"],
        fc=COLORS["reliability_fill"],
        ec=COLORS["reliability_edge"],
        fontsize=6.2,
        weight="bold",
    )
    box(
        ax,
        50.5,
        17.0,
        11.2,
        8.3,
        labels["scores"],
        fc="#FFFFFF",
        ec=COLORS["reliability_edge"],
        fontsize=6.5,
    )
    box(
        ax,
        65.4,
        17.0,
        15.6,
        8.3,
        labels["rank_loss"],
        fc=COLORS["neutral_fill"],
        ec=COLORS["reliability_edge"],
        fontsize=6.1,
    )
    arrow(ax, (11.4, 21.1), (15.3, 21.1), color=COLORS["reliability_edge"], dashed=True)
    arrow(ax, (7.8, 18.6), (8.8, 12.5), color=COLORS["neutral_edge"], lw=0.9, dashed=True)
    arrow(ax, (30.3, 21.1), (34.0, 21.1), color=COLORS["reliability_edge"])
    arrow(ax, (47.0, 21.1), (50.5, 21.1), color=COLORS["reliability_edge"])
    arrow(ax, (61.7, 21.1), (65.4, 21.1), color=COLORS["reliability_edge"], dashed=True)

    # Per-sample auxiliary allocation row.
    box(
        ax,
        3.0,
        4.3,
        11.6,
        8.2,
        labels["teachers"],
        fc=COLORS["neutral_fill"],
        ec=COLORS["neutral_edge"],
        fontsize=6.2,
        dashed=True,
    )
    tag(ax, 8.8, 13.5, labels["frozen"], fc="#FFFFFF", ec=COLORS["neutral_edge"], fontsize=5.3)
    box(
        ax,
        18.0,
        3.5,
        18.2,
        9.8,
        labels["losses"],
        fc=COLORS["aux_fill"],
        ec=COLORS["aux_edge"],
        fontsize=5.9,
    )
    box(
        ax,
        40.0,
        3.5,
        20.0,
        9.8,
        labels["budget"],
        fc=COLORS["aux_fill"],
        ec=COLORS["aux_edge"],
        fontsize=5.75,
        weight="bold",
    )
    box(
        ax,
        63.6,
        3.5,
        15.0,
        9.8,
        labels["weighted"],
        fc=COLORS["aux_fill"],
        ec=COLORS["aux_edge"],
        fontsize=5.9,
    )
    box(
        ax,
        82.2,
        3.5,
        15.0,
        9.8,
        labels["objective"],
        fc="#FFFFFF",
        ec=COLORS["ink"],
        fontsize=5.75,
        weight="bold",
        lw=1.2,
    )
    arrow(ax, (14.6, 8.4), (18.0, 8.4), color=COLORS["neutral_edge"], dashed=True)
    arrow(ax, (36.2, 8.4), (40.0, 8.4), color=COLORS["aux_edge"])
    arrow(ax, (60.0, 8.4), (63.6, 8.4), color=COLORS["aux_edge"])
    arrow(ax, (78.6, 8.4), (82.2, 8.4), color=COLORS["aux_edge"])
    arrow(ax, (56.1, 17.0), (50.0, 13.3), color=COLORS["reliability_edge"], lw=1.2)
    arrow(ax, (73.2, 17.0), (89.1, 13.3), color=COLORS["reliability_edge"], lw=1.0, dashed=True, rad=-0.05)
    arrow(ax, (81.0, 41.7), (89.5, 13.3), color=COLORS["neutral_edge"], lw=1.0, dashed=True, rad=-0.08)

    # Clean embeddings feed the per-sample losses; no score arrow goes to fusion.
    arrow(ax, (36.0, 41.7), (27.0, 13.3), color=COLORS["aux_edge"], lw=0.95, dashed=True, rad=0.10)

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.012, top=0.992)
    return fig


def export(fig: plt.Figure, stem: Path) -> None:
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path)
    # Matplotlib emits trailing spaces in multiline SVG path data. Normalize
    # generated text so repository whitespace checks remain useful.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)


def main() -> None:
    payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    for locale in ("en", "zh"):
        fig = draw(locale, payload[locale])
        export(fig, FIG_DIR / f"f1_method_overview_{locale}")
        plt.close(fig)
    print(f"Wrote F1 exports to {FIG_DIR}")


if __name__ == "__main__":
    main()
