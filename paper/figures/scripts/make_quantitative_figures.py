#!/usr/bin/env python3
"""Generate F2--F5 manuscript figures from frozen aggregate evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "figures"
DATA_DIR = FIG_DIR / "data"

MM = 1 / 25.4
WIDTH_MM = 183

METHODS = ["Repaired MFON", "P4 Constant", "P4 Learned"]
METHOD_COLORS = {
    "Repaired MFON": "#484878",
    "P4 Constant": "#9AA8CE",
    "P4 Learned": "#D58EAA",
}
METHOD_HATCHES = {"Repaired MFON": "//", "P4 Constant": "..", "P4 Learned": "xx"}
MODALITIES = ["Vision", "Audio"]
MODALITY_COLORS = {"Vision": "#665B9A", "Audio": "#C18432"}
MODALITY_HATCHES = {"Vision": "//", "Audio": ".."}
INK = "#27333B"
MUTED = "#65727B"
GRID = "#D9DEE2"
POSITIVE = "#3B7895"
NEGATIVE = "#B85C5C"


TEXT = {
    "en": {
        "methods": METHODS,
        "vision": "Vision",
        "audio": "Audio",
        "mean_sd": "Mean ± sample SD (3 seeds)",
        "primary": "Primary regression endpoints",
        "delta_title": "Learned versus equal-budget Constant",
        "delta_x": "Favourable-direction delta",
        "delta_note": "Right favours Learned; lower-is-better metrics are sign-reversed",
        "auroc": "Clean/corrupt AUROC",
        "monotonic": "Monotonicity (-Spearman)",
        "below": "Highest severity below clean",
        "confounds": "Clean-score correlations with simple statistics",
        "correlation": "Pearson correlation",
        "length": "Length",
        "energy": "Energy",
        "abs_label": "|Sentiment label|",
        "cross_auroc": "Held-out corruption detection",
        "cross_mae": "Task sensitivity to corruption",
        "delta_mae": "ΔMAE (corrupt − clean)",
        "chance": "chance",
        "single_session": "Single GPU session; no uncertainty estimate",
        "added_params": "Added optimized parameters",
        "percent_baseline": "% of MFON optimizer parameters",
        "latency": "Latency",
        "throughput": "Throughput",
        "memory": "Peak allocated memory",
        "ms_batch": "ms per batch",
        "samples_s": "samples/s",
        "gib": "GiB",
        "inference": "Inference",
        "fb": "Forward + backward",
        "corruptions": {
            "Timestep dropout": "Timestep dropout\n(0.75)",
            "Contiguous mask": "Contiguous mask\n(0.75)",
            "Temporal shift": "Temporal shift\n(1.00)",
            "Modality missing": "Modality missing\n(1.00)",
        },
    },
    "zh": {
        "methods": ["修复版 MFON", "P4 Constant", "P4 Learned"],
        "vision": "视觉",
        "audio": "音频",
        "mean_sd": "均值 ± 样本标准差（3 个种子）",
        "primary": "主要回归终点",
        "delta_title": "Learned 相对等预算 Constant",
        "delta_x": "有利方向差值",
        "delta_note": "向右有利于 Learned；MAE 与 Loss 已反转符号",
        "auroc": "干净/退化 AUROC",
        "monotonic": "单调性（-Spearman）",
        "below": "最高严重度低于干净的比例",
        "confounds": "干净分数与简单统计量的相关性",
        "correlation": "Pearson 相关系数",
        "length": "长度",
        "energy": "能量",
        "abs_label": "|情感标签|",
        "cross_auroc": "未见扰动检测",
        "cross_mae": "任务预测对扰动的敏感度",
        "delta_mae": "ΔMAE（退化 - 干净）",
        "chance": "随机水平",
        "single_session": "单 GPU 会话；无不确定性估计",
        "added_params": "新增优化参数",
        "percent_baseline": "占 MFON 优化参数的百分比",
        "latency": "延迟",
        "throughput": "吞吐量",
        "memory": "峰值已分配显存",
        "ms_batch": "ms/批",
        "samples_s": "样本/s",
        "gib": "GiB",
        "inference": "推理",
        "fb": "前向 + 反向",
        "corruptions": {
            "Timestep dropout": "时间步丢弃\n(0.75)",
            "Contiguous mask": "连续片段遮蔽\n(0.75)",
            "Temporal shift": "时间错位\n(1.00)",
            "Modality missing": "模态缺失\n(1.00)",
        },
    },
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )


def light_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
    ax.tick_params(length=3, width=0.7)


def save_figure(fig: plt.Figure, stem: str) -> None:
    # Preserve the declared 183 mm canvas exactly; subplot margins are managed
    # explicitly above so the journal-size contract is not changed by cropping.
    fig.savefig(FIG_DIR / f"{stem}.svg")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def format_bar_labels(ax: plt.Axes, bars, fmt: str, pad: float = 2.0) -> None:
    ax.bar_label(bars, fmt=fmt, padding=pad, fontsize=5.8, color=INK)


def draw_metric_panel(ax: plt.Axes, data: pd.DataFrame, metric: str, locale: str) -> None:
    labels = TEXT[locale]
    subset = data[data["metric"] == metric].set_index("method").loc[METHODS]
    x = np.arange(len(METHODS))
    values = subset["mean"].to_numpy()
    sd = subset["sd"].to_numpy()
    bars = ax.bar(
        x,
        values,
        yerr=sd,
        capsize=2.4,
        width=0.68,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor=INK,
        linewidth=0.65,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    for patch, method in zip(bars, METHODS):
        patch.set_hatch(METHOD_HATCHES[method])
    spread = max(values.max() + sd.max() - (values.min() - sd.max()), 0.002)
    lower = values.min() - sd.max() - 0.34 * spread
    upper = values.max() + sd.max() + 0.42 * spread
    ax.set_ylim(lower, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(labels["methods"], rotation=16, ha="right")
    ax.set_title(metric, fontweight="bold", pad=5)
    format_bar_labels(ax, bars, "%.4f")
    light_axis(ax)


def make_f2(locale: str) -> None:
    configure(locale)
    labels = TEXT[locale]
    data = pd.read_csv(DATA_DIR / "f2_mosei_main_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM * MM, 122 * MM))
    for ax, metric, letter in zip(axes.flat[:3], ["MAE", "Corr", "Loss"], "abc"):
        draw_metric_panel(ax, data, metric, locale)
        panel_label(ax, letter)

    ax = axes.flat[3]
    order = [
        "Has0 Acc-2",
        "Has0 F1",
        "Non0 Acc-2",
        "Non0 F1",
        "Acc-5",
        "Acc-7",
        "MAE",
        "Corr",
        "Loss",
    ]
    pivot = data.pivot(index="metric", columns="method", values="mean")
    direction = data.drop_duplicates("metric").set_index("metric")["direction"]
    delta = pivot["P4 Learned"] - pivot["P4 Constant"]
    favourable = delta.copy()
    favourable[direction == "lower"] *= -1
    values = favourable.loc[order].to_numpy()
    y = np.arange(len(order))[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in values]
    bars = ax.barh(y, values, color=colors, edgecolor=INK, linewidth=0.6, height=0.68)
    for patch, value in zip(bars, values):
        patch.set_hatch("//" if value >= 0 else "\\\\")
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(labels["delta_x"])
    ax.set_title(labels["delta_title"], fontweight="bold", pad=5)
    ax.text(0.5, -0.22, labels["delta_note"], transform=ax.transAxes, ha="center", va="top", fontsize=5.7, color=MUTED)
    for yi, value in zip(y, values):
        label_x = value + 0.00022 if value >= 0 else 0.00016
        ax.text(
            label_x,
            yi,
            f"{value:+.4f}",
            ha="left",
            va="center",
            fontsize=5.5,
        )
    ax.set_xlim(-0.0020, 0.0077)
    light_axis(ax, xgrid=True, ygrid=False)
    panel_label(ax, "d")

    fig.suptitle(labels["mean_sd"], y=0.995, fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.12, hspace=0.48, wspace=0.34)
    save_figure(fig, f"f2_mosei_main_results_{locale}")


def make_f3(locale: str) -> None:
    configure(locale)
    labels = TEXT[locale]
    data = pd.read_csv(DATA_DIR / "f3_mosei_gaussian_audit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM * MM, 112 * MM))
    audit = data[data["section"] == "audit"]
    metric_specs = [
        ("AUROC", labels["auroc"], (0.78, 1.025)),
        ("-Spearman", labels["monotonic"], (0.72, 1.0)),
        ("Highest below clean", labels["below"], (0.92, 1.012)),
    ]
    for ax, (metric, title, ylim), letter in zip(axes.flat[:3], metric_specs, "abc"):
        subset = audit[audit["metric"] == metric].set_index("modality").loc[MODALITIES]
        x = np.arange(2)
        bars = ax.bar(
            x,
            subset["mean"],
            yerr=subset["sd"],
            capsize=2.5,
            width=0.62,
            color=[MODALITY_COLORS[m] for m in MODALITIES],
            edgecolor=INK,
            linewidth=0.7,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        for patch, modality in zip(bars, MODALITIES):
            patch.set_hatch(MODALITY_HATCHES[modality])
        ax.set_xticks(x)
        ax.set_xticklabels([labels["vision"], labels["audio"]])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", pad=5)
        format_bar_labels(ax, bars, "%.3f")
        light_axis(ax)
        panel_label(ax, letter)

    ax = axes.flat[3]
    conf = data[data["section"] == "confound"]
    metrics = ["Length", "Energy", "Absolute label"]
    x = np.arange(len(metrics))
    width = 0.34
    for i, modality in enumerate(MODALITIES):
        subset = conf[conf["modality"] == modality].set_index("metric").loc[metrics]
        bars = ax.bar(
            x + (i - 0.5) * width,
            subset["mean"],
            yerr=subset["sd"],
            width=width,
            capsize=2.3,
            color=MODALITY_COLORS[modality],
            edgecolor=INK,
            linewidth=0.65,
            label=labels["vision"] if modality == "Vision" else labels["audio"],
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
        )
        for patch in bars:
            patch.set_hatch(MODALITY_HATCHES[modality])
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels["length"], labels["energy"], labels["abs_label"]])
    ax.set_ylabel(labels["correlation"])
    ax.set_ylim(-0.50, 0.34)
    ax.set_title(labels["confounds"], fontweight="bold", pad=5)
    ax.legend(loc="lower left", ncol=2)
    light_axis(ax)
    panel_label(ax, "d")

    fig.suptitle(labels["mean_sd"], y=0.995, fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.10, hspace=0.44, wspace=0.31)
    save_figure(fig, f"f3_mosei_gaussian_audit_{locale}")


def make_f4(locale: str) -> None:
    configure(locale)
    labels = TEXT[locale]
    data = pd.read_csv(DATA_DIR / "f4_mosei_cross_corruption.csv")
    corruptions = ["Timestep dropout", "Contiguous mask", "Temporal shift", "Modality missing"]
    display = [labels["corruptions"][name] for name in corruptions]
    y = np.arange(len(corruptions))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * MM, 82 * MM))

    ax = axes[0]
    height = 0.34
    for i, modality in enumerate(MODALITIES):
        subset = data[data["modality"] == modality].set_index("corruption").loc[corruptions]
        bars = ax.barh(
            y + (0.5 - i) * height,
            subset["auroc"],
            xerr=subset["auroc_sd"],
            height=height,
            capsize=2.2,
            color=MODALITY_COLORS[modality],
            edgecolor=INK,
            linewidth=0.65,
            label=labels["vision"] if modality == "Vision" else labels["audio"],
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        for patch in bars:
            patch.set_hatch(MODALITY_HATCHES[modality])
    ax.axvline(0.5, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(display)
    ax.set_xlim(0.15, 1.06)
    ax.set_xlabel("AUROC")
    ax.set_title(labels["cross_auroc"], fontweight="bold", pad=6)
    light_axis(ax, xgrid=True, ygrid=False)
    panel_label(ax, "a")

    ax = axes[1]
    for i, modality in enumerate(MODALITIES):
        subset = data[data["modality"] == modality].set_index("corruption").loc[corruptions]
        bars = ax.barh(
            y + (0.5 - i) * height,
            subset["delta_mae"],
            xerr=subset["delta_mae_sd"],
            height=height,
            capsize=2.2,
            color=MODALITY_COLORS[modality],
            edgecolor=INK,
            linewidth=0.65,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        for patch in bars:
            patch.set_hatch(MODALITY_HATCHES[modality])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(display)
    ax.set_xlim(-0.0041, 0.0173)
    ax.set_xticks([-0.0025, 0.0, 0.0050, 0.0100, 0.0150])
    ax.set_xticklabels(["-0.0025", "0", "0.0050", "0.0100", "0.0150"])
    ax.set_xlabel(labels["delta_mae"])
    ax.set_title(labels["cross_mae"], fontweight="bold", pad=6)
    light_axis(ax, xgrid=True, ygrid=False)
    panel_label(ax, "b")

    handles = [
        Patch(facecolor=MODALITY_COLORS[m], edgecolor=INK, hatch=MODALITY_HATCHES[m], label=labels["vision"] if m == "Vision" else labels["audio"])
        for m in MODALITIES
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.90), ncol=2)
    fig.suptitle(labels["mean_sd"], y=0.995, fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.86, bottom=0.18, wspace=0.42)
    save_figure(fig, f"f4_mosei_cross_corruption_{locale}")


def make_f5(locale: str) -> None:
    configure(locale)
    labels = TEXT[locale]
    data = pd.read_csv(DATA_DIR / "f5_mosei_efficiency.csv").set_index("method").loc[METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM * MM, 112 * MM))
    x = np.arange(3)
    colors = [METHOD_COLORS[m] for m in METHODS]

    ax = axes.flat[0]
    overhead = data["added_reliability_parameters"].to_numpy() / data.loc["Repaired MFON", "optimized_parameters"] * 100
    bars = ax.bar(x, overhead, color=colors, edgecolor=INK, linewidth=0.65, width=0.68)
    for patch, method in zip(bars, METHODS):
        patch.set_hatch(METHOD_HATCHES[method])
    ax.set_xticks(x)
    ax.set_xticklabels(labels["methods"], rotation=16, ha="right")
    ax.set_ylabel(labels["percent_baseline"])
    ax.set_ylim(0, 0.0305)
    ax.set_title(labels["added_params"], fontweight="bold", pad=5)
    for bar, count, pct in zip(bars, data["added_reliability_parameters"], overhead):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.0010, f"{int(count):,}\n({pct:.3f}%)", ha="center", va="bottom", fontsize=5.5)
    light_axis(ax)
    panel_label(ax, "a")

    ax = axes.flat[1]
    latency = np.vstack([data["inference_ms_per_batch"], data["forward_backward_ms_per_batch"]]).T
    width = 0.34
    for i, label in enumerate([labels["inference"], labels["fb"]]):
        bars = ax.bar(x + (i - 0.5) * width, latency[:, i], width=width, color=colors, edgecolor=INK, linewidth=0.65, label=label)
        for patch in bars:
            patch.set_hatch("//" if i == 0 else "xx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels["methods"], rotation=16, ha="right")
    ax.set_ylabel(labels["ms_batch"])
    ax.set_ylim(920, 1190)
    ax.set_title(labels["latency"], fontweight="bold", pad=5)
    ax.legend(loc="upper left", ncol=2)
    light_axis(ax)
    panel_label(ax, "b")

    ax = axes.flat[2]
    throughput = np.vstack([data["inference_samples_per_second"], data["forward_backward_samples_per_second"]]).T
    for i, label in enumerate([labels["inference"], labels["fb"]]):
        bars = ax.bar(x + (i - 0.5) * width, throughput[:, i], width=width, color=colors, edgecolor=INK, linewidth=0.65, label=label)
        for patch in bars:
            patch.set_hatch("//" if i == 0 else "xx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels["methods"], rotation=16, ha="right")
    ax.set_ylabel(labels["samples_s"])
    ax.set_ylim(26.5, 33.0)
    ax.set_title(labels["throughput"], fontweight="bold", pad=5)
    light_axis(ax)
    panel_label(ax, "c")

    ax = axes.flat[3]
    memory = np.vstack([data["inference_peak_gib"], data["forward_backward_peak_gib"]]).T
    for i, label in enumerate([labels["inference"], labels["fb"]]):
        bars = ax.bar(x + (i - 0.5) * width, memory[:, i], width=width, color=colors, edgecolor=INK, linewidth=0.65, label=label)
        for patch in bars:
            patch.set_hatch("//" if i == 0 else "xx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels["methods"], rotation=16, ha="right")
    ax.set_ylabel(labels["gib"])
    ax.set_ylim(0, 5.35)
    ax.set_title(labels["memory"], fontweight="bold", pad=5)
    light_axis(ax)
    panel_label(ax, "d")

    fig.suptitle(labels["single_session"], y=0.995, fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.12, hspace=0.48, wspace=0.32)
    save_figure(fig, f"f5_mosei_efficiency_{locale}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for locale in ("en", "zh"):
        make_f2(locale)
        make_f3(locale)
        make_f4(locale)
        make_f5(locale)
    print("Generated F2--F5 in English and Chinese (SVG, PDF, PNG).")


if __name__ == "__main__":
    main()
