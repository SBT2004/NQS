"""Reusable plotting and export helpers for the split exercise notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def ensure_report_output_dir(name: str = "notebook_reports") -> Path:
    output_dir = Path(__file__).resolve().parent / "report_outputs" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_report_table(
    table: pd.DataFrame,
    stem: str,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    destination = ensure_report_output_dir() if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}.csv"
    html_path = destination / f"{stem}.html"
    table.to_csv(csv_path, index=False)
    table.to_html(html_path, index=False)
    return {"csv": csv_path, "html": html_path}


def save_report_figure(
    figure: Figure,
    stem: str,
    output_dir: str | Path | None = None,
    dpi: int = 150,
) -> Path:
    destination = ensure_report_output_dir() if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / f"{stem}.png"
    figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    return figure_path


def build_output_manifest(entries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(entries).sort_values(by=["section", "name"]).reset_index(drop=True)


def add_report_figure_context(
    figure: Figure,
    *,
    distinction: str,
    context: str,
) -> Figure:
    figure.text(
        0.01,
        0.01,
        f"{distinction} | {context}",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4A5568",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return figure


def _label_column(table: pd.DataFrame) -> str:
    for column in ("sweep_label", "model", "subsystem_size", "step"):
        if column in table.columns:
            return column
    raise ValueError("table must contain a label column for plotting.")


def plot_energy_benchmark(summary_table: pd.DataFrame, title: str = "Energy Benchmark") -> Figure:
    label_column = _label_column(summary_table)
    labels = summary_table[label_column].astype(str).tolist()
    x_positions = np.arange(len(labels), dtype=np.float64)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        x_positions,
        summary_table["final_energy"],
        marker="o",
        linewidth=1.6,
        color="#1f77b4",
        label="Sampled VMC final energy",
    )
    axis.plot(
        x_positions,
        summary_table["exact_ground_energy"],
        marker="s",
        linewidth=1.4,
        color="#ff7f0e",
        label="Exact diagonalization",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("Energy expectation value")
    axis.set_title(title)
    axis.legend(title="Estimator")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_entropy_scan(
    entropy_scan_table: pd.DataFrame,
    *,
    line_column: str = "model",
    title: str = "Renyi-2 Entropy Scan",
) -> Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, group in entropy_scan_table.groupby(line_column):
        ordered_group = group.sort_values("subsystem_size").copy()
        ordered_group = ordered_group.loc[np.isfinite(ordered_group["renyi2"])].copy()
        if ordered_group.empty:
            continue
        if "renyi2_ci95" in ordered_group.columns:
            yerr = np.asarray(ordered_group["renyi2_ci95"], dtype=np.float64)
        elif "renyi2_std" in ordered_group.columns:
            yerr = np.asarray(ordered_group["renyi2_std"], dtype=np.float64)
        else:
            yerr = np.zeros(len(ordered_group), dtype=np.float64)
        yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)
        axis.errorbar(
            ordered_group["subsystem_size"],
            ordered_group["renyi2"],
            yerr=yerr,
            marker="o",
            capsize=3,
            label=str(label),
        )
    axis.set_xlabel("Subsystem Size")
    axis.set_ylabel("Renyi-2 entropy")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(title=line_column.replace("_", " ").title())
    figure.tight_layout()
    return figure


def plot_architecture_summary(summary_table: pd.DataFrame, title: str = "Architecture Comparison") -> Figure:
    labels = summary_table["model"].astype(str).tolist()
    x_positions = np.arange(len(labels), dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(x_positions, summary_table["half_partition_renyi2"], color="#2B6CB0")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Half-partition Renyi-2 entropy")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for bar, parameter_count in zip(bars, summary_table["parameter_count"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(parameter_count)} params",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    return figure


def plot_training_history(
    training_history_table: pd.DataFrame,
    value_column: str,
    *,
    line_column: str = "sweep_label",
    title: str | None = None,
) -> Figure:
    if value_column not in training_history_table.columns:
        raise ValueError(f"{value_column!r} is not present in the training history table.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, group in training_history_table.groupby(line_column):
        axis.plot(group["step"], group[value_column], marker="o", label=str(label))
    axis.set_xlabel("Training Step")
    ylabel = value_column.replace("_", " ").title()
    if value_column == "energy":
        ylabel = "Sampled variational energy"
    elif value_column == "renyi2_entropy":
        ylabel = "Half-partition Renyi-2 entropy"
    axis.set_ylabel(ylabel)
    axis.set_title(title or value_column.replace("_", " ").title())
    axis.grid(alpha=0.25)
    axis.legend(title=line_column.replace("_", " ").title())
    figure.tight_layout()
    return figure


def plot_lattice_graph(
    graph: Any,
    *,
    title: str,
    edge_specs: dict[int, str] | tuple[tuple[int, str], ...],
    legend_entries: Sequence[tuple[str, str]],
    legend_columns: int = 1,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    if figsize is None:
        figure, axis = graph.draw(
            edge_specs=edge_specs,
            title=title,
            node_size=500,
            font_size=9,
        )
    else:
        figure, axis = plt.subplots(figsize=figsize)
        figure, axis = graph.draw(
            edge_specs=edge_specs,
            ax=axis,
            title=title,
            node_size=500,
            font_size=9,
        )
    axis.legend(
        handles=[Line2D([0], [0], color=color, lw=2.3, label=label) for color, label in legend_entries],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=legend_columns,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return figure


__all__ = [
    "add_report_figure_context",
    "build_output_manifest",
    "ensure_report_output_dir",
    "plot_architecture_summary",
    "plot_energy_benchmark",
    "plot_lattice_graph",
    "plot_entropy_scan",
    "plot_training_history",
    "save_report_figure",
    "save_report_table",
]
