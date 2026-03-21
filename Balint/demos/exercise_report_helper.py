"""Reusable plotting and export helpers for the split exercise notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


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
    axis.plot(x_positions, summary_table["final_energy"], marker="o", label="VMC")
    axis.plot(x_positions, summary_table["exact_ground_energy"], marker="s", label="Exact")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("Energy")
    axis.set_title(title)
    axis.legend()
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
        axis.errorbar(
            group["subsystem_size"],
            group["renyi2"],
            yerr=group.get("renyi2_std"),
            marker="o",
            capsize=3,
            label=str(label),
        )
    axis.set_xlabel("Subsystem Size")
    axis.set_ylabel("Renyi-2")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_architecture_summary(summary_table: pd.DataFrame, title: str = "Architecture Comparison") -> Figure:
    labels = summary_table["model"].astype(str).tolist()
    x_positions = np.arange(len(labels), dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(x_positions, summary_table["half_partition_renyi2"], color="#2B6CB0")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Half-Partition Renyi-2")
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
    axis.set_ylabel(value_column.replace("_", " ").title())
    axis.set_title(title or value_column.replace("_", " ").title())
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


__all__ = [
    "build_output_manifest",
    "ensure_report_output_dir",
    "plot_architecture_summary",
    "plot_energy_benchmark",
    "plot_entropy_scan",
    "plot_training_history",
    "save_report_figure",
    "save_report_table",
]
