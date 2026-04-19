"""Plotting helpers for sandbox postprocess workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: plt.Figure, save_path: str | None) -> None:
    if not save_path:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')


def plot_series(
    values: Iterable[float],
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> plt.Figure:
    values = list(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values, marker='o', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_dual_series(
    values_a: Iterable[float],
    values_b: Iterable[float],
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> plt.Figure:
    values_a = list(values_a)
    values_b = list(values_b)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values_a) + 1), values_a, marker='o', linewidth=1.5, label=label_a)
    ax.plot(range(1, len(values_b) + 1), values_b, marker='s', linewidth=1.5, label=label_b)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_client_metric_heatmap(
    rows: list[dict],
    metric_key: str,
    title: str,
    colorbar_label: str,
    save_path: str | None = None,
) -> plt.Figure:
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No client metrics available', ha='center', va='center')
        ax.axis('off')
        return fig

    round_ids = sorted({int(row['round_idx']) for row in rows})
    client_ids = sorted({int(row['client_id']) for row in rows})
    matrix = np.full((len(client_ids), len(round_ids)), np.nan, dtype=float)

    round_to_col = {round_idx: i for i, round_idx in enumerate(round_ids)}
    client_to_row = {client_id: i for i, client_id in enumerate(client_ids)}

    for row in rows:
        value = row.get(metric_key)
        if value in ('', None):
            continue
        matrix[client_to_row[int(row['client_id'])]][round_to_col[int(row['round_idx'])]] = float(value)

    fig, ax = plt.subplots(figsize=(max(8, len(round_ids) * 0.14), max(4, len(client_ids) * 0.35)))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#f2f2f2')
    image = ax.imshow(masked, aspect='auto', interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel('Client ID')
    ax.set_xticks(range(len(round_ids)))
    ax.set_xticklabels(round_ids, rotation=90 if len(round_ids) > 20 else 0)
    ax.set_yticks(range(len(client_ids)))
    ax.set_yticklabels(client_ids)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_xy_series(
    x_values: Iterable[float],
    y_values: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | None = None,
) -> plt.Figure:
    x_values = list(x_values)
    y_values = list(y_values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, y_values, marker='o', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_mean_with_band(
    mean_values: Iterable[float],
    std_values: Iterable[float],
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> plt.Figure:
    mean_values = np.asarray(list(mean_values), dtype=float)
    std_values = np.asarray(list(std_values), dtype=float)
    rounds = np.arange(1, len(mean_values) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, mean_values, linewidth=1.8)
    ax.fill_between(rounds, mean_values - std_values, mean_values + std_values, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_round_boxplot(
    grouped_values: list[list[float]],
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(grouped_values) * 0.18), 4.8))
    positions = np.arange(1, len(grouped_values) + 1)
    filtered_positions = [pos for pos, values in zip(positions, grouped_values) if values]
    filtered_values = [values for values in grouped_values if values]
    if filtered_values:
        ax.boxplot(filtered_values, positions=filtered_positions, widths=0.6, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

__all__ = [
    'plot_client_metric_heatmap',
    'plot_dual_series',
    'plot_mean_with_band',
    'plot_round_boxplot',
    'plot_series',
    'plot_xy_series',
]
