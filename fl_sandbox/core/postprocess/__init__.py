"""Unified postprocess API for attacker_sandbox experiments."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_postprocess_hint_lines",
    "main",
    "parse_args",
    "plot_client_metric_heatmap",
    "plot_dual_series",
    "plot_mean_with_band",
    "plot_round_boxplot",
    "plot_series",
    "plot_xy_series",
]

_EXPORTS = {
    "build_postprocess_hint_lines": (".postprocess", "build_postprocess_hint_lines"),
    "main": (".postprocess", "main"),
    "parse_args": (".postprocess", "parse_args"),
    "plot_client_metric_heatmap": (".visualization", "plot_client_metric_heatmap"),
    "plot_dual_series": (".visualization", "plot_dual_series"),
    "plot_mean_with_band": (".visualization", "plot_mean_with_band"),
    "plot_round_boxplot": (".visualization", "plot_round_boxplot"),
    "plot_series": (".visualization", "plot_series"),
    "plot_xy_series": (".visualization", "plot_xy_series"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
