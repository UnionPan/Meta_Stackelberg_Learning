"""Core orchestration for the attacker sandbox."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "BatchRunRequest",
    "ATTACK_CHOICES",
    "ExperimentRunResult",
    "ClientRoundMetrics",
    "RoundContext",
    "MinimalFLRunner",
    "PROTOCOL_CHOICES",
    "RoundSummary",
    "RoundRuntimeState",
    "RoundTimer",
    "RoundUpdateStats",
    "ExperimentTimer",
    "RunConfig",
    "SandboxConfig",
    "Weights",
    "build_postprocess_hint_lines",
    "clone_args",
    "config_to_namespace",
    "execute_batch_run",
    "execute_experiment",
    "load_run_config",
    "merge_cli_overrides",
    "persist_experiment_artifacts",
    "summaries_to_dict",
    "client_metrics_to_rows",
    "write_batch_results",
]


_EXPORTS = {
    "BatchRunRequest": (".batch_runner", "BatchRunRequest"),
    "ATTACK_CHOICES": (".experiment_builders", "ATTACK_CHOICES"),
    "ClientRoundMetrics": (".runtime", "ClientRoundMetrics"),
    "RoundContext": (".runtime", "RoundContext"),
    "ExperimentTimer": (".runtime", "ExperimentTimer"),
    "ExperimentRunResult": (".experiment_service", "ExperimentRunResult"),
    "MinimalFLRunner": (".fl_runner", "MinimalFLRunner"),
    "PROTOCOL_CHOICES": ("fl_sandbox.config", "PROTOCOL_CHOICES"),
    "RoundRuntimeState": (".runtime", "RoundRuntimeState"),
    "RoundSummary": (".runtime", "RoundSummary"),
    "RoundTimer": (".runtime", "RoundTimer"),
    "RoundUpdateStats": (".runtime", "RoundUpdateStats"),
    "RunConfig": ("fl_sandbox.config", "RunConfig"),
    "SandboxConfig": (".fl_runner", "SandboxConfig"),
    "Weights": (".runtime", "Weights"),
    "build_postprocess_hint_lines": (".postprocess", "build_postprocess_hint_lines"),
    "clone_args": (".batch_runner", "clone_args"),
    "config_to_namespace": ("fl_sandbox.config", "config_to_namespace"),
    "execute_batch_run": (".batch_runner", "execute_batch_run"),
    "execute_experiment": (".experiment_service", "execute_experiment"),
    "load_run_config": ("fl_sandbox.config", "load_run_config"),
    "merge_cli_overrides": ("fl_sandbox.config", "merge_cli_overrides"),
    "persist_experiment_artifacts": (".experiment_service", "persist_experiment_artifacts"),
    "summaries_to_dict": (".runtime", "summaries_to_dict"),
    "client_metrics_to_rows": (".runtime", "client_metrics_to_rows"),
    "write_batch_results": (".batch_runner", "write_batch_results"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    if module_name.startswith("."):
        module = import_module(module_name, __name__)
    else:
        module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
