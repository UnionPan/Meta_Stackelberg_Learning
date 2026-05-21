import inspect

from fl_sandbox.config.schema import RunConfig
from fl_sandbox.core import experiment_service


def test_execute_experiment_accepts_run_config_as_first_argument():
    signature = inspect.signature(experiment_service.execute_experiment)

    assert list(signature.parameters)[:1] == ["run_config"]


def test_experiment_builders_no_longer_exports_build_config():
    import fl_sandbox.core.experiment_builders as builders

    assert not hasattr(builders, "build_config")


def test_run_config_can_build_benchmark_payload():
    cfg = RunConfig.from_flat_dict({"protocol": "rlfl", "warmup_rounds": 3})

    assert cfg.benchmark_protocol_payload() == {
        "name": "paper_aligned_accuracy_asr",
        "warmup_rounds": 3,
        "rl_distribution_steps": 10,
        "rl_policy_train_end_round": 30,
        "rl_attack_start_round": 10,
    }
