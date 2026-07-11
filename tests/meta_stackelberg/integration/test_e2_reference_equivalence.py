import numpy as np
import torch

from meta_stackelberg.core.random_state import RandomSnapshot
from meta_stackelberg.experiments.defense_response_matrix import DefenseGridPoint
from tests.meta_stackelberg.integration.test_backdoor_attack_validity import _run as run_e1_bfl
from tests.meta_stackelberg.integration.test_dba_attack_validity import _run as run_e1_dba
from tests.meta_stackelberg.integration.test_ipm_lmp_attack_validity import _run as run_e1_round_attack
from tests.meta_stackelberg.integration.test_untargeted_attack_validity import _run as run_e1_delta
from tests.meta_stackelberg.integration.test_backdoor_defense_matrix import _run as run_e2_backdoor
from tests.meta_stackelberg.integration.test_untargeted_defense_matrix import _run as run_e2_untargeted


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_trajectory_exact(e2_trajectory, e1_trajectory) -> None:
    assert len(e2_trajectory.transitions) == len(e1_trajectory.transitions)
    for e2_step, e1_step in zip(e2_trajectory.transitions, e1_trajectory.transitions):
        assert e2_step.sampled_clients == e1_step.sampled_clients
        np.testing.assert_array_equal(
            e2_step.aggregate_delta.vector(),
            e1_step.aggregate_delta.vector(),
        )
        np.testing.assert_array_equal(
            e2_step.state_after.global_model.vector(),
            e1_step.state_after.global_model.vector(),
        )
        _assert_snapshots_equal(
            e2_step.state_after.random_snapshot,
            e1_step.state_after.random_snapshot,
        )


def test_untargeted_matrix_references_are_exact_e1_fixtures() -> None:
    point = DefenseGridPoint(10.0, 0.0)
    cases = []
    for seed in (301, 302, 303):
        cases.extend((
            ('delta-reversal', seed, 'clean', run_e1_delta(seed, 'clean')),
            ('delta-reversal', seed, 'attack', run_e1_delta(seed, 'delta-reversal')),
        ))
    for seed in (501, 502, 503):
        cases.extend((
            ('ipm', seed, 'clean', run_e1_round_attack(seed, 'clean', 'fedavg')),
            ('ipm', seed, 'attack', run_e1_round_attack(seed, 'ipm', 'fedavg')),
            ('lmp', seed, 'clean', run_e1_round_attack(seed, 'clean', 'median')),
            ('lmp', seed, 'attack', run_e1_round_attack(seed, 'lmp', 'median')),
        ))
    for task, seed, branch, e1 in cases:
        e2 = run_e2_untargeted(task, seed, branch, point, reference=True)
        assert e2.observation.final_clean_loss == e1.metrics.loss
        assert e2.observation.final_clean_accuracy == e1.metrics.accuracy
        np.testing.assert_array_equal(
            e2.observation.final_model_vector,
            e1.trajectory.final_state.global_model.vector(),
        )
        _assert_trajectory_exact(e2.trajectory, e1.trajectory)


def test_backdoor_matrix_references_are_exact_e1_fixtures() -> None:
    point = DefenseGridPoint(10.0, 0.0)
    cases = []
    for seed in (401, 402, 403):
        cases.extend((
            ('bfl', seed, 'clean', run_e1_bfl(seed, 'clean')),
            ('bfl', seed, 'attack', run_e1_bfl(seed, 'bfl-1')),
        ))
    for seed in (601, 602, 603):
        cases.extend((
            ('dba', seed, 'clean', run_e1_dba(seed, 'clean')),
            ('dba', seed, 'attack', run_e1_dba(seed, 'dba-strong')),
        ))
    for task, seed, branch, e1 in cases:
        e2 = run_e2_backdoor(task, seed, branch, point, reference=True)
        assert e2.observation.final_clean_loss == e1.clean.loss
        assert e2.observation.final_clean_accuracy == e1.clean.accuracy
        expected_asr = (
            e1.backdoor.attack_success_rate
            if task == 'bfl'
            else e1.full_trigger.attack_success_rate
        )
        assert e2.observation.attack_metric == expected_asr
        np.testing.assert_array_equal(
            e2.observation.final_model_vector,
            e1.trajectory.final_state.global_model.vector(),
        )
        _assert_trajectory_exact(e2.trajectory, e1.trajectory)
