import pytest

from meta_stackelberg.stackelberg.algorithm2 import MetaSGAlgorithm2


def test_algorithm2_uses_exact_T_K_l_counts_and_one_meta_update() -> None:
    calls = {'sample': 0, 'clone': 0, 'adapt': 0, 'meta': 0}

    def sample_tasks(meta_iteration, count):
        calls['sample'] += 1
        return tuple(f'{meta_iteration}:{index}' for index in range(count))

    def clone_for_task(task):
        calls['clone'] += 1
        return [task]

    def adapt_task(task, clone, step):
        calls['adapt'] += 1
        clone.append(step)
        return f'{task}-step-{step}'

    def apply_meta_update(meta_iteration, adapted, meta_step):
        calls['meta'] += 1
        assert len(adapted) == 3
        assert meta_step == pytest.approx(0.25)
        return f'meta-{meta_iteration}'

    result = MetaSGAlgorithm2(T=2, K=3, l=4, meta_step=0.25).run(
        sample_tasks=sample_tasks,
        clone_for_task=clone_for_task,
        adapt_task=adapt_task,
        apply_meta_update=apply_meta_update,
    )

    assert calls == {'sample': 2, 'clone': 6, 'adapt': 24, 'meta': 2}
    assert len(result.iterations) == 2
    assert all(len(item.tasks) == 3 for item in result.iterations)
    assert all(len(task.adaptation_steps) == 4
               for item in result.iterations for task in item.tasks)
    assert [event.kind for event in result.events[:7]] == [
        'sample_tasks', 'clone_task', 'adapt_task', 'adapt_task',
        'adapt_task', 'adapt_task', 'clone_task',
    ]


@pytest.mark.parametrize('kwargs', [
    {'T': 0, 'K': 1, 'l': 1, 'meta_step': 1.0},
    {'T': 1, 'K': 0, 'l': 1, 'meta_step': 1.0},
    {'T': 1, 'K': 1, 'l': 0, 'meta_step': 1.0},
    {'T': 1, 'K': 1, 'l': 1, 'meta_step': 0.0},
])
def test_algorithm2_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        MetaSGAlgorithm2(**kwargs)


def test_algorithm2_does_not_accept_algorithm1_parameter_aliases() -> None:
    with pytest.raises(TypeError):
        MetaSGAlgorithm2(T=1, K=1, l=1, meta_step=1.0, N_D=10)
