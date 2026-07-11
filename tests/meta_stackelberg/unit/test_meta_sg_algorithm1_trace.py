from collections import Counter

from meta_stackelberg.stackelberg.algorithm1 import MetaSGAlgorithm1


def test_algorithm1_uses_nd_k_na_with_paper_meanings_and_order() -> None:
    calls = []

    def sample_tasks(leader_iteration, count):
        calls.append(('sample', leader_iteration, count))
        return tuple(f'task-{index}' for index in range(count))

    def current(leader_iteration):
        return f'theta-{leader_iteration}'

    def adapt(task, meta_defender, leader_iteration):
        assert meta_defender == f'theta-{leader_iteration}'
        calls.append(('adapt', leader_iteration, task))
        return f'adapted-{leader_iteration}-{task}'

    def attacker_update(task, meta_defender, response_step):
        calls.append(('attacker', int(meta_defender.split('-')[1]), task, response_step))

    def defender_gradient(task, adapted, response):
        calls.append(('gradient', leader_iteration_from(adapted), task, response))
        return float(len(task))

    applied = []
    result = MetaSGAlgorithm1(N_D=2, K=3, N_A=4).run(
        sample_tasks=sample_tasks,
        current_defender=current,
        adapt_defender=adapt,
        update_attacker=attacker_update,
        estimate_defender_gradient=defender_gradient,
        apply_leader_update=lambda iteration, gradients: applied.append((iteration, gradients)),
    )

    counts = Counter(call[0] for call in calls)
    assert counts == {'sample': 2, 'adapt': 6, 'attacker': 24, 'gradient': 6}
    assert applied == [(0, (6.0, 6.0, 6.0)), (1, (6.0, 6.0, 6.0))]
    assert len(result.iterations) == 2
    assert all(len(item.tasks) == 3 for item in result.iterations)
    assert all(len(task.attacker_steps) == 4 for item in result.iterations for task in item.tasks)
    assert [event.kind for event in result.events[:7]] == [
        'sample_tasks', 'adapt_defender', 'attacker_update', 'attacker_update',
        'attacker_update', 'attacker_update', 'defender_gradient',
    ]


def test_algorithm1_uses_only_final_phi_na_as_response() -> None:
    used = []
    result = MetaSGAlgorithm1(N_D=1, K=1, N_A=3).run(
        sample_tasks=lambda iteration, count: ('rl',),
        current_defender=lambda iteration: 'theta-meta',
        adapt_defender=lambda task, meta, iteration: 'theta-adapted',
        update_attacker=lambda task, meta, step: f'phi-{step + 1}',
        estimate_defender_gradient=lambda task, adapted, response: used.append((task, response)) or 1.0,
        apply_leader_update=lambda iteration, gradients: None,
    )
    task = result.iterations[0].tasks[0]
    assert task.attacker_steps == ('phi-1', 'phi-2', 'phi-3')
    assert task.approximate_best_response == 'phi-3'
    assert used == [('rl', 'phi-3')]


def test_algorithm1_rejects_invalid_counts_and_wrong_task_batch_size() -> None:
    for kwargs in ({'N_D': 0, 'K': 1, 'N_A': 1}, {'N_D': 1, 'K': 0, 'N_A': 1}, {'N_D': 1, 'K': 1, 'N_A': 0}):
        try:
            MetaSGAlgorithm1(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted invalid Algorithm 1 counts')
    try:
        MetaSGAlgorithm1(N_D=1, K=2, N_A=1).run(
            sample_tasks=lambda iteration, count: ('only-one',),
            current_defender=lambda iteration: 'theta-meta',
            adapt_defender=lambda task, meta, iteration: None,
            update_attacker=lambda task, meta, step: None,
            estimate_defender_gradient=lambda task, adapted, response: None,
            apply_leader_update=lambda iteration, gradients: None,
        )
    except ValueError as error:
        assert 'K' in str(error)
    else:
        raise AssertionError('accepted wrong task batch size')


def leader_iteration_from(adapted: str) -> int:
    return int(adapted.split('-', 2)[1])
