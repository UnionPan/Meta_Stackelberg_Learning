from meta_stackelberg.experiments.run_paper_meta_rl import (
    GLOBAL_TASKS,
    TASKS,
    build_parser,
)


def test_meta_rl_cli_declares_independent_fixed_table4_domain() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
    ])
    assert args.T == 100
    assert args.K == 5
    assert args.H == 200
    assert args.l == 10
    assert TASKS == ('na', 'ipm', 'lmp', 'bfl', 'dba')
    assert args.backdoor_poison_fraction == 1.0
    assert args.td3_batch_size == 256
    assert args.learning_starts == 100
    assert args.meta_update_step is None
    assert args.post_defense_mode == 'neuroclip'
    assert args.materialize_mnist is False
    assert (args.parallel_tasks, args.parallel_clients, args.cpu_threads) == (
        1, 1, 0,
    )
    assert args.deterministic_torch is False
    assert (args.workers, args.untargeted_attackers, args.sample_size) == (
        100, 20, 10,
    )


def test_meta_rl_cli_declares_global_model_poisoning_domain() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
        '--task-domain', 'global',
        '--attack-domain', 'attack-domain.pt',
        '--task-sampler', 'balanced',
    ])
    assert GLOBAL_TASKS == (
        'na', 'ipm', 'lmp', 'rl-krum', 'rl-clipmed',
    )
    assert args.task_sampler == 'balanced'


def test_meta_rl_cli_accepts_declared_reptile_stabilization_step() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
        '--meta-update-step', '0.1',
    ])
    assert args.meta_update_step == 0.1


def test_meta_rl_cli_accepts_scaled_client_topology() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
        '--workers', '20',
        '--untargeted-attackers', '4',
        '--sample-size', '4',
    ])
    assert (args.workers, args.untargeted_attackers, args.sample_size) == (
        20, 4, 4,
    )


def test_meta_rl_cli_can_disable_neuroclip_for_global_study() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
        '--post-defense-mode', 'identity',
        '--materialize-mnist',
        '--parallel-tasks', '5',
        '--parallel-clients', '4',
        '--cpu-threads', '1',
        '--deterministic-torch',
    ])
    assert args.post_defense_mode == 'identity'
    assert args.materialize_mnist is True
    assert (args.parallel_tasks, args.parallel_clients, args.cpu_threads) == (
        5, 4, 1,
    )
    assert args.deterministic_torch is True
