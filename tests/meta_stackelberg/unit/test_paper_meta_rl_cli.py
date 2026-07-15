from meta_stackelberg.experiments.run_paper_meta_rl import TASKS, build_parser


def test_meta_rl_cli_declares_independent_fixed_untargeted_domain() -> None:
    args = build_parser().parse_args([
        '--data-root', 'data',
        '--checkpoint', 'meta-rl.pt',
    ])
    assert args.T == 100
    assert args.K == 5
    assert args.H == 200
    assert args.l == 10
    assert TASKS == ('na', 'ipm', 'lmp')
