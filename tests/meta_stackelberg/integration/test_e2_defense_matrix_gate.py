from meta_stackelberg.experiments.defense_response_matrix import (
    evaluate_e2_gate,
    evaluate_task_matrix_gate,
)
from tests.meta_stackelberg.integration.test_backdoor_defense_matrix import (
    THRESHOLDS as BACKDOOR_THRESHOLDS,
    build_backdoor_matrix,
)
from tests.meta_stackelberg.integration.test_untargeted_defense_matrix import (
    THRESHOLDS as UNTARGETED_THRESHOLDS,
    build_untargeted_matrix,
)


REQUIRED_TASKS = ('delta-reversal', 'ipm', 'lmp', 'bfl', 'dba')


def test_total_e2_gate_has_all_tasks_active_dimensions_and_distinct_regions() -> None:
    matrices = (
        build_untargeted_matrix('delta-reversal'),
        build_untargeted_matrix('ipm'),
        build_untargeted_matrix('lmp'),
        build_backdoor_matrix('bfl'),
        build_backdoor_matrix('dba'),
    )
    task_gates = tuple(
        evaluate_task_matrix_gate(
            matrix,
            BACKDOOR_THRESHOLDS if matrix.task_id in ('bfl', 'dba') else UNTARGETED_THRESHOLDS,
        )
        for matrix in matrices
    )

    result = evaluate_e2_gate(
        matrices=matrices,
        task_gates=task_gates,
        required_task_ids=REQUIRED_TASKS,
    )

    assert result.passed
    assert set(result.task_ids) == set(REQUIRED_TASKS)
    assert set(result.independently_active_dimensions) == {'clip_radius', 'trim_ratio'}
    assert result.distinct_preferred_region_pairs
    assert result.failed_requirements == ()
