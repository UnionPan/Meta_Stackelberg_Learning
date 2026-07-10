from types import SimpleNamespace

import numpy as np

from fl_sandbox.attacks.mixed_backdoor import MixedBackdoorAttack
from meta_sg.scripts.evaluate_meta_sg_direct import _scenarios
from meta_sg.scripts.run_meta_sg_pretraining import attack_domain_from_name


class RecordingAttack:
    def __init__(self, base_value):
        self.base_value = base_value
        self.seen_selected = []

    def observe_round(self, ctx):
        self.seen_selected.append(("observe", list(ctx.selected_attacker_ids)))

    def execute(self, ctx, attacker_action=None):
        self.seen_selected.append(("execute", list(ctx.selected_attacker_ids)))
        return [[np.asarray([self.base_value + attacker_id], dtype=np.float32)] for attacker_id in ctx.selected_attacker_ids]


def test_mixed_backdoor_routes_sampled_attackers_by_fixed_roster_and_preserves_selected_order():
    bfl = RecordingAttack(10)
    dba = RecordingAttack(20)
    rl = RecordingAttack(30)
    attack = MixedBackdoorAttack(
        total_attackers=6,
        bfl_attack=bfl,
        dba_attack=dba,
        rl_backdoor_attack=rl,
    )
    ctx = SimpleNamespace(
        selected_attacker_ids=[0, 2, 3, 5],
        old_weights=[np.asarray([0.0], dtype=np.float32)],
    )

    attack.observe_round(ctx)
    weights = attack.execute(ctx)

    assert [float(item[0][0]) for item in weights] == [10.0, 22.0, 23.0, 35.0]
    assert bfl.seen_selected == [("observe", [0]), ("execute", [0])]
    assert dba.seen_selected == [("observe", [2, 3]), ("execute", [2, 3])]
    assert rl.seen_selected == [("observe", [5]), ("execute", [5])]


def test_direct_evaluator_builds_clean_mixed_backdoor_scenario_set():
    args = SimpleNamespace(
        scenario_set="clean_mixed_backdoor",
        scenario_filter=None,
        seed=42,
        rl_seed=506,
        H=50,
        num_clients=30,
        num_attackers=6,
        rl_distribution_dir="",
        rl_policy_checkpoint="",
    )

    scenarios = _scenarios(args)

    assert [scenario.name for scenario in scenarios] == ["clean", "mixed_backdoor"]
    assert scenarios[1].attack_name == "mixed_backdoor"
    assert scenarios[1].patch["num_attackers"] == 6


def test_pretraining_domain_can_sample_clean_single_backdoors_and_mixed_backdoor():
    domain = attack_domain_from_name("clean_backdoor_mixed")

    assert [attack.name for attack in domain] == [
        "clean",
        "bfl",
        "dba",
        "rl_backdoor",
        "mixed_backdoor",
    ]
    assert [attack.objective for attack in domain] == [
        "clean",
        "targeted",
        "targeted",
        "targeted",
        "targeted",
    ]
