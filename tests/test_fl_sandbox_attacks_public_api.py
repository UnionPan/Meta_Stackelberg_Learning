from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_attacker_config(attack_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        type=attack_type,
        ipm_scaling=2.0,
        lmp_scale=5.0,
        alie_tau=1.5,
        gaussian_sigma=0.01,
        bfl_poison_frac=1.0,
        dba_poison_frac=0.5,
        dba_num_sub_triggers=4,
        attacker_action=(0.0, 0.0, 0.0),
        rl_distribution_steps=0,
        rl_attack_start_round=0,
        rl_policy_train_end_round=0,
        rl_inversion_steps=1,
        rl_reconstruction_batch_size=1,
        rl_policy_train_episodes_per_round=1,
        rl_simulator_horizon=1,
    )


def test_package_level_attacker_imports():
    from fl_sandbox.attacks import (
        ALIEAttack,
        ATTACK_CHOICES,
        BFLAttack,
        BRLAttack,
        ClippedMedianGeometrySearchAttack,
        DBAAttack,
        GaussianAttack,
        IPMAttack,
        KrumGeometrySearchAttack,
        LMPAttack,
        RLAttack,
        SandboxAttack,
        SelfGuidedBRLAttack,
        SignFlipAttack,
        create_attack,
        supported_attack_types,
    )

    assert "rl" in ATTACK_CHOICES
    assert ATTACK_CHOICES == supported_attack_types()
    assert issubclass(IPMAttack, SandboxAttack)
    assert issubclass(LMPAttack, SandboxAttack)
    assert issubclass(ALIEAttack, SandboxAttack)
    assert issubclass(SignFlipAttack, SandboxAttack)
    assert issubclass(GaussianAttack, SandboxAttack)
    assert issubclass(BFLAttack, SandboxAttack)
    assert issubclass(DBAAttack, SandboxAttack)
    assert issubclass(BRLAttack, SandboxAttack)
    assert issubclass(SelfGuidedBRLAttack, SandboxAttack)
    assert issubclass(RLAttack, SandboxAttack)
    assert issubclass(KrumGeometrySearchAttack, SandboxAttack)
    assert issubclass(ClippedMedianGeometrySearchAttack, SandboxAttack)
    assert create_attack(_default_attacker_config("clean")) is None


def test_create_attack_constructs_every_non_clean_attack():
    from fl_sandbox.attacks import ATTACK_CHOICES, create_attack

    for attack_type in ATTACK_CHOICES:
        attack = create_attack(_default_attacker_config(attack_type))
        if attack_type == "clean":
            assert attack is None
        else:
            assert attack is not None
            assert attack.attack_type == attack_type


def test_no_old_attacker_import_paths_remain_in_python_sources():
    banned_prefixes = (
        "fl_sandbox.core.attacks",
        "fl_sandbox.core.rl",
        "fl_sandbox.attacks.vector",
        "fl_sandbox.attacks.backdoor",
        "fl_sandbox.attacks.adaptive",
    )
    allowed_files = {
        Path("tests/test_fl_sandbox_attacks_public_api.py"),
    }

    offenders: list[tuple[str, str]] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel_path = path.relative_to(PROJECT_ROOT)
        if rel_path in allowed_files or "__pycache__" in rel_path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(banned_prefixes):
                        offenders.append((str(rel_path), alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(banned_prefixes):
                    offenders.append((str(rel_path), node.module))

    assert offenders == []
