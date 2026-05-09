# fl_sandbox Attacks Phase 1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fl_sandbox.attacks` the only attacker API while preserving current attacker behavior and avoiding the Tianshou rewrite in this phase.

**Architecture:** Phase 1a is a layout and import migration only. Fixed attackers move into direct `fl_sandbox/attacks/*.py` files, geometry-search attackers move to the same flat package, and the current adaptive TD3 implementation moves unchanged into `fl_sandbox/attacks/rl_attacker/legacy_td3.py` behind a new `rl_attacker/attack.py` wrapper. Old `core/attacks`, `core/rl`, `attacks/vector`, `attacks/backdoor`, and `attacks/adaptive` paths are deleted after all imports and tests are updated.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, existing `fl_sandbox` runner/config/runtime modules. No Tianshou or Gymnasium dependency is added in Phase 1a.

---

## Scope Check

The approved spec covers multiple rollout stages. This plan covers **Phase 1a only**:

- Flat attacker layout.
- Package-level public API through `fl_sandbox.attacks`.
- Existing RL behavior preserved through `rl_attacker/legacy_td3.py`.
- Old attacker compatibility paths deleted.

Phase 1b, the SAC/TD3 Tianshou rewrite, needs a separate plan after Phase 1a lands and tests pass.

## File Map

Create:

- `fl_sandbox/attacks/ipm.py` — `craft_ipm`, `IPMAttack`.
- `fl_sandbox/attacks/lmp.py` — `_weights_to_vector`, `craft_lmp`, `LMPAttack`.
- `fl_sandbox/attacks/alie.py` — `craft_alie`, `ALIEAttack`.
- `fl_sandbox/attacks/signflip.py` — `SignFlipAttack`.
- `fl_sandbox/attacks/gaussian.py` — `GaussianAttack`.
- `fl_sandbox/attacks/bfl.py` — `BFLAttack`.
- `fl_sandbox/attacks/dba.py` — `DBAAttack`.
- `fl_sandbox/attacks/brl.py` — `BRLAttack`, `SelfGuidedBRLAttack`.
- `fl_sandbox/attacks/rl_attacker/__init__.py` — public RL attacker subpackage exports.
- `fl_sandbox/attacks/rl_attacker/attack.py` — `RLAttack` wrapper that delegates to `legacy_td3.PaperRLAttacker`.
- `fl_sandbox/attacks/rl_attacker/legacy_td3.py` — current `attacks/adaptive/td3_attacker.py` content, import paths updated.
- `fl_sandbox/attacks/rl_attacker/env.py` — current `attacks/adaptive/env.py` content, import paths updated.
- `fl_sandbox/attacks/rl_attacker/pz_env.py` — current `attacks/adaptive/pz_env.py` content, import paths updated.
- `tests/test_fl_sandbox_attacks_public_api.py` — package API and old-import guard tests.

Move:

- `fl_sandbox/attacks/adaptive/krum_geometry_search.py` to `fl_sandbox/attacks/krum_geometry_search.py`.
- `fl_sandbox/attacks/adaptive/clipped_median_geometry_search.py` to `fl_sandbox/attacks/clipped_median_geometry_search.py`.

Modify:

- `fl_sandbox/attacks/__init__.py` — re-export all public attacker classes and helpers from flat modules.
- `fl_sandbox/attacks/registry.py` — import flat classes directly and construct geometry attacks without wrapper classes.
- `fl_sandbox/__init__.py` — import `AttackerRLEnv` from `fl_sandbox.attacks.rl_attacker.env`.
- `fl_sandbox/run/run_experiment.py` — import `ATTACK_CHOICES` from `fl_sandbox.attacks`.
- `fl_sandbox/core/experiment_builders.py` — import `ATTACK_CHOICES` and `create_attack` from `fl_sandbox.attacks`.
- `fl_sandbox/core/experiment_service.py` — import `RLAttack` from `fl_sandbox.attacks`.
- `fl_sandbox/scripts/run_all_attacks_benchmark.py` — import `ATTACK_CHOICES` from `fl_sandbox.attacks`.
- `fl_sandbox/scripts/run_krum_geometry_search_compare.py` — import geometry search from `fl_sandbox.attacks`.
- `fl_sandbox/scripts/run_clipped_median_geometry_search_compare.py` — import geometry search from `fl_sandbox.attacks`.
- `fl_sandbox/README.md` — document `fl_sandbox.attacks` as the attacker API and `attacks/rl_attacker` as legacy TD3 location for Phase 1a.
- `tests/test_rl_attacker.py` — replace `fl_sandbox.core.attacks` and `fl_sandbox.core.rl` imports with new paths.
- `tests/test_attacker_sandbox_application.py` — replace `fl_sandbox.core.attacks.poisoning` import with `fl_sandbox.attacks`.
- `meta_sg` scripts and adapters that import attacker paths — migrate only attacker imports.

Delete:

- `fl_sandbox/core/attacks/`
- `fl_sandbox/core/rl/`
- `fl_sandbox/attacks/vector/`
- `fl_sandbox/attacks/backdoor/`
- `fl_sandbox/attacks/adaptive/`

Do not edit:

- `fl_sandbox/core/defender/`
- `fl_sandbox/aggregators/`
- `fl_sandbox/federation/runner.py`, except if an import-only change is required by a test failure.
- `fl_sandbox/config/schema.py`, except if an import-only change is required by a test failure.
- Tianshou/Gymnasium dependencies.

---

### Task 1: Add Public API and Old-Import Guard Tests

**Files:**
- Create: `tests/test_fl_sandbox_attacks_public_api.py`
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Write the failing package API tests**

Create `tests/test_fl_sandbox_attacks_public_api.py` with this content:

```python
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
    assert tuple(ATTACK_CHOICES) == supported_attack_types()
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
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/test_fl_sandbox_attacks_public_api.py -q
```

Expected: at least one failure because old import paths and missing flat exports still exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_fl_sandbox_attacks_public_api.py
git commit -m "test: capture attacker public api migration"
```

---

### Task 2: Split Model-Poisoning Attackers into Flat Modules

**Files:**
- Create: `fl_sandbox/attacks/ipm.py`
- Create: `fl_sandbox/attacks/lmp.py`
- Create: `fl_sandbox/attacks/alie.py`
- Create: `fl_sandbox/attacks/signflip.py`
- Create: `fl_sandbox/attacks/gaussian.py`
- Source: `fl_sandbox/attacks/vector/__init__.py`
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Create `ipm.py` from the current IPM code**

Move `craft_ipm` and `IPMAttack` from `fl_sandbox/attacks/vector/__init__.py` into `fl_sandbox/attacks/ipm.py`. The file should begin with:

```python
"""Inner Product Manipulation attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights
```

Keep the current `craft_ipm` body and `IPMAttack.execute` body unchanged.

- [ ] **Step 2: Create `lmp.py` from the current LMP code**

Move `_weights_to_vector`, `craft_lmp`, and `LMPAttack` into `fl_sandbox/attacks/lmp.py`. The file should begin with:

```python
"""Local Model Poisoning attacker."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader
from fl_sandbox.core.metrics import update_norm
```

Keep the current `craft_lmp` and `LMPAttack.execute` logic unchanged, including the local import of `median_aggregate`.

- [ ] **Step 3: Create `alie.py` from the current ALIE code**

Move `craft_alie` and `ALIEAttack` into `fl_sandbox/attacks/alie.py`. The file should begin with:

```python
"""A Little Is Enough model-poisoning attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights
```

Keep the current `craft_alie` and `ALIEAttack.execute` logic unchanged.

- [ ] **Step 4: Create `signflip.py` from the current SignFlip code**

Move `SignFlipAttack` into `fl_sandbox/attacks/signflip.py`. The file should begin with:

```python
"""Sign flipping model-poisoning attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights
```

Keep the current `SignFlipAttack.execute` logic unchanged.

- [ ] **Step 5: Create `gaussian.py` from the current Gaussian code**

Move `GaussianAttack` into `fl_sandbox/attacks/gaussian.py`. The file should begin with:

```python
"""Gaussian-noise attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights
```

Keep the current `GaussianAttack.execute` logic unchanged.

- [ ] **Step 6: Run focused import checks**

Run:

```bash
python - <<'PY'
from fl_sandbox.attacks.ipm import IPMAttack, craft_ipm
from fl_sandbox.attacks.lmp import LMPAttack, craft_lmp
from fl_sandbox.attacks.alie import ALIEAttack, craft_alie
from fl_sandbox.attacks.signflip import SignFlipAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
print(IPMAttack.attack_type, LMPAttack.attack_type, ALIEAttack.attack_type)
PY
```

Expected: imports succeed and the script prints dataclass field defaults, not an import error.

- [ ] **Step 7: Commit the split model-poisoning files**

```bash
git add fl_sandbox/attacks/ipm.py \
        fl_sandbox/attacks/lmp.py \
        fl_sandbox/attacks/alie.py \
        fl_sandbox/attacks/signflip.py \
        fl_sandbox/attacks/gaussian.py
git commit -m "refactor: split model poisoning attackers"
```

---

### Task 3: Split Backdoor Attackers into Flat Modules

**Files:**
- Create: `fl_sandbox/attacks/bfl.py`
- Create: `fl_sandbox/attacks/dba.py`
- Create: `fl_sandbox/attacks/brl.py`
- Source: `fl_sandbox/attacks/backdoor/__init__.py`
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Create `bfl.py`**

Move `BFLAttack` into `fl_sandbox/attacks/bfl.py`. The file should begin with:

```python
"""Fixed global-trigger backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader
```

Keep the current `BFLAttack.execute` logic unchanged.

- [ ] **Step 2: Create `dba.py`**

Move `DBAAttack` into `fl_sandbox/attacks/dba.py`. The file should begin with:

```python
"""Distributed backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader
```

Keep the current `DBAAttack.execute` logic unchanged.

- [ ] **Step 3: Create `brl.py`**

Move `_ACTION_LOW`, `_ACTION_HIGH`, `BRLAttack`, and `SelfGuidedBRLAttack` into `fl_sandbox/attacks/brl.py`. The file should begin with:

```python
"""Backdoor reinforcement-learning attackers."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from fl_sandbox.attacks.base import (
    SandboxAttack,
    Weights,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.core.metrics import update_norm
```

In `_ensure_policy`, replace the old adaptive import with:

```python
from fl_sandbox.attacks.rl_attacker.legacy_td3 import RLAttackerConfig, ReplayBuffer, TD3Agent
```

Keep the rest of `BRLAttack` and `SelfGuidedBRLAttack` logic unchanged.

- [ ] **Step 4: Run focused backdoor import checks**

Run:

```bash
python - <<'PY'
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.brl import BRLAttack, SelfGuidedBRLAttack
print(BFLAttack.attack_type, DBAAttack.attack_type, BRLAttack.attack_type, SelfGuidedBRLAttack.attack_type)
PY
```

Expected: imports succeed and print `bfl dba brl sgbrl`.

- [ ] **Step 5: Commit the split backdoor files**

```bash
git add fl_sandbox/attacks/bfl.py \
        fl_sandbox/attacks/dba.py \
        fl_sandbox/attacks/brl.py
git commit -m "refactor: split backdoor attackers"
```

---

### Task 4: Move RL Legacy TD3 and Geometry Search into New Paths

**Files:**
- Create: `fl_sandbox/attacks/rl_attacker/__init__.py`
- Create: `fl_sandbox/attacks/rl_attacker/attack.py`
- Create: `fl_sandbox/attacks/rl_attacker/legacy_td3.py`
- Create: `fl_sandbox/attacks/rl_attacker/env.py`
- Create: `fl_sandbox/attacks/rl_attacker/pz_env.py`
- Move: `fl_sandbox/attacks/adaptive/krum_geometry_search.py` to `fl_sandbox/attacks/krum_geometry_search.py`
- Move: `fl_sandbox/attacks/adaptive/clipped_median_geometry_search.py` to `fl_sandbox/attacks/clipped_median_geometry_search.py`
- Source: `fl_sandbox/attacks/adaptive/__init__.py`
- Source: `fl_sandbox/attacks/adaptive/td3_attacker.py`
- Source: `fl_sandbox/attacks/adaptive/env.py`
- Source: `fl_sandbox/attacks/adaptive/pz_env.py`
- Test: `tests/test_rl_attacker.py`

- [ ] **Step 1: Move geometry search files with history**

Run:

```bash
git mv fl_sandbox/attacks/adaptive/krum_geometry_search.py \
       fl_sandbox/attacks/krum_geometry_search.py
git mv fl_sandbox/attacks/adaptive/clipped_median_geometry_search.py \
       fl_sandbox/attacks/clipped_median_geometry_search.py
```

Expected: `git status --short` shows both files as renames or delete/add pairs.

- [ ] **Step 2: Create `rl_attacker` package directory**

Run:

```bash
mkdir -p fl_sandbox/attacks/rl_attacker
```

Create `fl_sandbox/attacks/rl_attacker/__init__.py`:

```python
"""Adaptive RL attacker package.

Phase 1a keeps the existing TD3 implementation in ``legacy_td3`` while the
public package path moves to ``fl_sandbox.attacks.rl_attacker``.
"""

from fl_sandbox.attacks.rl_attacker.attack import RLAttack
from fl_sandbox.attacks.rl_attacker.env import AttackerRLEnv
from fl_sandbox.attacks.rl_attacker.legacy_td3 import (
    ConvDenoiser,
    DecodedAction,
    GradientDistributionLearner,
    PaperRLAttacker,
    ProxyDatasetBuffer,
    ReplayBuffer,
    RLAttackerConfig,
    SimulatedFLEnv,
    TD3Agent,
    local_search_update,
)
from fl_sandbox.attacks.rl_attacker.pz_env import AttackerPolicyGymEnv, AttackerPolicyParallelEnv

__all__ = [
    "AttackerPolicyGymEnv",
    "AttackerPolicyParallelEnv",
    "AttackerRLEnv",
    "ConvDenoiser",
    "DecodedAction",
    "GradientDistributionLearner",
    "PaperRLAttacker",
    "ProxyDatasetBuffer",
    "RLAttack",
    "RLAttackerConfig",
    "ReplayBuffer",
    "SimulatedFLEnv",
    "TD3Agent",
    "local_search_update",
]
```

- [ ] **Step 3: Move current TD3 implementation to `legacy_td3.py`**

Copy the current contents of `fl_sandbox/attacks/adaptive/td3_attacker.py` into `fl_sandbox/attacks/rl_attacker/legacy_td3.py`. Then replace internal imports:

```python
from fl_sandbox.attacks.adaptive.pz_env import AttackerPolicyParallelEnv
```

with:

```python
from fl_sandbox.attacks.rl_attacker.pz_env import AttackerPolicyParallelEnv
```

Do not change class names, action decoding, reward logic, or training behavior in this task.

- [ ] **Step 4: Create the new RL wrapper in `attack.py`**

Create `fl_sandbox/attacks/rl_attacker/attack.py`:

```python
"""Public RL attacker wrapper for the attacker package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fl_sandbox.attacks.base import SandboxAttack


@dataclass
class RLAttack(SandboxAttack):
    """Paper-style RL attacker with online distribution and policy learning."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    config: Optional[object] = None
    name: str = "RL"
    attack_type: str = "rl"

    def __post_init__(self) -> None:
        from fl_sandbox.attacks.rl_attacker.legacy_td3 import PaperRLAttacker

        self._attacker = PaperRLAttacker(self.config)

    def observe_round(self, ctx) -> None:
        self._attacker.observe_round(ctx)

    def execute(self, ctx, attacker_action=None):
        return self._attacker.execute(ctx, attacker_action=attacker_action)

    def after_round(self, **kwargs):
        if hasattr(self._attacker, "after_round"):
            return self._attacker.after_round(**kwargs)
        return {}
```

- [ ] **Step 5: Move RL env wrappers**

Copy `fl_sandbox/attacks/adaptive/env.py` into `fl_sandbox/attacks/rl_attacker/env.py`. Replace imports:

```python
from fl_sandbox.attacks.adaptive.td3_attacker import RLAttackerConfig
```

with:

```python
from fl_sandbox.attacks.rl_attacker.legacy_td3 import RLAttackerConfig
```

Copy `fl_sandbox/attacks/adaptive/pz_env.py` into `fl_sandbox/attacks/rl_attacker/pz_env.py`. Replace the type-checking import:

```python
from fl_sandbox.attacks.adaptive.td3_attacker import RLAttackerConfig, SimulatedFLEnv
```

with:

```python
from fl_sandbox.attacks.rl_attacker.legacy_td3 import RLAttackerConfig, SimulatedFLEnv
```

- [ ] **Step 6: Run focused RL import checks**

Run:

```bash
python - <<'PY'
from fl_sandbox.attacks.rl_attacker import (
    AttackerRLEnv,
    GradientDistributionLearner,
    PaperRLAttacker,
    RLAttack,
    RLAttackerConfig,
)
from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack
from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack
print(RLAttack.attack_type, KrumGeometrySearchAttack.__name__, ClippedMedianGeometrySearchAttack.__name__)
PY
```

Expected: imports succeed and print `rl KrumGeometrySearchAttack ClippedMedianGeometrySearchAttack`.

- [ ] **Step 7: Commit RL and geometry moves**

```bash
git add fl_sandbox/attacks/rl_attacker \
        fl_sandbox/attacks/krum_geometry_search.py \
        fl_sandbox/attacks/clipped_median_geometry_search.py
git commit -m "refactor: move adaptive attackers to new paths"
```

---

### Task 5: Rebuild the Attacks Public API and Registry

**Files:**
- Modify: `fl_sandbox/attacks/__init__.py`
- Modify: `fl_sandbox/attacks/registry.py`
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Replace `attacks/__init__.py` with flat public exports**

Use this content for `fl_sandbox/attacks/__init__.py`:

```python
"""Sandbox attack implementations and public factory API."""

from fl_sandbox.attacks.alie import ALIEAttack, craft_alie
from fl_sandbox.attacks.base import (
    SandboxAttack,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.brl import BRLAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
from fl_sandbox.attacks.ipm import IPMAttack, craft_ipm
from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack
from fl_sandbox.attacks.lmp import LMPAttack, craft_lmp
from fl_sandbox.attacks.registry import ATTACK_CHOICES, create_attack, supported_attack_types
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.signflip import SignFlipAttack
from fl_sandbox.core.runtime import Weights

__all__ = [
    "ATTACK_CHOICES",
    "ALIEAttack",
    "BFLAttack",
    "BRLAttack",
    "ClippedMedianGeometrySearchAttack",
    "DBAAttack",
    "GaussianAttack",
    "IPMAttack",
    "KrumGeometrySearchAttack",
    "LMPAttack",
    "RLAttack",
    "SandboxAttack",
    "SelfGuidedBRLAttack",
    "SignFlipAttack",
    "Weights",
    "bounded_boost",
    "bounded_local_epochs",
    "bounded_local_lr",
    "craft_alie",
    "craft_ipm",
    "craft_lmp",
    "create_attack",
    "get_model_weights",
    "set_model_weights",
    "supported_attack_types",
    "train_on_loader",
]
```

- [ ] **Step 2: Replace `attacks/registry.py` imports and construction**

In `fl_sandbox/attacks/registry.py`, replace imports with:

```python
from fl_sandbox.attacks.alie import ALIEAttack
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.brl import BRLAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.gaussian import GaussianAttack
from fl_sandbox.attacks.ipm import IPMAttack
from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack
from fl_sandbox.attacks.lmp import LMPAttack
from fl_sandbox.attacks.rl_attacker import RLAttack
from fl_sandbox.attacks.signflip import SignFlipAttack
```

Replace the `rl` branch local import with:

```python
from fl_sandbox.attacks.rl_attacker.legacy_td3 import RLAttackerConfig
```

Replace the geometry branches with:

```python
    if attack_type == "krum_geometry_search":
        from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchConfig
        return KrumGeometrySearchAttack(config=KrumGeometrySearchConfig())
    if attack_type == "clipped_median_geometry_search":
        from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchConfig
        return ClippedMedianGeometrySearchAttack(config=ClippedMedianGeometrySearchConfig())
```

Keep `ATTACK_CHOICES` unchanged.

- [ ] **Step 3: Run public API tests**

Run:

```bash
pytest tests/test_fl_sandbox_attacks_public_api.py::test_package_level_attacker_imports -q
pytest tests/test_fl_sandbox_attacks_public_api.py::test_create_attack_constructs_every_non_clean_attack -q
```

Expected: both tests pass or fail only because remaining old modules still exist in imports that are addressed in Task 6.

- [ ] **Step 4: Commit public API updates**

```bash
git add fl_sandbox/attacks/__init__.py fl_sandbox/attacks/registry.py
git commit -m "refactor: expose flat attacker api"
```

---

### Task 6: Migrate All In-Repo Attacker Imports

**Files:**
- Modify: `fl_sandbox/__init__.py`
- Modify: `fl_sandbox/run/run_experiment.py`
- Modify: `fl_sandbox/core/experiment_builders.py`
- Modify: `fl_sandbox/core/experiment_service.py`
- Modify: `fl_sandbox/scripts/run_all_attacks_benchmark.py`
- Modify: `fl_sandbox/scripts/run_krum_geometry_search_compare.py`
- Modify: `fl_sandbox/scripts/run_clipped_median_geometry_search_compare.py`
- Modify: `fl_sandbox/scripts/run_rlfl_benchmark.py` if it imports old attacker paths.
- Modify: `tests/test_rl_attacker.py`
- Modify: `tests/test_attacker_sandbox_application.py`
- Modify: `meta_sg` files with old attacker imports.
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Inspect current old imports**

Run:

```bash
rg -n "fl_sandbox\\.core\\.attacks|fl_sandbox\\.core\\.rl|fl_sandbox\\.attacks\\.vector|fl_sandbox\\.attacks\\.backdoor|fl_sandbox\\.attacks\\.adaptive" \
  fl_sandbox meta_sg tests
```

Expected: output lists old import paths to migrate.

- [ ] **Step 2: Apply deterministic import replacements**

Make these replacements:

```text
from fl_sandbox.core.attacks import ATTACK_CHOICES
→ from fl_sandbox.attacks import ATTACK_CHOICES

from fl_sandbox.core.attacks import create_attack
→ from fl_sandbox.attacks import create_attack

from fl_sandbox.core.attacks.rl import RLAttack
→ from fl_sandbox.attacks import RLAttack

from fl_sandbox.core.attacks.poisoning import craft_lmp
→ from fl_sandbox.attacks import craft_lmp

from fl_sandbox.core.attacks import RLAttack
→ from fl_sandbox.attacks import RLAttack

from fl_sandbox.core.rl import (
    ...
)
→ from fl_sandbox.attacks.rl_attacker import (
    ...
)

from fl_sandbox.attacks.adaptive import RLAttack
→ from fl_sandbox.attacks import RLAttack

from fl_sandbox.attacks.adaptive.td3_attacker import RLAttackerConfig
→ from fl_sandbox.attacks.rl_attacker.legacy_td3 import RLAttackerConfig

from fl_sandbox.attacks.adaptive.td3_attacker_v2 import RLAttackerConfigV2
→ remove this import and the related V2 usage, because `td3_attacker_v2.py` is already deleted in the working tree

from fl_sandbox.attacks.adaptive.krum_geometry_search import KrumGeometrySearchAttack, KrumGeometrySearchConfig
→ from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack, KrumGeometrySearchConfig

from fl_sandbox.attacks.adaptive.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack, ClippedMedianGeometrySearchConfig
→ from fl_sandbox.attacks.clipped_median_geometry_search import ClippedMedianGeometrySearchAttack, ClippedMedianGeometrySearchConfig

from fl_sandbox.attacks.backdoor import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack
→ from fl_sandbox.attacks import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack

from fl_sandbox.attacks.vector import IPMAttack, LMPAttack
→ from fl_sandbox.attacks import IPMAttack, LMPAttack
```

For `fl_sandbox/__init__.py`, replace:

```python
from fl_sandbox.attacks.adaptive.env import AttackerRLEnv
```

with:

```python
from fl_sandbox.attacks.rl_attacker.env import AttackerRLEnv
```

- [ ] **Step 3: Re-run old import search**

Run:

```bash
rg -n "fl_sandbox\\.core\\.attacks|fl_sandbox\\.core\\.rl|fl_sandbox\\.attacks\\.vector|fl_sandbox\\.attacks\\.backdoor|fl_sandbox\\.attacks\\.adaptive" \
  fl_sandbox meta_sg tests
```

Expected: no matches in Python source files except this plan/spec document if the search includes `docs/`.

- [ ] **Step 4: Run old-import guard test**

Run:

```bash
pytest tests/test_fl_sandbox_attacks_public_api.py::test_no_old_attacker_import_paths_remain_in_python_sources -q
```

Expected: pass.

- [ ] **Step 5: Commit import migration**

```bash
git add fl_sandbox meta_sg tests
git commit -m "refactor: migrate attacker imports to public api"
```

---

### Task 7: Delete Old Attacker Paths

**Files:**
- Delete: `fl_sandbox/core/attacks/`
- Delete: `fl_sandbox/core/rl/`
- Delete: `fl_sandbox/attacks/vector/`
- Delete: `fl_sandbox/attacks/backdoor/`
- Delete: `fl_sandbox/attacks/adaptive/`
- Test: `tests/test_fl_sandbox_attacks_public_api.py`

- [ ] **Step 1: Remove tracked old directories**

Run:

```bash
git rm -r fl_sandbox/core/attacks \
          fl_sandbox/core/rl \
          fl_sandbox/attacks/vector \
          fl_sandbox/attacks/backdoor \
          fl_sandbox/attacks/adaptive
```

Expected: tracked Python files are staged for deletion. Ignored `__pycache__` directories may remain on disk; do not add them.

- [ ] **Step 2: Verify old paths are gone from git**

Run:

```bash
git ls-files fl_sandbox/core/attacks fl_sandbox/core/rl fl_sandbox/attacks/vector fl_sandbox/attacks/backdoor fl_sandbox/attacks/adaptive
```

Expected: no output.

- [ ] **Step 3: Run package API tests**

Run:

```bash
pytest tests/test_fl_sandbox_attacks_public_api.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit old path deletion**

```bash
git add -u fl_sandbox/core/attacks \
          fl_sandbox/core/rl \
          fl_sandbox/attacks/vector \
          fl_sandbox/attacks/backdoor \
          fl_sandbox/attacks/adaptive
git commit -m "refactor: remove old attacker import paths"
```

---

### Task 8: Update README and Developer Notes

**Files:**
- Modify: `fl_sandbox/README.md`
- Test: manual grep checks

- [ ] **Step 1: Update README folder layout**

In `fl_sandbox/README.md`, replace the current `core/` bullet text with language that keeps core as orchestration/runtime infrastructure, not attacker implementation storage:

```markdown
- `attacks/`
  Public attacker subsystem. External code should import attackers and the
  attack factory from `fl_sandbox.attacks`. Fixed attackers live as direct
  files, and the adaptive RL attacker lives under `attacks/rl_attacker/`.
- `core/`
  Shared runtime helpers, experiment builders, metrics, and postprocess entry
  points. Attacker implementations do not live under `core/`.
```

- [ ] **Step 2: Update current status**

In the Current Status section, replace the RL bullet with:

```markdown
- paper-style legacy TD3 `RL` attacker preserved under `attacks/rl_attacker/`
  for Phase 1a; the next phase replaces the hand-written TD3 backend with
  Tianshou SAC/TD3 trainers
```

- [ ] **Step 3: Verify README has no old attacker path language**

Run:

```bash
rg -n "core/attacks|core/rl|attacks/vector|attacks/backdoor|attacks/adaptive" fl_sandbox/README.md
```

Expected: no matches.

- [ ] **Step 4: Commit README update**

```bash
git add fl_sandbox/README.md
git commit -m "docs: document attacker public api"
```

---

### Task 9: Run Phase 1a Verification

**Files:**
- Test-only task.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_fl_sandbox_attacks_public_api.py \
       tests/test_attacker_sandbox_application.py \
       tests/test_rl_attacker.py -q
```

Expected: tests pass. If `tests/test_rl_attacker.py` fails because it still expects `core.rl`, fix the import in the test and rerun this command.

- [ ] **Step 2: Run import smoke check**

Run:

```bash
python - <<'PY'
from fl_sandbox.attacks import (
    ATTACK_CHOICES,
    BFLAttack,
    IPMAttack,
    RLAttack,
    create_attack,
)
from fl_sandbox.attacks.rl_attacker import RLAttackerConfig, AttackerRLEnv
print("attack choices:", ",".join(ATTACK_CHOICES))
print(IPMAttack.__name__, BFLAttack.__name__, RLAttack.__name__, RLAttackerConfig.__name__, AttackerRLEnv.__name__, create_attack is not None)
PY
```

Expected: command exits with code 0 and prints attacker choices plus class names.

- [ ] **Step 3: Run the Phase 1a smoke experiment**

Run:

```bash
python fl_sandbox/run/run_experiment.py \
  --attack_type ipm \
  --rounds 1 \
  --num_clients 4 \
  --num_attackers 1 \
  --max_client_samples_per_client 16 \
  --max_eval_samples 32 \
  --num_workers 0
```

Expected: experiment completes one round and prints completion lines. If the local CLI does not support `--max_client_samples_per_client` or `--max_eval_samples`, rerun without those two flags and record that the current parser lacks the faster smoke knobs.

- [ ] **Step 4: Final old-path search**

Run:

```bash
rg -n "fl_sandbox\\.core\\.attacks|fl_sandbox\\.core\\.rl|fl_sandbox\\.attacks\\.vector|fl_sandbox\\.attacks\\.backdoor|fl_sandbox\\.attacks\\.adaptive" \
  fl_sandbox meta_sg tests
```

Expected: no matches in Python source files.

- [ ] **Step 5: Commit verification fixes if any**

If Step 1 through Step 4 required import-only fixes, commit them:

```bash
git add fl_sandbox meta_sg tests
git commit -m "test: verify attacker api migration"
```

If no files changed, do not create an empty commit.

---

### Task 10: Prepare Phase 1b Handoff Notes

**Files:**
- Modify: `fl_sandbox/docs/superpowers/specs/2026-05-09-fl-sandbox-attacks-restructure-design.md` only if the implementation discovers a spec mismatch.
- Create: no file by default.

- [ ] **Step 1: Confirm legacy TD3 is isolated**

Run:

```bash
rg -n "TD3Agent|ReplayBuffer|Actor|Critic|legacy_td3" fl_sandbox/attacks
```

Expected: hand-written RL primitives appear only in `fl_sandbox/attacks/rl_attacker/legacy_td3.py` and imports from `fl_sandbox/attacks/rl_attacker/attack.py`, `env.py`, `pz_env.py`, or `brl.py`.

- [ ] **Step 2: Confirm Tianshou is not required yet**

Run:

```bash
python - <<'PY'
import sys
import fl_sandbox.attacks
assert "tianshou" not in sys.modules
print("tianshou not imported")
PY
```

Expected: prints `tianshou not imported`.

- [ ] **Step 3: Record Phase 1b start condition in the final implementation summary**

The implementation summary should include this exact sentence:

```text
Phase 1b can now replace fl_sandbox.attacks.rl_attacker.legacy_td3 with the Tianshou-backed SAC/TD3 trainer package described in the spec.
```

This is not a code change.

