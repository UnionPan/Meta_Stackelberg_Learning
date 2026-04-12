"""Compatibility wrapper for attack runs.

Prefer using ``main.py --attack_type <name>`` for new commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_sandbox.apps.run.main import main


if __name__ == "__main__":
    main(default_attack_type="ipm", description="Standalone attacker sandbox")
