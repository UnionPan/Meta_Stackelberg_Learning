"""Compatibility wrapper for clean-vs-attack postprocess."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.postprocess.postprocess import main as unified_main


def main() -> None:
    unified_main(sys.argv[1:])


if __name__ == "__main__":
    main()
