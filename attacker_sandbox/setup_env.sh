#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_VARIANT="${1:-cpu}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r "${ROOT_DIR}/attacker_sandbox/requirements.txt"

case "${TORCH_VARIANT}" in
  cpu)
    python -m pip install torch torchvision
    ;;
  cu118)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ;;
  cu121)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ;;
  *)
    echo "Unsupported torch variant: ${TORCH_VARIANT}" >&2
    echo "Supported values: cpu, cu118, cu121" >&2
    exit 1
    ;;
esac

VENV_DIR="${VENV_DIR}" python - <<'PY'
import os
from pathlib import Path

venv_dir = Path(os.environ["VENV_DIR"])
script_path = venv_dir / "bin" / "tensorboard"
python_bin = venv_dir / "bin" / "python"

wrapper = f"""#!{python_bin}
import io
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


class _FilteredStderr(io.TextIOBase):
    _SUPPRESSED_PREFIXES = (
        "TensorFlow installation not found - running with reduced feature set.",
        "NOTE: Using experimental fast data loading logic. To disable, pass",
        '    "--load_fast=false" and report issues on GitHub. More details:',
        "    https://github.com/tensorflow/tensorboard/issues/4784",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):
        if any(s.startswith(prefix) for prefix in self._SUPPRESSED_PREFIXES):
            return len(s)
        return self._wrapped.write(s)

    def flush(self):
        return self._wrapped.flush()


sys.stderr = _FilteredStderr(sys.stderr)

if "--load_fast" not in sys.argv:
    sys.argv.append("--load_fast=false")

from tensorboard.main import run_main

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    sys.exit(run_main())
"""

script_path.write_text(wrapper)
script_path.chmod(0o755)
PY

echo "Environment ready in ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Verify torch with: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
