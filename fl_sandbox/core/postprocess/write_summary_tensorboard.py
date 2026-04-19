"""Write TensorBoard logs from a saved sandbox summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.core.postprocess.tensorboard_utils import (
    build_summary_writer,
    payload_to_series,
    write_standard_tensorboard_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TensorBoard logs from a saved summary.json")
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    with summary_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    config = payload.get("config", {})
    run_label = args.run_name or payload.get("attack_type", config.get("attack_type", "clean"))
    series = payload_to_series(payload)
    total_seconds = float(payload.get("total_seconds", 0.0))
    tb_dir = Path(args.tb_dir) if args.tb_dir else summary_path.parent.parent.parent / "runs" / summary_path.parent.name
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = build_summary_writer(tb_dir)
    writer.add_text("run/name", run_label, global_step=0)
    writer.add_scalar("run/total_seconds", total_seconds, 0)
    write_standard_tensorboard_run(
        writer,
        run_label,
        config,
        series,
        include_attack=run_label != "clean",
        total_seconds=total_seconds or None,
    )
    writer.flush()
    writer.close()
    print(f"TensorBoard logs written to: {tb_dir}")


if __name__ == "__main__":
    main()
