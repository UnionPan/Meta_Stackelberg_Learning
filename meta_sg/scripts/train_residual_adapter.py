"""Train a context-conditioned residual adapter from JSONL supervision."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_sg.learning.residual_adapter import fit_residual_adapter, save_residual_adapter


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--act-dim", type=int, required=True)
    parser.add_argument("--bound", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Train only on samples whose query selector accepted the residual.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    samples = _load_samples(Path(args.input_jsonl))
    if bool(args.accepted_only):
        samples = [sample for sample in samples if bool(sample.get("accepted", False))]
    adapter = fit_residual_adapter(
        samples,
        act_dim=int(args.act_dim),
        bound=float(args.bound),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    metrics = {
        "num_samples": len(samples),
        "act_dim": int(args.act_dim),
        "bound": float(args.bound),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "accepted_only": bool(args.accepted_only),
    }
    save_residual_adapter(adapter, args.output_checkpoint, metrics=metrics)
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_samples(path: Path) -> list[dict]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


if __name__ == "__main__":
    main()
