#!/usr/bin/env python3
"""Record per-round 4D base-policy FL curves for the old full-mix checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.scripts import evaluate_meta_sg_direct as direct


DEFAULT_RUN_DIR = (
    "runs/meta_sg_goal/"
    "4d_both_clean_global_backdoor_mixed_h200_t100_k8_l10_c30_a6/"
    "20260707-000755"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--H", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=502)
    parser.add_argument("--rl-seed", type=int, default=506)
    parser.add_argument("--num-clients", type=int, default=30)
    parser.add_argument("--num-attackers", type=int, default=6)
    parser.add_argument("--subsample-rate", type=float, default=0.2)
    parser.add_argument("--client-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--scenarios",
        default="clean,ipm,lmp,rl,bfl,dba,rl_backdoor,mixed_backdoor",
        help="Comma-separated scenario names.",
    )
    return parser.parse_args(argv)


def _direct_args(args: argparse.Namespace, *, checkpoint: str, scenario_filter: str) -> argparse.Namespace:
    argv = [
        "--checkpoint",
        checkpoint,
        "--output-json",
        str(Path(args.output_dir) / "_unused_direct_eval.json"),
        "--scenario-set",
        "clean_global_backdoor_mixed",
        "--scenario-filter",
        scenario_filter,
        "--H",
        str(args.H),
        "--num-clients",
        str(args.num_clients),
        "--num-attackers",
        str(args.num_attackers),
        "--subsample-rate",
        str(args.subsample_rate),
        "--client-samples",
        str(args.client_samples),
        "--eval-samples",
        str(args.eval_samples),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--rl-seed",
        str(args.rl_seed),
        "--defender-third-action",
        "both",
        "--post-defense-mode",
        "model_aware_neuroclip",
        "--server-lr-min",
        "0.0",
        "--server-lr-max",
        "1.0",
        "--lambda-bd",
        "1.0",
        "--attacker-source",
        "native",
    ]
    return direct.parse_args(argv)


def _load_defender(args: argparse.Namespace, direct_args: argparse.Namespace, checkpoint: str) -> TD3Agent:
    device = direct._resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    probe = direct.FLSandboxCoordinatorAdapter(
        direct._sandbox_config(direct_args, "clean", seed=args.seed, patch={"num_attackers": 0})
    )
    obs_dim = direct.BSMGEnv(
        coordinator=probe,
        attack_type=direct._attack_type("clean"),
        attack_strategy=None,
        defense_strategy=direct.PaperDefenseStrategy(),
        config=direct._bsmg_config(direct_args),
        evaluator=None,
        post_training_evaluator=direct._post_training_evaluator_for_args(direct_args, probe),
    ).reset(seed=args.seed).shape[0]

    defender = TD3Agent(
        int(obs_dim),
        direct._defender_action_dim(direct_args),
        TD3Config(hidden_dim=args.hidden_dim, batch_size=16, buffer_capacity=4096, warmup_steps=0),
        device=device,
    )
    defender.load(direct._checkpoint_path(checkpoint))
    return defender


def _run_scenario(args: argparse.Namespace, direct_args: argparse.Namespace, defender: TD3Agent) -> list[dict]:
    scenario = direct._scenarios(direct_args)[0]
    env = direct._make_env(direct_args, scenario, seed=int(scenario.seed), horizon=int(args.H))
    obs = env.reset(seed=int(scenario.seed))
    if hasattr(defender, "reset"):
        defender.reset()
    records: list[dict] = []
    for _ in range(int(args.H)):
        raw_action = defender.get_action(obs, noise=0.0)
        attacker_action = direct._attacker_action_for_source(direct_args, source="native")
        obs, reward_d, reward_a, done, info = env.step(raw_action, attacker_action)
        decision = info.get("defense_decision")
        clean_acc = float(info.get("clean_acc", float("nan")))
        backdoor_acc = float(info.get("backdoor_acc", float("nan")))
        records.append(
            {
                "scenario": scenario.name,
                "attack_type": scenario.attack_name,
                "round": int(info.get("round", len(records) + 1)),
                "clean_acc": clean_acc,
                "backdoor_acc": backdoor_acc,
                "defense_score": clean_acc - float(direct_args.lambda_bd) * backdoor_acc,
                "defender_reward": float(reward_d),
                "attacker_reward": float(reward_a),
                "alpha": float(decision.norm_bound_alpha) if decision is not None else float("nan"),
                "beta": float(decision.trimmed_mean_beta) if decision is not None else float("nan"),
                "neuroclip": float(decision.neuroclip_epsilon)
                if decision is not None and decision.neuroclip_epsilon is not None
                else float("nan"),
                "server_lr": float(decision.server_lr)
                if decision is not None and decision.server_lr is not None
                else float("nan"),
            }
        )
        if done:
            break
    return records


def _write_outputs(records: list[dict], output_dir: Path, *, checkpoint: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "base_4d_round_curves.csv"
    json_path = output_dir / "base_4d_round_curves.json"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    df = pd.DataFrame(records)
    svg_path = output_dir / "base_4d_round_curves.svg"
    _plot_curves(df, svg_path)
    html_path = output_dir / "base_4d_round_curves.html"
    _write_html(df, html_path, checkpoint=checkpoint, csv_path=csv_path, svg_path=svg_path)
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"svg={svg_path}")
    print(f"html={html_path}")


def _plot_curves(df: pd.DataFrame, svg_path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("4D Base Initial Policy: Per-Round FL Curves", fontsize=14)
    metrics = [
        ("clean_acc", "Clean Accuracy", (0, 1.02)),
        ("backdoor_acc", "ASR / Backdoor Accuracy", (0, 1.02)),
        ("defense_score", "Defense Score: clean - ASR", None),
        ("defender_reward", "Defender Reward", None),
    ]
    for ax, (metric, title, ylim) in zip(axes.ravel(), metrics):
        for scenario, sub in df.groupby("scenario", sort=False):
            ax.plot(sub["round"], sub[metric], label=str(scenario), linewidth=1.15)
        ax.set_title(title)
        ax.set_xlabel("FL round")
        if ylim:
            ax.set_ylim(*ylim)
        if metric in {"defense_score", "defender_reward"}:
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax.legend(ncol=2, fontsize=7)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)


def _write_html(df: pd.DataFrame, html_path: Path, *, checkpoint: str, csv_path: Path, svg_path: Path) -> None:
    final = df.sort_values("round").groupby("scenario", sort=False).tail(1).copy()
    rows = []
    for _, row in final.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.scenario))}</td>"
            f"<td>{float(row.clean_acc):.4f}</td>"
            f"<td>{float(row.backdoor_acc):.4f}</td>"
            f"<td>{float(row.defense_score):.4f}</td>"
            f"<td>{float(row.defender_reward):.4f}</td>"
            f"<td>{float(row.alpha):.4f}</td>"
            f"<td>{float(row.beta):.4f}</td>"
            f"<td>{float(row.neuroclip):.4f}</td>"
            f"<td>{float(row.server_lr):.4f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>4D Base Initial Policy FL Curves</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    .meta {{ color: #5f6b7a; font-size: 13px; line-height: 1.45; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f6; }}
    img {{ width: 100%; max-width: 1200px; border: 1px solid #d9dee7; }}
    code {{ background: #eef2f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>4D Base Initial Policy: Per-Round FL Curves</h1>
  <div class="meta">
    Checkpoint: <code>{escape(checkpoint)}</code><br>
    Curves: 8 scenarios × 200 FL rounds. No adaptation. 4D action = alpha, beta, neuroclip, server_lr.<br>
    CSV: <code>{escape(str(csv_path))}</code>
  </div>
  <img src="{escape(svg_path.name)}" alt="4D base per-round curves">
  <h2>Final Round Summary</h2>
  <table>
    <thead><tr><th>Scenario</th><th>Clean</th><th>ASR</th><th>Score</th><th>Reward</th><th>Alpha</th><th>Beta</th><th>NeuroClip</th><th>Server LR</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or str(run_dir / "final")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(run_dir / "eval" / f"base_4d_round_curves_{stamp}")
    scenarios = [item.strip() for item in str(args.scenarios).split(",") if item.strip()]
    all_records: list[dict] = []
    defender = None
    for scenario in scenarios:
        dargs = _direct_args(args, checkpoint=checkpoint, scenario_filter=scenario)
        if defender is None:
            defender = _load_defender(args, dargs, checkpoint)
        all_records.extend(_run_scenario(args, dargs, defender))
    _write_outputs(all_records, Path(args.output_dir), checkpoint=checkpoint)


if __name__ == "__main__":
    main()
