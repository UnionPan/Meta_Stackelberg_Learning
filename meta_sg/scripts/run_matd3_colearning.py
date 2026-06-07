"""Run MATD3 attacker/defender co-learning on BSMGEnv."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import TD3Config
from meta_sg.learning.matd3 import JointReplayBuffer, MATD3AgentPair
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("stub", "fl_sandbox"), default="stub")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="runs/matd3_colearning")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=10_000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--subsample-rate", type=float, default=1.0)
    parser.add_argument("--client-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--compare-baselines", nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-horizon", type=int, default=0)
    parser.add_argument("--attack-name", choices=tuple(ATTACK_DOMAIN), default="rl")
    parser.add_argument("--lambda-bd", type=float, default=0.0)
    parser.add_argument("--reward-mode", choices=("loss", "accuracy"), default="loss")
    parser.add_argument("--action-prior-weight", type=float, default=0.0)
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument(
        "--comparison-mode",
        choices=("action_baselines", "fixed_attacker_defender", "fixed_defender", "best_response"),
        default="action_baselines",
    )
    parser.add_argument("--fixed-attacker-action", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--fixed-defender-action", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--defender-action-low", nargs=3, type=float, default=[-1.0, -1.0, -1.0])
    parser.add_argument("--defender-action-high", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--br-train-episodes", type=int, default=5)
    parser.add_argument("--br-train-horizon", type=int, default=0)
    parser.add_argument("--br-train-steps", type=int, default=0)
    parser.add_argument("--br-stage-steps", nargs="*", type=int, default=[])
    parser.add_argument("--br-eval-episodes", type=int, default=3)
    parser.add_argument("--br-eval-horizon", type=int, default=0)
    parser.add_argument("--br-updates-per-step", type=int, default=1)
    parser.add_argument("--br-exploration-noise", type=float, default=0.15)
    parser.add_argument("--br-attacker-init", choices=("learned", "fresh"), default="learned")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / f"{args.backend}_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(args, seed=args.seed)
    obs = env.reset(seed=args.seed)
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.exploration_noise),
    )
    pair = MATD3AgentPair(
        obs_dim=int(obs.shape[0]),
        defender_act_dim=env.act_dim,
        attacker_act_dim=env.act_dim,
        config=td3_cfg,
        device=torch.device(args.device),
    )
    buffer = JointReplayBuffer(
        capacity=int(args.buffer_capacity),
        obs_dim=int(obs.shape[0]),
        defender_act_dim=env.act_dim,
        attacker_act_dim=env.act_dim,
    )

    rows = _run_matd3(args, env, pair, buffer)

    csv_path = output_dir / "rounds.csv"
    _write_rows(csv_path, rows)
    torch.save(
        {
            "defender_actor": pair.defender.actor.state_dict(),
            "attacker_actor": pair.attacker.actor.state_dict(),
            "td3_config": td3_cfg.__dict__,
            "obs_dim": int(obs.shape[0]),
            "act_dim": int(env.act_dim),
        },
        output_dir / "matd3_policy.pt",
    )
    del env
    gc.collect()
    summary = _summarize(args, rows, output_dir)
    eval_rows = _evaluate_matd3(args, pair)
    eval_dir = output_dir / "eval_matd3"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(eval_dir / "rounds.csv", eval_rows)
    summary["eval"] = _summarize(args, eval_rows, eval_dir)
    baseline_summaries = {}
    for baseline in args.compare_baselines:
        baseline_rows = _run_baseline(args, baseline)
        baseline_dir = output_dir / f"baseline_{baseline}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(baseline_dir / "rounds.csv", baseline_rows)
        baseline_summary = _summarize(args, baseline_rows, baseline_dir)
        baseline_eval_rows = _run_baseline(args, baseline, eval_mode=True)
        baseline_eval_dir = baseline_dir / "eval"
        baseline_eval_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(baseline_eval_dir / "rounds.csv", baseline_eval_rows)
        baseline_summary["eval"] = _summarize(args, baseline_eval_rows, baseline_eval_dir)
        baseline_summaries[baseline] = baseline_summary
    if baseline_summaries:
        summary["baselines"] = baseline_summaries
        summary["comparison"] = compare_with_baselines(summary, baseline_summaries)
    if args.comparison_mode == "fixed_attacker_defender":
        fixed_summary = _run_fixed_attacker_defender_experiment(args, output_dir)
        summary["fixed_attacker_defender"] = fixed_summary
        summary["fixed_attacker_comparison"] = compare_fixed_attacker_defenders(
            co_learning_eval=summary["eval"],
            fixed_attacker_eval=fixed_summary["eval"],
        )
    if args.comparison_mode == "fixed_defender":
        fixed_defender_rows = _evaluate_fixed_defender_against_learned_attacker(args, pair)
        fixed_defender_dir = output_dir / "fixed_defender_eval"
        fixed_defender_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(fixed_defender_dir / "rounds.csv", fixed_defender_rows)
        fixed_defender_summary = _summarize(args, fixed_defender_rows, fixed_defender_dir)
        summary["fixed_defender"] = {"eval": fixed_defender_summary}
        summary["fixed_defender_comparison"] = compare_fixed_defender_baseline(
            co_learning_eval=summary["eval"],
            fixed_defender_eval=fixed_defender_summary,
        )
    if args.comparison_mode == "best_response":
        best_response_summary = _run_best_response_comparison(args, pair, output_dir)
        summary["best_response"] = best_response_summary
        summary["best_response_comparison"] = compare_best_response_defenders(
            co_learning_br_eval=best_response_summary["co_learning_defender"]["eval"],
            fixed_defender_br_eval=best_response_summary["fixed_defender"]["eval"],
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "MATD3_RESULT",
        f"backend={summary['backend']}",
        f"steps={summary['steps']}",
        f"final_clean_acc={summary['final_clean_acc']:.6f}",
        f"mean_defender_reward={summary['mean_defender_reward']:.6f}",
        f"mean_attacker_reward={summary['mean_attacker_reward']:.6f}",
        f"output={output_dir}",
    )


def _run_matd3(
    args: argparse.Namespace,
    env: BSMGEnv,
    pair: MATD3AgentPair,
    buffer: JointReplayBuffer,
) -> list[dict]:
    rows = []
    global_step = 0
    for episode in range(int(args.episodes)):
        obs = env.reset(seed=int(args.seed) + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(int(args.horizon)):
            defender_action, attacker_action = pair.get_actions(obs, noise=float(args.exploration_noise))
            defender_action = _clip_learned_defender_action(args, defender_action)
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, attacker_action)
            buffer.add(obs, defender_action, attacker_action, defender_reward, attacker_reward, next_obs, done)
            stats = pair.update(buffer, gradient_steps=int(args.updates_per_step))
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats=stats,
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _run_fixed_attacker_defender_experiment(args: argparse.Namespace, output_dir: Path) -> dict:
    env = _build_env(args, seed=int(args.seed) + 20_000)
    obs = env.reset(seed=int(args.seed) + 20_000)
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.exploration_noise),
    )
    pair = MATD3AgentPair(
        obs_dim=int(obs.shape[0]),
        defender_act_dim=env.act_dim,
        attacker_act_dim=env.act_dim,
        config=td3_cfg,
        device=torch.device(args.device),
    )
    buffer = JointReplayBuffer(
        capacity=int(args.buffer_capacity),
        obs_dim=int(obs.shape[0]),
        defender_act_dim=env.act_dim,
        attacker_act_dim=env.act_dim,
    )
    rows = _run_fixed_attacker_defender_training(args, env, pair, buffer)
    run_dir = output_dir / "fixed_attacker_defender"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(run_dir / "rounds.csv", rows)
    torch.save(
        {
            "defender_actor": pair.defender.actor.state_dict(),
            "td3_config": td3_cfg.__dict__,
            "obs_dim": int(obs.shape[0]),
            "act_dim": int(env.act_dim),
            "fixed_attacker_action": list(args.fixed_attacker_action),
        },
        run_dir / "defender_policy.pt",
    )
    del env
    gc.collect()
    summary = _summarize(args, rows, run_dir)
    eval_rows = _evaluate_defender_against_fixed_attacker(args, pair, seed_offset=30_000)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(eval_dir / "rounds.csv", eval_rows)
    summary["eval"] = _summarize(args, eval_rows, eval_dir)
    return summary


def _run_fixed_attacker_defender_training(
    args: argparse.Namespace,
    env: BSMGEnv,
    pair: MATD3AgentPair,
    buffer: JointReplayBuffer,
) -> list[dict]:
    rows = []
    global_step = 0
    fixed_attacker = np.asarray(args.fixed_attacker_action, dtype=np.float32)
    for episode in range(int(args.episodes)):
        obs = env.reset(seed=int(args.seed) + 20_000 + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(int(args.horizon)):
            defender_action, _ = pair.get_actions(obs, noise=float(args.exploration_noise))
            defender_action = _clip_learned_defender_action(args, defender_action)
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, fixed_attacker)
            buffer.add(obs, defender_action, fixed_attacker, defender_reward, attacker_reward, next_obs, done)
            stats = pair.update(buffer, gradient_steps=int(args.updates_per_step), update_attacker=False)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats=stats,
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _evaluate_matd3(args: argparse.Namespace, pair: MATD3AgentPair) -> list[dict]:
    if args.comparison_mode == "fixed_attacker_defender":
        return _evaluate_defender_against_fixed_attacker(args, pair, seed_offset=10_000)
    env = _build_env(args, seed=int(args.seed) + 10_000)
    rows = []
    global_step = 0
    horizon = int(args.eval_horizon or args.horizon)
    for episode in range(int(args.eval_episodes)):
        obs = env.reset(seed=int(args.seed) + 10_000 + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            defender_action, attacker_action = pair.get_actions(obs, noise=0.0)
            defender_action = _clip_learned_defender_action(args, defender_action)
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, attacker_action)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats={},
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _evaluate_fixed_defender_against_learned_attacker(
    args: argparse.Namespace,
    pair: MATD3AgentPair,
) -> list[dict]:
    env = _build_env(args, seed=int(args.seed) + 10_000)
    fixed_defender = np.asarray(args.fixed_defender_action, dtype=np.float32)
    rows = []
    global_step = 0
    horizon = int(args.eval_horizon or args.horizon)
    for episode in range(int(args.eval_episodes)):
        obs = env.reset(seed=int(args.seed) + 10_000 + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            _, attacker_action = pair.get_actions(obs, noise=0.0)
            next_obs, defender_reward, attacker_reward, done, info = env.step(fixed_defender, attacker_action)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats={},
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _run_best_response_comparison(
    args: argparse.Namespace,
    learned_pair: MATD3AgentPair,
    output_dir: Path,
) -> dict:
    br_dir = output_dir / "best_response"
    br_dir.mkdir(parents=True, exist_ok=True)
    co_learning = _train_and_eval_attacker_best_response(
        args,
        learned_pair,
        run_dir=br_dir / "co_learning_defender",
        fixed_defender_action=None,
        train_seed_offset=40_000,
        eval_seed_offset=60_000,
    )
    fixed_defender = _train_and_eval_attacker_best_response(
        args,
        learned_pair,
        run_dir=br_dir / "fixed_defender",
        fixed_defender_action=np.asarray(args.fixed_defender_action, dtype=np.float32),
        train_seed_offset=40_000,
        eval_seed_offset=60_000,
    )
    result = {
        "attacker_init": str(args.br_attacker_init),
        "train_episodes": int(args.br_train_episodes),
        "train_horizon": int(args.br_train_horizon or args.horizon),
        "train_steps_requested": int(args.br_train_steps or 0),
        "stage_steps": _br_stage_steps(
            args,
            total_steps=_phase_step_budget(
                args,
                episodes_attr="br_train_episodes",
                horizon=int(args.br_train_horizon or args.horizon),
                steps_attr="br_train_steps",
            ),
        )
        if args.br_stage_steps
        else [],
        "eval_episodes": int(args.br_eval_episodes),
        "eval_horizon": int(args.br_eval_horizon or args.eval_horizon or args.horizon),
        "co_learning_defender": co_learning,
        "fixed_defender": fixed_defender,
    }
    result["stage_comparisons"] = compare_best_response_stages(co_learning, fixed_defender)
    return result


def _train_and_eval_attacker_best_response(
    args: argparse.Namespace,
    source_pair: MATD3AgentPair,
    *,
    run_dir: Path,
    fixed_defender_action: np.ndarray | None,
    train_seed_offset: int,
    eval_seed_offset: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(args.seed) + train_seed_offset)
    torch.manual_seed(int(args.seed) + train_seed_offset)
    env = _build_env(args, seed=int(args.seed) + train_seed_offset)
    obs = env.reset(seed=int(args.seed) + train_seed_offset)
    pair = _build_br_pair(args, source_pair, obs_dim=int(obs.shape[0]), act_dim=env.act_dim)
    buffer = JointReplayBuffer(
        capacity=int(args.buffer_capacity),
        obs_dim=int(obs.shape[0]),
        defender_act_dim=env.act_dim,
        attacker_act_dim=env.act_dim,
    )
    stage_summaries: dict[str, dict] = {}

    def on_stage(step: int, train_rows: list[dict]) -> None:
        stage_dir = run_dir / "stages" / f"step_{step}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(stage_dir / "rounds.csv", train_rows)
        eval_rows = _evaluate_attacker_best_response(
            args,
            pair,
            fixed_defender_action=fixed_defender_action,
            seed_offset=eval_seed_offset,
        )
        eval_dir = stage_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(eval_dir / "rounds.csv", eval_rows)
        stage_summary = _summarize(args, train_rows, stage_dir)
        stage_summary["eval"] = _summarize(args, eval_rows, eval_dir)
        stage_summaries[str(step)] = stage_summary
        (run_dir / "progress.json").write_text(
            json.dumps({"latest_stage": step, "stages": stage_summaries}, indent=2),
            encoding="utf-8",
        )

    train_rows = _train_attacker_best_response(
        args,
        env,
        pair,
        buffer,
        fixed_defender_action=fixed_defender_action,
        seed_offset=train_seed_offset,
        stage_callback=on_stage if args.br_stage_steps else None,
    )
    _write_rows(run_dir / "rounds.csv", train_rows)
    torch.save(
        {
            "defender_actor": pair.defender.actor.state_dict(),
            "attacker_actor": pair.attacker.actor.state_dict(),
            "td3_config": pair.config.__dict__,
            "obs_dim": int(obs.shape[0]),
            "act_dim": int(env.act_dim),
            "fixed_defender_action": None
            if fixed_defender_action is None
            else np.asarray(fixed_defender_action, dtype=np.float32).tolist(),
        },
        run_dir / "attacker_best_response_policy.pt",
    )
    del env
    gc.collect()

    summary = _summarize(args, train_rows, run_dir)
    if stage_summaries:
        summary["stages"] = stage_summaries
    eval_rows = _evaluate_attacker_best_response(
        args,
        pair,
        fixed_defender_action=fixed_defender_action,
        seed_offset=eval_seed_offset,
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(eval_dir / "rounds.csv", eval_rows)
    summary["eval"] = _summarize(args, eval_rows, eval_dir)
    return summary


def _build_br_pair(
    args: argparse.Namespace,
    source_pair: MATD3AgentPair,
    *,
    obs_dim: int,
    act_dim: int,
) -> MATD3AgentPair:
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.br_exploration_noise),
    )
    pair = MATD3AgentPair(
        obs_dim=obs_dim,
        defender_act_dim=act_dim,
        attacker_act_dim=act_dim,
        config=td3_cfg,
        device=torch.device(args.device),
    )
    pair.defender.actor.load_state_dict(source_pair.defender.actor.state_dict())
    pair.defender.actor_target.load_state_dict(source_pair.defender.actor_target.state_dict())
    pair.defender.critic.load_state_dict(source_pair.defender.critic.state_dict())
    pair.defender.critic_target.load_state_dict(source_pair.defender.critic_target.state_dict())
    if args.br_attacker_init == "learned":
        pair.attacker.actor.load_state_dict(source_pair.attacker.actor.state_dict())
        pair.attacker.actor_target.load_state_dict(source_pair.attacker.actor_target.state_dict())
        pair.attacker.critic.load_state_dict(source_pair.attacker.critic.state_dict())
        pair.attacker.critic_target.load_state_dict(source_pair.attacker.critic_target.state_dict())
    return pair


def _train_attacker_best_response(
    args: argparse.Namespace,
    env: BSMGEnv,
    pair: MATD3AgentPair,
    buffer: JointReplayBuffer,
    *,
    fixed_defender_action: np.ndarray | None,
    seed_offset: int,
    stage_callback=None,
) -> list[dict]:
    rows = []
    global_step = 0
    horizon = int(args.br_train_horizon or args.horizon)
    max_steps = _phase_step_budget(
        args,
        episodes_attr="br_train_episodes",
        horizon=horizon,
        steps_attr="br_train_steps",
    )
    stage_steps = set(_br_stage_steps(args, total_steps=max_steps)) if stage_callback is not None else set()
    emitted_stages: set[int] = set()
    episode = 0
    while global_step < max_steps:
        obs = env.reset(seed=int(args.seed) + seed_offset + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            if global_step >= max_steps:
                break
            defender_action = _defender_action_for_br(args, pair, obs, fixed_defender_action)
            attacker_action = _attacker_action(pair, obs, noise=float(args.br_exploration_noise))
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, attacker_action)
            buffer.add(obs, defender_action, attacker_action, defender_reward, attacker_reward, next_obs, done)
            stats = pair.update(
                buffer,
                gradient_steps=int(args.br_updates_per_step),
                update_defender=False,
                update_attacker=True,
                fixed_defender_action=fixed_defender_action,
            )
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats=stats,
                )
            )
            if global_step in stage_steps and global_step not in emitted_stages:
                stage_callback(global_step, rows)
                emitted_stages.add(global_step)
            obs = next_obs
            if done:
                break
        episode += 1
    return rows


def _evaluate_attacker_best_response(
    args: argparse.Namespace,
    pair: MATD3AgentPair,
    *,
    fixed_defender_action: np.ndarray | None,
    seed_offset: int,
) -> list[dict]:
    env = _build_env(args, seed=int(args.seed) + seed_offset)
    rows = []
    global_step = 0
    horizon = int(args.br_eval_horizon or args.eval_horizon or args.horizon)
    for episode in range(int(args.br_eval_episodes)):
        obs = env.reset(seed=int(args.seed) + seed_offset + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            defender_action = _defender_action_for_br(args, pair, obs, fixed_defender_action)
            attacker_action = _attacker_action(pair, obs, noise=0.0)
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, attacker_action)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats={},
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _defender_action_for_br(
    args: argparse.Namespace,
    pair: MATD3AgentPair,
    obs: np.ndarray,
    fixed_defender_action: np.ndarray | None,
) -> np.ndarray:
    if fixed_defender_action is not None:
        return np.asarray(fixed_defender_action, dtype=np.float32)
    defender_action, _ = pair.get_actions(obs, noise=0.0)
    return _clip_learned_defender_action(args, defender_action)


def _attacker_action(pair: MATD3AgentPair, obs: np.ndarray, *, noise: float) -> np.ndarray:
    _, attacker_action = pair.get_actions(obs, noise=0.0)
    if noise > 0.0:
        attacker_action = attacker_action + np.random.normal(0.0, noise, size=attacker_action.shape)
    return np.clip(attacker_action, -1.0, 1.0).astype(np.float32)


def _clip_learned_defender_action(args: argparse.Namespace, action: np.ndarray) -> np.ndarray:
    low = np.asarray(args.defender_action_low, dtype=np.float32)
    high = np.asarray(args.defender_action_high, dtype=np.float32)
    if low.shape != high.shape:
        raise ValueError("defender action low/high bounds must have matching shapes")
    return np.clip(np.asarray(action, dtype=np.float32), low, high).astype(np.float32)


def _phase_step_budget(
    args: argparse.Namespace,
    *,
    episodes_attr: str,
    horizon: int,
    steps_attr: str,
) -> int:
    requested_steps = int(getattr(args, steps_attr, 0) or 0)
    if requested_steps > 0:
        return requested_steps
    return int(getattr(args, episodes_attr)) * int(horizon)


def _br_stage_steps(args: argparse.Namespace, *, total_steps: int) -> list[int]:
    stages = {int(step) for step in getattr(args, "br_stage_steps", []) if int(step) > 0}
    stages = {min(step, int(total_steps)) for step in stages}
    stages.add(int(total_steps))
    return sorted(stages)


def _evaluate_defender_against_fixed_attacker(
    args: argparse.Namespace,
    pair: MATD3AgentPair,
    *,
    seed_offset: int,
) -> list[dict]:
    env = _build_env(args, seed=int(args.seed) + seed_offset)
    fixed_attacker = np.asarray(args.fixed_attacker_action, dtype=np.float32)
    rows = []
    global_step = 0
    horizon = int(args.eval_horizon or args.horizon)
    for episode in range(int(args.eval_episodes)):
        obs = env.reset(seed=int(args.seed) + seed_offset + episode)
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            defender_action, _ = pair.get_actions(obs, noise=0.0)
            defender_action = _clip_learned_defender_action(args, defender_action)
            next_obs, defender_reward, attacker_reward, done, info = env.step(defender_action, fixed_attacker)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats={},
                )
            )
            obs = next_obs
            if done:
                break
    return rows


def _run_baseline(args: argparse.Namespace, baseline: str, *, eval_mode: bool = False) -> list[dict]:
    env = _build_env(args, seed=args.seed)
    rng = np.random.default_rng(int(args.seed))
    rows = []
    global_step = 0
    episodes = int(args.eval_episodes if eval_mode else args.episodes)
    horizon = int((args.eval_horizon or args.horizon) if eval_mode else args.horizon)
    seed_offset = 10_000 if eval_mode else 0
    for episode in range(episodes):
        obs = env.reset(seed=int(args.seed) + seed_offset + episode)
        del obs
        episode_defender_return = 0.0
        episode_attacker_return = 0.0
        for _ in range(horizon):
            defender_action, attacker_action = _baseline_actions(baseline, env.act_dim, rng)
            _, defender_reward, attacker_reward, done, info = env.step(defender_action, attacker_action)
            global_step += 1
            episode_defender_return += float(defender_reward)
            episode_attacker_return += float(attacker_reward)
            rows.append(
                _round_row(
                    global_step=global_step,
                    episode=episode,
                    info=info,
                    defender_reward=defender_reward,
                    attacker_reward=attacker_reward,
                    episode_defender_return=episode_defender_return,
                    episode_attacker_return=episode_attacker_return,
                    stats={},
                )
            )
            if done:
                break
    return rows


def _baseline_actions(baseline: str, act_dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    name = str(baseline).lower()
    if name == "fixed_mid":
        action = np.zeros(act_dim, dtype=np.float32)
        return action.copy(), action.copy()
    if name == "random":
        return (
            rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32),
            rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32),
        )
    raise ValueError(f"Unsupported baseline: {baseline}")


def _round_row(
    *,
    global_step: int,
    episode: int,
    info: dict,
    defender_reward: float,
    attacker_reward: float,
    episode_defender_return: float,
    episode_attacker_return: float,
    stats: dict,
) -> dict:
    decision = info["defense_decision"]
    return {
        "step": global_step,
        "episode": episode,
        "round": info["round"],
        "defender_reward": float(defender_reward),
        "attacker_reward": float(attacker_reward),
        "episode_defender_return": episode_defender_return,
        "episode_attacker_return": episode_attacker_return,
        "clean_acc": float(info["clean_acc"]),
        "clean_loss": float(info["clean_loss"]),
        "backdoor_acc": float(info["backdoor_acc"]),
        "alpha": float(decision.norm_bound_alpha),
        "beta": float(decision.trimmed_mean_beta),
        "epsilon": float(decision.neuroclip_epsilon or 0.0),
        "defender_critic_loss": float(stats.get("defender_critic_loss", np.nan)),
        "attacker_critic_loss": float(stats.get("attacker_critic_loss", np.nan)),
    }


def _build_env(args: argparse.Namespace, seed: int) -> BSMGEnv:
    attack_type = ATTACK_DOMAIN[str(args.attack_name)]
    pair_proxy = _ExternalActionAgent()
    attack_strategy = AdaptiveAttackStrategy(attack_type, pair_proxy)
    if args.backend == "stub":
        coordinator = StubCoordinator(
            num_clients=int(args.num_clients),
            num_attackers=int(args.num_attackers),
            subsample_rate=float(args.subsample_rate),
            seed=int(seed),
        )
    else:
        coordinator = FLSandboxCoordinatorAdapter(
            SandboxConfig(
                dataset="mnist",
                data_dir="data",
                device=str(args.device),
                num_clients=int(args.num_clients),
                num_attackers=int(args.num_attackers),
                subsample_rate=float(args.subsample_rate),
                local_epochs=1,
                batch_size=int(args.batch_size_fl),
                eval_batch_size=int(args.eval_batch_size),
                fltrust_root_size=0,
                max_client_samples_per_client=int(args.client_samples),
                max_eval_samples=int(args.eval_samples),
            )
        )
    return BSMGEnv(
        coordinator=coordinator,
        attack_type=attack_type,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=int(args.horizon),
            reward_mode=str(args.reward_mode),
            lambda_bd=float(args.lambda_bd),
            history_len=int(args.history_len),
            action_prior_weight=float(args.action_prior_weight),
        ),
    )


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(args: argparse.Namespace, rows: list[dict], output_dir: Path) -> dict:
    clean_accs = [row["clean_acc"] for row in rows] or [0.0]
    defender_rewards = [row["defender_reward"] for row in rows] or [0.0]
    attacker_rewards = [row["attacker_reward"] for row in rows] or [0.0]
    return {
        "backend": args.backend,
        "episodes": int(args.episodes),
        "horizon": int(args.horizon),
        "steps": len(rows),
        "final_clean_acc": float(clean_accs[-1]),
        "min_clean_acc": float(min(clean_accs)),
        "mean_defender_reward": float(np.mean(defender_rewards)),
        "mean_attacker_reward": float(np.mean(attacker_rewards)),
        "rounds_csv": str(output_dir / "rounds.csv"),
    }


def compare_with_baselines(matd3_summary: dict, baselines: dict[str, dict]) -> dict:
    if not baselines:
        return {}
    matd3_effect = prefer_eval_summary(matd3_summary)
    baseline_effects = {name: prefer_eval_summary(summary) for name, summary in baselines.items()}
    best_name, best_summary = max(
        baseline_effects.items(),
        key=lambda item: float(item[1].get("mean_defender_reward", float("-inf"))),
    )
    matd3_reward = float(matd3_effect.get("mean_defender_reward", 0.0))
    baseline_reward = float(best_summary.get("mean_defender_reward", 0.0))
    matd3_acc = float(matd3_effect.get("final_clean_acc", 0.0))
    baseline_acc = float(best_summary.get("final_clean_acc", 0.0))
    return {
        "best_baseline": best_name,
        "metric_source": "eval" if "eval" in matd3_summary else "train",
        "defender_reward_improvement_vs_best_baseline": matd3_reward - baseline_reward,
        "final_clean_acc_improvement_vs_best_baseline": matd3_acc - baseline_acc,
    }


def compare_fixed_attacker_defenders(co_learning_eval: dict, fixed_attacker_eval: dict) -> dict:
    co_reward = float(co_learning_eval.get("mean_defender_reward", 0.0))
    fixed_reward = float(fixed_attacker_eval.get("mean_defender_reward", 0.0))
    co_acc = float(co_learning_eval.get("final_clean_acc", 0.0))
    fixed_acc = float(fixed_attacker_eval.get("final_clean_acc", 0.0))
    return {
        "metric_source": "fixed_attacker_eval",
        "co_learning_mean_defender_reward": co_reward,
        "fixed_attacker_mean_defender_reward": fixed_reward,
        "co_learning_reward_improvement": co_reward - fixed_reward,
        "co_learning_final_clean_acc": co_acc,
        "fixed_attacker_final_clean_acc": fixed_acc,
        "co_learning_final_clean_acc_improvement": co_acc - fixed_acc,
    }


def compare_fixed_defender_baseline(co_learning_eval: dict, fixed_defender_eval: dict) -> dict:
    co_reward = float(co_learning_eval.get("mean_defender_reward", 0.0))
    fixed_reward = float(fixed_defender_eval.get("mean_defender_reward", 0.0))
    co_acc = float(co_learning_eval.get("final_clean_acc", 0.0))
    fixed_acc = float(fixed_defender_eval.get("final_clean_acc", 0.0))
    return {
        "metric_source": "learned_attacker_eval",
        "co_learning_mean_defender_reward": co_reward,
        "fixed_defender_mean_defender_reward": fixed_reward,
        "co_learning_reward_improvement": co_reward - fixed_reward,
        "co_learning_final_clean_acc": co_acc,
        "fixed_defender_final_clean_acc": fixed_acc,
        "co_learning_final_clean_acc_improvement": co_acc - fixed_acc,
    }


def compare_best_response_defenders(co_learning_br_eval: dict, fixed_defender_br_eval: dict) -> dict:
    co_reward = float(co_learning_br_eval.get("mean_defender_reward", 0.0))
    fixed_reward = float(fixed_defender_br_eval.get("mean_defender_reward", 0.0))
    co_acc = float(co_learning_br_eval.get("final_clean_acc", 0.0))
    fixed_acc = float(fixed_defender_br_eval.get("final_clean_acc", 0.0))
    return {
        "metric_source": "attacker_best_response_eval",
        "co_learning_br_mean_defender_reward": co_reward,
        "fixed_defender_br_mean_defender_reward": fixed_reward,
        "co_learning_br_reward_improvement": co_reward - fixed_reward,
        "co_learning_br_final_clean_acc": co_acc,
        "fixed_defender_br_final_clean_acc": fixed_acc,
        "co_learning_br_final_clean_acc_improvement": co_acc - fixed_acc,
    }


def compare_best_response_stages(co_learning_summary: dict, fixed_defender_summary: dict) -> dict:
    co_stages = co_learning_summary.get("stages", {})
    fixed_stages = fixed_defender_summary.get("stages", {})
    comparisons = {}
    for step in sorted(set(co_stages).intersection(fixed_stages), key=lambda value: int(value)):
        comparisons[step] = compare_best_response_defenders(
            co_learning_br_eval=co_stages[step].get("eval", {}),
            fixed_defender_br_eval=fixed_stages[step].get("eval", {}),
        )
    return comparisons


def prefer_eval_summary(summary: dict) -> dict:
    return summary.get("eval") or summary


class _ExternalActionAgent:
    def get_action(self, obs, noise: float = 0.0):
        del obs, noise
        return np.zeros(3, dtype=np.float32)


if __name__ == "__main__":
    main()
