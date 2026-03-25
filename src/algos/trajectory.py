import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Trajectory:
    """Stores trajectory data for a single agent over H steps."""
    obs: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def to_tensors(self):
        """Convert lists to stacked tensors for gradient computation."""
        return {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.stack(self.values),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.rewards)


class TrajectoryCollector:
    """
    Collects trajectories from a PettingZoo ParallelEnv.

    The collector rolls out H steps, handling episode boundaries
    by resetting the env when done. It returns per-agent Trajectory objects.

    Args:
        defender_id: agent id for the defender (leader)
        attacker_id: agent id for the attacker (follower)
    """
    def __init__(self, defender_id="player_1", attacker_id="player_2"):
        self.defender_id = defender_id
        self.attacker_id = attacker_id

    def collect(
        self,
        env,
        defender_policy,
        attacker_policy,
        H: int,
        attack_type: Optional[str] = None,
        gamma: float = 0.99,
    ) -> Dict[str, Trajectory]:
        """
        Roll out H steps and collect trajectories.

        Args:
            env: PettingZoo ParallelEnv (must support reset/step)
            defender_policy: PolicyAgent for the defender
            attacker_policy: PolicyAgent or FixedPolicyAgent for the attacker
            H: number of steps to collect
            attack_type: if provided, passed to env.reset via options
            gamma: discount factor (used for computing returns later)

        Returns:
            dict mapping agent_id -> Trajectory
        """
        trajectories = {
            self.defender_id: Trajectory(),
            self.attacker_id: Trajectory(),
        }

        # Reset env
        reset_options = {"attack_type": attack_type} if attack_type else None
        obs, infos = env.reset(options=reset_options)

        for step in range(H):
            # Get observations as tensors
            def_obs = torch.tensor(obs[self.defender_id], dtype=torch.float32)
            att_obs = torch.tensor(obs[self.attacker_id], dtype=torch.float32)

            # Get actions from policies (no gradient during collection)
            with torch.no_grad():
                def_action, def_log_prob, def_value = defender_policy.get_action(
                    def_obs.unsqueeze(0)
                )
                att_action, att_log_prob, att_value = attacker_policy.get_action(
                    att_obs.unsqueeze(0)
                )

            # Step the environment
            # Handle both discrete (scalar) and continuous (array) actions
            def_act_env = def_action.numpy() if def_action.dim() > 0 else def_action.item()
            att_act_env = att_action.numpy() if att_action.dim() > 0 else att_action.item()
            actions = {
                self.defender_id: def_act_env,
                self.attacker_id: att_act_env,
            }
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Store transitions
            for agent_id, policy_out in [
                (self.defender_id, (def_obs, def_action, def_log_prob, def_value)),
                (self.attacker_id, (att_obs, att_action, att_log_prob, att_value)),
            ]:
                agent_obs, agent_action, agent_lp, agent_val = policy_out
                traj = trajectories[agent_id]
                traj.obs.append(agent_obs)
                traj.actions.append(agent_action)
                traj.log_probs.append(agent_lp)
                traj.values.append(agent_val)
                traj.rewards.append(rewards[agent_id])
                traj.dones.append(terminations[agent_id])

            # Handle episode boundaries
            done = any(terminations.values())
            if done:
                obs, infos = env.reset(options=reset_options)
            else:
                obs = next_obs

        return trajectories
