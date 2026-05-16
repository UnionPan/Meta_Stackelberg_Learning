from fl_sandbox.core.experiment_service import rl_training_tensorboard_scalars


def test_rl_training_tensorboard_scalars_maps_td3_losses_and_rewards():
    metrics = {
        "rl_trainer_actor_loss": -0.25,
        "rl_trainer_critic1_loss": 1.5,
        "rl_trainer_critic2_loss": 2.5,
        "rl_trainer_loss": 3.75,
        "rl_trainer_last_update_loss": 3.0,
        "rl_trainer_reward_mean": 0.4,
        "rl_real_reward": -0.1,
        "rl_simulated_reward": 0.2,
        "rl_action_gamma_scale": 15.0,
        "rl_action_local_steps": 25.0,
        "rl_observation_norm": 3.5,
        "rl_observation_std": 0.7,
        "rl_trainer_replay_size": 128,
        "not_rl": 10,
    }

    scalars = dict(rl_training_tensorboard_scalars(metrics))

    assert scalars["rl_training/actor_loss"] == -0.25
    assert scalars["rl_training/critic1_loss"] == 1.5
    assert scalars["rl_training/critic2_loss"] == 2.5
    assert scalars["rl_training/critic_loss"] == 4.0
    assert scalars["rl_training/total_loss"] == 3.75
    assert scalars["rl_training/last_update_loss"] == 3.0
    assert scalars["rl_training/reward_mean"] == 0.4
    assert scalars["rl_training/real_reward"] == -0.1
    assert scalars["rl_training/simulated_reward"] == 0.2
    assert scalars["rl_action/gamma_scale"] == 15.0
    assert scalars["rl_action/local_steps"] == 25.0
    assert scalars["rl_observation/norm"] == 3.5
    assert scalars["rl_observation/std"] == 0.7
    assert scalars["rl_training/replay_size"] == 128.0
    assert "not_rl" not in scalars


def test_rl_training_tensorboard_scalars_ignores_non_finite_values():
    scalars = dict(
        rl_training_tensorboard_scalars(
            {
                "rl_trainer_actor_loss": float("nan"),
                "rl_trainer_critic1_loss": 1.0,
            }
        )
    )

    assert scalars == {"rl_training/critic1_loss": 1.0}
