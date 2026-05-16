import copy
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.attacks import RLAttack
from fl_sandbox.attacks.rl_attacker import (
    AttackerPolicyGymEnv,
    GradientDistributionLearner,
    RLAttackerConfig,
    SimulatedFLEnv,
)
from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.defenders import AggregationDefender
from fl_sandbox.core.fl_runner import MinimalFLRunner, SandboxConfig
from fl_sandbox.core.runtime import RoundContext, RoundSummary, summaries_to_dict
from src.models.cnn import MNISTClassifier


def _weights(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


class _TinyDataset:

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


class _ToyPoisonDataset:

    def __init__(self, targets):
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.data = torch.zeros((len(targets), 28, 28), dtype=torch.uint8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx].float().unsqueeze(0)
        return image, int(self.targets[idx].item())


class TestRLAttacker(unittest.TestCase):

    def test_distribution_learner_initializes_proxy_buffer(self):
        images = torch.randn(12, 1, 28, 28)
        labels = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        learner = GradientDistributionLearner(RLAttackerConfig(seed_samples=8))

        learner.initialize_from_loader(loader, torch.device("cpu"))

        self.assertTrue(learner.initialized)
        self.assertGreaterEqual(len(learner.buffer), 8)
        self.assertEqual(learner.buffer.num_classes, 10)

    def test_rl_attack_can_craft_manual_action_update(self):
        model = MNISTClassifier()
        weights = _weights(model)
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        fl_config = SandboxConfig(
            num_clients=4,
            num_attackers=1,
            batch_size=4,
            rl_attack_start_round=1,
            rl_distribution_steps=1,
            rl_policy_train_end_round=1,
        )
        attack = RLAttack(
            config=RLAttackerConfig(
                attack_start_round=1,
                distribution_steps=1,
                policy_train_end_round=1,
                inversion_steps=1,
                reconstruction_batch_size=4,
                episodes_per_observation=1,
                simulator_horizon=2,
                local_search_batch_size=4,
            ),
        )
        ctx = RoundContext(
            round_idx=1,
            old_weights=weights,
            benign_weights=[],
            selected_attacker_ids=[0],
            model=copy.deepcopy(model),
            device=torch.device("cpu"),
            fl_config=fl_config,
            defense_type="fedavg",
            lr=0.05,
            server_lr=1.0,
            local_epochs=1,
            attacker_train_iter=loader,
            all_attacker_train_iter=loader,
            attacker_action=np.asarray([-0.8, -0.9, -1.0], dtype=np.float32),
        )

        attack.observe_round(ctx)
        malicious = attack.execute(ctx, attacker_action=ctx.attacker_action)

        self.assertEqual(len(malicious), 1)
        self.assertEqual(len(malicious[0]), len(weights))
        deltas = [
            np.linalg.norm(new_layer - old_layer)
            for new_layer, old_layer in zip(malicious[0], weights)
        ]
        self.assertGreater(sum(deltas), 0.0)

    def test_action_dims_are_stealth_aware_for_robust_defenses(self):
        config = RLAttackerConfig()

        self.assertEqual(config.action_dim("krum"), 3)
        self.assertEqual(config.action_dim("multi_krum"), 3)
        self.assertEqual(config.action_dim("trimmed_mean"), 3)
        self.assertEqual(config.action_dim("clipped_median"), 3)
        decoded = decode_action(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), "clipped_median", config)
        self.assertGreater(decoded.gamma_scale, 0.0)
        self.assertGreater(decoded.local_steps, 0)
        self.assertGreater(decoded.lambda_stealth, 0.0)

    def test_simulator_samples_positive_attack_steps(self):
        model = MNISTClassifier()
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        learner = GradientDistributionLearner(RLAttackerConfig(seed_samples=8, reconstruction_batch_size=4))
        learner.initialize_from_loader(loader, torch.device("cpu"))
        sim = SimulatedFLEnv(
            model_template=model,
            proxy_buffer=learner.buffer,
            defender=AggregationDefender(defense_type="krum"),
            config=RLAttackerConfig(simulator_horizon=2, local_search_batch_size=4),
            fl_config=SandboxConfig(num_clients=10, num_attackers=2, subsample_rate=0.1, batch_size=4),
            device=torch.device("cpu"),
        )

        sim.reset(_weights(model))

        self.assertGreaterEqual(sim.current_num_attackers, 1)

    def test_legacy_clipped_median_simulator_copies_one_malicious_update_for_all_attackers(self):
        model = MNISTClassifier()
        weights = _weights(model)
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        learner = GradientDistributionLearner(
            RLAttackerConfig(
                attacker_semantics="legacy_clipped_median",
                seed_samples=8,
                reconstruction_batch_size=4,
                simulator_horizon=1,
                local_search_batch_size=4,
            )
        )
        learner.initialize_from_loader(DataLoader(TensorDataset(images, labels), batch_size=4), torch.device("cpu"))
        sim = SimulatedFLEnv(
            model_template=model,
            proxy_buffer=learner.buffer,
            defender=AggregationDefender(defense_type="clipped_median"),
            config=learner.config,
            fl_config=SandboxConfig(num_clients=4, num_attackers=2, subsample_rate=0.5, batch_size=4),
            device=torch.device("cpu"),
        )
        sim.reset(weights)
        sim.current_num_attackers = 2
        calls = 0

        def fake_malicious(old_weights, action):
            nonlocal calls
            calls += 1
            return [layer + calls for layer in old_weights]

        sim._simulate_malicious_weight = fake_malicious
        captured = {}

        def fake_aggregate(old_weights, new_weights, trusted_weights=None):
            captured["new_weights"] = new_weights
            return old_weights

        sim.defender.aggregate = fake_aggregate
        sim.step(np.zeros(2, dtype=np.float32))

        self.assertEqual(calls, 1)
        self.assertEqual(len(captured["new_weights"]), 2)
        self.assertTrue(
            all(
                np.allclose(captured["new_weights"][0][idx], captured["new_weights"][1][idx])
                for idx in range(len(weights))
            )
        )

    def test_legacy_clipped_median_live_observation_matches_tail_state_contract(self):
        model = MNISTClassifier()
        weights = _weights(model)
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        fl_config = SandboxConfig(
            num_clients=4,
            num_attackers=1,
            batch_size=4,
            rl_attack_start_round=1,
            rl_distribution_steps=1,
            rl_policy_train_end_round=0,
        )
        attack = RLAttack(
            config=RLAttackerConfig(
                attacker_semantics="legacy_clipped_median",
                attack_start_round=1,
                distribution_steps=1,
                policy_train_end_round=0,
                reconstruction_batch_size=4,
                local_search_batch_size=4,
            ),
        )
        ctx = RoundContext(
            round_idx=1,
            old_weights=weights,
            benign_weights=[],
            selected_attacker_ids=[0],
            model=copy.deepcopy(model),
            device=torch.device("cpu"),
            fl_config=fl_config,
            defense_type="clipped_median",
            lr=0.05,
            server_lr=1.0,
            local_epochs=1,
            attacker_train_iter=loader,
            all_attacker_train_iter=loader,
        )

        attack.observe_round(ctx)
        obs = attack._current_observation(ctx)
        expected_tail_dim = weights[-2].size + weights[-1].size + 1

        self.assertEqual(obs.shape, (expected_tail_dim,))
        self.assertEqual(obs[-1], 1.0)

    def test_strict_legacy_clipped_median_reward_uses_eval_loader(self):
        model = MNISTClassifier()
        weights = _weights(model)
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        learner = GradientDistributionLearner(
            RLAttackerConfig(
                attacker_semantics="legacy_clipped_median_strict",
                seed_samples=8,
                reconstruction_batch_size=4,
                simulator_horizon=1,
                local_search_batch_size=4,
            )
        )
        learner.initialize_from_loader(DataLoader(TensorDataset(images, labels), batch_size=4), torch.device("cpu"))
        eval_loader = DataLoader(TensorDataset(torch.randn(8, 1, 28, 28), torch.zeros(8, dtype=torch.long)), batch_size=4)
        sim = SimulatedFLEnv(
            model_template=model,
            proxy_buffer=learner.buffer,
            defender=AggregationDefender(defense_type="clipped_median"),
            config=learner.config,
            fl_config=SandboxConfig(num_clients=4, num_attackers=1, subsample_rate=0.25, batch_size=4),
            device=torch.device("cpu"),
            eval_loader=eval_loader,
        )
        eval_losses = iter([1.0, 3.5])
        eval_calls = []

        def fake_eval_metrics(_weights):
            eval_calls.append("eval")
            return next(eval_losses), 0.0

        def fail_proxy_metrics(_weights):
            raise AssertionError("strict reproduction reward must not use proxy metrics when eval_loader is available")

        sim._evaluate_eval_metrics = fake_eval_metrics
        sim._evaluate_proxy_metrics = fail_proxy_metrics
        sim._simulate_benign_update = lambda old_weights: [layer.copy() for layer in old_weights]
        sim._simulate_malicious_weight = lambda old_weights, action: [layer.copy() for layer in old_weights]
        sim.defender.aggregate = lambda old_weights, new_weights, trusted_weights=None: [layer.copy() for layer in old_weights]

        sim.reset(weights)
        _, reward, _ = sim.step(np.zeros(2, dtype=np.float32))

        self.assertEqual(eval_calls, ["eval", "eval"])
        self.assertEqual(reward, 2.5)

    def test_strict_legacy_clipped_median_proxy_view_expands_on_reset(self):
        model = MNISTClassifier()
        weights = _weights(model)
        config = RLAttackerConfig(
            attacker_semantics="legacy_clipped_median_strict",
            strict_reproduction_initial_samples=3,
            strict_reproduction_samples_per_epoch=2,
            local_search_batch_size=16,
        )
        learner = GradientDistributionLearner(config)
        images = torch.randn(8, 1, 28, 28)
        labels = torch.arange(8, dtype=torch.long)
        learner.initialize_from_loader(DataLoader(TensorDataset(images, labels), batch_size=8), torch.device("cpu"))
        sim = SimulatedFLEnv(
            model_template=model,
            proxy_buffer=learner.buffer,
            defender=AggregationDefender(defense_type="clipped_median"),
            config=config,
            fl_config=SandboxConfig(num_clients=4, num_attackers=1, batch_size=4),
            device=torch.device("cpu"),
        )

        sim.reset(weights)
        _, labels_epoch1 = sim.active_proxy_buffer().sample(16, torch.device("cpu"))
        sim.reset(weights)
        _, labels_epoch2 = sim.active_proxy_buffer().sample(16, torch.device("cpu"))

        self.assertTrue(sim.active_proxy_buffer().uses_dataloader)
        self.assertLessEqual(int(labels_epoch1.max().item()), 2)
        self.assertLessEqual(int(labels_epoch2.max().item()), 4)
        self.assertEqual(sim.strict_reproduction_sample_limit, 5)

    def test_gym_wrapper_exposes_flat_observation(self):
        model = MNISTClassifier()
        weights = _weights(model)
        images = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        buffer_learner = GradientDistributionLearner(RLAttackerConfig(seed_samples=8, reconstruction_batch_size=4))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        buffer_learner.initialize_from_loader(loader, torch.device("cpu"))
        fl_config = SandboxConfig(num_clients=4, num_attackers=1, batch_size=4)
        sim = SimulatedFLEnv(
            model_template=model,
            proxy_buffer=buffer_learner.buffer,
            defender=AggregationDefender(defense_type="krum"),
            config=RLAttackerConfig(simulator_horizon=2, local_search_batch_size=4),
            fl_config=fl_config,
            device=torch.device("cpu"),
        )
        env = AttackerPolicyGymEnv(sim, RLAttackerConfig(simulator_horizon=2, local_search_batch_size=4), "krum", weights)

        obs, info = env.reset()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.ndim, 1)
        self.assertEqual(env.observation_space.shape, obs.shape)
        self.assertEqual(info["defense_type"], "krum")

    def test_checkpoint_init_mode_loads_saved_weights(self):
        reference_model = MNISTClassifier()
        with torch.no_grad():
            for idx, parameter in enumerate(reference_model.parameters(), start=1):
                parameter.fill_(0.01 * idx)

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"state_dict": reference_model.state_dict()}, handle.name)

            runner = MinimalFLRunner.__new__(MinimalFLRunner)
            runner.config = SandboxConfig(
                dataset="mnist",
                init_mode="checkpoint",
                init_checkpoint_path=handle.name,
            )
            runner.device = torch.device("cpu")

            loaded_model, loaded_client_model = runner._initialize_model_pair()

        reference_weights = _weights(reference_model)
        self.assertTrue(
            all(np.allclose(expected, loaded) for expected, loaded in zip(reference_weights, _weights(loaded_model)))
        )
        self.assertTrue(
            all(
                np.allclose(expected, loaded)
                for expected, loaded in zip(reference_weights, _weights(loaded_client_model))
            )
        )

    def test_summaries_to_dict_exposes_asr_alias(self):
        summary = RoundSummary(
            round_idx=1,
            attack_name="RL",
            defense_name="clipped_median",
            sampled_clients=[0, 1],
            benign_clients=[1],
            selected_attackers=[0],
            clean_loss=1.0,
            clean_acc=0.75,
            backdoor_acc=0.6,
            round_seconds=0.1,
        )

        series = summaries_to_dict([summary])

        self.assertEqual(series["backdoor_acc"], [0.6])
        self.assertEqual(series["asr"], [0.6])

    def test_paper_q_split_assigns_every_example(self):
        runner = MinimalFLRunner.__new__(MinimalFLRunner)
        runner.config = SandboxConfig(num_clients=10, num_attackers=2, split_mode="paper_q", noniid_q=0.1, seed=7)
        runner.train_dataset = _TinyDataset([idx % 10 for idx in range(100)])
        runner.client_groups = list(range(10))

        client_splits = runner._split_data_paper_q()

        assigned = set()
        for indices in client_splits:
            assigned.update(indices)
        self.assertEqual(len(client_splits), 10)
        self.assertEqual(len(assigned), 100)

    def test_poisoned_train_loaders_stay_within_attacker_local_partitions(self):
        runner = MinimalFLRunner.__new__(MinimalFLRunner)
        runner.config = SandboxConfig(
            dataset="mnist",
            num_attackers=2,
            batch_size=2,
            base_class=1,
            target_class=7,
            bfl_poison_frac=1.0,
            dba_poison_frac=1.0,
            dba_num_sub_triggers=2,
        )
        runner.loader_kwargs = {"shuffle": False, "pin_memory": False, "num_workers": 0}
        runner.train_dataset = _ToyPoisonDataset([1, 1, 1, 2, 2, 1])
        runner.attacker_ids = [0, 1]
        runner.client_data_idxs = [
            {0, 3},
            {1, 4},
            {2, 5},
        ]

        loaders = runner._prepare_poisoned_train_loaders()

        attacker_zero_loader = loaders["global_by_attacker"][0]
        attacker_one_loader = loaders["global_by_attacker"][1]

        self.assertEqual(attacker_zero_loader.dataset.idxs, [0, 3])
        self.assertEqual(attacker_one_loader.dataset.idxs, [1, 4])
        self.assertEqual(int(attacker_zero_loader.dataset.dataset.targets[0].item()), 7)
        self.assertEqual(int(attacker_zero_loader.dataset.dataset.targets[1].item()), 1)
        self.assertEqual(int(attacker_one_loader.dataset.dataset.targets[1].item()), 7)
        self.assertEqual(int(attacker_one_loader.dataset.dataset.targets[5].item()), 1)


if __name__ == "__main__":
    unittest.main()
