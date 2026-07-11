import torch
from types import SimpleNamespace
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.paper_mnist_env import PaperMNISTDatasets
from meta_stackelberg.experiments import paper_mnist_evidence
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds


def test_mnist_evidence_entrypoint_uses_1290_1295_dims_and_fixed_factory(monkeypatch) -> None:
    inputs = torch.zeros(400, 1, 28, 28)
    labels = torch.arange(400) % 10
    dataset = TensorDataset(inputs, labels)
    datasets = PaperMNISTDatasets(dataset, TensorDataset(inputs[:40], labels[:40]), dataset, ())
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=1, l=1, N_A=1, N_D=1,
        workers=20, untargeted_attackers=10, sample_size=10,
        td3_batch_size=1, learning_starts=1, hidden_sizes=(8,),
        replay_capacity=64,
    )
    captured = {}
    sentinel = SimpleNamespace(
        training='training', scientific='scientific',
        parameter_snapshot={'H': 1}, query_seeds=(101,),
    )

    def fake_run(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(paper_mnist_evidence, 'run_scaled_evidence', fake_run)
    result = paper_mnist_evidence.run_paper_mnist_scaled_evidence(
        config=config,
        datasets=datasets,
        thresholds=ScientificGateThresholds(0, 0, 0, 0, 0, 0),
        query_seeds=(101,),
        training_support_seed=1000,
        scientific_support_seeds=tuple(range(2000, 2020)),
        seed=1,
        partition_seed=2,
        model_seed=3,
        local_search_batch_size=4,
    )

    assert result.training == 'training'
    assert result.protocol == 'paper-mnist-scaled-meta-sg-evidence-v1'
    assert captured['defender_obs_dim'] == 1291
    assert captured['attacker_obs_dim'] == 1295
    first = captured['env_factory']('rl', 11, 1)
    second = captured['env_factory']('rl', 12, 1)
    assert torch.equal(
        torch.from_numpy(first.state.global_model.vector()),
        torch.from_numpy(second.state.global_model.vector()),
    )
