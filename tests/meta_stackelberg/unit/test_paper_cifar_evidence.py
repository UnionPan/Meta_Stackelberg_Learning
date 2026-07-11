from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments import paper_cifar_evidence
from meta_stackelberg.experiments.paper_cifar_env import PaperCIFARDatasets
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds


def test_cifar_evidence_entrypoint_uses_5131_5135_dims(monkeypatch) -> None:
    inputs = torch.zeros(200, 3, 32, 32)
    labels = torch.arange(200) % 10
    dataset = TensorDataset(inputs, labels)
    datasets = PaperCIFARDatasets(
        dataset, TensorDataset(inputs[:20], labels[:20]), dataset, (),
    )
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=1, l=1, N_A=1, N_D=1,
        workers=20, untargeted_attackers=10, sample_size=2,
        td3_batch_size=1, learning_starts=1, hidden_sizes=(8,),
        replay_capacity=64,
    )
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            training='training', scientific='scientific',
            parameter_snapshot={'H': 1}, query_seeds=(101,),
        )

    monkeypatch.setattr(paper_cifar_evidence, 'run_scaled_evidence', fake_run)
    result = paper_cifar_evidence.run_paper_cifar_scaled_evidence(
        config=config,
        datasets=datasets,
        thresholds=ScientificGateThresholds(0, 0, 0, 0, 0, 0),
        query_seeds=(101,), training_support_seed=1000,
        scientific_support_seeds=tuple(range(2000, 2020)),
        seed=1, partition_seed=2, model_seed=3,
        local_search_batch_size=4,
    )

    assert captured['defender_obs_dim'] == 5131
    assert captured['attacker_obs_dim'] == 5135
    assert result.parameter_snapshot['batchnorm_float_buffers_aggregated']
    assert result.protocol == 'paper-cifar-scaled-meta-sg-evidence-v1'
