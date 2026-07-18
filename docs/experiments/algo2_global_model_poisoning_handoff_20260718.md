# Meta-Stackelberg Algo2 全局模型投毒实验

## 1. 范围与当前状态

本文介绍 Algorithm 2（Reptile Meta-RL）在 MNIST 全局模型投毒场景下的实现。

代码边界：

- `meta_stackelberg/`：重构后的正式 Python package，也是本文训练与评估代码的主体；
- `exp_scripts/`：正式实验的进程编排与运行入口；
- `meta_sg/`：重构前的历史实现，仅用于结果和实现对照，不是当前正式 Algo2 入口。

重构后，主链路已经跑通：

1. 预训练：`T=100, K=5, H=100, l=10`；
2. Frozen 评估：Clean、IPM、LMP、RL-ClipMed、RL-Krum；
3. 新版在线适应：每场景 1,000 FL rounds、100 次 TD3 更新；
4. 独立 reward guard：候选变差时回退 Frozen；
5. 训练与在线适应均支持按外层迭代保存、恢复 checkpoint。

正式运行目录：

```text
runs/meta_stackelberg/
└── algo2_global_formal_iid_standard_normmedian_t100_k5_h100_l10_w20_s4_a4_alpha075_meta025_seed41_20260718/
    ├── training.pt
    ├── training.log
    ├── training_metrics.json
    ├── evaluation_frozen_h100/
    └── evaluation_online_h100_v2_alpha_guarded/
```

## 2. 整体思路

```text
预训练攻击域
  NA / IPM / LMP / RL-Krum / RL-ClipMed
                    │
                    ▼
Algorithm 2：每轮采 K 个任务
  克隆 Meta Defender → 每任务适应 l 步 → Reptile 参数平均
                    │
                    ▼
              training.pt
                 θ_meta
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Frozen 直接评估      每场景在线适应
                              │
                    anti-saturation TD3
                              │
                    独立 reward guard
                      ┌───────┴───────┐
                      ▼               ▼
                  接受 adapted     回退 Frozen
```

训练与评估彼此独立：Algo2 输出 `algorithm2_defender`，最终评估不得加载 Algorithm 1 Defender。

## 3. 四个时间尺度

当前实现按以下四层组织：

```text
FL Round → Episode → Best Response / Task Adaptation → Meta Iteration
```

| 时间尺度 | 语义 | Algo2 正式训练中的数量 | 重构后代码 |
|---|---|---:|---|
| FL Round | 一次客户端采样、本地更新、攻击、聚合和 reward 计算 | 最内层 1 步 | `meta_stackelberg/environments/paper_bsmg.py`、`meta_stackelberg/federated/engine/round_kernel.py` |
| Episode | 固定任务和双方策略，连续运行 `H` 个 FL rounds，形成一条轨迹 | `H=100` | `PaperTD3TrajectoryCollector.collect()` |
| Best Response / Task Adaptation | 固定对手，只更新当前角色；Algo2 中具体为任务 Defender 适应 | `l=10` | `PolicyMetaSGAlgorithm2._adapt_jobs()`、`TD3Agent.update()` |
| Meta Iteration | 采样 `K` 个任务，分别适应后执行一次 Reptile 元更新 | `T=100, K=5` | `PolicyMetaSGAlgorithm2.run()`、`reptile_update_td3()` |

### 3.1 FL Round：联邦学习物理步

`PaperBSMGEnv.begin_round()` 与 `finish_round()` 负责把 Defender/Attacker 动作送入联邦学习环境；底层 round kernel 完成客户端采样、本地训练、攻击注入、防御、聚合和 reward 计算。只有这一层修改部署中的全局 FL 模型，不在此处做 TD3 或 Reptile 更新。

### 3.2 Episode：H 轮轨迹

`meta_stackelberg/experiments/paper_meta_sg.py::PaperTD3TrajectoryCollector.collect()` 连续执行 `H` 个 FL rounds，每轮由双方策略产生动作，并将 `(s, a, r, s', done)` 写入对应 Replay Buffer。Collector 只负责采样，不直接更新 Agent。

### 3.3 Best Response / Task Adaptation：角色内层更新

四时间尺度的通用定义是：冻结一方，只更新另一方；若双方同时更新，则属于 joint co-learning，不能称为 best response。

Algo2 没有 Algo1 的 attacker best-response 内循环。它冻结每个任务对应的 attacker response，只克隆并适应 Defender，因此在 Algo2 交接中更准确的名称是 **Task Adaptation**：

```python
for step in range(l):
    collect_H_round_trajectory(adapted_defender, frozen_attacker)
    adapted_defender.update(replay.sample(...))
```

该循环位于 `meta_stackelberg/stackelberg/policy_algorithm2.py::PolicyMetaSGAlgorithm2._adapt_jobs()`；具体 TD3 梯度更新位于 `meta_stackelberg/agents/td3/agent.py::TD3Agent.update()`。

### 3.4 Meta Iteration：跨任务 Reptile 更新

`PolicyMetaSGAlgorithm2.run()` 每轮采样 `K` 个任务，为每个任务克隆 Defender 并完成 `l` 步适应，然后调用 `meta_stackelberg/stackelberg/algorithm2.py::reptile_update_td3()`：

```text
θ_meta ← θ_meta + meta_step / K × Σ(θ_task − θ_meta)
```

`ScaledPaperMetaSGTrainingRunner.run()` 将这套 Algo2 循环接入环境、任务采样和 checkpoint callback。每个 Meta Iteration 结束后均可保存并从下一轮恢复。

这里的 `K` 是同一 Meta Iteration 内的任务批宽，不是第五个时间尺度；客户端 minibatch 更新属于 FL Round 内部，也不单独构成时间尺度。

### 3.5 训练与在线适应的尺度映射

| 阶段 | FL Round | Episode | Task Adaptation | Meta Iteration |
|---|---:|---:|---:|---:|
| Algo2 预训练 | 基本环境步 | 每条轨迹 `H=100` | 每任务 `l=10`；每步重新采一条 H 轨迹后更新 TD3 | `T=100`；每轮 `K=5` 后做 Reptile |
| 当前在线适应 | 基本环境步 | 每个窗口采 `H=100` | 在该场景的持久 Replay 上更新 `l=10` 次 | 无元更新；`online_T=10` 表示适应窗口，不是 Meta Iteration |

因此，正式预训练预算是：

```text
T × K × l × H = 100 × 5 × 10 × 100 = 500,000 FL rounds
```

当前每场景在线适应预算是：

```text
online_T × online_H = 10 × 100 = 1,000 FL rounds，共 100 次 TD3 updates
```

论文 Algorithm 2 的伪代码是在每个 inner step 重新采一条 H 轨迹，因此当前预训练实现与该伪代码一致。它随后直接使用任务参数差做 Reptile，没有额外的 query trajectory/query gate；这比项目通用四时间尺度设计中的 support/query 蓝图更简单。

## 4. 代码模块

| 模块 | 职责 |
|---|---|
| `exp_scripts/run_algo2_formal_pipeline.py` | 正式实验编排、监控、拉起训练/评估、汇总状态；路径为实验专用硬编码。 |
| `meta_stackelberg/experiments/run_paper_meta_rl.py` | Algo2 训练 CLI；构造数据、任务域、环境、TD3 Agent 和训练 Runner。 |
| `meta_stackelberg/stackelberg/policy_algorithm2.py` | Algo2 主循环：采 K 个任务、克隆 Defender、适应 l 步、执行 Reptile 更新。 |
| `meta_stackelberg/stackelberg/algorithm2.py` | 通用 Reptile 事件模型及 `reptile_update_td3` 参数插值。 |
| `meta_stackelberg/experiments/paper_meta_sg.py` | Algo1/Algo2 共用的轨迹采集、规模化训练 Runner、在线适应 Runner。文件名虽含 `meta_sg`，Algo2 也依赖它。 |
| `meta_stackelberg/agents/td3/` | TD3 actor/critic、Replay Buffer、快照、RNG、在线 logit anti-saturation。 |
| `meta_stackelberg/experiments/paper_mnist_env.py` | MNIST 数据、root set、客户端划分、初始模型和环境工厂。 |
| `meta_stackelberg/environments/paper_bsmg.py` | 单个 FL round：Defender 动作 → 客户端更新/攻击 → 聚合 → reward。 |
| `meta_stackelberg/environments/model_tail.py` | 将全局模型最后两个可学习参数张量与 round progress 编码为状态。 |
| `meta_stackelberg/federated/clients/trainer.py` | 客户端本地训练；当前 `local_steps=1` 表示恰好一个 minibatch 更新。 |
| `meta_stackelberg/security/defenses/paper_action.py` | 将 actor 的 `[-1,1]^3` 动作解码为 `(alpha, beta, epsilon)`。 |
| `meta_stackelberg/experiments/attack_domain.py` | 保存/加载 RL-Krum、RL-ClipMed attacker checkpoint 及来源。 |
| `meta_stackelberg/experiments/run_model_poisoning_evaluation.py` | Frozen/Online 评估入口、在线 checkpoint、reward guard、结果缓存。 |
| `meta_stackelberg/experiments/model_poisoning_evaluation.py` | 五场景逐 round 测试及 `summary.json` 汇总。 |
| `scaled_training_checkpoint.py` / `online_adaptation_checkpoint.py` | 原子保存和精确恢复训练/在线适应状态。 |

## 5. Algo2 预训练流程

正式任务域为：

```python
GLOBAL_TASKS = ('na', 'ipm', 'lmp', 'rl-krum', 'rl-clipmed')
```

当前使用 balanced sampler。因为 `K=5` 且任务域也是 5 个任务，所以每个外层迭代恰好覆盖全部任务，只改变顺序。

每个外层迭代：

1. 从 Meta Defender 克隆 5 个相互隔离的任务 Defender；
2. 每个任务执行 `l=10` 次适应；
3. 每次适应先生成一条 `H=100` 的完整 FL 轨迹，再从持久 Replay 中采样并更新 TD3；
4. attacker 在 Algo2 中固定，只作为任务响应策略，不参与更新；
5. 将 5 个 adapted TD3 网络与 Meta 网络做 Reptile 参数插值；
6. 每个外层迭代原子保存一次 `training.pt`。

Reptile 更新为：

```text
θ ← θ + meta_step / K × Σ(θ_task − θ)
```

当前插值 actor、target actor、双 critic 和 target critic；不平均 optimizer、计数器和 RNG。

正式预训练轨迹预算：

```text
T × K × l × H = 100 × 5 × 10 × 100 = 500,000 FL rounds
```

5 个任务的 FL 轨迹可以并行生成，单 GPU 上 TD3 optimizer update 仍按任务顺序执行，以保持确定性。

## 6. FL 环境与动作

### 状态

Defender 状态由两部分组成：

- MNIST 模型最后两个可学习参数张量，拼接并按 L2 norm 归一化；
- 当前 round progress：`round_index / H`。

Attacker 状态额外加入采样恶意客户端数和 Defender 原始动作。

### Defender 三维动作

| 维度 | 作用 | 当前全局实验 |
|---|---|---|
| `alpha` | 客户端更新 norm clipping 阈值 | 有效；下限为观测 norm 参考值的 75%。 |
| `beta` | coordinate-wise trimmed mean 比例 | 有效，范围 `[0, 0.45]`。 |
| `epsilon` | NeuroClip clipping range | `post_defense_mode=identity`，因此当前是无效维度。 |

当前 norm 参考值使用 sampled updates 的 median，而不是 max。

### 单轮执行

1. 20 个客户端中采样 4 个，并保证攻击场景至少有一个 benign reference；
2. 每个 benign client 只训练一个 minibatch：batch size 128、SGD learning rate 0.05；
3. NA 不注入恶意更新；IPM/LMP 使用 scale 2；RL 场景加载预训练 attacker；
4. 先 norm clip，再 coordinate-wise trimmed mean；
5. identity 模式不执行 NeuroClip；
6. 100 个 root samples 上计算交叉熵，Defender reward 为 `-post_loss_after`。

## 7. 在线适应与评估

每个场景从同一个 `θ_meta` 独立开始，场景之间不串联。

新版在线适应配置：

```text
online_T=10, online_H=100, online_l=10
TD3 updates=100, FL rounds=1,000
optimizer lr=0.001
actor pre-tanh logit L2=1.0
logit mask=(1,0,0)，只解除 alpha 饱和
```

每个 online outer iteration：采一条 H=100 轨迹，然后在同一 Replay 上更新 10 次。每个 outer iteration 保存一次场景 checkpoint。

适应结束后，用两个独立 seed 分别评估 Frozen 和 adapted 的 mean Defender reward：

- adapted reward 严格更高：部署 adapted；
- 否则：回退 Frozen；
- guard 不使用测试集 accuracy。

最终评估使用 seed 101、H=100，并记录每轮 clean accuracy/loss、动作、reward 和采样恶意客户端数。

## 8. 正式结果

第一次在线评估直接部署 adapted 候选，多数场景相对 Frozen 没有变化，LMP 从 90.69% 降至 83.07%（-7.62 pp）。该负结果保留在 `evaluation_online_h100/` 和 `final_results.json`，用于说明原在线适应流程存在无效更新和负迁移风险。

下表是加入 anti-saturation、稳定学习率和独立 reward guard 后的新版结果。Online 候选列仍保留下降结果；Guard 后结果才是最终选择：

| 场景 | Frozen | Online 候选 | 候选变化 | Guard 后结果 |
|---|---:|---:|---:|---:|
| Clean | 91.55% | 92.10% | +0.55 pp | 92.10% |
| IPM | 90.36% | 89.71% | -0.65 pp | 90.36%（回退） |
| LMP | 90.69% | 91.11% | +0.42 pp | 91.11% |
| RL-ClipMed | 87.69% | 89.16% | +1.47 pp | 89.16% |
| RL-Krum | 89.96% | 89.91% | -0.05 pp | 89.96%（回退） |

权威结果文件：

- `evaluation_frozen_h100/summary.json`
- `evaluation_online_h100_v2_alpha_guarded/summary.json`
- `evaluation_online_h100_v2_alpha_guarded/rejected_candidates.json`

## 9. 与论文原文对照

原文：`2410.17431v1.pdf`，Algorithm 2 在第 18 页，实验设置与 Table 4 在第 18–19 页，在线适应描述在 Section II-C。

| 项目 | 论文 | 当前正式实验 | 说明 |
|---|---|---|---|
| 方法关系 | Algo2 是独立 Meta-RL baseline | 只训练/加载 `algorithm2_defender` | 一致；不是 Algo1 的后续阶段。 |
| 元学习 | T 轮、K 个任务、任务副本适应 l 步、Reptile 平均 | 同一结构 | 一致。 |
| 轨迹采样 | Algorithm 2 第 8–11 行：每个 inner step 采 H 轨迹 | 每个 inner step 新采 H 轨迹 | 与伪代码一致。 |
| 预训练规模 | `T=100,K=10,H=200,l=10`（MNIST） | `T=100,K=5,H=100,l=10` | 为资源和运行时间缩小 K、H。 |
| FL 拓扑 | 100 clients、20 attackers、10% sampling | 20 clients、4 attackers、每轮采 4 | 攻击者占比同为 20%；采样率由 10% 变为 20%。 |
| 本地训练 | batch 128、local iteration 1、lr 0.05 | 完全相同；iteration 实现为一个 minibatch step | 已修正原先“完整 local epoch”的错误解释。 |
| 数据 | 默认 IID；预训练强调生成/模拟数据 | IID、真实 MNIST、standard normalization | 未复现 cGAN/self-generated data。 |
| Meta-RL 域 | Table 4：NA/IPM/LMP/BFL/DBA | NA/IPM/LMP/RL-Krum/RL-ClipMed | 当前是 global poisoning 专用扩展，不是 Table 4 原域。 |
| 状态压缩 | 使用模型最后两层信息 | 最后两个可学习参数张量 | 工程近似，不等同于“完整最后两层”。 |
| 动作 | `(alpha,beta,epsilon/sigma)` | `(alpha,beta,epsilon)` | 结构一致；identity 下 epsilon 无效。 |
| 后处理 | NeuroClip 或 Prun | identity | 当前只研究全局模型投毒，主动关闭 NeuroClip。 |
| 单任务步长 | `κ=0.001` | 0.001 | 一致。 |
| Meta step | 1.0 | 0.25 | 稳定性覆盖值，属于工程差异。 |
| 在线步长 | 0.01 | 0.001 | 0.01 在当前 actor 上导致过冲，改为稳定值。 |
| 在线适应 | `T=10,H=100,l=10`、100 TD3 steps | 同参数；10 条轨迹、1,000 FL rounds | 更新数一致；轨迹预算采用部署型工程解释。 |
| 在线 reward | 自生成数据及在线推断数据估计 | 100 个 root samples 的 CE loss | 未实现梯度反演/数据生成部分。 |
| 安全选择 | 未给出 reward guard | anti-saturation + reward guard | 当前 enhanced 工程版本，不应冒充论文原始 TD3。 |
| sim-to-real | 预训练小模拟 FL，在线真实大 FL | 训练/评估均为 20-client 同拓扑 | 当前没有复现论文主要分布偏移。 |

论文自身有两处需要保留说明：

1. Table 4 的 Meta-RL 域为 NA/IPM/LMP/BFL/DBA，但 Appendix C 的文字又列出 IPM/LMP/EB；
2. Algorithm 2 明确在每个 inner step 采轨迹，而 Appendix C 的概述容易被读成“一条轨迹后更新 10 步”。当前预训练以 Algorithm 2 伪代码为准。

## 10. 运行与恢复

正式参数的唯一维护入口应优先看：

- `exp_scripts/run_algo2_formal_pipeline.py::_common_training_args`
- `exp_scripts/run_algo2_formal_pipeline.py::_evaluation_args`

训练中断后，训练 CLI 在原命令末尾增加：

```bash
--resume
```

在线评估会自动检测 `output/online_adaptation/<scenario>.pt` 并恢复，无需额外 `--resume`。

注意：pipeline 脚本的目录常量是实验专用硬编码。开始新实验前必须换新的 `FORMAL_DIR`/评估目录；否则已有 `summary.json` 会被判定为已完成而跳过执行。当前新版结果目录是 `evaluation_online_h100_v2_alpha_guarded`，不是历史的 `evaluation_online_h100`。

## 11. 其他优先事项

1. 正式论文表需要扩展到多个 evaluation seed，报告均值和置信区间；
2. 将 historical paper-TD3 与当前 enhanced online adaptation 分开命名；
3. 若恢复 NeuroClip/后门任务，必须重新启用并训练 epsilon 维，不能沿用 `(1,0,0)` mask；
4. 若目标是严格论文复现，需要恢复 Table 4 任务域、100-client 拓扑、生成数据和 sim-to-real；
5. 新实验开始前同步更新 pipeline 的硬编码目录及 `protocol.json` 字段。
