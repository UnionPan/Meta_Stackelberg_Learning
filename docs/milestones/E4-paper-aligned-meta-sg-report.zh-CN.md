# E4：论文对齐 Meta-SG / TD3 实现报告

## 结论边界

本阶段完成的是论文算法语义和参数语义的可执行对齐，以及小规模确定性 conformance
验证；这不是论文规模的性能复现。完整论文规模仍需按原始数据集、100 个 worker 和论文给定的
长 horizon 独立运行。

## 参数设计

参数被分成互不混用的三组：

| 层级 | 参数 | 论文默认值 | 精确含义 |
|---|---:|---:|---|
| Algorithm 1 | `N_D` | 10 | leader/Defender 外迭代次数 |
| Algorithm 1 | `N_A` | 10 | 冻结一个 Defender policy 后，attacker policy 的连续更新次数 |
| Algorithm 2 | `T` | 100 | Reptile 元迭代次数 |
| Algorithm 2 | `K` | 10 | 每个元迭代采样的攻击任务数 |
| Algorithm 2 | `l` | 10 | 每个任务副本上的适应更新次数 |
| Episode | `H` | MNIST 200 / CIFAR-10 500 | 一条 trajectory 的 FL round 数 |
| TD3 | learning rate | 0.001 | actor/critic optimizer learning rate |
| TD3 | batch size | 256 | 每次 TD3 更新采样数 |
| TD3 | `gamma` | 0.99 | Markov return 折扣 |
| FL | workers / attackers | 100 / 20 | 客户端总体与恶意总体 |
| FL | subsampling | 10% | 每个 FL round 的客户端采样比例 |

`N_D` 不是“每个 FL step 再训练 Defender 10 次”，`N_A` 也不是从 10 个攻击方法中选一个。
`N_A` 次更新作用于同一个 attacker policy 参数序列
`φ(0) → … → φ(N_A)`，只有最终 `φ(N_A)` 被当作当前冻结 Defender 的近似 best response。
Algorithm 2 的 `l` 是另一套 Reptile 任务适应计数，不与 `N_D` 或 `N_A`互为别名。

Defender 每个 FL round 输出三维连续动作 `(alpha, beta, epsilon)`；Attacker 每个 FL round
输出三维连续动作 `(gamma, E, lambda)`。动作网络范围统一为 `[-1, 1]^3`，再由严格 codec
映射到物理参数范围。

## 算法契约

- Algorithm 1 的调用轨迹严格为 `N_D × K × N_A` attacker updates，每个 `N_D` 只做一次
  leader update。
- best-response 训练期间用完整 TD3 snapshot fingerprint 冻结 Defender；snapshot 包括网络、
  optimizer、更新计数与随机数状态。
- Algorithm 2 严格执行 `T × K × l` 个任务适应调用，每个 `T` 只做一次 Reptile 外更新。
- Reptile 更新 actor、双 critic 及其 target 网络；optimizer、计数器和 RNG 不做参数平均。
- 一个 Markov step 等于一个 FL round；该 round 内先 Defender 动作，再 Attacker 动作。
- Defender/Attacker 使用角色隔离 replay，且 transition 带 generation 标签。

## 验证证据

- 真实 `PaperBSMGEnv` 小规模 trajectory：2 个 FL rounds，每轮双方各执行一次三维连续动作；
  两个角色 replay 各得到 2 条 transition。
- Algorithm 1、policy-level BR、Algorithm 2 和 Reptile 隔离测试均通过。
- 2026-07-12 全仓库测试：`817 passed in 54.19s`。

## 尚未宣称的结果

当前证据只证明实现结构、调用次数、冻结边界和小规模执行路径符合设计。尚未执行论文规模训练，
因此不宣称达到论文准确率、攻击收益或 meta-adaptation 性能。后续科学 Gate 应使用独立 query
trajectories 比较 `φ(N_A)` 与 `φ(0)`、不同 Defender、adapted 与 initial Defender，以及在相同
预算下 meta initialization 与 random/no-adaptation；query 数据不得参与更新或模型选择。
