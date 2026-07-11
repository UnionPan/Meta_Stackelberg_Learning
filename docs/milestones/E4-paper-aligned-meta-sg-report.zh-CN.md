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
| RL environment | training rounds | 300 | Appendix C 的一般 RL 训练轮数 |
| Full FL | MNIST / CIFAR-10 rounds | 500 / 1000 | 完整 FL 实验训练轮数，不等于 trajectory `H` |
| FL | backdoor attackers | 5 | backdoor 实验恶意客户端数 |
| Root data | MNIST / CIFAR-10 | 100 / 200 | server root samples |
| Data split | default `q` | 0.5 | 默认 non-IID assignment bias |
| Generated data | seed samples / `q` | 200 / 0.1 | self-generated data 起点 |
| Backdoor reward | `lambda` | 0.5 | 默认 backdoor reward tradeoff |

`N_D` 不是“每个 FL step 再训练 Defender 10 次”，`N_A` 也不是从 10 个攻击方法中选一个。
`N_A` 次更新作用于同一个 attacker policy 参数序列
`φ(0) → … → φ(N_A)`，只有最终 `φ(N_A)` 被当作当前冻结 Defender 的近似 best response。
Algorithm 2 的 `l` 是另一套 Reptile 任务适应计数，不与 `N_D` 或 `N_A`互为别名。

Algorithm 1 严格区分两个 Defender 参数点：先按原文第 11 行用 `eta` 做一次适应得到
`theta_xi`；Attacker 随后按第 14 行针对未适应的 meta policy `theta_t` 做 `N_A` 次更新；
Defender 梯度再按第 17 行于 `theta_xi`、`phi_xi(N_A)` 条件下估计。各任务的
`kappa_D`-scaled 更新差量最后按第 24 行以 `1/K` 汇总到 `theta_t`。`kappa_A`、`eta`、
`kappa_D`各自只进入对应 optimizer 一次，不重复缩放。

Algorithm 2 的真实 TD3 runner 从同一个 `theta_t`隔离克隆 `K` 个任务策略；每个副本以
`kappa` 为 optimizer 步长执行恰好 `l` 次更新，任务 response policy 在整个适应块冻结。
完成后只执行一次 `meta_update_step/K` Reptile 参数更新。论文配置中 `meta_update_step=1`；
它不是任务 optimizer 的第二个 learning rate。

Defender 每个 FL round 输出三维连续动作 `(alpha, beta, epsilon)`；Attacker 每个 FL round
输出三维连续动作 `(gamma, E, lambda)`。动作网络范围统一为 `[-1, 1]^3`，再由严格 codec
映射到物理参数范围。

动作 decoder 的物理边界并非全部由 Meta-SG 论文给出，必须明确列为实现声明而不是论文参数：

| 动作参数 | 当前边界 | 来源与偏差 |
|---|---|---|
| Defender `alpha` | `[1e-6, observed_max_norm]` | 论文为 `(0,max norm]`；实现增加数值 floor |
| Defender `beta` | `[0,0.45]` | 论文写 `[0,1)`；对称 trimmed mean 必须 `<0.5`，实现保守 cap 0.45 |
| Defender `epsilon` | `[0.1,10]` | 论文未发布 decoder bounds，implementation-declared |
| Attacker `gamma` | `[0.1,2.9]` | RL-attacker-compatible declared；Meta-SG 未发布 bounds |
| Attacker `E` | integer `[1,19]` | RL-attacker-compatible declared；Meta-SG 未发布 bounds |
| Attacker `lambda` | `[0.05,0.95]` | RL-attacker-compatible declared；Meta-SG 未发布 bounds |

## 算法契约

- Algorithm 1 的调用轨迹严格为 `N_D × K × N_A` attacker updates，每个 `N_D` 只做一次
  leader update。
- best-response 训练期间用完整 TD3 snapshot fingerprint 冻结 Defender；snapshot 包括网络、
  optimizer、更新计数与随机数状态。
- role-local replay 另有不可变 snapshot，覆盖全部 transition、generation、ring cursor、size 与
  sampling RNG；联合 freeze guard 可同时检测 policy 或 replay 的任何变化。
- Algorithm 2 严格执行 `T × K × l` 个任务适应调用，每个 `T` 只做一次 Reptile 外更新。
- Reptile 更新 actor、双 critic 及其 target 网络；optimizer、计数器和 RNG 不做参数平均。
- 一个 Markov step 等于一个 FL round；该 round 内先 Defender 动作，再 Attacker 动作。
- Defender/Attacker 使用角色隔离 replay，且 transition 带 generation 标签。

## 验证证据

- 不可变缩放配置为 `T=2,K=2,H=8,l=2,N_A=2,N_D=2`，6 clients、3 attackers、
  sample size 4、TD3 batch/learning-start 16、hidden `(64,64)`、replay 4096；记录明确标为
  `scaled-conformance-only-v1`。
- 真实 `PaperBSMGEnv` 缩放 trajectory：8 个 FL rounds，每轮双方各执行一次三维连续动作；
  Algorithm 1 的 leader/attacker 次数与 Algorithm 2 的 meta/adaptation 次数均和上述配置精确相等。
- 真实 policy 训练 runner 已按上述缩放配置执行 48 条 support trajectories，共 384 个 FL
  rounds。`H=8`、batch/learning-start 16，因此每次 TD3 update 自动采集 2 条完整 trajectories。
  被训练角色使用 exploration，冻结对手使用 deterministic action，避免把对手 RNG 推进误当作参数更新。
- support seeds `(1,2)` 与 query seeds `(101,102)` 严格不相交；真实环境 query trajectories
  执行前后 Attacker 的完整 TD3 fingerprint 相同。任何 query 内 policy mutation 都会直接抛错。
- 使用 query seeds `(101,102)` 对 initial/learned policy pair 做了真实双策略冻结评估；learned
  Attacker 的 held-out action trajectories 与 initial 不同。该事实只证明行为变化，不单独作为性能通过证据。
- 科学 Gate 已实现六项不可调结果：attacker BR、Defender-conditioned response、Defender task
  adaptation、meta initialization、specialized-oracle regret、action/objective 双信号。
- 科学 Gate micro-run 使用预声明阈值 `0.001/0.001/0.001/0.001/0.1/0.001`、独立 query
  seeds `(101,102)` 执行了所有比较。结果为 **failed**，并原样保留：attacker BR improvement
  `0`、Defender adaptation improvement `0`、meta advantage `0`、behavior/objective signal `0`；
  Defender-conditioned response difference `0.0041427374` 和 specialized-oracle regret
  `0.0007139444` 通过。该 micro-run 的 `H=2,T=1,K=1,l=N_A=2,N_D=1`，只验证科学
  执行路径，不能支持性能结论。
- meta/random/no-adaptation 使用完全相同的 adaptation trajectory 数、FL-round 数和同一组
  support seeds；no-adaptation 消耗相同 rollout 预算但 TD3 update 数为零。每个被比较 Defender
  都从同一初始 Attacker 独立训练 fresh BR，finite specialized oracle 只在预声明网格内取最好值。
- 规定缩放训练 `T=2,K=2,H=8,l=N_A=N_D=2` 的 checkpoint 已继续输入同尺度独立 Gate。
  该 Gate 仍为 **failed**：attacker BR improvement `0`；Defender-conditioned response
  difference `0.0045558793`（通过）；Defender adaptation improvement `-0.0000438690`；meta
  advantage `-0.0000438690`；specialized-oracle regret `0.0005413890`（通过）；behavior/objective
  signal `-0.0000438690`。因此当前实现通过结构/conformance，但缩放性能 Gate 明确未通过。
- Algorithm 1、policy-level BR、Algorithm 2 和 Reptile 隔离测试均通过。
- 2026-07-12 全仓库测试：`837 passed in 65.02s`。

## 尚未宣称的结果

当前证据证明实现结构、调用次数、冻结/query 边界和缩放执行路径符合设计。科学 Gate 的执行器
已经存在，但尚未执行论文规模训练，因此没有伪造一个 performance pass，也不宣称达到论文准确率、
攻击收益或 meta-adaptation 性能。论文规模证据仍须训练并冻结所有比较策略，再使用预先声明的
阈值、独立 query seeds 和 held-out data 运行；query 数据不得参与更新或模型选择。
