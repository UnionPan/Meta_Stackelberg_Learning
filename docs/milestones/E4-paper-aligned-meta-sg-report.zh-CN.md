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
| Online | `T / l / steps` | 10 / 10 / 100 | 10 个连续块，每块 10 次更新，总计 100 次；不等于 pre-training `T` |
| Online | `H` | MNIST 100 / CIFAR-10 200 | online adaptation trajectory horizon |
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

Online adaptation 另有独立 runner，并强制 `online_steps = online_T × online_l`。同一个
Meta-SG policy copy 顺序执行全部 100 次更新，不在每个 online `T` 重新初始化，也不将
`online_T` 复用为 pre-training `T`。真实缩放测试 `T=2,l=2,H=2` 自动采集 8 条 trajectories
（16 个 FL rounds）完成 4 次连续更新，攻击策略全程冻结。

Defender 每个 FL round 输出三维连续动作 `(alpha, beta, epsilon)`；Attacker 每个 FL round
输出三维连续动作 `(gamma, E, lambda)`。动作网络范围统一为 `[-1, 1]^3`，再由严格 codec
映射到物理参数范围。

状态压缩按 Meta-SG 实现的 weight-list 语义取最后两个 parameter tensors，而不是最后两个
module blocks。论文 MNIST CNN 对应 `fc1.weight + fc1.bias = 1280 + 10 = 1290` 维；旧的
`conv3+fc` block 解释会产生约 41 万维状态并使 TD3 输入层不可接受，现已由测试锁定为 1290。

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
  其中 meta initialization 明确定义为 Algorithm 1 输出的 Meta-SG `theta_ND`；Algorithm 2
  meta-RL policy 只属于独立 baseline，不再错误替代 Meta-SG initialization。
- 规定缩放训练 `T=2,K=2,H=8,l=N_A=N_D=2` 的 checkpoint 已继续输入同尺度独立 Gate。
  预声明 Attacker oracle grid 为 raw gamma `{-0.8,+0.8}`（其余 raw dimensions 为 0），plateau
  gap threshold 为 `0.001`。`phi(N_A)` 的动作虽然变化，但 attacker objective 与 oracle plateau
  相同，因此 attacker BR 条件通过 plateau 分支，而不是 improvement 分支。总 Gate 仍为
  **failed**：attacker BR improvement `0`；Defender-conditioned response difference
  `0.0045560017`（通过）；Defender adaptation improvement `-0.0000438690`；meta
  advantage `-0.0000438690`；specialized-oracle regret `0.0005413890`（通过）；behavior/objective
  signal `-0.0000438690`。因此当前实现通过结构/conformance，但缩放性能 Gate 明确未通过。

为区分“2 步预算过小”和执行错误，另执行了保持 tiny environment、`T=2,K=2,H=8`，但恢复
论文关键计数 `l=N_A=N_D=10` 的扩大实验。训练实际消耗 560 条 trajectories，科学比较消耗
180 次 support rollout（meta/random/no-adaptation 的 matched seeds 会重复计入执行次数）。阈值未
改变。Gate 仍为 **failed**：attacker plateau 分支通过（improvement `0`）；Defender-conditioned
response difference `0.0170114445`；Defender adaptation improvement `0`；meta advantage
`-0.0000666976`；specialized-oracle regret `0.0007151961`；双 action/objective signal 失败。
因此失败不能用“只训练了 2 步”解释；当前 tiny 四客户端、固定 benign delta 环境对连续防御动作的
held-out objective 可辨识性不足。后续性能结论必须迁移到论文 MNIST/CIFAR 数据生成与客户端训练，
不能继续在该 tiny 环境上事后调 decoder 或阈值。

## MNIST 论文环境迁移

已增加不依赖 legacy package 的 canonical MNIST 路径：8×8、6×6、5×5 convolution kernels
及 10-logit classifier；Appendix C `paper_q` partition；root samples 从 client training indices
严格移除；真实 `TorchLocalTrainer` SGD；100 workers/20 untargeted attackers/10 sampled clients；
attackers 在 10 个 class groups 中均匀分布。Uniform sampling 允许合法的 zero-malicious round，
不再因为某轮未采到攻击者而终止 episode。

MNIST loader 默认 `download=False`，避免实验隐式访问网络；调用者可显式下载或传入已有 Dataset。
synthetic MNIST-shaped integration 已跑通一个真实 local-SGD/RL-local-search FL round。尚未执行
论文 cGAN 生成数据训练，因此扩大 tiny Gate 的失败仍不能被描述成 MNIST 复现结果。

随后已显式下载 torchvision MNIST 到临时目录并验证：client train 59,900、server root 100、test
10,000，root indices 全部唯一且不进入 client train。真实 seed 23 round 采到 0 attackers，root loss
`2.3053006 → 2.2957854`；seed 24 round 采到 1 attacker，1290 维状态与 RL local-search 均执行，
root loss `2.3098135 → 2.3081958`。该执行同时发现并修复 torchvision label 为 Python `int` 时
local-search 错误调用 `torch.stack` 的兼容 bug。以上仍是 one-round execution evidence，不是训练性能。

现已提供 `run_paper_mnist_scaled_evidence` 单一入口，并通过 reusable
`PaperMNISTEnvironmentFactory` 固定 paper-q partition 与 initial global model；rollout seed 只控制
trajectory randomness，不再错误地重分数据或重置不同模型。真实 MNIST 最小端到端配置
`T=K=H=l=N_A=N_D=1` 已完成 4 条 training trajectories 及全部 held-out comparisons。
Gate 为 **failed**：attacker plateau、Defender-conditioned response (`0.0027188196`) 与 oracle
regret (`0.0003176880`) 通过；adaptation/meta/action-objective signal 均为 `0`。由于 TD3
`policy_delay=2` 而该 smoke 每个 block 只有 1 update，actor 不变化是预期结果，不能作为性能证据。

进一步执行最小 actor-active MNIST 配置 `T=K=H=1,l=N_A=N_D=2`、TD3 batch/learning-start
2，共 20 条 training trajectories，阈值保持不变。Gate 仍为 **failed**：Attacker objective
`-0.0046645355 → -0.0046647644`，无 improvement 但在 finite-oracle plateau 内；不同 Defender
response difference `0.0039335111`；Defender adaptation 与 meta advantage 均为
`+0.0002442932`，方向转正但低于预声明 `0.001`；specialized-oracle regret
`-0.0000406647`；action/objective 双信号因 Attacker objective 未改善而失败。真实 MNIST 已出现
tiny 环境没有的正向 adaptation signal，但不能据此降低阈值宣布通过。

继续执行 `T=K=H=1,l=N_A=N_D=4`、batch/learning-start 2，共 56 条 training
trajectories。Gate 仍为 **failed**：Attacker BR improvement `+0.0000007629` 并满足 plateau；
Defender-conditioned response `0.0057763009`；Defender adaptation/meta advantage
`+0.0002378845`，仍低于 `0.001`；oracle regret `0.0001184845`；action 与 held-out objective
双信号首次同时通过。2→4 步没有提升 adaptation magnitude，因此不能继续用“步数必然解决”解释。

Scientific check record 现分别存储 primary 与 alternative observed/threshold：Attacker check 同时记录
improvement 与 oracle plateau gap；双信号 check 分别记录 action distance 与 objective improvement，
避免旧格式出现显示的 observed 与 threshold 不属于同一判据的问题。

该端到端执行最初还暴露了 local-search 数值根因：初始 `current==global` 时 deviation 为零，
cosine similarity 未定义，`eps=1e-12` 产生约 `1e12` 梯度并导致 MNIST malicious update 非有限。
现仅在 exact-zero deviation 处将 cosine 项定义为零值/零梯度，让 empirical loss 产生第一条方向；
线性回归案例的 update norm 从 `5.54e10` 降至 `<10`，相同真实 MNIST 端到端随后完整通过。

为支持更长 MNIST 运行，增加 trusted-local 原子 checkpoint：一次保存/恢复 online/target
actor、双 critics、optimizers、policy RNG/counter，以及 role-local replay 的全部 transitions、
generation、ring cursor 和 sampling RNG。加载前验证 role 与 policy/replay shapes，加载后重新计算
双方 fingerprint；临时文件通过 atomic replace 提交。PyTorch pickle checkpoint 明确只允许加载本地
可信产物。

Policy Algorithm 1/2 现支持 outer-boundary resume：`start_iteration` 使用全局 `N_D`/`T` index，
checkpoint callback 只在一个完整 leader/meta iteration（包含全部 `K` tasks、`N_A` BR 或 `l`
adaptation）结束后触发，禁止从半个 BR block 恢复而改变 Stackelberg 更新语义。

## CIFAR-10 / ResNet-18 路径

已实现 paper CIFAR ResNet-18 与 5130 维尾部状态（`linear.weight=5120`、`linear.bias=10`）。
为避免只聚合 parameters 而丢失 BatchNorm 状态，新增 `TorchModelStateCodec`：FL state 顺序为
全部 learnable parameters 后接所有 floating persistent buffers，包含 running mean/variance；整数
`num_batches_tracked` 不进入浮点聚合状态。local-search 的 cosine 几何只切取 parameter 部分，
但最终 benign/malicious delta、clip/trim 和 server update 都覆盖完整浮点 buffer state。

synthetic CIFAR-shaped integration 已用真实 ResNet-18、local SGD、RL local-search 和 buffer-aware
aggregation 跑通一轮；Defender/Attacker observation dimensions 分别为 5131/5135，并提供
`run_paper_cifar_scaled_evidence` 一键训练/Gate 入口。尝试显式下载 torchvision CIFAR-10 时外部源
仅约 40 KB/s，170 MB 文件预计超过一小时，已中止，因此不宣称完成真实 CIFAR 数据运行。

Scaled evidence 现在可原子持久化为 `policies.pt + manifest.json`：前者包含 Algorithm 1/2
Defenders 与全部 type-specific Attackers 的完整 TD3 snapshots；后者包含参数快照、training/query
seeds、trajectory budgets、fresh-BR 数量、oracle labels、六项 primary/alternative Gate records 和
所有 policy fingerprints。这样长运行结束后可继续独立诊断，而不再丢失进程内 policies。
- Algorithm 1、policy-level BR、Algorithm 2 和 Reptile 隔离测试均通过。
- 2026-07-12 全仓库测试：`866 passed in 69.63s`。

## 尚未宣称的结果

当前证据证明实现结构、调用次数、冻结/query 边界和缩放执行路径符合设计。科学 Gate 的执行器
已经存在，但尚未执行论文规模训练，因此没有伪造一个 performance pass，也不宣称达到论文准确率、
攻击收益或 meta-adaptation 性能。论文规模证据仍须训练并冻结所有比较策略，再使用预先声明的
阈值、独立 query seeds 和 held-out data 运行；query 数据不得参与更新或模型选择。
