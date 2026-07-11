# 证据链驱动的 Meta-Stackelberg 研究系统设计

## 1. 文档状态

本文档是新 `meta_stackelberg` 系统的权威设计规范。现有 `fl_sandbox`、`meta_sg` 和 `src` 只作为算法行为、实验配置和历史结果的参考，不约束新系统的模块边界。

设计采用已经确认的路线：先建立可证伪的证据链，再逐步实现 FL、攻击、防御、best response、单任务 RL、快速适应和 Meta-SG 泛化。

## 2. 研究目标

系统要回答的核心问题是：

> 在攻击类型、攻击配置和恶意客户端组成未知，并且攻击者可能针对防御策略进行适应的联邦学习中，能否学习一个防御策略初始化，使服务器只用有限在线反馈就能快速适应当前任务，同时保持 clean utility 并降低攻击效果？

目标不是训练一个在平均攻击上表现尚可的固定策略，也不是让两个 RL agent 同时更新后观察 reward 是否上升。

目标对象是：

```text
meta initialization θ_meta
    + limited support feedback
    + explicit adaptation rule A
    -> task-adapted defender θ_ξ
```

并且需要在 attacker 对 leader policy 的 best response 下评价 `θ_ξ`。

## 3. 必须成立的研究假设

系统按 E0-E6 七级证据链推进。

### E0：FL 正确性

需要证明：

- clean FL 在小型确定性设置下朝正确方向优化；
- client update、aggregate update 和 server update 的符号一致；
- loss、accuracy、ASR 的定义不依赖 batch size 或输出格式；
- snapshot/restore 能重放相同的 round；
- 聚合算法通过手工可计算样例和性质测试。

E0 不成立时，不允许解释任何攻击、防御或 RL 结果。

### E1：攻击有效性

需要证明：

- untargeted 攻击能降低 clean utility 或增加 clean loss；
- targeted 攻击能提高定义明确的 ASR；
- 攻击预算或强度与攻击效果存在可解释关系；
- clean/no-op attack 不改变基线行为；
- 攻击只能使用 threat model 允许的信息。

### E2：防御动作可控性

不训练 RL，使用网格或响应面实验验证：

- 不同防御动作确实改变 aggregate update；
- clean utility、attack harm 和 defense cost 之间存在可辨识 trade-off；
- 不同任务的较优防御区域不同；
- 动作空间没有大片无效区或完全平坦区。

如果响应面近似平坦，则问题在动作设计或 FL 信号，不在 RL 算法。

### E3：Follower Best Response 成立

固定 leader policy，训练 follower response，验证：

- follower objective 在独立评估轨迹上改善；
- policy 参数变化对应行为变化，而不只是数值变化；
- leader policy 改变时，follower response curve 随之改变；
- 固定攻击作为零训练 response oracle 能使用同一接口。

### E4：单任务 Defender 可学习

在单个固定任务和冻结 follower 下验证：

- learned defender 优于 random policy；
- learned defender 优于固定中点动作；
- learned defender 接近有限动作网格 oracle；
- 改善出现在独立 evaluation trajectory，而不是 replay 数据上。

### E5：快速适应成立

比较相同 support budget 下的：

- meta initialization；
- 普通多任务预训练初始化；
- 随机初始化；
- 不适应的 frozen policy。

需要证明 meta initialization 的 query performance 更高，或者达到同一目标所需 support 数量更少。

### E6：未知任务泛化

在未参与训练和选择的任务上验证：

- 未见 attack method；
- 未见 attack budget；
- 未见 attacker fraction；
- 未见 non-IID 程度；
- 未见 target/trigger 配置；
- simulator 到真实 FL 的分布变化。

最终比较关系至少包括：

```text
meta pre-training + online adaptation
vs. meta pre-training without adaptation
vs. multi-task pre-training + adaptation
vs. random initialization + adaptation
vs. fixed defenses
```

## 4. 四个时间尺度

### 4.1 FL Round

一个 round 是唯一允许改变 FL deployed state 的基本转换：

```text
RoundState_t
  + TaskSpec
  + DefenderAction_t
  + AttackerAction_t
  + RandomState_t
      -> RoundTransition_t
      -> RoundState_{t+1}
```

FL round 不计算 RL reward，不更新 RL agent，不写 TensorBoard，不选择 checkpoint。

### 4.2 Episode

固定一个任务 `ξ` 和双方 policy，执行 H 个 FL rounds：

```text
EpisodeSpec(ξ, H, initial_snapshot)
    -> Trajectory_ξ
```

Episode 负责 observation、action 调用、feedback reward 和 termination，但不执行梯度更新。

### 4.3 Best Response / Adaptation

两种互斥过程：

```text
follower best response:
    freeze leader policy
    update follower

leader adaptation:
    freeze follower response
    update leader from support data
```

任何同时更新双方的过程必须命名为 `joint co-learning`，不能作为严格 Stackelberg best response 的证据。

### 4.4 Meta Iteration

从训练任务分布采样 K 个任务，对每个任务独立执行 follower response、support adaptation 和 query evaluation，最后执行 meta update。

```text
TaskBatch
  -> per-task BestResponseResult
  -> per-task SupportResult
  -> per-task QueryResult
  -> MetaUpdateResult
```

## 5. Stackelberg 承诺与信息结构

Stackelberg 语义是 defender 先承诺 policy，attacker 知道或通过交互推断该 policy 并求 best response。

不默认 attacker 能看到当轮完整 defender action。任务必须显式声明：

```python
FollowerKnowledge(
    knows_defender_policy: bool,
    observes_current_defender_action: bool,
    observes_aggregator_family: bool,
    observes_defense_parameters: bool,
    observes_benign_updates: bool,
    observes_global_model: bool,
    has_white_box_gradients: bool,
)
```

三种建议的标准 threat model：

1. `black_box`：只看到 global model 和公开协议；
2. `gray_box`：知道 aggregation family，但不知道动态参数；
3. `white_box`：知道 defender policy 或参数，用于 worst-case pre-training。

所有实验报告必须记录使用的 threat model。

## 6. Task ξ 的定义

任务是 Meta-SG 的一等对象：

```python
TaskSpec(
    task_id,
    dataset,
    model,
    data_partition,
    num_clients,
    sample_fraction,
    malicious_population,
    attack_objective,
    attack_method,
    attack_budget,
    attacker_knowledge,
    defense_action_space,
    feedback_access,
    horizon,
    seed_family,
)
```

`attack_method` 相同不代表任务相同。恶意比例、target、trigger、non-IID 程度、攻击预算或信息能力不同，都构成不同任务。

任务分布分为：

```text
Q_train       用于训练 meta initialization
Q_validation  用于超参数、checkpoint 和候选选择
Q_test        只用于最终报告
```

`Q_test` 的任何数据不得进入 policy update、reward normalization、checkpoint selection 或 candidate ranking。

## 7. 权威数据模型

### 7.1 模型状态

```python
@dataclass(frozen=True)
class ModelState:
    tensors: tuple[np.ndarray, ...]
```

必须通过统一函数完成：clone、flatten、unflatten、difference、apply_delta、norm 和 cosine。模块不能各自定义权重正负方向。

### 7.2 ClientUpdate

```python
@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    delta: ModelState
    num_examples: int
    is_malicious: bool
    metadata: Mapping[str, Scalar]
```

`delta` 统一定义为：

```text
delta = local_model - global_model
```

server update 统一定义为：

```text
global_next = global + server_lr * aggregate_delta
```

### 7.3 RoundState

```python
@dataclass(frozen=True)
class RoundState:
    round_index: int
    global_model: ModelState
    server_optimizer_state: object
    random_state: RandomSnapshot
    component_states: Mapping[str, object]
```

### 7.4 RoundTransition

```python
@dataclass(frozen=True)
class RoundTransition:
    task_id: str
    state_before: RoundState
    sampled_clients: tuple[int, ...]
    benign_updates: tuple[ClientUpdate, ...]
    malicious_updates: tuple[ClientUpdate, ...]
    aggregate_delta: ModelState
    state_after: RoundState
    public_signals: Mapping[str, ArrayOrScalar]
    private_diagnostics: Mapping[str, ArrayOrScalar]
```

`public_signals` 可用于构造合法 observation。`private_diagnostics` 只用于 oracle evaluation 和调试，不能意外进入 policy input。

### 7.5 Trajectory

```python
@dataclass(frozen=True)
class Trajectory:
    task_id: str
    initial_snapshot: RoundState
    steps: tuple[GameStep, ...]
    final_state: RoundState
```

每个 `GameStep` 包含 observation、raw action、physical action、reward components、transition 和 termination flags。

## 8. 模块边界

```text
meta_stackelberg/
  core/
  federated/
  security/
  tasks/
  feedback/
  environments/
  agents/
  stackelberg/
  meta_learning/
  evaluation/
  experiments/
  cli/
  compat/
```

### 8.1 core

负责不可变数据类型、数组/模型状态操作、RNG capture/restore、序列化版本和协议基础类型。

不得依赖 Torch 数据集、FL、RL 或实验代码。

### 8.2 federated

负责 dataset、partition、client sampling、local training、server update、模型 registry 和 deterministic round engine。

不知道 reward、TD3、Meta-SG、TensorBoard 或 CLI。

### 8.3 security

负责 fixed/adaptive attack execution、attack capability、aggregation defense、post-training defense 和 physical action codec。

攻击产生 `ClientUpdate`；防御消费 updates 并产生 aggregate delta 或 evaluation-only model copy。

### 8.4 tasks

负责 TaskSpec、task distribution、train/validation/test split 和 reproducible sampling。

### 8.5 feedback

负责训练时可用反馈、reward components 和约束。它只使用 task 声明允许的服务器信息。

### 8.6 environments

负责将 FL transition 转换为 attacker、defender 或 joint RL 环境。环境不实现 RL update。

### 8.7 agents

负责 policy、replay、TD3、SAC、MATD3 和 scripted policy。agent 不加载数据集、不创建 FL engine。

### 8.8 stackelberg

负责 leader commitment、follower best response、response oracle 和 leader objective。

### 8.9 meta_learning

负责 support、query、adaptation、Reptile/meta update 和 task batch orchestration。

### 8.10 evaluation

负责 oracle metrics、baselines、response surface、response curve、adaptation curve、统计检验和 protocol version。

### 8.11 experiments

负责依赖注入和对象组装，运行阶段实验，写 checkpoint、JSON、CSV 和 TensorBoard。

### 8.12 cli

只解析参数、加载 ExperimentSpec 并调用 experiment service。

### 8.13 compat

只负责读取旧 config/checkpoint 或提供迁移 wrapper。canonical 模块不能依赖 compat。

## 9. 可插拔协议

核心组件通过 `typing.Protocol` 或等价抽象注入。

```python
class ClientSampler(Protocol):
    def sample(self, state: RoundState, task: TaskSpec, rng: RandomSource) -> tuple[int, ...]: ...

class LocalTrainer(Protocol):
    def train(self, client_id: int, state: RoundState, task: TaskSpec, rng: RandomSource) -> ClientUpdate: ...

class AttackStrategy(Protocol):
    capabilities: AttackCapabilities
    def craft(self, context: AttackContext, action: AttackerAction) -> tuple[ClientUpdate, ...]: ...

class Aggregator(Protocol):
    def aggregate(self, updates: Sequence[ClientUpdate], context: AggregationContext) -> ModelState: ...

class ObservationEncoder(Protocol):
    def encode(self, history: TransitionHistory, task: TaskSpec) -> np.ndarray: ...

class RewardFunction(Protocol):
    def evaluate(self, transition: RoundTransition, feedback: FeedbackRecord) -> RewardRecord: ...

class Policy(Protocol):
    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray: ...
```

每种插件必须通过共享 contract test suite。仅注册成功或返回正确 shape 不算功能正确。

## 10. Observation 设计

Observation 是实验变量，不写死为模型尾层。

Defender 可选特征：

- global model 的稳定低维编码；
- client update norm 分位数；
- pairwise distance/cosine 分布；
- aggregate 前后变化；
- 最近 clean proxy trend；
- 最近 attack proxy trend；
- 历史 defender actions；
- sampled client 数量和 round progress。

Attacker 可选特征：

- threat model 允许的 global model 编码；
- benign/reference update statistics；
- 公开 defense family 或参数；
- 上一轮 poison survival；
- attack budget 和剩余 horizon；
- target objective 的 proxy signal。

编码器至少支持：

1. `UpdateStatisticsEncoder`，作为第一默认；
2. `RandomProjectionEncoder`；
3. `ModelTailEncoder`，只作为论文基线。

E2 前不得选择最终 observation；应通过可辨识性和消融实验确定。

## 11. Action 设计

### 11.1 Defender action

E2 第一版只使用连续、可解释、能网格扫描的 aggregation action：

```text
clip_radius α
trim_ratio β
server_lr η_server（可选独立实验）
```

post-training defense 在第一版中作为独立决策阶段，不与 aggregation action 混成一个难以解释的连续空间。

确认 E2 后，再评估 hierarchical action：先选 defense family，再输出该 family 参数。

### 11.2 Attacker action

固定攻击使用确定性或脚本化参数，不伪装成 RL action。

自适应攻击的 action 必须对应可执行攻击 operator 参数，例如：

- malicious update scale；
- local optimization steps；
- stealth/bypass trade-off；
- poison ratio 或 trigger strength；
- mixture weights over attack operators。

第一阶段不允许 agent 直接输出完整高维模型更新，除非有单独的低维生成器和可验证 decoder。

## 12. Feedback、Reward 与 Oracle Evaluation

### 12.1 FeedbackEvaluator

训练时只允许使用服务器现实中拥有或可推断的信息：

- root/validation data；
- update statistics；
- trusted reference update；
- generated/inferred proxy data；
- task 中声明公开的协议参数。

### 12.2 OracleEvaluator

只用于离线科学报告，可使用：

- 完整 clean test set；
- 真实 attacker identity；
- 真实 target label 和 trigger；
- 真正的 ASR denominator；
- private transition diagnostics。

Oracle 结果不能进入 replay、gradient、normalizer、checkpoint selection 或 candidate ranking。

### 12.3 RewardRecord

不只存一个 scalar：

```python
RewardRecord(
    scalar,
    clean_utility,
    attack_harm,
    defense_cost,
    action_cost,
    constraint_violations,
    source,
)
```

Defender reward 由明确组件组合：

```text
clean utility improvement
- attack harm penalty
- defense/action cost
- constraint violation penalty
```

Attacker reward 由：

```text
attack harm
- detectability/stealth cost
- attack budget cost
```

每个组件单独记录，避免只看到总 reward 而无法解释策略行为。

## 13. 数据流

### 13.1 E0 Clean FL

```text
ExperimentSpec
 -> TaskSpec(clean)
 -> DatasetProvider
 -> DataPartitioner
 -> InitialModelFactory
 -> RoundState_0
 -> ClientSampler
 -> LocalTrainer[]
 -> ClientUpdate[]
 -> Aggregator
 -> ServerOptimizer
 -> RoundState_1
 -> OracleEvaluator
 -> RoundReport
```

### 13.2 E1 Attack FL

```text
RoundState
 -> sample clients
 -> benign local updates
 -> AttackContextBuilder
 -> AttackStrategy.craft
 -> benign + malicious updates
 -> Aggregator
 -> next RoundState
 -> oracle clean/attack metrics
```

### 13.3 E2 Defense Response Surface

```text
matched InitialSnapshot
 × TaskSpec
 × DefenderAction grid
 × repeated seeds
 -> independent episodes
 -> clean utility / attack harm / cost surface
 -> controllability report
```

### 13.4 E3 Best Response

```text
committed DefenderPolicy
 -> clone follower initialization
 -> collect follower support trajectories
 -> follower updates
 -> frozen follower evaluation
 -> BestResponseResult
```

### 13.5 E4 Single-task Defender

```text
fixed TaskSpec
 + frozen ResponseOracle
 -> defender rollouts
 -> defender updates
 -> independent query evaluation
 -> compare random / fixed / grid oracle
```

### 13.6 E5 Meta Iteration

```text
Q_train.sample(K)
 -> for each task:
      follower response
      clone θ_meta
      support adaptation
      query evaluation
 -> MetaUpdater
 -> θ_meta_next
 -> Q_validation checkpoint selection
```

### 13.7 E6 Held-out Evaluation

```text
frozen selected checkpoint
 + frozen protocol
 + Q_test
 + repeated seeds
 -> adaptation curves
 -> fixed/multitask/random baselines
 -> confidence intervals
 -> final report
```

## 14. 单模块验证要求

### core

- model flatten/unflatten round trip；
- delta sign hand calculation；
- clone 不共享内存；
- RNG capture/restore 重放 Python、NumPy、Torch。

### federated

- partition reproducibility；
- one-client local SGD 与手工梯度方向一致；
- client order 不影响对称 aggregation；
- round snapshot 重放一致。

### security

- attack capability violation 立即失败；
- no-op attack 与 clean 相同；
- fixed attack numerical fixtures；
- aggregation clipping/trim bounds；
- post-training defense 不改变 deployed state。

### tasks

- task serialization round trip；
- train/validation/test 无交集；
- sampling seed 可重复；
- task fingerprint 稳定。

### feedback

- public/private 信息隔离；
- reward component sum 等于 scalar；
- OracleEvaluator 不能被训练对象引用；
- metric denominator hand fixtures。

### environments

- observation/action space contract；
- reset/step/snapshot；
- terminated/truncated 区分；
- environment step 不更新 agent。

### agents

- fixed replay batch 的 target/loss/gradient；
- deterministic checkpoint reload；
- terminal mask；
- delayed update/target smoothing/Polyak update。

### stackelberg

- leader freeze guard；
- follower freeze guard；
- fixed response zero training；
- response improvement 使用独立 evaluation。

### meta_learning

- support/query buffer 隔离；
- query 无梯度和无 mutation；
- exact Reptile interpolation；
- task failure 不污染其他 task state。

## 15. 集成验证要求

集成顺序固定：

```text
core
 -> clean round
 -> clean episode
 -> fixed attack round
 -> defense response surface
 -> attacker best response
 -> single-task defender
 -> support/query adaptation
 -> meta iteration
```

每次只增加一个新边界。失败时能够定位到最近增加的两个模块之间。

## 16. 预期结果和失败解释

每个阶段报告必须同时写“成功预期”和“失败意味着什么”。

| 阶段 | 成功预期 | 失败优先排查 |
|---|---|---|
| E0 | clean loss 下降、确定性重放、聚合性质成立 | update 符号、optimizer、数据、metric |
| E1 | 攻击相对 clean baseline 产生方向正确的 harm | attack 实现、ASR 定义、预算 |
| E2 | response surface 非平坦且存在 trade-off | action codec、aggregation、observation signal |
| E3 | follower eval objective 提升且随 leader 改变 | reward、replay、information set、freeze |
| E4 | learned defender 优于 random/fixed | observation、reward、action range、RL update |
| E5 | meta init 在相同 support budget 更快 | task relatedness、support/query、meta update |
| E6 | held-out 平均和不确定性优于基线 | distribution shift、selection leakage、过拟合 |

## 17. 阶段交付文档

每个 E 阶段生成中文报告：

```text
docs/milestones/E<N>-<name>-report.zh-CN.md
```

固定章节：

1. 阶段假设；
2. 模块和接口；
3. 数据流；
4. 测试配置；
5. 预期结果；
6. 实际结果；
7. 单元测试证据；
8. 集成测试证据；
9. 与基线的差异；
10. 已知限制；
11. 是否通过阶段 gate；
12. 下一阶段前置条件。

## 18. 可复现性规则

每次运行记录：

- resolved config；
- task fingerprint；
- code commit；
- Python/Torch/NumPy 版本；
- device；
- initial model hash；
- data partition hash；
- seed family；
- checkpoint schema version；
- evaluation protocol version。

机器本地路径和被忽略 artifact 不能成为默认测试前提。

## 19. 错误处理

- shape、action dimension、capability 和 checkpoint schema 错误必须在 round 开始前失败；
- metric 非有限值必须携带来源并中止对应实验，不能静默替换为 0；
- task 失败必须隔离，不得复用部分更新后的 agent 或 replay；
- snapshot 缺少组件状态时禁止声明 deterministic replay；
- evaluation protocol 不匹配时禁止比较结果。

## 20. 实施顺序

```text
E0.1 core types + RNG
E0.2 data/model/client contracts
E0.3 aggregation + server update
E0.4 deterministic clean round
E0.5 clean episode + report

E1 fixed untargeted attacks
E1 fixed backdoor attacks
E1 attack validity report

E2 action codec
E2 response surface runner
E2 controllability report

E3 follower environment
E3 best response solver
E3 response curve report

E4 defender environment
E4 single-task training
E4 oracle baseline comparison

E5 task distributions
E5 support/query adaptation
E5 meta update

E6 frozen protocol
E6 repeated-seed evaluation
E6 final evidence report
```

## 21. 当前代码的使用原则

现有代码仅按以下方式使用：

- 作为 numerical fixture 的生成参考；
- 作为 checkpoint/config migration 来源；
- 作为论文复现 baseline；
- 作为已知问题样本。

不直接复制以下结构：

- 超大 runner/service/script；
- 多套重复 weights/update helper；
- test set 直接作为训练 reward；
- 脚本内部 business logic；
- 同时更新 attacker/defender 却称为 best response；
- 未保存 RNG 的 snapshot；
- 静默缓存或替换非有限评估值。

## 22. 设计完成标准

本设计实施完成时，应能回答：

1. 每个数值从哪里产生、流向哪里、是否公开给 policy；
2. 每个模块能否替换且通过同一 contract tests；
3. 每个研究结论由哪个阶段 gate 和哪个 protocol 支撑；
4. 任意失败能否定位到单模块或相邻模块边界；
5. Meta-SG 的改善是否来自快速适应，而不是测试泄漏、共享 replay、随机状态或选择偏差。
