# E0.4–E1 攻击有效性详细设计

## 1. 文档状态与目的

本文细化已经批准的“证据链驱动 Meta-Stackelberg 研究系统”中 E1 攻击有效性阶段。总体研究问题、四个时间尺度和 E0–E6 gate 不变。

本阶段要建立的不是“攻击代码能运行”，而是以下可证伪命题：

1. matched clean/no-op 对照在相同初始模型、客户端采样和本地训练随机性下完全一致；
2. untargeted attack 在不使用越权信息时，能对 clean utility 造成方向正确的 harm；
3. targeted backdoor attack 能在独立 held-out source-class 数据上提高 ASR；
4. attack budget 在算子层具有精确含义，在 Episode 层产生可解释的 endpoint effect；
5. oracle clean/ASR 指标不能流回攻击生成、训练反馈或状态选择。

## 2. 发现的前置缺口：E0.4

E0.3 已用测试中的显式循环证明 10 个 clean rounds 可学习并精确续跑，但 canonical package 尚无第一类 Episode runner。另一个更隐蔽的问题是：当前 `RoundEngine` 把同一个 `RandomSource` 依次传给 sampler 和所有 local trainers。若 attack 分支比 clean 分支多消耗随机数，后续客户端采样和 honest local shuffle 就可能不同，matched comparison 会受到随机路径混杂。

因此 E1 前增加 E0.4，而不是把这些职责塞进 attack engine：

- `RandomSource.spawn()`：父流只产生一个确定性 child seed；child 内消费多少随机数都不再影响父流；
- round engine 每轮固定派生一个 sampling child stream，再按 sampled order 为每个客户端派生一个 client child stream；
- `EpisodeRunner`：固定 task、initial state、horizon 和 round executor，运行 H 个 rounds 并返回 immutable `FederatedTrajectory`；
- Episode 不计算 reward、不更新 agent、不选择 checkpoint、不写 TensorBoard。

clean 与 attacked Episode 从同一个 snapshot 开始时，父流每轮消费数量一致，因此 sample schedule 和 per-client child seeds 一致。攻击内部额外随机操作只发生在该客户端 child stream 中。

## 3. 方案比较

### 方案 A：所有攻击都是 post-training update transform

接口只接收 honest `ClientUpdate` 并返回变换后的 update。优点是简单，delta reversal、sign flip、scale 等攻击容易验证。缺点是无法诚实表达 backdoor local training；用固定随机方向冒充 backdoor 不能证明 ASR，正是 legacy `meta_sg` stub 的问题。

### 方案 B：恶意更新生成器（推荐）

attack engine 按 malicious population 把客户端路由到 `MaliciousUpdateGenerator`。untargeted generator 可以包装 honest trainer 后变换 delta；backdoor generator 可以使用仅属于恶意客户端的 poisoned local dataset 训练。两者都返回同一个 `ClientUpdate` 类型，但内部机理不被强行统一。

优点：

- 同一 round engine 支持 no-op、update poisoning 和 data poisoning；
- 不需要让高层 runner 理解触发器或优化器；
- 每种 generator 可独立做 numerical fixture；
- 后续 adaptive attacker 可以实现同一协议。

代价是要明确 capability、malicious population 和 poisoned-data ownership，但这些本来就是可信研究结论所需的信息。

### 方案 C：通用多阶段 attack hook pipeline

为 pre-data、pre-train、post-train、pre-aggregate、post-aggregate 都提供 hook。它最灵活，但 E1 尚未证明两个基本攻击，提前引入 hook ordering、共享 mutable context 和组合语义会显著扩大验证面。

结论：采用方案 B。仅在未来确有第三种不同生命周期攻击时，才从已验证 generator 抽象 hook pipeline。

## 4. 阶段拆分

### E0.4：Clean Episode 与 RNG 隔离

成功条件：

- Episode 运行 H 个 rounds，final state 等于逐轮手工调用；
- horizon=0、task mismatch、非连续 round index 立即失败；
- snapshot/resume 后 trajectory suffix 完全一致；
- 两个 child stream 内部消费不同数量的随机数，不改变父流下一次 spawn；
- 替换 clean round executor 不需要修改 Episode runner。

### E1.1：Fixed untargeted attack

实现 delta reversal budget：

```text
base_delta = honest local model - global model
crafted_delta(b) = (1 - b) * base_delta
attack displacement = crafted_delta - base_delta = -b * base_delta
```

其中 `b = 0` 是严格 identity/no-op，`b = 1` 抑制该客户端更新，`b = 2` 是等范数 sign flip，`b > 2` 是 amplified reversal。

算子级预算定义为：

```text
||crafted_delta - base_delta|| / ||base_delta|| = b
```

当 base norm 为零时，只允许返回零 delta，并记录 ratio=0，不能除零或制造任意方向。

### E1.2：Fixed targeted backdoor attack

使用真实 poisoned local training，不使用随机权重方向模拟 backdoor：

- 固定 source class、target class 和 deterministic trigger；
- 仅从恶意客户端自己的 source-class local samples 中选择 poison subset；
- trigger 后将标签改为 target class；
- `poison_fraction=0` 时 dataset 与 honest dataset 逐样本相同；
- held-out ASR dataset 与所有 client datasets 分离；
- 可选 update boost 必须作为独立 budget 参数记录，第一版默认 1.0。

### E1.3：Matched Episode 与有效性报告

每个 seed 从同一 initial snapshot 运行：

```text
clean/no-op episode
untargeted budget episodes
backdoor poison-fraction episodes
```

报告 paired harm、ASR、clean utility、update norm 和实际 sampled malicious count。E1 不训练 RL，也不定义 defender reward。

## 5. 权威类型和协议

### 5.1 FederatedTrajectory

```python
@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    horizon: int
    sample_size: int
    server_lr: float
    initial_state: RoundState

@dataclass(frozen=True)
class FederatedTrajectory:
    task_id: str
    initial_state: RoundState
    transitions: tuple[RoundTransition, ...]
    final_state: RoundState
```

要求：相邻 transition 的 `state_after` 与下一步 `state_before` 参数、round index 和 RNG snapshot 连续。

round executor 使用现有调用形状：

```python
class RoundExecutor(Protocol):
    def run_round(self, request: RoundRequest, rng: RandomSource) -> RoundTransition: ...
```

### 5.2 AttackCapabilities

```python
@dataclass(frozen=True)
class AttackCapabilities:
    needs_global_model: bool
    needs_local_data: bool
    observes_benign_updates: bool
    observes_other_client_data: bool
    observes_private_diagnostics: bool
    uses_oracle_data: bool
```

E1 两种标准攻击必须满足：

```text
needs_global_model = true
needs_local_data = true
observes_benign_updates = false
observes_other_client_data = false
observes_private_diagnostics = false
uses_oracle_data = false
```

### 5.3 AttackKnowledge

task 显式声明与 capability 对应的六个 `allows_*` 布尔字段。`validate_capabilities(capabilities, knowledge)` 在 round 开始前执行；任一 `needs/observes/uses=True` 但对应 `allows=False` 时失败，不能先训练再警告。

### 5.4 MaliciousUpdateGenerator

```python
@dataclass(frozen=True)
class AttackContext:
    client_id: int
    round_index: int
    global_model: ModelState

class MaliciousUpdateGenerator(Protocol):
    capabilities: AttackCapabilities

    def craft(
        self,
        context: AttackContext,
        rng: RandomSource,
    ) -> ClientUpdate: ...
```

`AttackContext` 是从 server state 构造的最小 client view，不包含 task id、`component_states`、其他客户端 update、恶意身份全集或 private diagnostics。所有 clean、benign 和 malicious local training 都使用统一的 `component_states={}` client state adapter，不得接收原始 server `RoundState`。

内建 generator 只接受 `ScopedLocalTrainer`，其 immutable allowed IDs 与 malicious population 对齐；实验装配必须分别构造 benign-only 和 malicious-only dataset mappings。generator 必须返回对应 client id、`is_malicious=True`、有限浮点 delta 和正 `num_examples`。

capability/knowledge 是可审计、可测试的研究 contract，不是对任意第三方 Python 插件的安全沙箱。任意外部插件进入可信实验 registry 前仍需代码审查；engine 会在构造时冻结 capability manifest，并在每轮采样前验证 manifest 未变化且仍符合 task knowledge。

### 5.5 MaliciousPopulation

```python
class MaliciousPopulation(Protocol):
    def contains(self, client_id: int) -> bool: ...
```

第一版提供 immutable `FixedMaliciousPopulation(client_ids)`。恶意身份是真实实验条件，只进入 engine 路由和 oracle diagnostics，不进入 public signals。

## 6. 模块布局

```text
meta_stackelberg/
  core/
    random_state.py                 # spawn child RNG
  federated/
    episode.py                      # EpisodeSpec, FederatedTrajectory, EpisodeRunner
    engine/
      round_engine.py               # child RNG routing
  security/
    types.py                        # capability / knowledge records
    protocols.py                    # malicious generator / population protocols
    population.py                   # fixed malicious population
    engine/
      attack_round_engine.py        # sampled-order routing and aggregation
    attacks/
      identity.py                   # IdentityMaliciousUpdateGenerator
      delta_reversal.py             # DeltaReversalAttack
      backdoor.py                   # BackdoorLocalUpdateGenerator
    data/
      trigger.py                    # trigger operator and poison selection
  evaluation/
    attack_metrics.py               # paired harm and targeted ASR oracle
```

`security` 不得导入 `evaluation`。`AttackRoundEngine` 构造函数中不得出现 oracle evaluator。oracle 只在 Episode 完成后由实验/测试层调用。

## 7. Round 数据流

### 7.1 Clean / no-op matched path

```text
parent RoundState RNG snapshot
 -> spawn sampling RNG
 -> sample ordered client ids
 -> for each sampled slot spawn one client RNG
 -> route honest or identity malicious generator
 -> ordered ClientUpdate tuple
 -> FedAvg in sampled order
 -> ServerSGD
 -> capture parent RNG only
 -> RoundTransition
```

虽然 transition 分别暴露 `benign_updates` 和 `malicious_updates`，aggregation 必须消费原 sampled order 的内部 tuple，避免按身份重新分组造成浮点求和差异。

### 7.2 Untargeted path

```text
malicious client RNG
 -> honest base trainer on its own data
 -> base ClientUpdate
 -> delta reversal operator(b)
 -> malicious ClientUpdate
 -> common ordered aggregation
```

attack 不观察本轮其他 benign updates，因此不是假设服务器把预聚合更新泄露给 attacker 的 IPM white-box 版本。

### 7.3 Backdoor path

```text
malicious client clean local indices
 + deterministic poison mask
 + trigger(source -> target)
 -> poisoned local dataset
 -> ordinary Torch local optimization
 -> malicious ClientUpdate
 -> common ordered aggregation
```

### 7.4 Oracle path

```text
RoundState / FederatedTrajectory
 -> clean held-out evaluator
 -> clean loss + clean accuracy

RoundState / FederatedTrajectory
 + held-out source-class examples
 + trigger operator
 -> targeted predictions
 -> ASR numerator / eligible denominator
```

oracle outputs不写入 `RoundState`、`public_signals`、generator metadata 或下一轮输入。

## 8. ASR 与 harm 定义

### 8.1 Untargeted paired harm

同 seed、同 round 的主要量：

```text
loss_harm_t = attacked_clean_loss_t - matched_clean_loss_t
accuracy_harm_t = matched_clean_accuracy_t - attacked_clean_accuracy_t
```

E1.1 gate 以最终和 area-under-harm-curve 为主。只要求强攻击 endpoint 的 loss harm 为正且跨重复 seed 方向一致；不要求复杂 FL 中每个相邻 budget 的最终 accuracy 严格单调。算子 budget ratio 则必须精确成立。

### 8.2 Targeted ASR

eligible denominator 只包含 held-out source-class 样本，并排除原本就是 target class 的样本：

```text
ASR = triggered eligible samples predicted as target / eligible source samples
```

denominator 为零时抛错，不能返回 0 或 NaN。报告同时给出 clean held-out accuracy，避免把“所有输入都预测 target”误判为成功 backdoor。

## 9. 单模块测试

### RandomSource / Episode

- child stream replay；
- child consumption isolation；
- clean manual-loop equivalence；
- executor plug-in contract；
- snapshot suffix replay。

### Capability / population

- 每个 capability flag 的 allowed/rejected fixture；
- fixed population immutable、去重、拒绝负 id；
- malicious identity 不出现在 public signals。

### Identity / delta reversal

- hand-calculated vector fixtures；
- `b=0` 逐元素等于 base update；
- `b=1` 为零；
- `b=2` 为负 base update；
- displacement ratio 精确；
- NaN、负 budget 和 client-id mismatch 拒绝。

### Trigger / backdoor dataset

- 只修改 source-class poison subset；
- 未选样本 bitwise unchanged；
- poisoned labels 全为 target；
- fixed seed mask replay；
- poison fraction 0 与 clean dataset 相同；
- 不修改底层 dataset。

### Oracle evaluator

- 手工 logits 的 ASR numerator/denominator；
- source/target 相同拒绝；
- 空 denominator 拒绝；
- batch size 不影响结果；
- evaluator 无 mutation。

### AttackRoundEngine

- sampled-order aggregation；
- returned id/flag/shape validation；
- no malicious sampled 等同 clean；
- identity attack model state 等同 clean；
- 替换 malicious generator 不修改 engine；
- private diagnostics 与 public signals 隔离。

## 10. 集成实验与预期结果

### E1.1 controlled untargeted fixture

使用 E0.3 的 held-out clean protocol和小型数据，但固定 seed family、malicious population 与 matched initial snapshot。

预期：

- `b=0` 与 clean trajectory 的 sampled clients、parameters、metrics 完全相同；
- `b=2` 或更强 budget 的最终 clean loss 高于 matched clean；
- malicious update displacement ratio 与 budget 一致；
- 多 seed 的 paired loss harm 方向一致。

失败含义：若算子单测正确但 Episode 无 harm，优先调整 malicious fraction、budget 或受控任务，不把失败归因于未来 defender/RL。

### E1.2 controlled backdoor fixture

构造独立 train/held-out 图像、固定 source/target/trigger 和包含 source samples 的 malicious clients。

预期：

- poison fraction 0 与 identity attack 等价；
- strong poison 的 ASR 明显高于 matched no-op；
- clean accuracy 同时报告且为有限值；
- ASR 只来自 held-out triggered source samples。

失败含义：优先检查 trigger 可学习性、source samples、local poison ratio、恶意参与频率和 ASR denominator；不能用随机方向权重扰动替代真实 backdoor。

## 11. 错误处理与不变量

- capability mismatch 在采样或训练前失败；
- non-finite model/update/metric 携带来源并失败；
- generator 返回错误 client id、`is_malicious=False` 或 incompatible shape 时拒绝 round；
- `poison_fraction` 必须在 `[0,1]`；
- source class 不能等于 target class；
- matched protocol 的 initial model hash、partition hash、sample history 不一致时，禁止计算 paired harm；
- Episode 失败不得返回部分 trajectory 冒充成功结果；
- oracle protocol version 不同的 ASR 不可直接比较。

## 12. 可插拔边界

以下替换必须无需修改 orchestration：

- `RoundEngine` ↔ `AttackRoundEngine` 通过同一个 round executor contract；
- `IdentityMaliciousUpdateGenerator` ↔ `DeltaReversalAttack` ↔ `BackdoorLocalUpdateGenerator` 通过 malicious generator protocol；
- fixed population ↔ future stochastic population 通过 population protocol；
- FedAvg ↔ future defense aggregator 通过现有 aggregator protocol；
- clean evaluator ↔ targeted oracle evaluator 位于 round 外部。

contract test 不只检查 `isinstance(Protocol)`，还必须运行共享行为 fixture。

## 13. 报告交付物

阶段完成后生成：

```text
docs/milestones/E0.4-clean-episode-report.zh-CN.md
docs/milestones/E1-attack-validity-report.zh-CN.md
```

E1 报告固定包含：

1. threat model 和 capability matrix；
2. clean/no-op/attack 数据流；
3. matched seed、initial model、partition 和 sample schedule；
4. untargeted budget operator 与 Episode harm curve；
5. backdoor poison fraction、ASR denominator 与 clean utility；
6. 单模块和集成测试证据；
7. 实际结果与预期差异；
8. gate 是否通过；
9. E2 前置条件。

## 14. 非目标

E1 不实现：

- adaptive/RL attacker；
- defense action；
- reward；
- best response；
- test set 反馈；
- DBA、多攻击混合或 defense-aware geometry search；
- 敌对 checkpoint 文件加载。

这些内容只有在 fixed attack validity 成立后才有解释基础。

## 15. Gate

E0.4 通过后才允许运行 E1 matched experiment。E1 通过需要同时满足：

```text
identity/no-op exact equivalence
+ untargeted operator budget correctness
+ untargeted matched harm direction
+ backdoor held-out ASR improvement
+ clean utility co-reporting
+ capability compliance
+ oracle isolation
+ exact replay/resume
```

其中任何一项缺失，都只能报告对应子模块进度，不能宣称“攻击有效性已证明”。
