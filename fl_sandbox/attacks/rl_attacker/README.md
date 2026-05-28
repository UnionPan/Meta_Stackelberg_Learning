# Paper-Aligned RL Attacker

本目录实现的是 `attack_type=rl` 的 paper-style model poisoning attacker。当前版本对齐的是作者代码仓库里可复现的工程实现：先使用 Phase 1 learned distribution 构建代理 FL simulator，再离线训练 TD3 policy，最后冻结 policy 在真实 FL deployment 中攻击。

## 当前定位

论文文字描述的是一个在线框架：

```text
FL epoch 1-100:
  学 distribution
  同时学 policy

FL epoch 101-400:
  开始攻击
  policy 继续学习

FL epoch 401-1000:
  停止 policy learning
  使用最终 policy 继续攻击
```

作者代码的实验实现更接近离线 pipeline：

```text
Phase 1:
  生成 learned distribution artifacts

Phase 2:
  用 learned distribution 构建 simulator
  TD3.learn(total_timesteps=80000)

Phase 3:
  加载/使用 TD3 policy
  在真实 FL deployment 中执行攻击
```

本实现采用第二种方式。也就是说，`80000` 是 TD3 与 simulator 交互的总步数，不是真实 FL 的 80000 轮；`simulator_horizon=1000` 是一个 simulator episode 的长度，也不和真实 FL 的 `1-400` 做乘法。

## 目录结构

```text
rl_attacker/
  __init__.py
  config.py
  observation.py
  paper_attack.py
  trainer.py
  proxy/
    paper_dataset.py
  simulator/
    fl_dynamics.py
    paper_env.py
    distribution_learning/
      ...
  tianshou_backend/
    common.py
    td3.py
```

### `paper_attack.py`

`PaperRLAttack` 是 `attack_type=rl` 的正式入口。

主要职责：

- 校验 `rl_distribution_dir`，加载 Phase 1 artifacts。
- 构建 `PaperDistributionDataset` 和 `PaperDistributionSampler`。
- 第一次 `observe_round()` 时训练或加载 TD3 policy。
- offline TD3 训练完成后设置：
  - `rl_policy_warmup_done = 1`
  - `rl_policy_frozen = 1`
- `execute()` 阶段使用 deterministic policy action。
- 根据 action 构造 malicious weights。
- `after_round()` 写出 diagnostics 到 CSV/TensorBoard。

当前 policy learning 是一次性 offline：

```text
observe_round()
  -> _train_policy_offline()
     -> _build_policy_env()
        -> PaperFLSimulator
        -> PaperAttackerPolicyGymEnv
     -> build_trainer(TD3)
     -> warmup_collect(random actions)
     -> collect + update until policy_warmup_steps
  -> freeze policy
```

部署阶段：

```text
execute()
  -> current real FL observation
  -> trainer.act(obs, deterministic=True)
  -> decode action to gamma/local_steps
  -> craft_paper_malicious_update()
  -> return same malicious weights for selected attackers
```

### `config.py`

`RLAttackerConfig` 保存 paper RL attacker 的参数。

当前 paper-aligned TD3 默认值：

```text
algorithm = td3
policy_warmup_steps = 80000
policy_warmup_random_steps = 100
simulator_horizon = 1000
policy_lr = 1e-7
critic_lr = 1e-7
gamma = 1.0
hidden_sizes = (256, 128)
train_freq_steps = 5
batch_size = 256
replay_capacity = 100000
local_search_batch_size = 128
paper_local_lr = 0.01
```

当前只支持 paper clipped-median 复现：

```text
defense_type = clipped_median
action_dim = 2
```

### `proxy/paper_dataset.py`

`PaperDistributionDataset` 是 Phase 1 learned distribution 的 disk-backed 数据源。

期望目录格式：

```text
distribution_dir/
  metadata.json
  data.csv
  train/
    0.png
    1.png
    ...
```

它会：

- 读取 `metadata.json` 中的 `mean/std`。
- 读取 `data.csv` 中的 sample id 到 label 映射。
- 加载 `{split}/*.png`，转换成归一化 tensor。
- 提供 `sample(batch_size, device)` 给 simulator 使用。

`PaperDistributionSampler` 在 dataset 上提供采样和 paper growth schedule：

```text
paper_growth:
  available_samples = min(N, 200 + (episode - 1) * 80)

full:
  available_samples = N
```

`advance_episode()` 只在 simulator episode 结束时调用，不在每次 reset 或每个真实 FL round 调用。

### `simulator/paper_env.py`

`PaperFLSimulator` 是 policy learning 使用的 paper-style FL simulator。

一个 `step(action)` 约等于一次 simulated FL round：

```text
old_weights
  -> sample selected attackers
  -> simulate benign client updates on learned distribution
  -> decode action
  -> simulate malicious update
  -> clipped median aggregate
  -> evaluate proxy loss
  -> reward = new_loss - old_loss
  -> next state
```

动作解码对齐作者代码：

```text
action space = [-1, 1]^2
gamma = action[0] * 14.9 + 15
local_steps = int(action[1] * 24 + 25)
```

因此：

```text
gamma in [0.1, 29.9]
local_steps in [1, 49]
```

恶意更新公式：

```text
trained = attacker local trained weights
malicious = old + gamma * (old - trained)
```

reward：

```text
reward = new_loss - old_loss
```

这里的 loss 默认在 learned distribution proxy 上评估，不在真实 test loader 上评估。真实 clean loss delta 会在 deployment 的 `after_round()` 中记录为 `rl_real_reward`。

`PaperAttackerPolicyGymEnv` 是 gymnasium wrapper，供 Tianshou TD3 调用。

重要语义：

- simulator episode reset 到第一次看到的 initial weights。
- 不会每个真实 FL round reset 到当前 `ctx.old_weights`。
- `simulator_horizon=1000` 表示一个 simulator episode 最长 1000 个 simulated FL rounds。

### `simulator/fl_dynamics.py`

低层 FL 权重工具：

- 根据模板模型和 weights 构建模型。
- 捕获模型 weights。
- 计算 update norm。
- 实现 paper reversal-style malicious update。

`paper_env.py` 使用这些 helper 保持 simulator 和真实模型权重格式一致。

### `observation.py`

构造 policy state。

当前 state 近似作者代码的 clipped-median state：

```text
last model layers
  -> flatten
  -> min-max normalize to [-1, 1]
  -> append selected attacker count
```

作者原始 SB3 实现使用 `MultiInputPolicy` 和 dict observation：

```text
{
  "pram": normalized last layer parameters,
  "num_attacker": selected attacker count
}
```

当前 Tianshou backend 使用 flat Box observation，因此把这些信息拼成一个向量。

### `trainer.py`

定义 trainer 协议和 `build_trainer()` 工厂。

当前 `rl` 只允许 TD3：

```text
build_trainer()
  -> TianshouTD3Trainer
```

### `tianshou_backend/common.py`

Tianshou trainer 的通用实现。

关键点：

- TD3 使用 recency-weighted replay buffer。
- `warmup_collect()` 用随机 action 填 replay。
- `collect()` 用当前 actor + exploration noise 采样。
- `update()` 在 replay size 小于 batch size 时不更新。
- `reset_on_start=False` 时连续使用上一次 env state，避免每个 `train_freq` 都 reset episode。
- diagnostics 会写入 TensorBoard/CSV，例如：
  - `rl_trainer_collect_steps`
  - `rl_trainer_update_steps`
  - `rl_trainer_replay_size`
  - `rl_trainer_reward_mean`

### `tianshou_backend/td3.py`

TD3 trainer 的薄封装。

当前核心网络和 optimizer 在 `common.py` 中构造：

```text
actor hidden sizes = (256, 128)
critic hidden sizes = (256, 128)
actor lr = 1e-7
critic lr = 1e-7
gamma = 1.0
```

### `simulator/distribution_learning/`

Phase 1 learned distribution 相关代码。

这一部分应该保留。它负责从 FL 信息中生成 paper-style learned distribution artifacts。Phase 2 policy learning 不再在线调用 inversion 或重建，而是通过 `rl_distribution_dir` 读取 Phase 1 产物。

## 数据流

完整数据流：

```text
Phase 1 artifacts
  train/*.png + data.csv + metadata.json
      |
      v
PaperDistributionDataset
      |
      v
PaperDistributionSampler
      |
      v
PaperFLSimulator
      |
      v
PaperAttackerPolicyGymEnv
      |
      v
Tianshou TD3 trainer
      |
      v
frozen policy
      |
      v
PaperRLAttack.execute()
      |
      v
real FL deployment
```

训练时的数据流：

```text
state_t
  -> actor outputs action_t
  -> decode action_t to gamma/local_steps
  -> simulator runs one FL round
  -> reward_t = new_loss - old_loss
  -> replay.add(state_t, action_t, reward_t, state_{t+1})
  -> TD3 critic/actor update
```

部署时的数据流：

```text
real ctx.old_weights
  -> build observation
  -> deterministic policy action
  -> gamma/local_steps
  -> attacker local train on learned distribution
  -> old + gamma * (old - trained)
  -> malicious weights returned to FL runner
```

## 轮次含义

这里有三种容易混淆的计数器。

### 真实 FL round

真实实验里的 round，例如：

```text
round 101 -> 600
```

这是 deployment/evaluation 的真实 FL 时间轴。

### simulator step

一个 `env.step(action)` 是一次 simulated FL round。

```text
policy_warmup_steps = 80000
```

表示 TD3 和 simulator 交互 80000 次。

### simulator episode

一个 simulator episode 最长：

```text
simulator_horizon = 1000
```

所以：

```text
80000 simulator steps ~= 80 simulator episodes
```

每个 episode reset 到 initial weights，不 reset 到当前真实 FL round 的 weights。

## 与论文描述的差异

论文描述的 policy learning window 是真实 FL epoch `1-400`。作者代码没有逐真实 FL epoch 在线更新 policy，而是：

```text
Distribution_set(...)
env = FL_mnist_clipping_median(...)
model = TD3(...)
model.learn(total_timesteps=80000)
```

因此当前实现对齐的是作者代码的离线可复现路径：

```text
offline distribution
offline TD3 policy learning
frozen deployment
```

不是每个真实 FL round 都把 simulator reset 到当前真实 global model 的 online/Dyna-style RL。

## 运行方式

先准备 Phase 1 artifacts，例如：

```text
fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1_init_pre_label
```

再运行 RL attacker：

```bash
PYTHONPATH=. .venv/bin/python -m fl_sandbox.run.run_experiment \
  --dataset mnist \
  --split_mode paper_q \
  --noniid_q 0.1 \
  --attack_type rl \
  --defense_type clipped_median \
  --rounds 500 \
  --start_round_idx 101 \
  --num_clients 100 \
  --num_attackers 20 \
  --subsample_rate 0.1 \
  --local_epochs 1 \
  --lr 0.01 \
  --batch_size 128 \
  --parallel_clients 4 \
  --distribution_dir fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1_init_pre_label \
  --rl_simulator_horizon 1000 \
  --rl_checkpoint_interval 5 \
  --output_root fl_sandbox/outputs/paper_rl_author_aligned_500r \
  --tb_root fl_sandbox/runs/paper_rl_author_aligned_500r
```

每 5 轮 checkpoint 会写到：

```text
<output_dir>/checkpoints/
  rl_policy_latest.pt
  rl_policy_round_XXXXXX.pt
  global_model_latest.pt
  global_model_round_XXXXXX.pt
```

## 监控指标

主要看：

```text
metrics/accuracy
metrics/loss
rl_action/gamma
rl_action/local_steps
rl_training/real_reward
rl_training/simulated_reward
rl_training/sim2real_gap
rl_policy/warmup_done
rl_policy/frozen
attack_only/mean_malicious_norm
```

model poisoning 任务不要主要看 `asr`。`asr`/`backdoor_accuracy` 是 backdoor 任务指标。

典型成功信号：

```text
rl_policy/warmup_done = 1
rl_policy/frozen = 1
rl_action/gamma 不贴 0.1 边界
rl_action/local_steps 不贴 1 边界
metrics/accuracy 明显下降
metrics/loss 明显上升
rl_training/real_reward 出现大正值
```

后期如果模型被打崩，norm/cosine 统计可能出现 `inf` 或 `nan`。这种情况下应优先看 `accuracy`、`loss`、`real_reward` 和 action 曲线。

## 最近一次复现实验结论

最近一次 500-round author-aligned run：

```text
output:
  fl_sandbox/outputs/paper_rl_author_aligned_500r_20260528_batch128/

tensorboard:
  fl_sandbox/runs/paper_rl_author_aligned_500r_20260528_batch128/
```

结果摘要：

```text
rounds: 101 -> 600
policy steps: 80000
final clean acc: 0.1010
max clean acc: 0.9116
first acc <= 0.5 after round 150: round 199
first acc <= 0.2 after round 150: round 347
first acc <= 0.11 after round 150: round 369
gamma range: 24.72 -> 29.72
local_steps range: 30 -> 34
```

这说明当前 policy 已经不再退化成 `gamma=0.1, local_steps=1`，而是学到了强 model-poisoning 策略。

