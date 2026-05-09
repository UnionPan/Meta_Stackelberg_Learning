# RL Attacker: Tianshou 结构说明

当前 `fl_sandbox` 只保留一个 adaptive RL attacker 入口：

```python
from fl_sandbox.attacks import RLAttack, create_attack
from fl_sandbox.attacks.rl_attacker import RLAttackerConfig
```

旧的 hand-written `Actor`、`Critic`、`TD3Agent`、`ReplayBuffer`、target-network
更新逻辑已经移除。`RLAttack` 通过 `Trainer` 协议调用 Tianshou 后端，默认算法是
SAC，实验需要时可以把 `rl_algorithm` 设置为 `td3`。

## 模块职责

- `attack.py`
  真实 FL round 的入口。负责观察 server update、维护 proxy learner、训练策略、
  执行 deploy guard，并把 action 转成 malicious client weights。
- `config.py`
  RL attacker 的统一配置，包括算法选择、proxy reconstruction、simulator horizon、
  reward 权重、action 解码尺度和 deploy guard 阈值。
- `action_decoder.py`
  把连续 action 解码成攻击参数：`gamma_scale`、`local_steps`、
  `lambda_stealth`、`local_search_lr`。不同 defense 使用不同尺度。
- `observation.py`
  把模型压缩状态、攻击者数量和 defense 信息整理成策略 observation。
- `proxy/`
  维护 proxy dataset buffer，并用 seed attacker data / server update reconstruction
  补充模拟环境的数据分布。
- `simulator/`
  构建单步 FL world model：采样 benign / malicious updates，经 defender 聚合后
  计算 proxy loss、accuracy、norm penalty 和 reward。
- `trainer.py`
  定义 `Trainer` 协议和 `build_trainer(config)` 工厂；Tianshou 只在这里之后懒加载。
- `tianshou_backend/`
  提供 `TianshouSACTrainer` 和 `TianshouTD3Trainer`。外部代码不直接依赖这些实现。
- `diagnostics.py`
  输出 sim2real gap、deploy guard 状态、trainer 训练计数和 replay buffer 规模。

## 执行流

1. 每轮开始前，runner 构建 `RoundContext` 并调用 `attack.observe_round(ctx)`。
2. `RLAttack` 从 attacker loader 初始化 proxy buffer，并在 warmup 阶段吸收 server
   update 的 reconstruction 样本。
3. 当 proxy buffer 和模型模板可用时，`SimulatedFLEnv` 用当前 global weights 构建
   训练环境。
4. `build_trainer(config)` 按 `config.algorithm` 创建 SAC 或 TD3 trainer。
5. trainer 在 simulator 中 collect/update，诊断信息记录到 `RLSim2RealDiagnostics`。
6. 真实攻击轮中，deploy guard 检查 proxy 样本量和 sim2real gap。
7. guard 通过后，policy action 经 `decode_action(...)` 变成 local search 参数。
8. `craft_malicious_update(...)` 在 proxy data 上搜索恶意更新，并按 benign norm 目标
   做缩放。

## 为什么默认 SAC

adaptive attacker 的 action 是连续的，reward 又来自一个会随 global model 和 defense
状态变化的近似 simulator。SAC 的熵项能在这种非平稳场景中保持探索，比把 TD3 作为默认
更稳。TD3 仍然保留为 ablation 选项，便于对比连续控制算法。

## 运行示例

```bash
python fl_sandbox/run/run_experiment.py \
  --attack_type rl \
  --defense_type krum \
  --rl_algorithm sac \
  --rounds 50 \
  --rl_distribution_steps 10 \
  --rl_attack_start_round 11 \
  --rl_policy_train_end_round 10
```
