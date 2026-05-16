# Adaptive Attacker：clipping-median 防御下的 RL 攻击者

> 论文：*Learning to Attack Federated Learning: A Model-based Reinforcement Learning Attack Framework*（NeurIPS 2022，Li / Sun / Zheng）
> 代码：<https://github.com/SliencerX/Learning-to-Attack-Federated-Learning>（commit `db329b5`），本仓库镜像位于 `~/rl/analysis_learning_to_attack_fl/repo/`
> 本笔记聚焦 **clipping coordinate-wise median** 这一种防御对应的 attacker 设计。

---

## 0. 一句话定位

论文的 RL attacker **不是一个泛化设计**，而是 **per-defense 设计**：每种聚合规则（Krum、coord-wise median、clipping median、FLTrust）都对应一套独立的动作空间和环境实现。本笔记只覆盖 clipping median 这一支。

证据：仓库里 `exp_environments.py` 为每种 (数据集, 防御) 组合写了独立的 gym `Env` 子类，例如 `FL_mnist_clipping_median`、`FL_mnist_krum`、`FL_mnist_fltrust_g_td3_acsend_real_test`，动作维度、动作语义、ε 区间互不相同。

---

## 1. 论文怎么定义这个 RL 问题

对应章节：**§4.1（MDP 形式化）、§4.3（policy learning 的动作设计）、§6.1（实验默认设置）**。

### 1.1 MDP 形式（§4.1）

- **状态**：$s_\tau = (\theta^{t(\tau)},\ A^{t(\tau)})$
  - $\theta$：当前全局模型参数
  - $A^{t(\tau)}$：第 τ 步采样到的攻击者集合
  - τ 只在「至少一个攻击者被抽中」时推进
- **动作**：每个被选中的攻击者发送一个 $d$ 维 local update $\tilde g$，未被抽中的攻击者动作为 $\bot$
- **Reward**：

$$
r_\tau = f(\theta^{t(\tau+1)}) - f(\theta^{t(\tau)})
$$

  目标是最大化 $H$ 步累计 reward。$f(\theta) = \sum_k p_k F_k(\theta)$ 是 FL 在最小化的全局经验损失（论文 §3）
- **Transition / Reward 都依赖良性 worker 的数据分布** $\{P_k\}$，attacker 用 IG 学到的 $\tilde P$ 替代它（这是 model-based 的核心）

### 1.2 状态压缩（§4.3）

- 用神经网络**最后一层参数**取代整套 $\theta$
- 用**本轮采样到的攻击者数量** $m^{t(\tau)}$ 取代攻击者集合 $A^{t(\tau)}$
- 论文明确说：压缩后的 state **只用于策略输入**，真正决定 transition / reward 的仍是完整 FL 模型

### 1.3 动作设计：为什么"只动最后一层"不够（§4.3）

论文原话：

> a policy that manipulates the model parameters of the last hidden layer only works well for certain aggregation rules such as Krum and coordinate-wise median. However, it becomes less effective for stronger defenses such as coordinate-wise median with clipping.

原因：
- clipping median 先把每条 update 整体 L2 范数裁到 ≤ 2，再逐维取中位数
- 把所有扰动堆在最后一层 → 整条 update 的范数被这一层主导 → clipping 把绝大部分扰动砍掉
- 怎么从 $d$ 维里挑出"扰动哪几个维度最伤害精度"是个组合搜索问题，RL 直接学不来

### 1.4 新动作设计：3 层结构

**第 1 层：本地搜索目标 $L(\theta)$**

候选攻击落点记为 $\theta$，定义

$$
L(\theta) = (1-\lambda)\,F(\theta) \;+\; \lambda \cdot \cos\!\big(\theta^{t(\tau)} - \theta,\ g(\theta^{t(\tau)})\big)
$$

- $F(\theta) = \mathbb{E}_{z \sim \tilde P}[\ell(\theta; z)]$：在估出的分布 $\tilde P$ 上的经验损失（让模型变烂）
- $g(\theta^{t(\tau)}) = \mathbb{E}_{z \sim \tilde P}[\nabla_\theta \ell(\theta^{t(\tau)}; z)]$：正常人在当前模型下的平均梯度方向（一个 $d$ 维向量）
- cosine 项：让攻击方向和正常方向同向，避免被服务器从方向上识别
- $\lambda \in [0,1]$：在"伤害"和"隐蔽"之间权衡

**第 2 层：用梯度上升解 $L(\theta)$**

- 起点：$\theta = \theta^{t(\tau)}$
- 生成 $G$ 条轨迹，每条做 $E$ 步梯度上升，每步用 $\tilde P$ 采一个大小 $B'$ 的 minibatch
- 第 $k$ 条轨迹结束时落在 $\theta_k$，取平均 $\bar\theta = \tfrac{1}{G}\sum_{k=1}^G \theta_k$
- $G$ 条平均的作用是降方差

**第 3 层：把搜索结果包装成"伪梯度"发给服务器**

$$
\tilde g^{t(\tau)+1} = \gamma\Big(\theta^{t(\tau)} - \bar\theta\Big)
$$

服务器执行 $\theta_{\text{new}} = \theta^{t(\tau)} - \eta\,\mathrm{Aggr}(\dots)$，相当于把全局模型往 $\bar\theta$ 这个恶意落点拉。$\gamma$ 控制拉的力度。

**RL 只学 3 个标量**：$(\gamma, E, \lambda)$。$G$、$B'$ 是固定超参。动作空间从 $d$ 维 → 3 维。

### 1.5 实验默认设置（§6.1）

| 项 | 值 |
|---|---|
| Clients | 100 |
| Attackers | 20 |
| 采样率 κ | 10% |
| 学习率 η | 0.01 |
| FL 总轮数 T | 1000 |
| 防御 | coord-wise median + norm clipping，阈值默认 2 |
| Distribution learning | epoch 0–100 |
| Policy learning | 至 epoch 400 结束 |
| 攻击执行 | epoch 100 起 |
| RL 算法 | TD3（也试过 PPO） |

### 1.5.1 在线 test 与 checkpoint replay

论文主实验的攻击效果曲线看的是 **FL 训练过程中的 global model test accuracy**：distribution learning / policy learning 从前期开始，攻击执行从约第 100 个 epoch 后开始，后续 FL epoch 使用当时已经学到的最新 policy 生成恶意更新。

所以严格说，最自然、也最接近攻击执行语义的评估是 **边训练 policy、边执行攻击、边记录 test accuracy/loss**。我们训练 run 里的 `train/clean_acc`、`train/clean_loss` 就是这种在线轨迹，也是当前项目应该优先看的 RL attacker 结果。

论文 §6.3 在讨论 policy learning 不稳定时提到，可以用 separate testing environment 去识别 `best trained policies`。这句话的意思是：如果不同训练步的 policy 好坏震荡，可以额外开一个独立 FL 测试环境来挑某个 checkpoint。它是离线 model selection / 稳定性诊断，不是 RL 攻击主测试口径，也不要求每一轮都保存 checkpoint。

原仓库的实现也不是每轮全量保存：`sim_train.py` 周期性保存 TD3 checkpoint，`test.py` 在测试 FL 的第 `rnd` 轮加载当时可用的 checkpoint，例如 `min(int(rnd / 5) * 1000, 80000)`。这相当于“rolling latest-policy eval”：测试环境独立，但不会在第 101 轮偷看最终 policy。这个过程可以复现，但它更像额外对照实验。

本仓库因此保留两种 test 方式：

| 模式 | 用途 | CLI |
|---|---|---|
| 在线轨迹 | 主测试口径；边学边攻边测 | 正常 train run |
| 固定最终 policy | 检查最后导出的 attacker 部署能力 | `--rl_policy_checkpoint_path ... --rl_freeze_policy` |
| rolling checkpoint | 离线 checkpoint selection / 复刻原仓库 test 脚本 | `--rl_policy_checkpoint_dir ... --rl_freeze_policy` |

不需要每一轮都存 checkpoint。为了做离线对照，按固定间隔保存即可，例如 `rl_checkpoint_interval=25` 或 `50`；间隔越小，rolling eval 越接近真实在线策略更新，磁盘和测试开销也越高。正式汇报攻击效果时，优先报告在线 run 的趋势曲线。

### 1.6 "只动最后一层" vs 新方案对比

| 维度 | 只动最后一层 | $L$ + 梯度上升 + $(\gamma, E, \lambda)$ |
|---|---|---|
| RL 动作维度 | 1290 维向量 | 3 维标量 |
| 改的参数范围 | 仅最后一层 | 整模型（由梯度上升自然分布） |
| 范数预算 | 全堆最后一层 → 被 clipping 砍掉 | 分散到所有维度，clipping 后仍有效 |
| 方向伪装 | 无 | 有 cosine 项，由 $\lambda$ 控制 |
| 选维度的方式 | 人工切分 | 由 $\nabla L$ 在 $\tilde P$ 上自动选 |

**核心思想**：把"哪些维度该被攻击、各改多少"从 RL 身上转移到"对 $L$ 做梯度上升"这个数值优化器身上，RL 只调几个超参。

---

## 2. 代码怎么实现

对应文件：
- 环境：`exp_environments.py` 的 `FL_mnist_clipping_median`（行 `44-207`）
- 防御聚合：`utilities.py` 的 `Clipping_Median`（行 `964-1000`）
- 攻击梯度生成：`utilities.py` 的 `craft_att`（行 `1117-1133`）
- 评估函数：`utilities.py` 的 `test`（行 `542-557`）

### 2.1 观测空间（`exp_environments.py:60-63`）

```python
observation_space = Dict(
    pram         = Box(-inf, +inf, shape=(1290,)),  # 最后一层 weight(10×128)+bias(10)
    num_attacker = Discrete(11),                    # 0..10
)
```

最后一层向量做 **per-frame min-max 归一化** 到 $[-1, 1]$（`exp_environments.py:151-155`）。

### 2.2 动作空间（`exp_environments.py:53-58, 82-83`）

```python
action_space = Box(-1, +1, shape=(2,))
# step() 内的解码
action[0] = action[0]*14.9 + 15   # → ε ∈ [0.1, 29.9]  对应论文 γ
action[1] = action[1]*24   + 25   # → E ∈ [1,  49]
```

**2 维**：缩放系数 ε 和本地步数 E。**没有 λ**。

### 2.3 一步 `step()` 内发生的事情（`exp_environments.py:80-161`）

1. **良性客户端跑 1 epoch SGD**（行 `84-90`）：所有非攻击者共享 `self.train_iter`
2. **生成攻击方向**（行 `92-106`）：
   - 网络重置到 `aggregate_weights`
   - 用 `train(net, all_train_iter, epochs=E)` 跑 $E$ 步 **下降**（注意：不是上升）
   - 得到 `new_weight = θ̄`
3. **构造伪梯度**（行 `108-110`）：调 `craft_att(old, avg, -1, ε)`，展开为

   ```
   diff       = old − new                  # ≈ +η·grad，下降方向
   crafted_d  = (−1) · ε · diff
   crafted_w  = old − crafted_d = old + ε · diff
   ```

   即"下降方向取反、放大 ε 倍"。**所有攻击者发同一份 `crafted_weight`**
4. **服务器聚合**（行 `116`，调 `Clipping_Median`）
5. **采样下一轮 clients**（行 `121-144`）：`is_attack=True` 硬编码，跳过"无攻击者"分支
6. **算 reward**（行 `146-149`）：

   ```python
   new_loss, new_acc = test(net, valiloader)
   reward = new_loss − self.loss
   ```

7. **构造下一步 state**（行 `151-155`）：最后一层 + min-max 归一化 + `num_attacker`
8. **终止条件**：`rnd >= 1000` 时 `done=True`（行 `156-160`）

### 2.4 防御实现 `Clipping_Median`（`utilities.py:964-1000`）

```python
max_norm = 2                                   # 写死
for new_weight in new_weights:
    g_i      = old − new_weight                # 该 client 的"梯度"
    g_i_clip = g_i · min(1, max_norm / ‖vec(g_i)‖)
med_grad   = coord-wise median of {g_i_clip}   # 逐维中位数
new_global = old − med_grad
```

### 2.5 评估函数 `test`（`utilities.py:542-557`）

```python
def test(net, valloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()   # sum, not mean
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss, correct / total
```

注意点：
- `valiloader = DataLoader(validset, batch_size=len(validset))`：**整个验证集一次性 forward**（`exp_environments.py:73`）
- **`loss` 是累加和**而不是平均，validset 大小变了会跳变
- **没有 `net.eval()`**：dropout 仍是随机的，reward 里多了一份固定方差噪声
- `Net.forward` 末行是 `F.log_softmax(x)`，再过 `CrossEntropyLoss` 会**再做一次 log_softmax**，loss 绝对值偏移，但 reward 取的是差值，影响有限

### 2.6 Reset（`exp_environments.py:163-207`）

- 加载固定初始模型 `mnist_init`
- 攻击者可见数据量随 RL episode 线性增长：`min(200 + (epoch-1)*32*2.5, len(att_trainset))`
- `starts_step=0`、`is_attack=True` 都写死，**episode 一开始就允许攻击**

### 2.7 论文符号 ↔ 代码对照

| 论文 | 代码 |
|---|---|
| $s_\tau$ | 最后一层归一化向量 + `num_attacker`（`exp_environments.py:151-155`） |
| $a_\tau = (\gamma, E, \lambda)$ | 2 维 `[ε, E]`，λ 被省略（`exp_environments.py:82-83`） |
| $\tilde g = \gamma(\theta - \bar\theta)$ | `craft_att(old, avg, -1, ε)`（`utilities.py:1117-1133`） |
| $G$ 条轨迹 | $G = 1$（代码只跑一次 `train`） |
| $E$ 步梯度上升 | `train(net, all_train_iter, epochs=E)` + 在 `craft_att` 里取反号 |
| $r_\tau = f(\theta') - f(\theta)$ | `new_loss − self.loss`（validset 上的 CE 之和） |
| $\mathrm{Aggr}$ | `Clipping_Median(old, new_weights)`，阈值 2 |

---

## 3. 论文 vs 代码：差异及原因

### 总览

| 维度           | 论文                              | 代码（clipping median）                 | 原因                          |
| ------------ | ------------------------------- | ----------------------------------- | --------------------------- |
| 动作维度         | 3D $(\gamma, E, \lambda)$       | 2D $(\varepsilon, E)$               | clipping median 不看方向，λ 无收益  |
| 局部搜索方式       | 梯度上升 E 步 + $\tilde P$ minibatch | 普通 SGD 下降 + 取反                      | 数学等价，工程省事                   |
| 轨迹数 $G$      | 公式留接口                           | $G = 1$                             | 算力成本                        |
| cos 正则项      | 有                               | 无                                   | 防御不看方向                      |
| 良性 client 采样 | 各自 $\hat P_k$                   | 共享 `train_iter`                     | env 默认 q=0.1 接近 IID         |
| τ 推进规则       | 仅当采到 attacker 推进                | 硬编码 `is_attack=True`                | 简化 RL 经验语义                  |
| state 喂给策略   | 最后一层参数                          | 最后一层 + min-max 归一化                  | 基本一致                        |
| 攻击启动时机       | epoch 100 起                     | 已在本地实现：env 串联 distribution / policy / attack 三阶段 | 与论文一致                       |
| Clipping 阈值  | 默认 2                            | 写死 2                                | 与论文一致                       |
| γ / ε 区间     | 未给                              | $[0.1, 29.9]$（vs Krum $[0.1, 9.9]$） | clipping 把 norm 裁到 2，需要更大 ε |

### 3.1 动作 3D → 2D（去掉 λ-cos 项）

cosine 项设计目的是"让恶意方向像正常方向、躲过防御过滤"。clipping median **只看 norm + 逐维中位数**，对方向不敏感，所以 λ 没有收益。

→ 代码把 λ 留给了 FLTrust 那条线（`FL_mnist_fltrust_*` 行 `730-762` 动作是 3 维并带 `alpha`）。**典型的 per-defense 设计。**

### 3.2 G 条轨迹 → G = 1

每条轨迹要在攻击者本地跑 $E$ 步训练。$G=10$、$E=49$ 时单步开销已经够大；论文实验里 $G$ 也是定值，代码取 $G=1$ 节省时间。

### 3.3 梯度上升 → SGD 下降 + 取反

论文："从 $\theta^{t(\tau)}$ 出发做 gradient ascent on $L$"。
代码：调 `train()` 跑 SGD **下降**，然后在 `craft_att` 里乘 −1。

数学上方向严格相反（梯度场互为相反数），只在 step size 几何含义上略有差别。工程上能直接复用 PyTorch optimizer / lr schedule。

### 3.4 良性 client 共享 `train_iter`

论文 §3：每个 worker $k$ 独立从自己的 $\hat P_k$ 采 minibatch。
代码：所有良性 client 共用 `self.train_iter`（`exp_environments.py:85-90`）。

后果（最严重的一条，单独展开）：
- **方差被压低**：仿真出的 10 份良性 update 几乎重合
- **clipping median 在仿真里显得"过强"**：良性 update 密集 → 中位数稳 → attacker 必须用更大 ε 才能顶过去
- **真实非 IID（q=0.5）时 mismatch 加剧**：仿真用 $\tilde P$ 模拟所有 worker，没有按 $\hat P_k$ 分别建模，model-based 框架的"模型"和真实环境差距是结构性的
- **`sim_train.py` 默认 q=0.1 接近 IID**，所以这个简化只有在低 q 时影响小

修法：`reset()` 里按论文 §6.1 的 q 公式把数据切给每个 `cid`，step 用 `self.train_iters[cid]` 代替共享迭代器。

### 3.5 τ 推进硬编码

论文：τ 只在采到 attacker 时推进，其余轮让 FL 自己跑。
代码：`is_attack=True`（`exp_environments.py:125`），`while is_attack==False` 永远不进。

默认设置下没采到任何攻击者的概率 ≈ $\binom{80}{10}/\binom{100}{10} \approx 9.5\%$，被简化掉的部分不大。代价是策略隐式假设"每步都能攻击"。

### 3.6 ε 区间 $[0.1, 29.9]$（vs Krum 的 $[0.1, 9.9]$）

clipping median 有 norm 上限 2 → attacker 发出去的 update 范数总会被裁到 2 → 想让恶意值在某些维度顶过中位数，**就需要更大的 ε 让原始方向更"长"**，被裁完之后剩下的有效幅度才足。Krum 不裁 norm，反而怕 ε 太大被距离判定踢掉。

→ **这是把 defense 知识手工编进 env 的最直接证据。**

---

## 4. 哪些该改、哪些不用改

### 🟢 可以接受（保持原样）

| 项                  | 理由                              |
| ------------------ | ------------------------------- |
| 下降+取反 替代真梯度上升      | 数学等价                            |
| λ-cos 项被砍          | clipping median 不看方向            |
| 攻击启动时机 epoch 100   | 已在本地代码里实现整套分阶段流程（distribution learning → policy learning → attack execution），与论文一致 |
| Clipping 阈值 = 2 写死 | 与论文一致，属于"环境常量"                  |

### 🟡 看场景决定

| 项 | 触发条件 |
|---|---|
| 动作维度 2D | 想做 defense-agnostic / meta-RL → 升回 3D |
| $G = 1$ | 训练曲线方差大时 → $G$ 加到 3–5 |
| `is_attack=True` | 想评估真实采样分布下的攻击 |
| ε 区间 $[0.1, 29.9]$ | 跨防御实验 → 用 log-uniform 等更大的统一区间 |

### 🔴 真有影响，建议改

| 项                                         | 原因                                                  | 修法                                                  |
| ----------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| **良性 client 共享 `train_iter`**             | 非 IID 下结构性低估 clipping median 脆弱程度，attacker 学不到利用异质性 | `reset()` 里按 q 切分数据 + per-client `train_iters[cid]` |
| **state per-frame min-max 归一化**           | 不同 FL 阶段的最后一层尺度被抹平，critic 看不出"早期/晚期"                | 用 `VecNormalize` 或自己维护 EMA mean/std                 |
| forward 已 log_softmax 又过 CrossEntropyLoss | reward 绝对值偏移（差值大致抵消）                                | `Net.forward` 末行改成 `return x`                       |
| `test` 没 `net.eval()`                     | dropout 仍随机 → reward 含噪                             | `test` 入口加 `net.eval()`                             |
| `loss` 是 sum 不是 mean                      | validset 大小变会跳变                                     | 改成 `/= total`                                       |
| `batch_size = len(validset)`              | 大数据集会爆显存                                            | 切成多个小 batch 循环累加                                    |

### 建议优先级

**良性 client 切分（#5）> state 归一化（#7）> 动作维度+cos 项+ε 区间一起做（#1+#4+#10）> 其它**

前两项是正确性问题，后一组是为了让 attacker 不再绑死单一 defense。

### 4.1 深入：state per-frame min-max 归一化为什么是隐性问题

**代码原貌**（`exp_environments.py:151-155`）：

```python
last_layer = np.concatenate([
    self.aggregate_weights[-2].flatten(),  # FC 层 weight: 10×128 = 1280
    self.aggregate_weights[-1]             # FC 层 bias  : 10
]).reshape(1, self.weights_dimension)      # 拼成 1290 维

state_min  = np.min(last_layer)            # ← 这一帧自己的 min
state_max  = np.max(last_layer)            # ← 这一帧自己的 max
norm_state = [
    2.0 * ((i - state_min) / (state_max - state_min)) - 1.0
    for i in last_layer
]
```

"**per-frame**" 的含义：**每一步都用当前这一帧自己的 min/max** 归一化，没有跨步共享。

**为什么是个问题**：FL 训练随时间的演化里，最后一层参数的尺度是有变化的：

| 真实 $\theta$ 区间 | 阶段 | 归一化后 critic 看到的 |
|---|---|---|
| $[-0.1,\ 0.1]$ | epoch 0–50（早期） | $[-1, 1]$ |
| $[-1.5,\ 1.5]$ | epoch 200–400（中期） | $[-1, 1]$ |
| $[-15,\ 12]$ | 被 attacker 打烂后 | $[-1, 1]$ |

→ 三种完全不同的 FL 状态被映射成同一个值域，**critic / actor 彻底丢掉"模型在训练曲线哪个位置"这一关键信息**。

**具体后果**：

1. **价值函数估计混乱**：同一组归一化后的 state，但底层 FL 阶段完全不同，对应的真实 reward 分布差很多 → TD3 的 $Q(s, a)$ 没法稳定收敛
2. **策略对阶段不敏感**：早期攻击和晚期攻击需要不同的 $\varepsilon$、$E$ 组合，但 actor 看不到阶段信号，只能学一个"平均策略"
3. **跨 episode 的尺度漂移**：episode 1 某帧 min/max 是 $[-0.05, 0.05]$，episode 50 可能是 $[-0.8, 0.8]$，归一化后都长得一样 → 经验回放里学不到 episode-level 的规律
4. **形状信息保留，幅度信息丢失**：归一化保留了"哪几维相对大"，但"绝对幅度多大"才是判断"模型是否被打坏"的最重要特征

**正常做法（running statistics）**：

```
维护全局 running_mean、running_var（跨 episode & step 累积）
norm_state = (state − running_mean) / sqrt(running_var)
```

不同帧用**同一套** mean/std 归一化 → 跨帧尺度一致 → critic 能区分阶段。

**修法**：用 `VecNormalize` ，env 内部不再做归一化：

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = DummyVecEnv([lambda: FL_mnist_clipping_median(...)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

注意把 env 里 `exp_environments.py:151-155` 那段 min-max 改成直接返回 raw `last_layer`，否则 `VecNormalize` 拿到的输入已经被压到 $[-1,1]$，再归一化就没意义了。

**为什么论文没踩这个坑**：论文目标是证明 "RL > 手工 baseline"，只要统计上显著就行，归一化方式不影响主结论。但如果你要做更长 horizon 的攻击、跨数据集/非 IID 程度泛化、或者稳定的 ablation 曲线，这一处会成为隐性瓶颈。

---

## 5. 关联背景：为什么 Krum 是另一条独立线（顺手记一下）

Krum 原始设计（Blanchard et al., NeurIPS 2017）：对每个 client update 计算到最近 $n - f - 2$ 个 update 的距离和（score），选 score 最小的那个作为唯一聚合结果。

防御逻辑：恶意 update 想破坏模型时会**远离 benign cluster** → score 变大 → 被排除。

→ 这种防御**只看距离，不看 norm**，所以 attacker 需要"伪装得像正常人"（cosine 重要），ε 不能太大（怕被距离判定踢掉）。这正是代码里 `FL_mnist_krum` 的 ε 区间被压缩到 $[0.1, 9.9]$ 的原因。

参考：[Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.neurips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent)

---

## 6. 论文实现思路细节（把整个 pipeline 串起来）

前面 §1 只列了 MDP 形式和动作设计；这里把论文"为什么这么搭"的整体工程思路串一遍，便于在自己复现 / 修改时心里有数。

### 6.1 三阶段流水线（§3）

论文把整个攻击 pipeline 切成三段，**三段同时与 FL 训练并行**，不是顺序串行：

```
FL epoch:    0 ─────────────────────── 100 ─────────── 400 ──────────── 1000
             │                          │               │                 │
Dist. learn: ├──── 从 server 模型更新中重构数据（IG）─┐
                                        │              │
Policy learn:├────────── TD3/PPO 在仿真器里学策略 ────┤
                                        │              │
Attack exec:                            ├── 用最新策略打真 FL ──────────┤
```

三阶段并行而不是顺序的原因：FL 训练一旦开始就一直在进行，攻击者**不能等**到分布完全学完再开始策略学习——否则可攻击窗口被吃掉一半。

### 6.2 为什么要 model-based 而不是 model-free（§4.1 结尾）

model-free RL（直接和真实 FL 交互、TD3 在线学）的问题：

- **样本效率太低**：TD3 通常需要 $10^5\sim10^6$ 步经验才收敛
- **真实 FL 里 attacker 不是每轮都被抽中**：默认设置下 9 成轮次 attacker 才被采到 0–2 个，"有效经验"密度极低
- **FL 单步成本高**：真实 FL 一轮要等所有 client 上传，分钟级别

所以必须 **off-policy 在仿真器里训练**：
- 仿真器近似真实 FL 的 transition + reward
- 一次 TD3 step ≈ 一次仿真 FL 轮 ≈ 毫秒级
- 训完再上线打真 FL

仿真器的核心未知量是良性 client 的数据分布 $\{P_k\}$ —— 这就是 distribution learning 要解决的。

### 6.3 为什么用 IG（gradient inversion）学分布（§4.2）

attacker 能拿到的"信号"只有服务器在每轮广播的全局模型 $\theta^t$。从相邻两个模型差能反推出**那一轮聚合后的批量梯度**：

$$
\bar g^\tau \approx \frac{\theta^{t(\tau-1)} - \theta^{t(\tau)}}{\eta \cdot (t(\tau) - t(\tau-1))}
$$

IG（Inverting Gradients, Geiping et al. 2020）原本用于隐私攻击：给定梯度，反推出生成这梯度的训练样本。论文把它"借"来：
- 维护一批 dummy 样本 $D_{\text{dummy}}$
- 优化目标：让 $D_{\text{dummy}}$ 上算出的梯度方向（cosine）接近观测到的 $\bar g^\tau$
- 加 TV (total variation) 正则保持图像平滑

得到的 dummy 样本就近似来自 $\hat P = \sum_k \tfrac{N_k}{N}\hat P_k$ 这个混合分布。注意：

- 论文反的是**聚合后**的批量梯度（已经经过 server aggregation + 多 client 平均），不是单个 client 的梯度——所以**只能反出混合分布 $\hat P$**，反不出单个 $\hat P_k$
- 反出的样本带噪声，论文用 **denoising autoencoder** 去噪：拿 attacker 已有的干净样本 + 人工高斯噪声训自编码器，再用它过滤 IG 输出
- 最终 $\tilde P$ = IG 重构样本（去噪后） + attacker 自有数据

### 6.4 simulator 怎么"用 $\tilde P$ 装 FL"（§4.3 开头）

仿真器要近似真实 FL 的两件事：
- **良性 client 行为**：每个 worker 从 $\tilde P$ 采 minibatch、跑 1 步 SGD
- **服务器行为**：对收到的 update 跑 `Aggr(...)`（这块是已知的，attacker 在 white-box 设定下知道）

attacker 在仿真器里**已知** attacker 自己的数据 + 服务器算法，**未知**良性 client 的真实数据 → 用 $\tilde P$ 填这一格。Theorem 1 给出了"$\tilde P$ 越准（Wasserstein 距离越小），仿真器训出的策略性能损失越小"的界。

### 6.5 为什么选 TD3（§6.1）

论文说 "TD3 在大多数 setting 更好"。原因（论文没展开，但能推出）：

- 动作空间 3 维连续 → 排除 DQN 类离散方法
- 候选：DDPG / TD3 / SAC / PPO
- TD3 是 DDPG 的改进版（双 Q + 延迟更新 + 目标平滑），off-policy + 经验回放，**样本效率高**
- PPO 是 on-policy，每个 batch 用过就丢，仿真器虽然便宜但仍然慢于 TD3
- SAC 通常和 TD3 一档，论文没特别说为什么没选 SAC，可能就是先跑通 TD3 没必要再换

### 6.6 关键超参的物理含义

| 超参 | 论文默认值 | 含义 |
|---|---|---|
| $H$ | ~900（= 1000 − 100） | 单 episode 攻击步数 |
| $G$ | 论文未显式给值 | 本地搜索轨迹数（降方差） |
| $B'$ | 论文未显式给值 | 本地搜索每步 minibatch 大小 |
| $\tau_E$ | 100 | distribution learning 阶段长度 |
| $m$ | 200（MNIST） | attacker 初始持有的真样本数 |
| $q$ | 0.5（默认非 IID 度） | 非 IID 度，每个类有 $q$ 概率落在指定组 |

### 6.7 论文的关键 trick / 设计选择总览

| 选择 | 为什么 |
|---|---|
| state 用最后一层而不是整模型 | 整模型 $d$ 维（百万级），全喂给策略网络不可行 |
| 动作用 $(\gamma, E, \lambda)$ 而不是 $d$ 维向量 | 直接学 $d$ 维方向 RL 学不来；改成"标量控制的本地搜索"后只有 3 维 |
| cos 正则项 | 让恶意方向像正常方向，对 Krum / FLTrust 这种"看方向"的防御有效 |
| 三阶段并行 | 攻击窗口不能等分布完全学好 |
| denoise autoencoder | IG 重构样本带高频噪声，直接用会让 $\tilde P$ 严重偏离 |
| 所有 attacker 共享同一份策略 | 减少训练开销；论文实验显示效果已经够好 |
| TD3 + off-policy + replay | 仿真器再快也比 SGD 慢，需要样本效率 |

---

## 7. 工程优化方案（针对 🔴 项的修法）

**前提**：本节目标不是做"defense-agnostic attacker"，而是把 `FL_mnist_clipping_median` 这一支 attacker 在**正确性 / 训练稳定性 / 实验严谨度**这三件事上修干净。下面伪代码以 `FL_mnist_clipping_median` 为模板。

7.1–7.3 + 7.5 是 clipping median 单防御场景下也直接受益的修法；7.4 和 7.6 列出来只是"如果将来想扩展再回来看"，不属于必做项。

### 7.1 良性 client 按 $\hat P_k$ 切分（最高优先级）

**问题**：所有良性 client 共用 `self.train_iter`，方差被压低，非 IID 实验失真。

**改法**：在 `__init__` 接收 `q` 参数，`reset()` 里按论文 §6.1 q 公式分发数据。

```python
def __init__(self, att_trainset, validset, full_trainset, q=0.5, num_classes=10):
    ...
    self.q = q
    self.num_classes = num_classes
    self.full_trainset = full_trainset   # 良性 client 总数据池

def _partition_by_q(self):
    """按 q 把 full_trainset 切给 num_clients 个 client"""
    by_class = defaultdict(list)
    for idx, (_, y) in enumerate(self.full_trainset):
        by_class[y].append(idx)

    # 按论文 §6.1：标签 c 的样本以 q 概率落入第 c 组，否则均分到其它组
    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in by_class.items():
        for idx in idxs:
            if np.random.rand() < self.q:
                target_group = c
            else:
                target_group = np.random.choice(
                    [g for g in range(self.num_classes) if g != c]
                )
            # group 内 client 间均分
            cid = target_group * (num_clients // self.num_classes) + \
                  np.random.randint(num_clients // self.num_classes)
            client_indices[cid].append(idx)
    return client_indices

def reset(self):
    ...
    client_indices = self._partition_by_q()
    self.train_iters = {}
    for cid in range(num_clients):
        if cid in att_ids:
            continue
        subset = Subset(self.full_trainset, client_indices[cid])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        self.train_iters[cid] = mit.seekable(loader)
    ...

def step(self, action):
    ...
    for cid in exclude(self.cids, att_ids):
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.train_iters[cid], epochs=1, lr=self.lr)  # ← per-client iter
        new_weights.append(get_parameters(self.net))
    ...
```

**收益**：仿真器真正建模异质性，$q$ 越大良性 update 越散，clipping median 在仿真里表现接近真实部署。

### 7.2 state 用 running statistics 归一化（去掉 per-frame min-max）

**问题**：per-frame min-max 抹掉 FL 阶段尺度信息。

**改法**：env 内部直接返回 raw 最后一层，外面套 `VecNormalize`。

```python
# env 内：去掉 min-max，直接给原始值
def step(self, action):
    ...
    last_layer = np.concatenate([
        self.aggregate_weights[-2].flatten(),
        self.aggregate_weights[-1]
    ]).astype(np.float32)
    return {"pram": last_layer, "num_attacker": ...}, reward, done, {}

# 训练入口：套 VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: FL_mnist_clipping_median(...)])
env = VecNormalize(env, norm_obs=True, norm_reward=False,  # reward 不归一化避免改变物理意义
                   clip_obs=10.0)
model = TD3("MultiInputPolicy", env, ...)
model.learn(total_timesteps=80000)

# 评估时记得 env.training = False、env.norm_reward = False
```

**注意**：`VecNormalize` 自带 `save/load`，保存策略时一定要把它的 running stats 一起存，部署时复用。

### 7.3 评估函数 `test` 修复（一组小改动）

**问题**：`test` 里 `loss` 是 sum 不是 mean、没 `net.eval()`、forward 已 log_softmax 又过 CE。

**改法**：

```python
# utilities.py
def test(net, valloader):
    net.eval()                           # ← 关 dropout
    criterion = torch.nn.NLLLoss(reduction='sum')   # ← 因为 forward 已 log_softmax
    # 或者改 Net.forward 末行为 return x，criterion 保持 CrossEntropyLoss
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()                          # ← 恢复
    return loss / total, correct / total  # ← 平均 loss，不是 sum
```

**注意**：`loss / total` 改成平均后，reward 物理意义变了（之前是"总 CE"，现在是"平均 CE"），ε 区间可能需要重新扫一遍。

### 7.4 [可选] action 升级到 3D（仅当将来想扩展到其它防御）

> ⚠️ **不是必做项**。只针对 clipping median 训 attacker 的话，2D 动作 $(\varepsilon, E)$ 完全够用——$\lambda$ 在 clipping median 上没有收益（防御不看方向）。
> 仅当未来想把同一份代码扩展到 Krum / FLTrust 时，再回头读这一节。

**改法**：所有 env 统一 3D 动作 $(\gamma, E, \lambda)$，clipping median 上 $\lambda$ 大概率被 RL 学成接近 0，不影响效果但保留接口。

```python
self.action_space = spaces.Box(-1, +1, shape=(3,), dtype=np.float32)

def step(self, action):
    # 用 log-uniform 让动作幅度不绑死防御
    gamma  = np.exp(action[0] * np.log(50.0))    # γ ∈ [1/50, 50]
    E      = int(np.clip(action[1] * 24 + 25, 1, 49))
    lam    = (action[2] + 1) / 2                 # λ ∈ [0, 1]

    # 本地搜索目标 L(θ) = (1-λ)·F(θ) + λ·cos(θ_old - θ, g(θ_old))
    set_parameters(self.net, self.aggregate_weights)
    g_normal = compute_avg_grad(self.net, self.all_train_iter, self.lr)  # 论文里的 g(θ^t)
    bar_theta = local_search(
        net=self.net, init=self.aggregate_weights, iter=self.all_train_iter,
        E=E, lam=lam, g_normal=g_normal
    )
    new_weight = craft_att(self.aggregate_weights, bar_theta, -1, gamma)
    ...
```

**收益**：换防御时不用动 action space 维度，只换 `Aggr` 函数。

### 7.5 把 `is_attack=True` 改回真采样

**问题**：硬编码假设每步都有攻击者被采到。

**改法**：

```python
self.cids = random.sample(range(num_clients), int(num_clients * subsample_rate))
is_attack = check_attack(self.cids, att_ids)

while not is_attack:
    # 让 FL 自己跑一轮良性聚合，不进 RL 经验
    new_weights = []
    for cid in self.cids:
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.train_iters[cid], epochs=1, lr=self.lr)
        new_weights.append(get_parameters(self.net))
    self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
    self.rnd += 1
    if self.rnd >= 1000:
        return self._terminal_obs()
    self.cids = random.sample(range(num_clients), int(num_clients * subsample_rate))
    is_attack = check_attack(self.cids, att_ids)
```

**收益**：仿真器分布更接近真实部署。代价是 step 时间变长（部分 step 内含多轮 FL）。

### 7.6 [可选] 防御聚合规则抽象成可注入（仅当将来想扩展）

> ⚠️ **不是必做项**。clipping median 单防御场景下 `FL_mnist_clipping_median` 这一个类够用了。
> 仅当未来想加 Krum / FLTrust / 别的聚合规则、想避免代码重复时，再回头读这一节。

**问题**：每个防御一个 env 类，10 行代码 90% 重复。

**改法**：

```python
class FL_env(gym.Env):
    def __init__(self, ..., aggr_fn, aggr_kwargs=None):
        self.aggr_fn = aggr_fn
        self.aggr_kwargs = aggr_kwargs or {}
        ...

    def step(self, action):
        ...
        self.aggregate_weights = self.aggr_fn(
            self.aggregate_weights, new_weights, **self.aggr_kwargs
        )
        ...

# 使用
env_cm   = FL_env(..., aggr_fn=Clipping_Median, aggr_kwargs={"max_norm": 2})
env_krum = FL_env(..., aggr_fn=Krum, aggr_kwargs={"num_attacker": 20})
env_ft   = FL_env(..., aggr_fn=FLtrust, aggr_kwargs={"root_iter": root_iter})
```

**收益**：新加防御只需要写一个 `Aggr` 函数；attacker 端代码完全复用。

---

## 8. 这是白盒还是黑盒攻击？

### 8.1 论文的明确定位：**白盒**（§1, §3）

论文原话（§1）：

> we take a first step in this direction by considering the **white-box attack setting** where the attacker has some global knowledge about the FL system and the server's algorithm, but has no access to the private data of benign devices, a reasonable assumption for real-world FL systems.

§3 进一步具体列出了 attacker 在白盒设定下知道什么：

| 信息 | attacker 是否知道 |
|---|---|
| 服务器学习率 η | ✓ |
| 采样率 κ | ✓ |
| 总 client 数 K | ✓ |
| 总训练轮数 T | ✓ |
| **聚合规则 `Aggr`**（Krum / clipping median / FLTrust 等） | ✓ |
| 自己的本地数据 $\hat P_k\ (k \in A)$ | ✓ |
| 自己的 batch size B | ✓ |
| 每轮全局模型 $\theta^t$（服务器广播） | ✓ |
| 良性 client 的私有数据 $\hat P_k\ (k \notin A)$ | ✗ |
| 良性 client 单独的 update（IPM/LMP 假设了这点，本文 RL 不依赖） | ✗ |

attacker 唯一缺失的核心信息是**良性 client 的数据**——这正是 distribution learning 阶段要补的（用 IG 反推混合分布 $\tilde P$）。

### 8.2 代码层面的白盒证据

仓库里有多处直接体现"attacker 知道 server 算法"：

| 代码位置 | 体现 |
|---|---|
| `FL_mnist_clipping_median.step()` 行 `116` 直接调 `Clipping_Median(...)` | env 内部就是真实 server 的聚合规则——attacker 完全知道防御长什么样 |
| ε 区间 $[0.1, 29.9]$ vs Krum 的 $[0.1, 9.9]$ | 攻击参数按防御特性人工调过，**这本身就是白盒假设**——attacker 知道当前是 clipping median 才会选这个区间 |
| craft_att 的方向取反 + 放缩公式 | 设计前提是 attacker 知道服务器会做"old − Aggr(...)" 这种更新规则 |
| FLTrust env 里专门写 `att_acsend_root` 把方向投影回 root 梯度模长 | 直接利用了 FLTrust 用 root 梯度评分这一防御内部机制 |

→ 整个仓库的 env 设计**不能在不知道防御类型的前提下复用**。

### 8.3 黑盒 vs 灰盒会是什么样

为方便对比，三种威胁模型：

| 模型 | attacker 知道 | attacker 不知道 |
|---|---|---|
| **白盒**（论文 + 仓库） | 服务器算法（含 `Aggr`）+ 超参 + 自己数据 + 每轮 $\theta^t$ | 良性 client 私有数据 |
| **灰盒** | 服务器算法部分已知（比如知道用了"某种鲁棒聚合"但不知道是哪种）+ 自己数据 + 每轮 $\theta^t$ | 良性数据 + 具体防御 |
| **黑盒** | 仅 $\theta^t$ 和自己数据 | 服务器算法、聚合规则、超参一概不知 |

论文 §1 最后一段也说了：

> our proposed framework can potentially be applied to other types of attacks in both the white-box and the more challenging black-box settings.

但论文**没有**给黑盒实验。黑盒留给后续工作。

### 8.4 这个白盒假设强不强？

业内常见看法：

- **不算特别强**。FL 场景里聚合规则、η、采样率通常是协议级公开信息（开源 SDK 比如 Flower、FedML 默认行为），攻击者只要能读代码 / 抓包就能知道；真正的"私有"主要是良性 client 数据
- **比"知道良性 update"弱**。IPM[Xie 2020] 和 LMP[Fang 2020] 这两个 baseline 假设 attacker 能拿到良性 client 的本轮 update（这才是不太现实的强假设）；RL 攻击不需要这个，只需要 server 广播的 $\theta^t$
- **比真正黑盒强**。如果不知道用了什么防御，那 attacker 没法针对性设计动作空间和 ε 区间

→ 论文这个白盒定位是个**合理而有意义的中间点**：既现实（不假设能偷良性数据），又有足够信息让 RL 跑得起来。

### 8.5 想做黑盒会需要改什么

如果你想把这套框架推到黑盒，主要要补三件事：

1. **聚合规则识别**：attacker 没法直接调用 `Clipping_Median(...)`，只能从观测到的 $\theta^{t-1} \to \theta^t$ 变化反推服务器在做什么聚合。可能要：
   - 用主动探测（发不同幅度的 update 看哪些被压住）
   - 用 supervised learning 训一个"聚合规则分类器"
2. **超参盲打**：不能再为 clipping median 单独调 ε 区间，得让 RL 自己探索（log-uniform、宽区间）
3. **更强的 distribution learning**：白盒下 IG 用的是"已知 η、κ 反推批量梯度"，黑盒里 η 都不一定知道，需要联合估计

这一块属于研究方向，不属于工程优化，超出当前笔记范围。

### 8.6 一句话总结

**论文 + 仓库 = 严格的白盒攻击**：attacker 知道服务器跑的是 clipping median + 学习率 0.01 + 采样率 0.1 等所有协议级信息，**但不知道良性 client 的私有数据**——这一缺口正是 distribution learning（IG + denoise）要填的。代码里"每防御一个 env 类、ε 区间手调"这种做法本身就预设了 attacker 知道当前防御是什么，是白盒最直接的证据。

---

## 9. FL Sandbox 当前沉淀方式

### 9.1 strict reproduction 与 optimized variant 分开

当前工程里保留两条线：

| Defense | 论文复刻 / strict | 优化版 |
|---|---|---|
| `clipped_median` | `legacy_clipped_median_strict` | `legacy_clipped_median_scaleaware` |
| `krum` | `legacy_krum_strict` | `legacy_krum_geometry` |

严格复刻线只负责对齐原仓库动作、reward、数据调度和攻击生成逻辑；优化线只改实现思路，不改外层训练框架。Krum 的优化版把 simulator 里的几何近似抽到了 `fl_sandbox/attacks/rl_attacker/krum_projection.py`，live attack 仍在真实 Krum 选区上做 projection，因此训练近似和真实执行边界分开。

### 9.2 Benchmark matrix

统一比较入口：

```bash
python -m fl_sandbox.scripts.run_robust_defense_attack_matrix \
  --defenses clipped_median krum \
  --attacks clean ipm lmp dba bfl \
    rl_clipped_median_scaleaware rl_krum_geometry \
  --rounds 150 \
  --num-clients 100 \
  --num-attackers 20 \
  --subsample-rate 0.1 \
  --krum-attackers 20 \
  --policy-train-steps-per-round 50
```

默认就是 full data；只有 smoke run 才显式加 `--max-client-samples-per-client` / `--max-eval-samples`。输出结构按 `defense/attack_plan/run_name` 分目录，避免 `rl` 这类相同 `attack_type` 在不同 semantics 下互相覆盖。

默认比较集只包含 `clean`、`IPM/LMP/DBA/BFL` 和两条优化版 RL attacker：`rl_clipped_median_scaleaware`、`rl_krum_geometry`。严格复刻版 `rl_clipped_median_strict` / `rl_krum_strict` 以及 geometry-search heuristic 仍可通过 `--attacks` 显式加入，用于论文复刻或消融，不混入默认 benchmark。

如果 RL attacker 已经完成 full-data 训练，可以用 `--reuse-summary defense:attack_name:path` 复用既有 run，并由 matrix runner 统一写 summary 和 TensorBoard。例如本次 150 轮 benchmark 复用已完成的 clipped/Krum optimized RL：

```bash
python -m fl_sandbox.scripts.run_robust_defense_attack_matrix \
  --defenses clipped_median krum \
  --attacks clean ipm lmp dba bfl \
    rl_clipped_median_scaleaware rl_krum_geometry \
  --rounds 150 \
  --reuse-summary clipped_median:rl_clipped_median_scaleaware:fl_sandbox/outputs/paper_clippedmedian_scaleaware_1000r_policy50/train/mnist_rl_clipped_median_paper_q_q0.1_1000r \
  --reuse-summary krum:rl_krum_geometry:fl_sandbox/outputs/krum_geometry_full_policy50_150r/train/mnist_rl_krum_paper_q_q0.1_150r
```

复用的长 run 会按 `--rounds` 截断系列数据，避免 1000 轮 RL 和 150 轮 benchmark 混在同一个 final/tail 口径里。

TensorBoard 记录两层：

| 路径 | 内容 |
|---|---|
| `runs/.../<defense>/<attack_plan>/...` | 每个 run 自己的训练趋势：accuracy、loss、RL loss、projection/bypass 等 |
| `runs/.../matrix_trends` | matrix 级 per-round 趋势，包含每个 attack 的原始数值和 clean-relative accuracy drop |
| `runs/.../matrix_summary` | matrix 级最终汇总，方便扫最终 drop 和 tail drop |

`IPM`、`LMP`、`DBA`、`BFL` 是固定 benchmark；`clean` 是每个 defense 的基线；RL 对比看当前训练出的 optimized attacker。正式报告优先看 per-round trend 和 tail mean，不只看单点 final accuracy。
