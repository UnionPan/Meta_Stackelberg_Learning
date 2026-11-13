## 1. 研究目标与评价口径

在有限候选攻击集合中，防御方不知道当前使用哪一种或哪几种攻击。目标是学习一个防御初始化，用少量部署反馈适应实际单一／混合攻击，同时保持正常任务准确率并抑制攻击。
非定向攻击报告正常测试准确率 ACC；后门攻击同时报告 ACC、ASR ，后续综合效用记为 Value=ACC−ASR：

$$
V(\mathrm{ACC},\mathrm{ASR})\coloneqq\mathrm{ACC}-\mathrm{ASR}.
$$

ACC 越高越好，ASR 越低越好，Value 越高越好。ACC、ASR 均用百分数尺度，Value 的单位为百分点（pp）。

## 2. 原论文算法参考

### 2.1 Algorithm 1：Meta-Stackelberg Learning

![[algorithm1_meta_stackelberg_original.png|475]]

Meta-SG 分支：**防御适应 → 固定防御、更新攻击者 → 计算响应后的防御梯度 → 汇总更新初始化。**

| 符号 | 含义 |
| --- | --- |
| $\Xi$、$\xi$、$Q(\Xi)$ | 候选攻击类型集合、其中一个类型、类型采样分布 |
| $K$、$\pi_\xi$ | 每批采样类型数、预定义攻击策略 |
| $J_{\mathcal D}$、$J_{\mathcal A}$ | 防御方、攻击方的期望回报目标 |
| $\theta^0$、$\theta^t$ | 初始防御参数、第 $t$ 轮共同初始化 |
| $\theta_\xi^t$ | 任务 $\xi$ 中适应后的防御参数 |
| $\phi_\xi^0$、$\phi_\xi^t$ | 预训练攻击参数、第 $t$ 轮攻击参数 |
| $\phi_\xi^t(0)$、$\phi_\xi^t(k)$、$\phi_\xi^t(N_{\mathcal A})$ | 本次响应的起点、第 $k$ 次更新后、响应结束后的攻击参数 |
| $\phi_\xi^*$ | 理论最佳响应；有限次更新得到其近似 |
| $\hat\nabla J_{\mathcal D}(\xi)$ | 任务 $\xi$ 提供的外层防御梯度 |
| $N_{\mathcal D}$、$N_{\mathcal A}$ | 外循环次数、每任务攻击者响应更新次数 |
| $t$、$k$ | 外循环编号、攻击者响应内循环编号 |
| $\eta$ | 防御任务内适应步长 |
| $\kappa_{\mathcal A}$、$\kappa_{\mathcal D}$ | 攻击者响应步长、防御外层步长 |
| $1/K$、$\theta^{N_{\mathcal D}}$ | 任务贡献平均系数、最终防御初始化 |

当前实验采用一阶响应后梯度，执行参数见 6.5。

### 2.2 Algorithm 2：多步 Reptile Meta-RL

![[algorithm2_reptile_meta_rl_original.png|413]]

**复制初始化 → 各任务内适应 $l$ 步 → 平均参数差值 → 更新初始化。**

| 符号 | 含义 |
| --- | --- |
| $Q$、$\xi$、$\hat\Xi$、$K$ | 类型分布、一个攻击类型、采样任务批、批内任务数 |
| $\tau$、$H$、MC | 一条轨迹、轨迹长度、Monte Carlo 回报／梯度估计 |
| $\theta^0$、$\theta^t$ | 随机初始参数、第 $t$ 轮共同初始化 |
| $\theta_\xi^t(0)$、$\theta_\xi^t(k)$、$\theta_\xi^t(l)$ | 任务内起点、第 $k$ 步参数、适应终点 |
| $T$、$l$ | 元训练外循环次数、每任务防御适应步数 |
| $k$、$\kappa$ | 防御任务内更新编号、步长 |
| $\eta$ | 输入中列出，但伪代码更新式未使用 |
| $\hat\nabla_\theta J_{\mathcal D}(\tau)$ | 根据轨迹估计的防御梯度 |
| $\theta_\xi^t(l)-\theta^t$、$\theta^T$ | 任务内参数变化、最终防御初始化 |

$$
\theta^{t+1}=\theta^t+\frac{1}{K}\sum_{\xi\in\hat\Xi}
\left(\theta_\xi^t(l)-\theta^t\right).
$$

算法 2 未展开攻击者响应循环。算法 1 的 Reptile 分支汇总适应后梯度；算法 2 汇总参数差值，外层系数为 1。

项目采用自定义观测、risk／mix／strength 动作及 root／risk 奖励，以真实 FL 生成轨迹。

## 3. 防御观测：Compact8 与 Histogram40

### 3.1 客户端原始特征

服务器拥有小型、可信且带标签的 root 数据集。设全局模型为 $w$，客户端更新为 $\Delta_i$，root 参考更新为 $\Delta_{root}$；每轮构造 $X_t\in\mathbb R^{n_t\times4}$，四列依次为 C、N、B、Z。

| 特征    | 当前定义                                                 | 含义                   |
| ----- | ---------------------------------------------------- | -------------------- |
| $C_i$ | $1-\cos(\Delta_i,\Delta_{root})$                     | 相对可信更新的方向偏离          |
| $N_i$ | $\log\lVert\Delta_i\rVert$                           | 更新幅度的对数，零范数使用正数下界    |
| $B_i$ | $\max_c\lvert\bar p_c(w+\Delta_i)-\bar p_c(w)\rvert$ | root 平均类别预测概率的最大绝对变化 |
| $Z_i$ | $L_{root}(w+\Delta_i)-L_{root}(w)$                   | root 交叉熵损失变化         |

$\bar p_c$ 为 root 上第 $c$ 类预测概率的均值。当前 N 对应 `signed_log_norm=True`。

### 3.2 Compact8

对 C、N、B、Z 各取 P50、P90，按 `[C50,N50,B50,Z50,C90,N90,B90,Z90]` 拼接。

### 3.3 Histogram40

1. 汇总训练校准轨迹的客户端特征，每列取 P10～P90，得到 9 个固定边界、10 个箱。
2. 每轮统计客户端落入各箱的比例；边界值归右侧箱，首尾箱延伸至无穷。重复边界允许空箱。
3. 四列各 10 个占比，组成 $10\times4$ 数组，按行展开为 40 维。

对特征 $f$、箱号 $b$：

$$
b_{i,f}=\sum_{j=1}^{9}\mathbf1[x_{i,f}\ge e_{f,j}],\qquad
h_{f,b}(t)=\frac{1}{n_t}\sum_i\mathbf1[b_{i,f}=b].
$$

实际输入顺序为：

$$
u_t^{hist}=[h_{C,0},h_{N,0},h_{B,0},h_{Z,0},\ldots,
h_{C,9},h_{N,9},h_{B,9},h_{Z,9}]\in\mathbb R^{40}.
$$

标准化前，各特征的 10 个箱占比之和为 1；直方图保留粗粒度分布，丢失箱内细节和跨特征关联。

### 3.4 输入标准化

Meta-SG 两种表示均用共同训练校准数据拟合逐维均值和标准差：$o_d=(u_d-\mu_d)/\max(\sigma_d,10^{-4})$；分箱和标准化参数在部署时冻结，支持／查询数据不参与拟合。8.3 的历史 Compact8 使用原始摘要。

## 4. 防御网络：Actor 与双 Critic

Meta-SG 使用以下 temporal TD3 网络，$d=8$ 或 $40$、γ=0.99；缓存 RL 使用 γ=0 的分解网格 Critic。

### 4.1 Actor

![[output/meta_sg_architecture_20261107/meta_sg_actor_architecture_v1.png|1000]]

```text
标准化观测 d → Linear(d,64) → ReLU
             → Linear(64,64) → ReLU
             → Linear(64,3) → Tanh → 三维动作
```

| 动作 | 控制对象 |
| --- | --- |
| risk_weight | 客户端风险降权强度 |
| norm_mix | 中位数范数与可信 root 范数规则的混合比例 |
| norm_strength | 范数修正强度 |

动作范围为 $[-1,1]$，由执行模块转换为客户端权重与更新缩放。探索噪声在 Actor 输出之后添加。

[PDF](output/meta_sg_architecture_20261107/meta_sg_actor_architecture_v1.pdf) · [SVG](output/meta_sg_architecture_20261107/meta_sg_actor_architecture_v1.svg)

### 4.2 双 Critic

![[output/meta_sg_architecture_20261107/meta_sg_critic_architecture_v1.png|1000]]

将标准化观测逐元素做固定 asinh 变换，与三维动作拼接，分别输入 Q1、Q2：

```text
[asinh(o), a]，d+3 维
    → Linear(d+3,64) → ReLU
    → Linear(64,64) → ReLU
    → Linear(64,1) → 标量 Q(o,a)
```

Q1、Q2 及 Actor 参数独立；动作不做 asinh，Critic 线性输出 $Q(o,a)$。

## 5. 攻击方式

### 5.1 攻击定义与指标

| 攻击 | 构造方式 | 指标 |
| --- | --- | --- |
| IPM-5 | 恶意更新为本轮正常客户端平均更新的 −5 倍 | ACC↑ |
| LMP-Krum | 沿正常平均更新的反符号方向搜索，以 Krum 为选择判据 | ACC↑ |
| RL Attacker（MNIST） | Li2022 策略控制反向更新倍率与代理 SGD 步数 | ACC↑ |
| RL／BRL（Meta 任务） | 分别为可学习非定向投毒／后门攻击；攻击者在元训练响应阶段更新 | ACC↑／ACC↑＋ASR↓ |
| DBA | 分布式子触发器投毒；结果区分 simultaneous 与 cyclic | ACC↑／ASR↓ |
| BFL | 使用完整触发器投毒 | ACC↑／ASR↓ |

### 5.2 各实验攻击配置

- **MNIST，20 全参与**：覆盖 IPM、LMP、RL Attacker、DBA simultaneous／cyclic、BFL；RL Attacker 的 IID、Non-IID 实验使用不同攻击者。
- **MNIST，100 抽 10**：攻击从第 3 轮开始；冻结 Li2022 SB3 10k 攻击者，使用 200 个既有训练样本；两组攻击代理 SGD 学习率分别为 0.01、0.05。
- **CIFAR-10，100 抽 10**：攻击从第 3 轮开始；LMP-Krum 使用 f=2、二分搜索 20 步、最大倍率 20。DBA simultaneous 的 5 个恶意身份分为四个触发器组，人数为 2／1／1／1；恶意本地训练 10 steps、投毒比例 0.5、更新放大 10 倍。BFL 恶意本地训练 10 steps、投毒比例 1.0、更新缩放 1.0。DBA／BFL 恶意本地 lr=0.05，源标签 0、目标标签 9；ASR 为源类别 0 的测试图像加完整触发器后被预测为 9 的比例。

- **Tiny-ImageNet／ResNet18，100 抽 10**：IPM-5 从第 11 轮开始，恶意更新为正常平均更新的 −5 倍；LMP 从第 11 轮开始，Krum 搜索预设 f=2、搜索 20 步、最大倍率 100；DBA 从第 101 轮开始，20 个恶意身份均分四组，使用 20 像素子触发器；BFL v2 分别从第 101 轮或第 1 轮开始，使用 80 像素完整白色触发器，更新缩放为 `10 / 当轮入选恶意数`。DBA／BFL 的投毒比例为 0.3125，目标类别为 9；ASR 分母为 9,950 张非目标验证图像。

### 5.3 固定混合攻击调度

20 个客户端中，16 个正常、4 个恶意；客户端 0／1 执行第一类攻击，2／3 执行第二类，从第 3 轮起同时攻击，轨迹内身份与类型固定。测试组合为 IPM+LMP、IPM+DBA、LMP+DBA、DBA+BFL。

LMP 搜索包含另一类已构造的恶意更新；两个 DBA 客户端各持有四块子触发器中的两块，测试使用完整触发器。DBA+BFL 共用触发图案及源 0→目标 9 的测试语义。

## 6. 防御方法

### 6.1 聚合规则

| 方法 | 规则 |
| --- | --- |
| FedAvg | 对入选客户端更新取平均 |
| Median | 按坐标取中位数；偶数个客户端取两个中央值的均值 |
| Trimmed Mean | 按坐标去掉最大、最小各 f 个值后平均 |
| Krum | 按更新间距离评分，选择一个更新 |
| Multi-Krum | 选择评分最小的若干更新并平均 |
| RFA | 平滑几何中位数 |
| FLTrust | 用可信 root 更新确定方向信任权重，并归一化客户端更新范数 |
| FLTrust+NC | 共享 FLTrust 训练权重，用 root 校准推理时的 ReLU 激活裁剪 |
| ClipMed | 先裁剪各更新的 L2 范数，再取逐坐标中位数 |
| RL | 根据观测输出三维防御动作 |
| RL transfer | 按当前攻击选用对应的冻结专用策略，迁移至目标客户端配置 |

### 6.2 防御训练与参数

| 训练配置 | 方法与参数 |
| --- | --- |
| 旧版三动作 | 原始 Compact8；动作控制重加权、范数裁剪和 NeuroClip；每攻击独立训练 10k TD3 步后冻结 |
| 缓存 mixed-norm | Compact8，关闭 NeuroClip；20 客户端、每攻击 10,000 步、seed=42；TD3 使用分解网格 Critic、γ=0、缓存上下文决策 |
| 缓存训练奖励 | 0.25 倍即时 root 损失改善＋0.75 倍三步后 root 损失改善＋参考风险与保留风险之差 |
| 表示对照 | Compact8 不作输入标准化；Histogram40 冻结分箱边界和输入标准化参数；各攻击独立训练后冻结评价 |
| 冻结迁移 | 将 20 客户端配置下训练的专用策略迁移至 100 抽 10，部署时不更新 |

- **Tiny-ImageNet／ResNet18**：Compact8，risk／norm_mix／norm_strength 三动作，NeuroClip 关闭；每攻击专用策略训练 10,000 步，γ=0、缓存网格 Critic，评价时冻结。BFL 首轮投毒分别保留匹配训练策略和从第 101 轮投毒训练的旧策略。

### 6.3 常规聚合器参数

100 抽 10 实验：Trimmed Mean／Krum／Multi-Krum 预设 f=2，Multi-Krum 保留 8 个更新，RFA 最多迭代 4 次；实际抽样恶意数可超过 2。FLTrust 与 RL 共用可信 root。

已列 ClipMed 配置将更新 L2 范数裁至不超过 0.1。

### 6.4 缓存联合 RL

**RL-A：单类轨迹联合。** 合并三种单攻击轨迹的缓存状态，训练一个共享策略。

| 训练场景 | 恶意客户端的攻击方式 | 缓存状态数 |
| --- | --- | --- |
| IPM-5 | 4 个恶意客户端全部执行 IPM，scale=5 | 6 |
| LMP-Krum | 4 个恶意客户端全部执行以 Krum 为目标的 LMP | 6 |
| DBA simultaneous | 4 个恶意客户端同时执行 DBA 子触发器投毒 | 6 |
| Clean | 无攻击；两组共用相同的正常状态 | 18 |

**RL-B：双攻击轨迹联合。** 合并三种同轮双攻击轨迹的缓存状态，训练另一个共享策略。

| 训练组合 | 客户端 0、1 | 客户端 2、3 | 缓存状态数 |
| --- | --- | --- | --- |
| IPM+LMP | IPM-5 | LMP-Krum | 6 |
| IPM+DBA | IPM-5 | DBA simultaneous | 6 |
| LMP+DBA | LMP-Krum | DBA simultaneous | 6 |
| Clean | 正常训练 | 正常训练 | 18 |

BFL 家族及 DBA+BFL 组合未参与训练。

| 共同训练设置 | 记录值 |
| --- | --- |
| 数据 | 18 攻击＋18 clean 状态；攻击状态取第 26、101、176、276、376、476 轮 |
| 优化 | 10k 个缓存训练步；更新次数如下表 |
| 反馈 | 25% 即时＋75% 三轮终点效用及风险反馈；来自可信 root |
| FL | 本地／root 5 步、恶意本地 10 步、batch 128、SGD lr 0.05 |
| 观测 | 均为 Histogram40；两组分别拟合分箱和标准化参数 |

用动作 `[1,0,1]` 采集 FL 状态，每个状态测量 27 点动作网格；三轮反馈的后两轮沿用采集动作。策略优化复用缓存。

每个训练步以 1/2 概率选择攻击池或 clean 池，再均匀抽取状态，读取动作对应的网格插值反馈后终止；攻击池内三场景各占 1/3。

| 缓存训练步范围 | Critic 更新 | Actor 更新 |
| --- | ---: | ---: |
| 1–1,000 | 0 | 0 |
| 1,001–3,000 | 2,000 次 | 0 |
| 3,001–10,000 | 每步 1 次，共 7,000 次 | 每两次 Critic 更新 1 次，共 3,500 次 |
| 合计 | 9,000 次 | 3,500 次 |

训练 γ=0；评价时冻结 Actor 和 Critic。

### 6.5 Meta-SG 元训练与初始化

| 单位／设置 | 实际含义 |
| --- | --- |
| 1 个 FL 轮 | 客户端本地训练，服务器观察更新、执行一次防御动作并聚合 |
| 1 条轨迹（episode） | 从初始 FL 模型重新开始，最多运行 500 轮；轨迹内不更新攻防策略参数 |
| 1 个 Meta 外循环 | 将 RL、BRL 随机排列，各处理一次，再更新共同防御初始化一次；共 10 个外循环 |
| 任务内防御状态 | 每任务复制共同 Actor＋双 Critic；重新建立优化器和回放池 |
| 攻击者状态 | 从随机初始化开始；RL、BRL 各保留自己的网络、优化器和回放池，跨外循环继续训练 |
| TD3 参数 | lr=0.001，batch=256，γ=0.99，τ=0.005，policy delay=2 |

一次 Critic 更新同时优化双 Q；每两次常规 Critic 更新伴随一次 Actor 更新。

每条元训练轨迹仅含 RL 或 BRL；任务名称不输入 Defender。

**外循环更新。** 每个任务 ξ 按下表执行，每次采集均重置 FL 模型。

| 阶段 | 新采集轨迹数 | 防御更新 | 攻击者更新 |
| --- | ---: | --- | --- |
| 防御任务内适应 | 2 | 每条采完后做 1 次 TD3 更新：合计 Critic 2 次、Actor 1 次 | 参数冻结 |
| 攻击者响应 | 2 | Actor 参数冻结 | 每条采完后做 1 次 TD3 更新：合计 Critic 2 次、Actor 1 次 |
| 响应后评价 | 1 | 用新回放拟合 Critic 20 次，Actor 不更新；再计算 1 份 Actor 梯度 gξ | 参数冻结 |
| 两任务汇总 | 0 | 将两份 gξ 平均，对共同 Actor 做 1 次外层更新；两个任务的响应后 Critic 取均值 | 两套攻击者保留到下一轮 |

外层 Actor 更新为：

$$
g_\xi=\nabla_{\theta_\xi}\bigl[-\mathbb E_o Q_{1,\xi}(o,\pi_{\theta_\xi}(o))\bigr],\qquad
\theta^{t+1}=\theta^t-\frac{\eta_{out}}{2}(g_{RL}+g_{BRL}),\quad\eta_{out}=1.
$$

Actor 使用一阶响应后梯度，不反传任务内优化；响应后 Critic 参数取均值并同步目标网络。

每个外循环：10 条轨迹、最多 5,000 FL 轮；防御副本共 44 次 Critic、2 次内层 Actor 更新，共同 Actor 更新 1 次。10 个外循环上限为 50,000 FL 轮／初始化，数值终止按实际有效轮数计。

**初始化对照**

| 初始化 | 离线训练 | 部署阶段 |
| --- | --- | --- |
| Meta-SG | 两任务分别复制防御初始化，按上述外循环汇总梯度 | 使用 6.6 的相同支持数据和预算 |
| Joint | 同样处理 RL／BRL；防御网络、优化器和回放跨任务连续保留，无 Meta 外层更新 | 同上 |
| Random | 随机 Actor＋Critic；不做离线策略训练，仍使用共同训练校准得到的观测变换 | 同上 |

Joint 在真实 FL 的 RL／BRL 任务上连续预训练；RL-A／RL-B 使用固定攻击缓存。

### 6.6 Meta-SG 部署适应

1. 每个测试场景先用独立的均匀随机动作采集 20 轮支持数据；S=5 取其前 5 轮。Meta、Joint、Random 及两种表示共用同一物理支持轨迹。
2. 加载各自 Actor＋双 Critic，重置优化器／回放池；收齐 S 条转移后做 C 次 TD3 更新，其中常规 Actor 更新 A=C/2 次；再冻结 Critic，追加 E=C/2 次 Actor 更新，最大化平均双 Q。
3. 适应后重置 FL 模型，冻结攻防策略独立查询 500 轮。

| 预算 | 支持 FL 轮 S | Critic 更新 C | 常规 Actor A | 追加 Actor E | Actor 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0：冻结 | 0 | 0 | 0 | 0 | 0 |
| B1 | 5 | 50 | 25 | 25 | 50 |
| B2 | 20 | 10 | 5 | 5 | 10 |
| B3 | 20 | 50 | 25 | 25 | 50 |
| B4 | 20 | 100 | 50 | 50 | 100 |

部署 lr=0.001、batch=256，重复抽样支持回放；反馈仅来自 root／风险，不使用攻击标签、恶意身份、触发器或测试标签。B1／B3 比较数据量，B2／B3／B4 比较更新次数；Clean 仅测 B0／B3。

## 7. MNIST／MNISTCNN

### 7.1 数据集与评价协议

MNISTCNN，Normalize(0.1307, 0.3081)，batch 128；可信 root 共 200 张、每类 20 张，评价 seed=43。

- **历史全攻击实验**：20 客户端全参与、4 恶意；IID／Non-IID（Dirichlet α=0.5）；200 轮，统计第 191–200 轮均值。本地训练 5 步、lr=0.05，训练 seed=42。
- **RL Attacker 长程实验**：IID；100 客户端中固定 20 个恶意身份，每轮均匀无放回抽 10 个；500 轮，统计第 491–500 轮均值。正常客户端每轮 SGD 1 步、root 5 步；客户端／root 学习率分为 0.01、0.05 两组。

### 7.2 实验结果

**历史全攻击结果：第 191–200 轮均值，后门项为 ACC／ASR。** RL 列中，IPM／LMP／DBA／BFL 为旧版三动作；RL Attacker 的 IID 为 mixed-norm，Non-IID 为标准化 Compact8。

| 划分 | 攻击 | FedAvg | Median | Trimmed Mean | Multi-Krum | RFA | FLTrust | RL |
| ------- | ---------------- | --------------: | -------------: | --------------: | -------------: | --------------: | -------------: | -------------: |
| IID | IPM-5 | 数值停止@33 | 96.883 | 96.296 | 98.149 | 96.588 | 95.243 | 98.214 |
| IID | LMP-Krum | 97.632 | 97.005 | 97.292 | 97.432 | 97.461 | 97.845 | 97.625 |
| IID | RL Attacker | 数值停止@68 | 97.333 | 96.665 | 98.365 | 97.352 | 97.513 | 98.579 |
| IID | DBA simultaneous | 95.362 / 99.847 | 98.505 / 0.000 | 98.541 / 0.102 | 98.371 / 0.000 | 98.454 / 0.337 | 98.361 / 0.112 | 98.353 / 0.000 |
| IID | DBA cyclic | 95.831 / 66.684 | 98.331 / 0.000 | 98.353 / 0.000 | 98.352 / 0.000 | 98.372 / 0.000 | 98.085 / 3.561 | 98.384 / 0.000 |
| IID | BFL | 98.479 / 99.133 | 98.415 / 2.673 | 98.444 / 11.296 | 98.368 / 0.000 | 98.414 / 93.367 | 98.366 / 5.633 | 98.376 / 0.000 |
| Non-IID | IPM-5 | 数值停止@16 | 97.250 | 94.831 | 96.273 | 97.447 | 97.108 | 97.640 |
| Non-IID | LMP-Krum | 97.082 | 96.401 | 96.612 | 构造失败@111 | 96.880 | 97.704 | 97.038 |
| Non-IID | RL Attacker | 数值停止@10 | 95.244 | 93.973 | 98.018 | 96.077 | 97.438 | 97.999 |
| Non-IID | DBA simultaneous | 93.720 / 99.796 | 98.099 / 0.633 | 98.158 / 0.745 | 98.013 / 0.000 | 98.190 / 0.939 | 98.025 / 0.235 | 97.998 / 0.000 |
| Non-IID | DBA cyclic | 93.808 / 83.071 | 98.072 / 0.051 | 98.119 / 0.143 | 98.120 / 0.010 | 98.153 / 0.184 | 98.025 / 0.143 | 97.951 / 0.000 |
| Non-IID | BFL | 98.220 / 99.520 | 98.002 / 3.704 | 98.066 / 12.408 | 98.119 / 0.082 | 98.158 / 97.265 | 98.033 / 0.571 | 98.126 / 0.276 |

本组多数有效防御的 ACC 接近 97%–99%；IID BFL 的 ASR 为 RL 0%、RFA 93.367%。

[历史逐项数据](outputs/single_attack_rl_20261107/mnist_iid_results.csv) · [Non-IID 数据](outputs/single_attack_rl_20261107/mnist_noniid_results.csv)

**RL Attacker 长程结果：旧 Compact8／mixed-norm，100 客户端抽 10，500 轮。** 两个子图分别对应 lr=0.01、0.05。RL 的末 10 轮 ACC 分别为 94.838%、98.156%；FLTrust 为 95.257%、96.956%。

![[outputs/meta_sg_review_20261107/mnist_sampled100_bars_review_v1.png|1400]]

[高清 PNG](outputs/meta_sg_review_20261107/mnist_sampled100_bars_review_v1.png) · [PDF](outputs/meta_sg_review_20261107/mnist_sampled100_bars_review_v1.pdf) · [SVG](outputs/meta_sg_review_20261107/mnist_sampled100_bars_review_v1.svg)

**0–500 轮 ACC 轨迹**

![[outputs/meta_sg_review_20261107/mnist_sampled100_curves_review_v1.png|1400]]

[曲线 PDF](outputs/meta_sg_review_20261107/mnist_sampled100_curves_review_v1.pdf) · [长程逐方法数据](outputs/mnist_sampled100_long_20261107_v2/mnist_sampled100_long_results.csv)

## 8. CIFAR-10／CIFAR10CNN

### 8.1 数据集与评价协议

CIFAR-10：10 类、32×32 RGB 图像，各通道 Normalize(0.5, 0.5)。模型为 CIFAR10CNN：卷积通道 32→64→64，前两层后接最大池化，全连接层 4096→128→10，ReLU，无 BatchNorm。

8.2／8.3 均采用 IID、SGD、lr=0.05、batch=128，评价 500 轮，seed=43。可信 root 每类 20 张、共 200 张，每轮更新 5 steps。

| 已有结果 | 观测 | 客户端参与 | 正常客户端每轮更新 | 固定恶意身份 | 统计窗口 |
| --- | --- | --- | --- | --- | --- |
| 冻结迁移（8.2） | 旧 Compact8 | 100 个客户端，每轮无放回抽 10 个 | 1 step | IPM／LMP 各 20 个；DBA／BFL 各 5 个 | 498–500 轮 |
| 表示对照（8.3） | Compact8、Histogram40 | 20 个客户端全部参与 | 5 steps | 各攻击 4 个 | 491–500 轮 |

### 8.2 Compact8 专用策略：100 抽 10 冻结迁移

第 **498–500 轮均值**。

| 防御方法 | IPM ACC↑ | LMP ACC↑ | DBA ACC↑ | DBA ASR↓ | BFL ACC↑ | BFL ASR↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 数值失败@37 | 36.267 | 57.210 | 96.533 | 56.203 | 91.700 |
| Median | 27.917 | 37.213 | 53.293 | 19.467 | 53.040 | 28.767 |
| Trimmed Mean | 26.100 | 32.843 | 53.823 | 25.533 | 53.600 | 36.100 |
| Krum | 36.203 | 29.387† | 50.697 | **4.867** | 50.320 | 6.000 |
| Multi-Krum | 22.887 | 24.393 | 54.367 | 9.700 | 52.280 | 6.200 |
| RFA | 33.553 | 33.653 | 53.137 | 27.267 | 52.870 | 52.733 |
| FLTrust | **56.127** | **54.157** | **57.553** | 25.100 | **56.463** | 60.233 |
| ClipMed | 43.090 | 44.573 | 50.350 | 16.167 | 50.293 | 24.233 |
| RL transfer | 53.710 | 53.277 | 52.187 | 5.733 | 55.353 | **3.500** |

† LMP/Krum 为 2026-11-07、seed=43 的完整 500 轮重跑；允许 10 次额外搜索重试，实际 0 次。其余方法沿用历史结果。

[历史 Compact8 图](outputs/meta_sg_review_20261107/cifar_sampled100_bars_review_v1.png)

[PDF](outputs/meta_sg_review_20261107/cifar_sampled100_bars_review_v1.pdf) · [SVG](outputs/meta_sg_review_20261107/cifar_sampled100_bars_review_v1.svg)

历史表中，IPM／LMP 的最高 ACC 均为 FLTrust；DBA 的最低 ASR 为 Krum；BFL 的最低 ASR 为 RL。

#### 8.2.1 确定性补跑：Compact8／Histogram40（attack_start=3）

复用既有单攻击专用策略，未重训；按 8.1 的 100 抽 10 协议冻结评价。本表为同批确定性执行的第 498–500 轮均值，单位 %。

旧评估器重复 5 轮存在数值差异，因此补跑 Compact8；新评估器与未改动原评估器在相同确定性设置下，四种攻击的 5 轮 ACC／ASR／动作逐项一致。旧表保留为历史参考。

| 攻击 | Compact8 ACC↑ | Histogram40 ACC↑ | Compact8 ASR↓ | Histogram40 ASR↓ |
| --- | ---: | ---: | ---: | ---: |
| IPM-5 | 55.630 | 55.740 | — | — |
| LMP-Krum | 51.640 | 55.547 | — | — |
| DBA simultaneous | 52.053 | 56.850 | 5.333 | 4.133 |
| BFL | 55.280 | 53.820 | 5.233 | 4.800 |

![[outputs/frozen100_todo_20261108/start3_frozen100_all_defenses_v1.png|1600]]

[PDF](outputs/frozen100_todo_20261108/start3_frozen100_all_defenses_v1.pdf) · [SVG](outputs/frozen100_todo_20261108/start3_frozen100_all_defenses_v1.svg)

灰色柱为历史常规防御参考；两种 RL 为本次同批结果，Compact8 用斜线、Histogram40 用实色。

![[outputs/frozen100_todo_20261108/frozen100_start3_fl_curves_v1.png|1600]]

[PDF](outputs/frozen100_todo_20261108/frozen100_start3_fl_curves_v1.pdf) · [SVG](outputs/frozen100_todo_20261108/frozen100_start3_fl_curves_v1.svg)

上排为四种攻击的完整 ACC；下排前两格为 IPM／LMP 的 100–500 轮 ACC 放大，后两格为 DBA／BFL ASR。策略全程冻结，曲线未平滑。

[同批逐项结果](outputs/frozen100_todo_20261108/start3_results.csv) · [完整协议与来源](outputs/frozen100_todo_20261108/report.md)

### 8.3 Compact8／Histogram40 表示对照：20 全参与

第 **491–500 轮均值**。

| 攻击 | Compact8 ACC↑ | Histogram40 ACC↑ | Compact8 ASR↓ | Histogram40 ASR↓ |
| --- | ---: | ---: | ---: | ---: |
| IPM-5 | 69.435 | 69.910 | — | — |
| LMP-Krum | 68.654 | 68.976 | — | — |
| DBA simultaneous | 69.469 | 69.929 | 3.190 | 3.220 |
| BFL | 69.684 | 69.754 | 3.300 | 3.630 |

Histogram40 在四种攻击下的 ACC 均高于该组 Compact8；DBA／BFL 的 ASR 也略高。

### 8.4 混合攻击联合 RL：单类轨迹与双攻击轨迹

**CIFAR10CNN、IID、20 客户端全参与、4 恶意；Histogram40；训练／评价 seed 42／43。** RL-A、RL-B 的训练配置见 6.4。

#### 8.4.1 IPM+LMP 与 Clean：ACC

以下为 **500 FL 轮的第 491–500 轮均值**；Value 按未舍入均值计算。

| 防御方式         | IPM+LMP ACC↑ | Clean ACC↑ |
| ------------ | ------------ | ---------- |
| FLTrust      | 68.579       | 70.037     |
| Median       | 58.861       | 70.569     |
| Trimmed Mean | 60.491       | 70.420     |
| Krum         | 攻击构造失败@267   | 62.342     |
| RL-A：单类轨迹联合  | 69.324       | 71.070     |
| RL-B：双攻击轨迹联合 | 68.783       | 70.827     |

#### 8.4.2 含后门的混合攻击：ACC、ASR、Value

##### 8.4.2.1 IPM + DBA

| 防御方式         | ACC↑   | ASR↓   | Value↑  |
| ------------ | ------ | ------ | ------- |
| FLTrust      | 69.099 | 93.610 | -24.511 |
| Median       | 67.986 | 87.130 | -19.144 |
| Trimmed Mean | 67.239 | 92.950 | -25.711 |
| Krum         | 62.234 | 3.580  | 58.654  |
| RL-A：单类轨迹联合  | 69.987 | 3.270  | 66.717  |
| RL-B：双攻击轨迹联合 | 69.318 | 2.950  | 66.368  |

##### 8.4.2.2 LMP + DBA

| 防御方式         | ACC↑       | ASR↓   | Value↑  |
| ------------ | ---------- | ------ | ------- |
| FLTrust      | 69.173     | 93.070 | -23.897 |
| Median       | 68.001     | 84.640 | -16.639 |
| Trimmed Mean | 68.958     | 88.910 | -19.952 |
| Krum         | 攻击构造失败@262 | —      | —       |
| RL-A：单类轨迹联合  | 69.425     | 3.120  | 66.305  |
| RL-B：双攻击轨迹联合 | 68.664     | 3.170  | 65.494  |

##### 8.4.2.3 DBA + BFL (held out)

| 防御方式         | ACC↑   | ASR↓   | Value↑  |
| ------------ | ------ | ------ | ------- |
| FLTrust      | 69.499 | 96.610 | -27.111 |
| Median       | 71.438 | 97.590 | -26.152 |
| Trimmed Mean | 71.283 | 97.780 | -26.497 |
| Krum         | 62.535 | 4.370  | 58.165  |
| RL-A：单类轨迹联合  | 69.545 | 3.360  | 66.185  |
| RL-B：双攻击轨迹联合 | 69.473 | 3.010  | 66.463  |

DBA+BFL 为留出测试；Krum 失败轮次见表。

**RL-B 相对 RL-A：** 四种攻击组合的 ACC 均下降；IPM+DBA、LMP+DBA、DBA+BFL 的 Value 分别变化 −0.349、−0.811、＋0.278 pp。双攻击轨迹联合训练没有一致优于单类轨迹联合训练。

#### 8.4.3 最终结果柱形图

**ACC↑**

![[outputs/mixed_attack_rl_display_20261107/mixed_joint_acc_bars_v1.png|1400]]

[PDF](outputs/mixed_attack_rl_display_20261107/mixed_joint_acc_bars_v1.pdf) · [SVG](outputs/mixed_attack_rl_display_20261107/mixed_joint_acc_bars_v1.svg)

**ASR↓**

![[outputs/mixed_attack_rl_display_20261107/mixed_joint_asr_bars_v1.png|1400]]

[PDF](outputs/mixed_attack_rl_display_20261107/mixed_joint_asr_bars_v1.pdf) · [SVG](outputs/mixed_attack_rl_display_20261107/mixed_joint_asr_bars_v1.svg)

**Value↑**

![[outputs/mixed_attack_rl_display_20261107/mixed_joint_value_bars_v1.png|1400]]

[PDF](outputs/mixed_attack_rl_display_20261107/mixed_joint_value_bars_v1.pdf) · [SVG](outputs/mixed_attack_rl_display_20261107/mixed_joint_value_bars_v1.svg)

#### 8.4.4 FL 曲线与数据

![[outputs/mixed_attack_rl_display_20261107/mixed_joint_fl_curves_v2.png|1400]]

三列为 IPM+DBA、LMP+DBA、DBA+BFL；各行依次为全程 ACC、350–500 轮 ACC、全程 ASR。

[曲线 PDF](outputs/mixed_attack_rl_display_20261107/mixed_joint_fl_curves_v2.pdf) · [全部 30 条展示结果](outputs/mixed_attack_rl_display_20261107/displayed_30_results.csv) · [逐柱数据](outputs/mixed_attack_rl_display_20261107/plotted_bars.csv) · [核验记录](outputs/mixed_attack_rl_display_20261107/audit.json)

30 条轨迹中 28 条完成 500 轮，均为单 seed。

### 8.5 首轮投毒：Compact8／Histogram40（attack_start=1）

其余配置与本次 attack_start=3 对照相同；从第 1 轮允许攻击，当轮未抽到恶意客户端时无投毒。两种表示、四种攻击均冻结评价 500 轮，seed=43；统计第 498–500 轮均值，单位 %。

| 攻击 | Compact8 ACC↑ | Histogram40 ACC↑ | Compact8 ASR↓ | Histogram40 ASR↓ |
| --- | ---: | ---: | ---: | ---: |
| IPM-5 | 55.917 | 56.877 | — | — |
| LMP-Krum | 53.593 | 55.077 | — | — |
| DBA simultaneous | 52.163 | 55.487 | 6.200 | 3.733 |
| BFL | 56.143 | 56.797 | 5.733 | 3.933 |

![[outputs/frozen100_todo_20261108/start1_frozen100_compact8_histogram40_v1.png|1600]]


## 9. Tiny-ImageNet／ResNet18

### 9.1 数据集与评价协议

Tiny-ImageNet，200 类、64×64 RGB 图像；标准 torchvision ResNet18，保留 BatchNorm，分类头为 200 类，无预训练。

| 配置 | 实际设置 |
| --- | --- |
| 客户端 | IID，100 个客户端，每轮抽 10 个，固定 20 个恶意身份 |
| 本地训练 | SGD，每客户端每轮 5 epochs／50 steps，batch=100，momentum=0，weight decay=0.001 |
| 学习率 | 第 r 轮为 `0.1 × 0.998^(r−1)` |
| 可信 root | 每类 2 张，共 400 张，每轮更新 5 steps |
| FL 预算与评价 seed | 200 轮，seed=43 |
| 统计窗口 | 第 198–200 轮均值 |

### 9.2 IPM／LMP 结果

第 **198–200 轮均值**。IPM 固定防御与 Median 补充项来自 2026-10-02 批次；IPM 的 RL 专用策略尚无完整评价结果。

| 攻击 | FedAvg | Median | Trimmed Mean | FLTrust | FLTrust+NC | ClipMed | RL specialist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IPM-5 | 0.483 | 25.463 | 16.260 | 27.170 | 27.170 | 4.457 | — |
| LMP | 25.720 | 25.390 | 26.483 | 27.090 | 27.090 | 5.697 | 27.147 |

IPM 中 FLTrust／FLTrust+NC 的末期 ACC 最高；LMP 中 RL specialist 最高。

### 9.3 DBA／BFL v2 结果

第 **198–200 轮均值**；RL specialist 按对应投毒起点训练。DBA 固定防御和 BFL 首轮投毒的 Median 使用 2026-10-02 补充批次。

| 防御方法 | DBA ACC↑ | DBA ASR↓ | BFL 第101轮 ACC↑ | BFL 第101轮 ASR↓ | BFL 第1轮 ACC↑ | BFL 第1轮 ASR↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 31.640 | 90.399 | 27.230 | 96.576 | 28.747 | 95.742 |
| Median | 31.373 | 79.762 | — | — | 32.103 | 92.992 |
| Trimmed Mean | 31.213 | 83.447 | 31.393 | 94.245 | 31.720 | 94.653 |
| FLTrust | 27.597 | 90.409 | 27.787 | 92.700 | 28.227 | 93.229 |
| FLTrust+NC | 27.597 | 90.409 | 27.787 | 92.700 | 28.227 | 93.229 |
| ClipMed | 9.473 | 11.866 | 9.643 | 32.034 | 9.563 | 68.308 |
| RL specialist | 28.950 | 0.074 | 28.957 | 0.094 | 30.130 | 0.094 |

BFL 首轮投毒下，训练起点为第 101 轮的旧冻结 RL 的 ACC 为 27.837%，ASR 为 0.087%。— 表示缺少对应方法记录。

### 9.4 无攻击参考

| 无攻击评价 | FL 轮数 | root 张数 | 统计窗口 | ACC↑（%） |
| --- | ---: | ---: | --- | ---: |
| FedAvg，无 root | 1000 | 0 | 998–1000 | 32.387 |
| FedAvg，LMP 配对无攻击 | 200 | 400 | 198–200 | 32.030 |
| DBA RL，无攻击 | 200 | 400 | 198–200 | 29.637 |
| BFL 第101轮策略，无攻击 | 200 | 400 | 198–200 | 29.860 |
| BFL 第1轮策略，无攻击 | 200 | 400 | 198–200 | 30.487 |

旧版黑色触发器的 FedAvg ASR 为 1.333%，详见[[outputs/meta_sg_review_20261107/legacy_bfl_diagnostic|历史诊断]]。

[本组原始结果与来源核验](outputs/resnet18_document_results_20261107_v2/audit.json)

## 10. CIFAR-10：Meta-SG 结果

CIFAR10CNN、IID、20 客户端全参与、4 恶意；本地 SGD 5 步、batch 128、lr=0.05；训练／查询 seed=42／43。

测试场景为 5.3 的四种固定组合及 Clean；元训练仅使用 RL／BRL，属于跨攻击实现与组合的迁移。

本节报告共享支持数据的 **132 条配对结果**。表格为独立 500 轮查询的 **491–500 轮均值**，后门格为 ACC % / ASR %；IPM+LMP、Clean 只报 ACC。表头 B1 等处的“5/50”表示 S/C；末 50 轮及早期 84 条结果保留在[完整数据](output/meta_learning_results_20261107/shared_132.csv)、[历史 84 条](output/meta_learning_results_20261107/historical_84.csv)中。

| 配置 | 支持数据 $S$（FL 轮） | Critic 更新 $C$（次） | Actor 总更新（次） |
| :--- | ---: | ---: | ---: |
| B0：冻结 | 0 | 0 | 0 |
| B1 | 5 | 50 | 50 |
| B2 | 20 | 10 | 10 |
| B3 | 20 | 50 | 50 |
| B4 | 20 | 100 | 100 |

### 10.1 Histogram40：全部预算结果

| 测试场景    | 初始化     | B0：冻结           | B1：5/50         | B2：20/10        | B3：20/50        | B4：20/100       |
| ------- | ------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| IPM+DBA | Meta-RL | 70.316 / 76.900 | 69.925 / 20.100 | 69.828 / 26.670 | 69.376 / 8.940  | 69.332 / 7.520  |
| IPM+DBA | Joint   | 70.486 / 96.520 | 70.671 / 89.310 | 70.844 / 91.130 | 68.126 / 21.660 | 67.970 / 16.360 |
| IPM+DBA | Random  | 68.603 / 97.830 | 69.172 / 98.040 | 68.805 / 97.940 | 70.730 / 92.720 | 70.709 / 92.710 |
| LMP+DBA | Meta-RL | 69.978 / 83.950 | 70.183 / 33.990 | 69.476 / 41.320 | 68.712 / 13.470 | 68.621 / 12.370 |
| LMP+DBA | Joint   | 70.463 / 96.100 | 70.699 / 92.600 | 70.629 / 93.880 | 70.049 / 78.890 | 70.562 / 82.350 |
| LMP+DBA | Random  | 69.576 / 97.590 | 70.043 / 96.310 | 69.603 / 97.580 | 69.562 / 97.410 | 70.001 / 96.260 |
| IPM+LMP | Meta-RL | 68.523          | 69.640          | 68.515          | 68.108          | 68.185          |
| IPM+LMP | Joint   | 67.588          | 68.897          | 68.486          | 69.533          | 69.874          |
| IPM+LMP | Random  | 63.982          | 65.790          | 65.095          | 69.461          | 69.734          |
| DBA+BFL | Meta-RL | 70.630 / 91.290 | 70.074 / 64.780 | 70.142 / 73.670 | 69.729 / 34.690 | 69.690 / 32.330 |
| DBA+BFL | Joint   | 70.899 / 97.410 | 70.788 / 95.250 | 70.873 / 95.890 | 68.495 / 76.100 | 67.900 / 71.340 |
| DBA+BFL | Random  | 70.056 / 98.060 | 70.546 / 97.660 | 69.971 / 97.850 | 70.676 / 95.360 | 70.283 / 93.490 |

| Clean：初始化 | 冻结 ACC | B3 ACC |
| --- | --- | --- |
| Meta-SG | 71.127 | 70.560 |
| Joint | 71.355 | 71.376 |
| Random | 70.934 | 71.152 |

### 10.2 Compact8：全部预算结果

| 测试场景 | 初始化 | B0：冻结 | B1：5/50 | B2：20/10 | B3：20/50 | B4：20/100 |
| ------- | ------- | --------------- | --------------- | --------------- | ------------------- | --------------- |
| IPM+DBA | Meta-SG | 70.633 / 91.350 | 70.294 / 83.830 | 70.335 / 85.580 | **69.698 / 38.160** | 69.539 / 19.450 |
| IPM+DBA | Joint | 70.558 / 96.550 | 70.406 / 95.070 | 70.543 / 95.670 | **70.218 / 83.230** | 69.934 / 31.400 |
| IPM+DBA | Random | 69.267 / 97.810 | 68.888 / 97.910 | 69.661 / 97.840 | **70.542 / 96.330** | 70.683 / 90.840 |
| LMP+DBA | Meta-SG | 69.916 / 88.730 | 69.234 / 63.270 | 69.463 / 74.630 | 68.579 / 14.800 | 68.416 / 6.450 |
| LMP+DBA | Joint | 70.572 / 95.780 | 70.186 / 95.310 | 70.582 / 95.110 | 70.005 / 88.100 | 69.121 / 19.870 |
| LMP+DBA | Random | 69.992 / 97.420 | 69.457 / 97.140 | 69.971 / 97.370 | 69.419 / 98.020 | 68.890 / 98.170 |
| IPM+LMP | Meta-SG | 68.535 | 68.284 | 68.355 | 68.180 | 68.105 |
| IPM+LMP | Joint | 68.041 | 67.203 | 68.042 | 67.242 | 68.392 |
| IPM+LMP | Random | 65.143 | 63.559 | 65.088 | 58.859 | 58.511 |
| DBA+BFL | Meta-SG | 70.658 / 93.880 | 70.259 / 87.100 | 70.418 / 89.110 | 69.769 / 46.060 | 69.574 / 21.220 |
| DBA+BFL | Joint | 71.018 / 97.090 | 71.048 / 96.290 | 71.122 / 96.410 | 70.764 / 95.320 | 70.279 / 85.760 |
| DBA+BFL | Random | 70.104 / 97.800 | 70.365 / 97.890 | 70.185 / 97.830 | 70.331 / 97.570 | 70.879 / 96.880 |

| Clean：初始化 | 冻结 ACC | B3 ACC |
| --- | --- | --- |
| Meta-SG | 70.884 | 70.515 |
| Joint | 71.320 | 71.039 |
| Random | 71.083 | 70.796 |

### 10.3 适应后 FL 曲线与预算曲线

![[output/meta_learning_results_20261107/meta_fl_queries_v1.png|1100]]

Histogram40 的冻结／B3 适应后 FL 曲线。

![[output/meta_learning_results_20261107/meta_samples_v1.png|1100]]

固定 C=50，比较 S=5 与 S=20；零点为冻结参考。

![[output/meta_learning_results_20261107/meta_updates_v1.png|1100]]

固定 S=20，比较 C=10／50／100；零点为冻结参考。实线为 Histogram40，虚线为 Compact8。

### 10.4 结果要点

- **后门抑制**：B3 的三个后门组合中，Histogram40 Meta 的 ASR 均低于 Joint／Random；IPM+DBA 相对 Joint 的 ACC 高 1.250 pp、ASR 低 12.720 pp。
- **ACC 代价**：LMP+DBA 相对 Joint，Meta 的 ASR 低 65.420 pp，但 ACC 低 1.337 pp；IPM+LMP、Clean 的 B3 ACC 也低于 Joint／Random。
- **表示取舍**：B4 的 LMP+DBA、DBA+BFL，Compact8 Meta 的 ASR 为 6.450、21.220，低于 Histogram40 的 12.370、32.330。

待补同接口、同部署预算的专用 RL 对照及从零训练达标曲线。

### 10.9 IPM+LMP：适应更新诊断（2026-11-08）

复用 Histogram40 Meta 第 10 轮 checkpoint；除冻结项外，共用原有 20 轮支持数据、root／risk 奖励和观测变换。四个新方案先通过 5 轮检查，再独立冻结查询 500 轮；冻结与原 B3 沿用已核验记录。下表为 491–500 轮均值，seed=43。未新增元训练。

| 方案 | Critic 更新 | 常规 Actor | 额外 Actor | ACC↑（%） | Δ原 B3（pp） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结初始化 | 0 | 0 | 0 | 68.523 | +0.415 |
| 原 B3 适应 | 50 | 25 | 25 | 68.108 | +0.000 |
| 去掉额外 Actor | 50 | 25 | 0 | 68.120 | +0.012 |
| 重置 Critic＋去掉额外 Actor | 50 | 25 | 0 | 69.649 | +1.541 |
| 重置 Critic＋预热 100 次＋去掉额外 Actor | 150 | 25 | 0 | 69.642 | +1.534 |
| policy delay=10＋去掉额外 Actor | 50 | 5 | 0 | 68.523 | +0.415 |

重置仅替换 Critic／目标 Critic，保留预训练 Actor。预热组的 100 次 Critic-only 更新已计入总数 150；policy delay=10 同时降低 Actor 更新与目标网络同步频率。

重置 Critic 后，相对同为 Critic 50／Actor 25 的不重置组，ACC 高 1.529 pp；相对冻结初始化高 1.126 pp。额外预热 100 次后为 69.642%，与不预热的 69.649% 相近。原 B1 为 69.640%，本次修复后也处于这一水平。

同一支持观测与初始 Actor 动作上，继承 Critic 拟合后对 norm_mix 的平均 Q1 梯度为 −0.06319，重置后为 ＋0.03144，优化方向相反。这支持检查 Critic 迁移；不证明奖励已最优。结果仅为单 seed 的 IPM+LMP 诊断，Joint／Random 尚未按新更新规则重测。

[完整结果与配置](outputs/ipm_lmp_adaptation_repair_20261108/report.md) · [逐项数据](outputs/ipm_lmp_adaptation_repair_20261108/comparison.csv) · [核验记录](outputs/ipm_lmp_adaptation_repair_20261108/verification.json)