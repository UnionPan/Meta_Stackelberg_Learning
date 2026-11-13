# Meta-SG

**[Quickstart：运行 MNIST Meta-SG Demo](QUICKSTART.md)** · [研究与历史实验文档](Meta-SG.md) · [图片与数据归档](Outputs/README.md)

本目录用于逐步整理 Meta-SG 开源代码。目前包含三个数据集、公共数据划分、联邦训练与评估、攻击与固定防御、可插拔 observation／action／reward 的 RL Defender、一阶 Meta-SG 元训练，以及 JSON 入口和实验保存恢复。

## 开发环境

需要 Python 3.11 和 uv。在本目录执行：

```bash
uv sync --group dev
uv run pytest
```

完整 Demo 使用 `uv sync --locked --group dev --extra rl --extra plots`；见 [Quickstart](QUICKSTART.md)。

## 数据集

| 配置名称 | 模块 | 输入 | 分类数 |
| --- | --- | --- | --- |
| `mnist` | [MNIST](src/meta_sg/datasets/mnist/README.md) | `[B, 1, 28, 28]` | 10 |
| `cifar10` | [CIFAR-10](src/meta_sg/datasets/cifar10/README.md) | `[B, 3, 32, 32]` | 10 |
| `tiny_imagenet` | [Tiny ImageNet](src/meta_sg/datasets/tiny_imagenet/README.md) | `[B, 3, 64, 64]` | 200 |

三个模块各自管理加载、归一化、模型和触发器，统一输出图像与标签，复用公共划分与训练代码。
CIFAR-10 和 Tiny ImageNet 均支持 CNN 与 ResNet-18；Tiny 使用有标签的 validation 集评估。
CIFAR-10 的 ResNet-18 使用 3×3、步长 1 的开头并移除首个最大池化；Tiny 使用标准 7×7、步长 2 的开头与最大池化。两者均保留 BatchNorm，从随机初始化开始。

[MNIST 模块](src/meta_sg/datasets/mnist/README.md) 集中数据处理、模型代码、架构图与数据集介绍。
其中 `model.py` 提供 `MNISTCNN`，保持旧实现的三层卷积结构与初始化方式。
输入为 `[batch, 1, 28, 28]`，输出为 `[batch, 10]` 的 logits，可直接交给交叉熵损失。

```python
import torch

from meta_sg.datasets.mnist import MNISTCNN

model = MNISTCNN()
logits = model(torch.randn(4, 1, 28, 28))  # Shape: [4, 10]
```

## 数据处理

[数据模块](src/meta_sg/data/README.md) 提供跨数据集共享的均衡可信根数据抽样，以及 IID 和 Dirichlet 非 IID 客户端划分。
Dirichlet 划分通过 `alpha` 控制类别偏斜，返回与 IID 相同的客户端索引列表。
各数据集的加载函数返回 `(train_dataset, test_dataset)`；单样本为 `(image, label)`。
DataLoader 输出 `float32 [B, C, H, W]` 图像与 `int64 [B]` 标签。
划分函数返回索引，直接配合 PyTorch `Subset` 和 `DataLoader` 使用。

## 配置与入口

[入口说明](src/meta_sg/experiments/README.md) 列出配置字段与各攻击、防御的参数。
三份示例分别位于 [MNIST](configs/mnist.json)、[CIFAR-10](configs/cifar10.json) 和 [Tiny ImageNet](configs/tiny_imagenet.json)。
ResNet 示例位于 [CIFAR-10 ResNet-18](configs/cifar10_resnet18.json) 和 [Tiny ImageNet ResNet-18](configs/tiny_imagenet_resnet18.json)。
在本目录执行配置检查：

```bash
uv run meta-sg --config configs/mnist.json --dry-run
```

修改 JSON 中的数据集、`partition.method`（`iid` 或 `dirichlet`）、`alpha`、攻击、防御和学习率等训练参数即可切换配置。
训练支持 `local_steps` 或 `local_epochs`，以及 SGD 的 `momentum`、`weight_decay` 和按轮乘法衰减 `learning_rate_decay`。设置 `local_epochs` 时优先执行完整 epoch，保留最后不足一批的数据；root 始终按 `root_steps` 训练。
`model.name` 选择 `cnn` 或 `resnet18`；省略时保持 `cnn`。Tiny 的 ResNet 示例显式启用 `tiny` 归一化与数据增强，模型选择本身不改写预处理。
`--dry-run` 输出合并后的配置与模型选择，不加载数据或启动训练。
示例数据路径为相对当前工作目录的 `../data`；Tiny ImageNet 需提供官方 zip。
训练在开发机执行，移除 `--dry-run` 后入口逐轮输出 JSON Lines，包含 ACC、损失、攻击与聚合诊断以及适用时的 ASR。
示例配置会在 `output.directory` 保存实验。每轮写入指标，默认每 50 轮及初始／结束时保存 `latest.pt`。新实验目录须为空；使用 `--output` 指定另一个目录。
后门的源／目标类别、投毒比例、样本选择、更新缩放和恶意客户端训练预算均可配置。完整示例见 [Tiny BFL 配置](configs/tiny_imagenet_bfl.json)，沿用 `bfl` 方法，无需增加攻击类型。

## 实验保存与恢复

保存配置、实际数据划分、数据指纹、逐轮 JSON 指标、状态与 checkpoint。恢复包含模型与 BatchNorm、轮次、随机状态和采样进度；RL 实验还包含策略、优化器、目标网络和 replay。

```bash
uv run meta-sg --resume outputs/mnist/latest.pt
```

从最后保存的完整通信轮继续；该轮之后的指标会回退后重算。详见 [保存与恢复](src/meta_sg/experiments/README.md#saving-and-resuming)。

## 可学习防御与 RL Defender

[RL 模块](src/meta_sg/rl/README.md) 提供可配置动作的 TD3 Defender。`rl.mode` 选择训练、冻结推理或固定动作参照；每份实验配置对应一个攻击任务、一份独立策略。
[MNIST RL 配置](configs/mnist_rl.json) 展示完整参数。安装可选依赖：`uv sync --group dev --extra rl`。

`rl.observation` 可选 compact8、分位数、统计量、直方图；`rl.action` 可选 mixed norm、median clipping、trusted norm、四维 CNB；`rl.reward` 可选 root 损失改善、效用差、绝对成本。三个组件独立注册，通过名称和参数选择，无需修改训练循环。
原版 Histogram40 已接入；训练期拟合、旧 JSON 加载、保存恢复见 [Histogram40 说明](src/meta_sg/rl/DISTRIBUTIONS.md)。[Histogram40 示例](configs/mnist_rl_histogram40.json) 与原来的简化 `histogram` 组件分别命名。
动作的 `controls` 选择学习的控制量及顺序，`fixed` 指定其余控制量；策略、回放、保存与恢复自动使用对应维度。默认仍为 risk／mix／strength 三动作。
默认输入与奖励只使用客户端更新和可信 root 数据；测试 ACC／ASR 独立评估。旧离线训练与历史校准分支尚未全部迁入。

## Meta-SG 元训练

[元训练模块](src/meta_sg/rl/META.md) 接入 TD3 一阶 Meta-SG：任务内 Defender 更新、固定 Defender 的独立 follower 响应、响应后 critic 拟合，以及共享 actor 初始化的外层梯度更新。模型投毒和后门 follower 分别保留策略；Defender 复用现有 Histogram40、动作和奖励组件。

[MNIST 元训练配置](configs/mnist_meta.json) 通过 `meta` 配置任务与各阶段预算；使用独立入口检查：

```bash
uv run meta-sg-meta --config configs/mnist_meta.json --dry-run
```

完整外循环边界保存并恢复 Defender、各 follower、replay、优化器、随机状态与冻结统计量。该入口尚不包含部署期快速适应评估，示例参数也不代表已复现历史实验。

## 联邦训练与评估

[联邦训练模块](src/meta_sg/federated/README.md) 中，客户端接收模型、全局参数与 DataLoader，按指定步数或 epoch 执行交叉熵和 SGD 训练。
返回客户端参数增量、平均训练损失及实际处理样本数；批次采样由 DataLoader 决定。
服务端每轮无放回采样客户端，从同一份全局参数训练，调用传入的防御函数聚合，并更新全局模型。
`run_federated()` 逐轮返回客户端、聚合及评估记录；评估统一输出平均损失、准确率和样本数。
对 ResNet，客户端与 root 训练前分别恢复全局 BatchNorm 状态。参数增量由所选防御聚合，浮点缓冲等权平均，整数计数器取最大值；root 缓冲不并入全局统计。

## 固定防御

每种方法位于 `src/meta_sg/defenses/` 下的独立模块，直接调用函数并传入参数。
各实现文件开头简述设计思路和数据流程。

| 方法 | 模块 | 默认参数与行为 |
| --- | --- | --- |
| FedAvg | `fedavg.py` | 客户端等权平均 |
| Median | `median.py` | 逐坐标中位数；偶数个值取两个中央值的平均数 |
| Trimmed Mean | `trimmed_mean.py` | `byzantine_bound=4`，每个坐标两端各移除 4 个值；要求 `n > 2f` |
| Krum | `krum.py` | `byzantine_bound=4`；要求 `n >= 2f+3` |
| Multi-Krum | `multi_krum.py` | `byzantine_bound=4, selected=16`；要求 `n >= 2f+3` 且 `selected <= n` |
| RFA | `rfa.py` | 平滑 Weiszfeld 迭代；`iterations=4, epsilon=1e-5, ftol=1e-6` |
| FLTrust | `fltrust.py` | 需要可信根更新；按正余弦相似度赋权并归一化到根更新范数 |
| ClipMed | `clipmed.py` | 完整客户端更新的 L2 范数上限 `norm_bound=0.1`，再取坐标中位数 |

`n` 为本轮参与聚合的客户端数，`f` 为配置的 Byzantine 客户端数上界。
Multi-Krum 沿用旧实验的一次评分、选取前 `selected` 个更新再平均的实现。

### 统一接口

```python
import torch

from meta_sg.defenses import clipmed

updates = [(torch.tensor([6., 8.]),), (torch.tensor([0., 20.]),)]
update, diagnostics = clipmed(updates, norm_bound=5)

print(update)       # Aggregated update: (tensor([1.5000, 4.5000]),)
print(diagnostics)  # Aggregate norm, clipped client count, and other metrics
```

每个函数接收客户端更新，返回 `(update, diagnostics)`。每个客户端更新是按固定参数顺序组成的张量元组，包含参数变化量。
调用方保证更新非空、参数形状一致、设备一致、数值有限且为浮点数，并提供有效的算法参数。
只有 FLTrust 额外接收必填的 `root_update`，例如 `fltrust(updates, root_update=root_update)`。
训练流程负责将返回的 `update` 加到全局模型。
FLTrust 的信任权重总和为零时返回零更新。

共享工具只负责展开张量和恢复参数形状；诊断字典包含聚合范数和当前方法的实际指标。
`federated/aggregation.py` 只保留通用的加权更新应用操作；防御算法由各自模块管理。
算法及默认配置迁自上一级目录的 `src/meta_sg/defenses/fixed_aggregators.py`，原目录保留。

## 攻击与 ASR

[攻击模块](src/meta_sg/attacks/README.md) 分为两类：`model_poisoning/` 包含 IPM、LMP-Krum、Min-Max 和归档 Li2022 冻结 RL；`backdoor/` 包含 DBA、源类别定向 BFL、投毒训练及 ASR 评估。
通过 `run_federated(..., attack=...)` 注入；攻击者使用固定的全局客户端身份，支持攻击开始轮次和随机参与。
DBA、BFL 使用对应数据集模块中的归一化触发器，ASR 通过独立评估函数统计，并排除目标类别原样本。
RL 攻击需额外安装 `rl` 依赖并提供 checkpoint 与已知训练样本。
攻击代理 SGD 学习率通过 `attacker_learning_rate` 单独配置，默认 `0.01`，支持历史实验的 `0.05` 设置。
