# IPM+LMP：已有 Meta-SG checkpoint 的适应更新诊断

CIFAR10CNN、IID、20 客户端全参与、4 恶意；Histogram40 Meta T10；训练 seed 42，查询 seed 43。

保留 Meta Actor、重置 Critic 后，ACC 为 69.649%，比原 B3 高 1.541 pp，比冻结初始化高 1.126 pp。仅取消额外 Actor 更新为 68.120%；重置后再预热 100 次为 69.642%，未增加实测收益。

四个预设诊断方案复用同一预训练 Actor、同一份 20 轮随机行为支持数据、相同 root／risk 奖励及冻结观测变换。无新元训练；查询期间策略不更新。

| 方案 | Critic 更新 | Actor 更新合计 | ACC：491–500 | ACC：451–500 | Δ原 B3（pp） | Δ冻结（pp） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结初始化 | 0 | 0 | 68.523 | 68.250 | +0.415 | +0.000 |
| 原 B3 适应 | 50 | 50 | 68.108 | 67.848 | +0.000 | -0.415 |
| 去掉额外 Actor 更新 | 50 | 25 | 68.120 | 67.924 | +0.012 | -0.403 |
| 重置 Critic，去掉额外 Actor | 50 | 25 | 69.649 | 69.473 | +1.541 | +1.126 |
| 重置 Critic＋预热 100 次，去掉额外 Actor | 150 | 25 | 69.642 | 69.447 | +1.534 | +1.119 |
| policy delay=10，去掉额外 Actor | 50 | 5 | 68.523 | 68.241 | +0.415 | +0.000 |

原 B3 为 25 次常规＋25 次额外 mean-Q Actor 更新；新四组均无额外 Actor-only 更新。Critic 预热 100 次的成本已计入 150 次。重置只替换 Critic／目标 Critic，Actor 初始化保留。policy delay=10 同时降低 Actor 更新与目标网络同步频率，不能把差异只归因于 Actor 更新次数。

历史 Joint B3 ACC 为 69.533%，Random B3 为 69.461%；两者均为 Critic 50／Actor 合计 50，未在新四种规则下重新评价。

原 Meta B1（5 支持轮、Critic 50／Actor 合计 50）ACC 为 69.640%；本次是修复 20 轮支持条件下的适应退化，结果也应与这一既有较好预算比较。

全部为单 seed、同一已观察任务上的探索性诊断，全部候选均报告；不能据此宣称跨攻击提升或对 Joint 的同更新规则优势。改动若改善效果，只能说明适应流程可改善，不能证明奖励已最优。

[逐项数据](comparison.csv) · [独立核验](verification.json) · [预训练 checkpoint](readback/pretrained/training/histogram40/meta/meta_checkpoints/010/td3.zip)

## Critic 的动作梯度检查

固定相同 20 个支持观测与初始 Meta Actor 动作，继承 Critic 在 50 次更新后对 norm_mix 的平均 Q1 梯度为 -0.06319，重置 Critic 后为 +0.03144。两者给出相反的优化方向；估计 Q 值不等于实测效用。此检查不训练模型、不生成新 FL 数据。
