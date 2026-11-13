# CIFAR-10：100 抽 10 冻结迁移补跑

CIFAR10CNN、IID、100 客户端每轮抽 10；IPM／LMP 固定 20 恶意身份，DBA／BFL 固定 5。正常本地 SGD 1 步、batch 128、lr=0.05；root 200 张／5 步；seed=43。

复用四套 Histogram40 与四套 Compact8 专用策略，均在 20 客户端／4 恶意配置下训练 10k 步。网络、分箱、输入标准化及风险校准冻结；只推进每客户端历史状态，无策略训练或部署适应。

旧评估器相同配置连续两次 5 轮复放也有微小差异。本批使用确定性执行；补跑 Compact8 的 start=3 对照。新评估器与原评估器在相同确定性设置下，四个攻击的 5 轮 ACC／ASR／动作精确一致。旧结果不覆盖。

LMP 仍用原 f=2、搜索 20 步、倍率上限 20 的主搜索；失败后最多允许 10 次确定性重启，实际额外次数逐轮记录。
本批 LMP 完整评价实际额外搜索合计 0 次。

## attack_start=3

| 攻击 | Compact8 ACC↑ | Histogram40 ACC↑ | Compact8 ASR↓ | Histogram40 ASR↓ |
| --- | ---: | ---: | ---: | ---: |
| IPM-5 | 55.630 | 55.740 | — | — |
| LMP-Krum | 51.640 | 55.547 | — | — |
| DBA simultaneous | 52.053 | 56.850 | 5.333 | 4.133 |
| BFL | 55.280 | 53.820 | 5.233 | 4.800 |

[逐项结果](start3_results.csv)

## attack_start=1

| 攻击 | Compact8 ACC↑ | Histogram40 ACC↑ | Compact8 ASR↓ | Histogram40 ASR↓ |
| --- | ---: | ---: | ---: | ---: |
| IPM-5 | 55.917 | 56.877 | — | — |
| LMP-Krum | 53.593 | 55.077 | — | — |
| DBA simultaneous | 52.163 | 55.487 | 6.200 | 3.733 |
| BFL | 56.143 | 56.797 | 5.733 | 3.933 |

[逐项结果](start1_results.csv)

统计均为第 498–500 轮实测均值；单 seed，无误差条。attack_start=1 表示从首轮允许攻击，某轮未抽到恶意客户端时无投毒。

本批 16 条完整查询均为 500 轮，每 50 轮保存 FL 检查点；checkpoint 留在开发机原目录。采样、实际攻击、动作、ACC／ASR、LMP 重试及冻结输入观测均已保存。

[完整 16 项](all_16_results.csv) · [表示差值](paired_comparison.csv) · [起点变化差值](onset_deltas.csv) · [独立核验](local_audit.json)
