# Algo2 首次正式实验结果

## 实验范围

本目录记录重构后的 `meta_stackelberg/` 在 MNIST 全局模型投毒场景中的第一次完整 Algo2 正式实验。训练配置为 `T=100, K=5, H=100, l=10`，攻击任务为 NA、IPM、LMP、RL-Krum 和 RL-ClipMed；联邦学习拓扑为 20 个客户端、4 个攻击者、每轮采样 4 个客户端。本地训练的 `local iteration=1` 按论文 Appendix C 实现为每客户端一个 minibatch step。

训练 checkpoint 的 SHA-256 为：

```text
8daa81ede3ff1cf42163610a17c488a20b26a98279d6f15bff768c1b133526ac
```

## 第一次在线适应结果

第一次评估直接部署在线适应候选，得到以下结果：

| 场景 | Frozen | 在线适应后 | 变化 |
|---|---:|---:|---:|
| Clean | 91.55% | 91.55% | 0.00 pp |
| IPM | 90.36% | 90.36% | 0.00 pp |
| LMP | 90.69% | 83.07% | -7.62 pp |
| RL-ClipMed | 87.69% | 87.69% | 0.00 pp |
| RL-Krum | 89.96% | 89.96% | 0.00 pp |

这组负结果没有被删除：它说明原在线适应流程并不能保证优于 Frozen。多数场景没有变化，LMP 还出现明显退化，因此不能把“完成 TD3 更新”等同于“得到更好的部署策略”。原始记录位于 `evaluation_online_h100/` 和 `final_results.json`。

## 修正后的在线适应与评估

后续排查与修正集中在在线阶段，不重新训练 Algo2 元策略：

- 每场景从同一个 Frozen 元策略独立开始，场景之间不串联；
- 使用 `online_T=10, online_H=100, online_l=10`，共采集 1,000 个 FL rounds，并执行 100 次 TD3 更新；
- 将在线学习率从 0.01 降为 0.001，并只对当前有效的 `alpha` 动作维加入 actor logit anti-saturation；
- 使用与最终测试独立的 reward guard 比较 Frozen 与 adapted，候选 reward 未提高时回退 Frozen。

修正后得到：

| 场景 | Frozen | Online 候选 | 候选变化 | Guard 后结果 |
|---|---:|---:|---:|---:|
| Clean | 91.55% | 92.10% | +0.55 pp | 92.10% |
| IPM | 90.36% | 89.71% | -0.65 pp | 90.36%（回退） |
| LMP | 90.69% | 91.11% | +0.42 pp | 91.11% |
| RL-ClipMed | 87.69% | 89.16% | +1.47 pp | 89.16% |
| RL-Krum | 89.96% | 89.91% | -0.05 pp | 89.96%（回退） |

这说明在线候选仍可能下降；reward guard 的作用是让这类负迁移不进入最终部署结果，而不是把负结果从实验记录中删除。修正后的权威结果位于 `evaluation_online_h100_v2_alpha_guarded/`。

## Git 中保留的产物

本次提交保留实验协议、训练指标、日志和逐场景 JSON，以便复核数值与轨迹。`.pt` checkpoint 体积较大，继续保存在本机原运行目录中，不进入普通 Git 历史；可通过上述 SHA-256 校验正式训练 checkpoint。

完整代码结构、四个时间尺度及论文对照见：

```text
docs/experiments/algo2_global_model_poisoning_handoff_20260718.md
```
