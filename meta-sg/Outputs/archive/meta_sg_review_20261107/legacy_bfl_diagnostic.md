# 旧版 BFL 诊断归档

### 9.5 旧版 BFL：源类别 0→9

100 抽 10，200 轮，第 11 轮开始投毒；完整黑色触发器、更新不放大，ASR 分母为 50 张源类别验证图像。统计第 198–200 轮均值。

| 防御方法 | ACC↑（%） | ASR↓（%） |
| --- | ---: | ---: |
| FedAvg | 32.093 | 1.333 |
| Trimmed Mean | 31.940 | 0.000 |
| FLTrust | 28.860 | 0.667 |
| FLTrust+NC | 28.860 | 0.667 |
| ClipMed | 9.933 | 0.000 |
| RL specialist | 30.220 | 0.000 |

该版本 FedAvg 的 ASR 为 1.333%；BFL v2 的攻击设置与结果见 5.2、9.3。

[完整攻防结果](../resnet18_document_results_20261107_v2/results.csv) · [补充批次](../resnet18_document_results_20261107_v2/additional_results.csv) · [无攻击及旧版本](../resnet18_document_results_20261107_v2/reference_results.csv) · [原始来源核验](../resnet18_document_results_20261107_v2/audit.json)

