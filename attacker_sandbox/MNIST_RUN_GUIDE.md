# MNIST 运行指南

这份文档只讲一件事：怎么跑 `attacker_sandbox` 里的 MNIST 结果，以及跑完后去哪里看结果。

## 先决条件

在仓库根目录下执行：

```bash
cd /home/antik/rl/Meta_Stackelberg_Learning
source .venv/bin/activate
```

如果你还没有准备好环境，可以先看 `attacker_sandbox/setup_env.sh` 的说明。

## 1. 跑纯 MNIST 联邦学习

这是最适合先做的基线。

```bash
python attacker_sandbox/benchmark_clean.py \
  --dataset mnist \
  --device auto \
  --rounds 2 \
  --num_clients 6 \
  --subsample_rate 0.5 \
  --local_epochs 1 \
  --batch_size 512 \
  --eval_batch_size 4096 \
  --num_workers 8 \
  --output_dir attacker_sandbox/outputs/mnist_clean_demo \
  --tb_dir attacker_sandbox/runs/mnist_clean_demo
```

如果你想把 GPU 强制指定到某一张卡，可以把 `--device auto` 改成 `--device cuda:0`。

## 2. 跑 MNIST + IPM 演示

这个入口会先跑一遍 clean FL，再跑一遍 IPM attacker。

```bash
python attacker_sandbox/run_sandbox.py \
  --dataset mnist \
  --device auto \
  --rounds 5 \
  --num_clients 10 \
  --num_attackers 2 \
  --subsample_rate 0.5 \
  --local_epochs 1 \
  --batch_size 512 \
  --eval_batch_size 4096 \
  --num_workers 8 \
  --output_dir attacker_sandbox/outputs/mnist_ipm_demo
```

## 3. 跑完以后看什么

### `benchmark_clean.py` 的结果

跑完后，目录里最重要的是：

```text
attacker_sandbox/outputs/mnist_clean_demo/
  summary.json
  client_metrics.csv
  clean_acc.png
  clean_loss.png
  round_seconds.png
  ...
```

你可以这样看：

```bash
cat attacker_sandbox/outputs/mnist_clean_demo/summary.json
```

重点看这些字段：

```json
{
  "total_seconds": 12.34,
  "rounds": [
    {
      "clean_loss": 0.123,
      "clean_acc": 0.981,
      "round_seconds": 2.31,
      "evaluated": true
    }
  ]
}
```

最常看的是：

- `clean_acc`
- `clean_loss`
- `round_seconds`
- `client_metrics.csv` 里的每个 client 的 `train_loss`、`train_acc`、`update_norm`

如果你想生成图：

```bash
python attacker_sandbox/postprocess_clean.py \
  --input_dir attacker_sandbox/outputs/mnist_clean_demo \
  --tb_dir attacker_sandbox/runs/mnist_clean_demo
```

生成后会多出这些图：

- `clean_acc.png`
- `clean_loss.png`
- `clean_acc_vs_time.png`
- `client_update_norm_heatmap.png`
- `client_train_acc_heatmap.png`

你也可以直接开 TensorBoard：

```bash
tensorboard --logdir attacker_sandbox/runs/mnist_clean_demo
```

### `run_sandbox.py` 的结果

这个入口会把 clean 和 IPM 放到同一个 `summary.json` 里。

```bash
python attacker_sandbox/postprocess_sandbox.py \
  --input_dir attacker_sandbox/outputs/mnist_ipm_demo
```

最关键的对比图是：

- `clean_acc_compare.png`
- `clean_loss_compare.png`
- `malicious_norm_compare.png`
- `ipm_cosine_to_benign.png`

你主要看这几个值：

- `final.clean_acc`
- `final.ipm_acc`
- `final.ipm_mean_malicious_norm`

## 4. 怎么判断结果对不对

一个正常的 MNIST 跑法，通常会看到：

- `clean_acc` 随轮次上升
- `clean_loss` 随轮次下降
- `round_seconds` 在前几轮后趋于稳定
- `client_metrics.csv` 里被采样的 client 才会有训练指标

如果你跑的是 `run_sandbox.py`，那么还要看：

- IPM 的 `mean_malicious_norm` 是否明显高于 clean
- `clean_acc` 和 `ipm_acc` 的差距是否符合你的攻击预期

## 5. 常用调参

如果你只是想先快速验证：

```bash
python attacker_sandbox/benchmark_clean.py --dataset mnist --rounds 1 --batch_size 512 --num_workers 8
```

如果你想更像正式实验一点：

- 把 `--rounds` 提高到 `10` 或 `50`
- 保持 `--batch_size 512`
- `--num_workers` 设成 `8`
- 如果机器上 GPU 合适，就用 `--device cuda:0`

## 6. 数据位置

MNIST 会默认下载到仓库下的 `data/` 目录。`benchmark_clean.py` 里显式传的是：

```python
data_dir="data"
```

如果你第一次跑，看见数据下载日志是正常的。
