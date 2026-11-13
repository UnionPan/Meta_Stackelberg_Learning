# MNIST integration demo

See the repository [Quickstart](../../QUICKSTART.md) for exact commands and budgets.

- [Measured summary](run/summary.json): CUDA/RTX 3090, seed 42, 36 total FL rounds.
- [Smoke summary](smoke_summary.json): reduced 16-round validation completed first.
- [Frozen evaluation](run/evaluation/metrics.jsonl): measured test accuracy/loss,
  rounds 0–3, IPM-5 enabled from round 2.
- [Meta phase metrics](run/meta/metrics.jsonl) and [physical rounds](run/meta/rounds.jsonl).
- [Histogram40](run/histogram40.json): frozen bins, without standardization for this small demo.
- [Plot](mnist_meta_demo_v1.png), [PDF](mnist_meta_demo_v1.pdf), [SVG](mnist_meta_demo_v1.svg),
  and [plot audit](mnist_meta_demo_v1.audit.json).

These are curated metrics and figures, copied from the completed development-machine
run. Full run checkpoints remain there; this directory is not a resume directory.
Absolute paths in saved run metadata record the execution environment. To generate
fresh paths, policies and checkpoints, rerun the Quickstart into `Outputs/runs/`.

The run checks finite metrics, every stage's completion, checkpoint presence and
a nonzero outer actor update. It is not a convergence or robustness benchmark.
