"""Design: plot only measured frozen-policy evaluation from the MNIST demo.
Data flow: evaluation JSONL and summary -> ACC/loss panels -> PNG/PDF/SVG and audit.
Use the shared project palette; do not smooth, extrapolate or rank a single run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from meta_sg_plot_style import STYLE_VERSION, apply_style, export_figure, finish_figure, format_axis, line_style


def main():
    parser = argparse.ArgumentParser(description='Plot the measured MNIST Quickstart evaluation')
    parser.add_argument('--run', default='Outputs/runs/quickstart')
    parser.add_argument('--output', default='Outputs/quickstart/mnist_meta_demo_v1')
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    metrics = run / 'evaluation/metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    summary = json.loads((run / 'summary.json').read_text())
    rounds = np.asarray([row['round'] for row in rows])
    accuracy = np.asarray([row['evaluation']['accuracy'] * 100 for row in rows])
    losses = np.asarray([row['evaluation']['loss'] for row in rows])
    assert np.array_equal(rounds, np.arange(summary['evaluation_fl_rounds'] + 1))
    assert np.isfinite(accuracy).all() and np.isfinite(losses).all()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    style = {**line_style('rl'), 'markevery': 1}
    handle, = axes[0].plot(rounds, accuracy, **style)
    axes[1].plot(rounds, losses, **style)
    format_axis(axes[0], title='Clean test accuracy', ylabel='ACC (%)', xlim=(0, rounds[-1]))
    format_axis(axes[1], title='Clean test cross-entropy', ylabel='Loss', xlim=(0, rounds[-1]), ylim=(0, losses.max() * 1.15))
    for ax in axes:
        ax.set_xticks(rounds)
        ax.axvline(2, color='#A4AFBD', linestyle=':', linewidth=1)
        ax.text(.98, .94, 'IPM-5 from round 2', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    finish_figure(fig, title='MNIST Meta-SG · Quickstart',
                  subtitle='MNISTCNN · IID · 4/4 clients · 1 malicious · Histogram40 · seed 42',
                  handles=[handle], labels=['Frozen Meta-SG defender'], columns=1,
                  footnotes=[f"Calibration: {summary['calibration_fl_rounds']} FL rounds; offline meta-training: {summary['meta']['fl_rounds']} FL rounds; frozen evaluation: {summary['evaluation_fl_rounds']} rounds.",
                             'Single-seed integration demo; this short run does not establish robustness.'])
    fig.texts[1].set_y(.91)
    fig.subplots_adjust(top=.80, bottom=.24, wspace=.24, left=.07)
    export_figure(fig, output)
    audit = {'style_version': STYLE_VERSION, 'method': 'frozen Meta-SG defender', 'seed': summary['seed'],
             'metric_units': {'accuracy': 'percent', 'loss': 'mean cross-entropy'},
             'measured_rounds': rounds.tolist(), 'smoothing': False, 'extrapolation': False,
             'source': str(metrics), 'source_sha256': hashlib.sha256(metrics.read_bytes()).hexdigest(),
             'outputs_sha256': {ext: hashlib.sha256(output.with_suffix('.' + ext).read_bytes()).hexdigest()
                                for ext in ['png', 'pdf', 'svg']}}
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
