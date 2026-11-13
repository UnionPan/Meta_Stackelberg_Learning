"""Shared Meta-SG figure style and measured-window ranking helpers."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

STYLE_VERSION = 'meta-sg-muted-v1'
PALETTE = {
    'fedavg': '#8C9BAA', 'median': '#D5AB86', 'trimmed_mean': '#8FAB86',
    'krum': '#A18AB7', 'multi_krum': '#6DADA5', 'rfa': '#AA8B7A',
    'fltrust': '#5B80AF', 'policy': '#B95670', 'custom_fixed': '#776B91',
    'clipmed': '#92945B',
    'reptile': '#D62728', 'pooled': '#E6B800', 'random': '#1F77B4',
    'specialist_reference': '#F28E2B',
    'meta_static': '#AA8B7A',
    'positive_fixed': '#C09A62',
    'fixed_mixed': '#C09A62',
    'legacy8': '#8C9BAA', 'histogram40': '#B95670', 'deep_sets40': '#5B80AF',
}
ALIASES = {'fixed': 'custom_fixed', 'rl': 'policy', 'rl_defender': 'policy'}
PALETTE_THEMES = {
    'comparison_rl_only_v5': {
        **PALETTE,
        **dict.fromkeys(('fedavg', 'median', 'trimmed_mean', 'krum', 'multi_krum',
                         'rfa', 'custom_fixed', 'clipmed', 'fltrust'), '#B8C4D1'),
        'policy': '#D45A43',
    },
    'comparison_slate_v2': {
        **PALETTE,
        **dict.fromkeys(('fedavg', 'median', 'trimmed_mean', 'krum', 'multi_krum',
                         'rfa', 'custom_fixed', 'clipmed'), '#B8C4D1'),
        'fltrust': '#2C5F8A',
        'policy': '#D45A43',
    },
    'adaptation_muted': {
        **PALETTE,
        'reptile': PALETTE['policy'],
        'random': PALETTE['fedavg'],
        'pooled': PALETTE['trimmed_mean'],
    },
}


def canonical_method(method):
    return ALIASES.get(method, method)


def apply_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ('Arial Unicode MS', 'Noto Sans CJK SC', 'PingFang SC', 'DejaVu Sans') if name in available]
    plt.rcParams.update({
        'font.family': fonts or ['sans-serif'], 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.edgecolor': '#A4AFBD', 'axes.linewidth': .8,
        'axes.labelcolor': '#465364', 'text.color': '#344254',
        'xtick.color': '#667385', 'ytick.color': '#667385',
        'grid.color': '#DDE3EA', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })


def line_style(method, *, theme=None):
    method = canonical_method(method)
    palette = PALETTE if theme is None else PALETTE_THEMES[theme]
    # Unknown methods need an explicit palette choice, not a positional color cycle.
    style = {'color': palette[method], 'linewidth': 1.25,
             'linestyle': '--' if method == 'custom_fixed' else '-', 'zorder': 2}
    if method == 'policy':
        style.update(linewidth=1.7, marker='o', markevery=(24, 50),
                     markersize=3.5, markerfacecolor='white', zorder=5)
    return style


def format_axis(ax, *, title, ylabel, xlim, ylim=(0, 100)):
    ax.set(title=title, xlabel='FL round', ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.grid(alpha=.6, linewidth=.6)


def window_mean(rounds, values, start, end):
    """Return None unless every measured round in the window exists exactly once."""
    rounds, values = np.asarray(rounds), np.asarray(values, dtype=float)
    if rounds.ndim != 1 or values.shape != rounds.shape or start < 1 or end < start:
        raise ValueError('Expected aligned one-dimensional data and a positive round window')
    selected = (rounds >= start) & (rounds <= end)
    if not np.array_equal(rounds[selected], np.arange(start, end + 1)):
        return None
    if not np.isfinite(values[selected]).all():
        return None
    return float(values[selected].mean())


def top_k_box(ax, series, *, start, end, metric='ACC', k=3, position=(.98, .05)):
    """series: (label, measured rounds, percentage values); include cutoff ties."""
    if metric not in ('ACC', 'ASR'):
        raise ValueError('Ranking direction must be explicitly ACC or ASR')
    ranking = []
    for label, rounds, values in series:
        mean = window_mean(rounds, values, start, end)
        if mean is not None:
            ranking.append((label, mean))
    ranking.sort(key=lambda item: -item[1] if metric == 'ACC' else item[1])
    if not ranking:
        return []
    cutoff = ranking[min(k, len(ranking)) - 1][1]
    top = [item for item in ranking if (item[1] >= cutoff - 1e-10 if metric == 'ACC' else item[1] <= cutoff + 1e-10)]
    heading = f'Top {k} · rounds {start}–{end} mean · {metric} {"↑" if metric == "ACC" else "↓"}'
    ax.text(*position, heading + '\n' + '\n'.join(f'{label}: {mean:.3f}%' for label, mean in top),
            transform=ax.transAxes, ha='right' if position[0] > .5 else 'left',
            va='bottom', fontsize=9, zorder=10,
            bbox={'facecolor': 'white', 'alpha': .92, 'edgecolor': '#DDE3EA', 'pad': 5})
    return top


def finish_figure(fig, *, title, subtitle, handles, labels, footnotes, columns=4):
    fig.suptitle(title, fontsize=18, y=.985)
    fig.text(.5, .935, subtitle, ha='center', fontsize=10, color='#475569')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055),
               ncol=columns, frameon=False, fontsize=10)
    fig.text(.5, .012, '\n'.join(footnotes), ha='center', fontsize=9, color='#475569')
    fig.subplots_adjust(top=.875, bottom=.17, hspace=.30, wspace=.16, left=.065, right=.985)


def export_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
