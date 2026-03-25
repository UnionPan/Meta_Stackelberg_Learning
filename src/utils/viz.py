"""
Visualization utilities for meta-Stackelberg learning.

Includes:
- Ternary (simplex) plots for 3-action policy evolution
- Exploitability computation for zero-sum games
- TensorBoard figure logging helpers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
from typing import List, Dict, Optional, Tuple
import torch
import io
from PIL import Image


# --- Simplex (Ternary) Plot Utilities ---

def _simplex_to_cartesian(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert probability vectors (N, 3) to 2D Cartesian coordinates
    for plotting on an equilateral triangle simplex.

    Vertices:
        action 0 (e.g., Rock)     -> top        (0.5, sqrt(3)/2)
        action 1 (e.g., Paper)    -> bottom-left (0, 0)
        action 2 (e.g., Scissors) -> bottom-right(1, 0)
    """
    probs = np.array(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, 3)

    x = 0.5 * probs[:, 0] + probs[:, 2]
    y = (np.sqrt(3) / 2) * probs[:, 0]
    return x, y


def draw_simplex_frame(ax, labels=("Rock", "Paper", "Scissors")):
    """Draw the equilateral triangle frame on the given axes."""
    # Triangle vertices
    vertices = np.array([
        [0.5, np.sqrt(3) / 2],  # top (action 0)
        [0.0, 0.0],             # bottom-left (action 1)
        [1.0, 0.0],             # bottom-right (action 2)
    ])
    triangle = plt.Polygon(vertices, fill=False, edgecolor='gray',
                           linewidth=1.5, zorder=1)
    ax.add_patch(triangle)

    # Labels
    offset = 0.06
    ax.text(vertices[0, 0], vertices[0, 1] + offset, labels[0],
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(vertices[1, 0] - offset, vertices[1, 1] - offset, labels[1],
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(vertices[2, 0] + offset, vertices[2, 1] - offset, labels[2],
            ha='center', va='top', fontsize=11, fontweight='bold')

    # Uniform point (Nash equilibrium for RPS)
    ux, uy = _simplex_to_cartesian(np.array([1/3, 1/3, 1/3]))
    ax.plot(ux, uy, 'k+', markersize=12, markeredgewidth=2, zorder=5,
            label='Nash (1/3, 1/3, 1/3)')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, np.sqrt(3) / 2 + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_policy_trajectory(
    policy_history: List[np.ndarray],
    ax=None,
    label: str = "policy",
    color: str = "blue",
    marker_start: str = "o",
    marker_end: str = "s",
    alpha: float = 0.6,
    show_arrows: bool = True,
):
    """
    Plot a policy trajectory on the simplex.

    Args:
        policy_history: list of (3,) probability vectors over training
        ax: matplotlib axes (created if None)
        label: legend label
        color: line/marker color
        marker_start: marker for the first policy
        marker_end: marker for the last policy
        alpha: line transparency
        show_arrows: whether to draw arrows showing direction
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        draw_simplex_frame(ax)

    probs = np.array(policy_history)
    x, y = _simplex_to_cartesian(probs)

    # Draw trajectory line
    ax.plot(x, y, '-', color=color, alpha=alpha * 0.5, linewidth=1, zorder=2)

    # Start and end markers
    ax.plot(x[0], y[0], marker_start, color=color, markersize=8,
            zorder=4, label=f'{label} (start)')
    ax.plot(x[-1], y[-1], marker_end, color=color, markersize=10,
            markeredgewidth=2, zorder=4, label=f'{label} (end)')

    # Arrows along trajectory
    if show_arrows and len(x) > 5:
        step = max(1, len(x) // 8)
        for i in range(0, len(x) - step, step):
            dx = x[i + step] - x[i]
            dy = y[i + step] - y[i]
            ax.annotate('', xy=(x[i + step], y[i + step]),
                        xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        alpha=alpha, lw=1.2))

    return ax


def plot_all_policies_on_simplex(
    defender_history: List[np.ndarray],
    attacker_histories: Dict[str, List[np.ndarray]],
    title: str = "Policy Evolution on Simplex",
    labels: Tuple[str, str, str] = ("Rock", "Paper", "Scissors"),
    save_path: Optional[str] = None,
):
    """
    Plot defender and all attacker type policies on one simplex.

    Args:
        defender_history: list of defender policy vectors
        attacker_histories: {type_name: list of policy vectors}
        title: plot title
        labels: action labels
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    draw_simplex_frame(ax, labels=labels)

    # Color map for attacker types
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(attacker_histories), 3)))

    # Plot defender
    plot_policy_trajectory(defender_history, ax=ax, label="Defender",
                           color="blue", marker_start="o", marker_end="*")

    # Plot each attacker type
    for i, (type_name, history) in enumerate(attacker_histories.items()):
        plot_policy_trajectory(history, ax=ax, label=f"Attacker ({type_name})",
                               color=colors[i], marker_start="^",
                               marker_end="D")

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# --- Exploitability ---

def compute_rps_exploitability(policy_probs: np.ndarray) -> float:
    """
    Compute exploitability of a mixed strategy in standard RPS.

    Exploitability = max payoff a best-responder can achieve.

    RPS payoff matrix (row = defender, col = attacker):
        R  P  S
    R [ 0 -1  1]
    P [ 1  0 -1]
    S [-1  1  0]

    For defender playing (p_R, p_P, p_S), attacker's expected payoff
    for each pure strategy:
        play R: 0*p_R + 1*p_P + (-1)*p_S = p_P - p_S
        play P: (-1)*p_R + 0*p_P + 1*p_S = p_S - p_R
        play S: 1*p_R + (-1)*p_P + 0*p_S = p_R - p_P

    Exploitability = max(p_P - p_S, p_S - p_R, p_R - p_P)
    At Nash: exploitability = 0
    """
    p = np.array(policy_probs).flatten()
    attacker_payoffs = [
        p[1] - p[2],  # attacker plays R
        p[2] - p[0],  # attacker plays P
        p[0] - p[1],  # attacker plays S
    ]
    return max(attacker_payoffs)


# --- TensorBoard Helpers ---

def fig_to_tensor(fig) -> torch.Tensor:
    """Convert a matplotlib figure to a torch tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    # HWC -> CHW
    return torch.tensor(img_array).permute(2, 0, 1).float() / 255.0


def get_policy_probs(agent, obs_dim: int, obs_index: int = -1) -> np.ndarray:
    """
    Get the policy distribution from a PolicyAgent.

    Args:
        agent: PolicyAgent
        obs_dim: observation dimension
        obs_index: which one-hot obs to use (-1 = last = "start" state for RPS)

    Returns:
        (act_dim,) probability vector
    """
    with torch.no_grad():
        obs = torch.zeros(1, obs_dim)
        if obs_index >= 0:
            obs[0, obs_index] = 1.0
        else:
            obs[0, obs_dim - 1] = 1.0  # "start" observation
        logits, _ = agent(obs)
        probs = torch.softmax(logits, dim=-1)
    return probs.squeeze().numpy()
