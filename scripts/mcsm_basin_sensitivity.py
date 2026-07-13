"""
Basin sensitivity sweep — how does the I91/I157 basin split depend on
the starting distribution of the 4 background taxa?

The existing full phase portrait fixes I44/I89/I78/I49 at equal abundance
and shows a roughly 50/50 basin split between I91 and I157.  This script
sweeps over N_BACKGROUNDS Dirichlet samples of background weights, runs
a full triangular (I157, I91) phase portrait for each, and records how
the basin split changes.

Key benchmark: the surrogate's vectorized Euler integration (one forward
pass per step over the entire grid) makes this sweep feasible in seconds.
The equivalent LP computation would require ~weeks.

Outputs in --plots-dir:
  basin_sensitivity_histogram.png    — distribution of splits + benchmark annotation
  basin_sensitivity_portraits.png    — side-by-side maps for baseline / extreme cases
  basin_sensitivity_correlations.png — per-background-taxon correlation with I91 fraction

Results saved to --output (npz).
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

TAXA_IDS = ['I44', 'I89', 'I157', 'I78', 'I91', 'I49']
IDX_I157 = TAXA_IDS.index('I157')   # 2
IDX_I91  = TAXA_IDS.index('I91')    # 4
IDX_BG   = [0, 1, 3, 5]            # indices of I44, I89, I78, I49
BG_NAMES = ['I44', 'I89', 'I78', 'I49']
N_TAXA   = len(TAXA_IDS)
N_BG     = len(IDX_BG)

MIN_ABUND = 1e-8
MIN_OTHER = 1e-3   # each background taxon gets at least this much abundance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "mcsm_community_input-6_output-6_hidden-64.pt")
    p.add_argument("--n-backgrounds", type=int, default=1000,
                   help="Number of background weight distributions to sample")
    p.add_argument("--grid-n", type=int, default=20,
                   help="Grid points per axis on (I91, I157) simplex face")
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--dt",      type=float, default=0.1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "plots" / "mcsm")
    p.add_argument("--output",    type=Path,
                   default=REPO_ROOT / "results" / "mcsm_basin_sensitivity.npz")
    p.add_argument("--lp-step-time", type=float, default=0.2,
                   help="Estimated LP solve time per step in seconds (benchmark annotation)")
    return p.parse_args()


def euler_step_batch(X, mu, dt):
    """Euler step on a batch of states. X, mu: (n_starts, n_taxa)."""
    X_new = X * (1.0 + mu * dt)
    X_new = np.maximum(X_new, MIN_ABUND)
    X_new /= X_new.sum(axis=1, keepdims=True)
    return X_new


def build_batch_x0(start_i91, start_i157, bg_weights):
    """
    Build (n_starts, 6) initial state matrix for a given background weight vector.

    bg_weights: (4,) summing to 1, for [I44, I89, I78, I49].
    The leftover abundance (1 - I91 - I157) is distributed proportionally to bg_weights.
    """
    n  = len(start_i91)
    X0 = np.zeros((n, N_TAXA))
    leftover = 1.0 - start_i91 - start_i157           # (n,)
    X0[:, IDX_I91]  = start_i91
    X0[:, IDX_I157] = start_i157
    X0[:, IDX_BG]   = leftover[:, None] * bg_weights  # (n, 4) via broadcast
    return X0


def run_batch(model_nn, x_scaler, y_scaler, X0, n_steps, dt):
    """
    Run n_steps of surrogate Euler on the full batch of starting states.
    Returns final abundance matrix (n_starts, n_taxa).
    """
    X = X0.copy()
    for _ in range(n_steps):
        mu = surrogate_predict(model_nn, x_scaler, y_scaler, X)
        X  = euler_step_batch(X, mu, dt)
    return X


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def fmt_bg(w):
    return "  ".join(f"{BG_NAMES[j]}={w[j]:.3f}" for j in range(N_BG))


def plot_basin_map(ax, start_i91, start_i157, i91_wins, max_sum, title, subtitle):
    """Draw a binary basin scatter (I157 on x, I91 on y) on the given axes."""
    ax.scatter(start_i157[i91_wins],  start_i91[i91_wins],
               color="tab:red",  s=18, alpha=0.8, linewidths=0)
    ax.scatter(start_i157[~i91_wins], start_i91[~i91_wins],
               color="tab:blue", s=18, alpha=0.8, linewidths=0)
    ax.plot([0, max_sum], [max_sum, 0], color="grey",
            linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xlim(-0.01, max_sum + 0.02)
    ax.set_ylim(-0.01, max_sum + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Starting I157", fontsize=9)
    ax.set_ylabel("Starting I91",  fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.text(0.5, -0.14, subtitle, transform=ax.transAxes,
            fontsize=7, ha="center", va="top", style="italic",
            wrap=True)


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"  Checkpoint : {args.checkpoint.name}")
    print(f"  hidden_dim : {metadata.get('hidden_dim')}  "
          f"n_layers : {metadata.get('n_layers', 1)}")

    section("2. Building triangular grid")
    max_sum = 1.0 - N_BG * MIN_OTHER
    vals    = np.linspace(0.0, max_sum, args.grid_n)
    starts  = [(i91, i157) for i91 in vals for i157 in vals
               if i91 + i157 <= max_sum]
    n_starts       = len(starts)
    start_i91_arr  = np.array([s[0] for s in starts])
    start_i157_arr = np.array([s[1] for s in starts])
    print(f"  grid_n={args.grid_n}  →  {n_starts} valid starting points  "
          f"(max_sum={max_sum:.4f})")

    section("3. Sampling background distributions")
    rng         = np.random.default_rng(args.seed)
    backgrounds = np.empty((args.n_backgrounds, N_BG))
    backgrounds[0] = np.ones(N_BG) / N_BG   # equal-weight reference (explicit)
    backgrounds[1:] = rng.dirichlet(np.ones(N_BG), size=args.n_backgrounds - 1)
    print(f"  n_backgrounds={args.n_backgrounds}  (idx 0 = equal-weight reference)")

    section("4. Running sweep (vectorized surrogate)")
    frac_i91 = np.zeros(args.n_backgrounds)
    t0       = time.time()

    for i, w in enumerate(backgrounds):
        X0           = build_batch_x0(start_i91_arr, start_i157_arr, w)
        X_fin        = run_batch(model_nn, x_scaler, y_scaler, X0, args.n_steps, args.dt)
        frac_i91[i]  = (X_fin[:, IDX_I91] > X_fin[:, IDX_I157]).mean()
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1:>4}/{args.n_backgrounds}]  "
                  f"frac_I91={frac_i91[i]:.3f}  bg: {fmt_bg(w)}")

    surrogate_time = time.time() - t0
    frac_i157      = 1.0 - frac_i91

    n_lp_solves   = args.n_backgrounds * n_starts * args.n_steps
    lp_time_est_s = n_lp_solves * args.lp_step_time
    speedup       = lp_time_est_s / surrogate_time

    print(f"\n  Surrogate sweep : {surrogate_time:.1f} s")
    print(f"  LP estimate     : {lp_time_est_s/3600:.1f} h  "
          f"({n_lp_solves:,} solves × {args.lp_step_time} s)")
    print(f"  Speedup         : {speedup:,.0f}×")

    section("5. Extremes")
    eq_idx      = 0
    max_i91_idx = int(np.argmax(frac_i91))
    min_i91_idx = int(np.argmin(frac_i91))
    print(f"  Equal-weight baseline (idx {eq_idx}):  "
          f"frac_I91={frac_i91[eq_idx]:.3f}")
    print(f"  Most I91-biased (idx {max_i91_idx}):   "
          f"frac_I91={frac_i91[max_i91_idx]:.3f}  bg: {fmt_bg(backgrounds[max_i91_idx])}")
    print(f"  Most I157-biased (idx {min_i91_idx}):  "
          f"frac_I91={frac_i91[min_i91_idx]:.3f}  bg: {fmt_bg(backgrounds[min_i91_idx])}")
    print(f"  Range : [{frac_i91.min():.3f}, {frac_i91.max():.3f}]  "
          f"mean={frac_i91.mean():.3f}  std={frac_i91.std():.3f}")

    section("6. Saving results")
    np.savez_compressed(
        args.output,
        backgrounds         = backgrounds,
        frac_i91            = frac_i91,
        frac_i157           = frac_i157,
        background_taxa_ids = BG_NAMES,
        grid_starts         = np.column_stack([start_i91_arr, start_i157_arr]),
        n_steps             = args.n_steps,
        dt                  = args.dt,
        surrogate_time_s    = surrogate_time,
        lp_time_est_s       = lp_time_est_s,
        speedup             = speedup,
    )
    print(f"  Saved: {args.output}")

    # ── Figure 1: histogram of basin splits ──────────────────────────────────────
    section("7. Plotting")
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    ax.hist(frac_i91, bins=40, color="steelblue", edgecolor="white", linewidth=0.4,
            label=f"N={args.n_backgrounds} backgrounds")
    ax.axvline(frac_i91[eq_idx], color="black", linewidth=2.0, linestyle="--",
               label=f"Equal-weight baseline  {frac_i91[eq_idx]:.3f}")
    ax.axvline(frac_i91[max_i91_idx], color="tab:red",  linewidth=1.5, linestyle=":",
               label=f"Max I91 bias  {frac_i91[max_i91_idx]:.3f}")
    ax.axvline(frac_i91[min_i91_idx], color="tab:blue", linewidth=1.5, linestyle=":",
               label=f"Max I157 bias  {frac_i91[min_i91_idx]:.3f}")
    ax.set_xlabel("Fraction of grid starting points converging to I91", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"Basin split sensitivity — {args.n_backgrounds} background distributions\n"
        f"mean={frac_i91.mean():.3f}  std={frac_i91.std():.3f}  "
        f"range=[{frac_i91.min():.3f}, {frac_i91.max():.3f}]  |  "
        f"Surrogate: {surrogate_time:.1f} s  vs  LP est.: {lp_time_est_s/3600:.1f} h  "
        f"(×{speedup:,.0f} speedup)",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    path = args.plots_dir / "basin_sensitivity_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Figure 2: side-by-side phase portraits ────────────────────────────────────
    case_indices = [eq_idx, max_i91_idx, min_i91_idx]
    case_titles  = [
        f"Equal weight\nI91={frac_i91[eq_idx]:.1%}  I157={frac_i157[eq_idx]:.1%}",
        f"Max I91 bias\nI91={frac_i91[max_i91_idx]:.1%}  "
        f"I157={frac_i157[max_i91_idx]:.1%}",
        f"Max I157 bias\nI91={frac_i91[min_i91_idx]:.1%}  "
        f"I157={frac_i157[min_i91_idx]:.1%}",
    ]
    case_subtitles = [fmt_bg(backgrounds[i]) for i in case_indices]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, idx, title, subtitle in zip(axes, case_indices, case_titles, case_subtitles):
        w        = backgrounds[idx]
        X0       = build_batch_x0(start_i91_arr, start_i157_arr, w)
        X_fin    = run_batch(model_nn, x_scaler, y_scaler, X0, args.n_steps, args.dt)
        i91_wins = X_fin[:, IDX_I91] > X_fin[:, IDX_I157]
        plot_basin_map(ax, start_i91_arr, start_i157_arr,
                       i91_wins, max_sum, title, subtitle)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
               markersize=9, label="→ I91 dominant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markersize=9, label="→ I157 dominant"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Basin of attraction maps — baseline vs extreme background distributions",
                 fontsize=11)
    path = args.plots_dir / "basin_sensitivity_portraits.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Figure 3: per-taxon correlation ──────────────────────────────────────────
    bg_colors = plt.get_cmap("tab10")(np.arange(N_BG) / 10)
    fig, axes = plt.subplots(1, N_BG, figsize=(13, 3.5), constrained_layout=True,
                             sharey=True)
    for j, (ax, name, col) in enumerate(zip(axes, BG_NAMES, bg_colors)):
        x_j = backgrounds[:, j]
        r   = float(np.corrcoef(x_j, frac_i91)[0, 1])
        ax.scatter(x_j, frac_i91, color=col, s=5, alpha=0.35, linewidths=0)
        ax.axhline(frac_i91[eq_idx], color="black", linewidth=0.9,
                   linestyle="--", alpha=0.6)
        ax.set_xlabel(f"Weight of {name}", fontsize=10)
        ax.set_title(f"Pearson r = {r:+.3f}", fontsize=10)
        if j == 0:
            ax.set_ylabel("Fraction I91 dominant", fontsize=10)
    fig.suptitle(
        "Correlation between each background taxon's weight and the I91 basin fraction\n"
        "(dashed line = equal-weight baseline)",
        fontsize=10,
    )
    path = args.plots_dir / "basin_sensitivity_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\nAll outputs written to {args.plots_dir}")
    print(f"Surrogate: {surrogate_time:.1f}s  |  LP est.: {lp_time_est_s/3600:.1f}h  "
          f"|  Speedup: {speedup:,.0f}×")


if __name__ == "__main__":
    main()
