"""
Full phase portrait via surrogate simulation from a triangular grid of
(taxon_x, taxon_y) starting abundances.

The remaining N-2 taxa share the leftover abundance equally:
    x_other = (1 - a_x - a_y) / (n_taxa - 2)

Produces in --plots-dir:
  full_phase_portrait.png  — all grid trajectories in taxon_x vs taxon_y space,
                             coloured by final dominant taxon
  full_basin_map.png       — scatter of starting (a_x, a_y) coloured by
                             which taxon the trajectory converges to

Usage:
    python scripts/hvsc1_full_phase_portrait.py
    python scripts/hvsc1_full_phase_portrait.py --taxon-x 946 --taxon-y 644
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

MIN_ABUND = 1e-8
MIN_OTHER = 1e-3   # minimum abundance floor for each background taxon


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taxon-x",   type=str,   default="946")
    p.add_argument("--taxon-y",   type=str,   default="644")
    p.add_argument("--grid-n",    type=int,   default=25,
                   help="Grid points per axis (→ ~grid_n²/2 valid starts)")
    p.add_argument("--n-steps",   type=int,   default=80)
    p.add_argument("--dt",        type=float, default=0.1)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--lp-trajectories", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_trajectories.npz")
    p.add_argument("--plots-dir", type=Path,
                   default=REPO_ROOT / "plots" / "hvsc1")
    return p.parse_args()


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def simulate(model_nn, x_scaler, y_scaler, x0, n_steps, dt):
    """Return final abundance vector after n_steps of surrogate Euler integration."""
    x = x0.copy()
    for _ in range(n_steps):
        mu = surrogate_predict(model_nn, x_scaler, y_scaler, x).flatten()
        x  = euler_step(x, mu, dt)
    return x


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load surrogate ────────────────────────────────────────────────────────
    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"Checkpoint loaded: {args.checkpoint.name}")

    # ── Load LP trajectories (for overlay) ───────────────────────────────────
    lp            = np.load(args.lp_trajectories, allow_pickle=True)
    taxa_ids      = list(lp["taxa_ids"])
    n_taxa        = len(taxa_ids)
    lp_trajs      = lp["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    lp_names      = list(lp["profile_names"])

    idx_x = taxa_ids.index(args.taxon_x)
    idx_y = taxa_ids.index(args.taxon_y)
    n_other = n_taxa - 2
    max_sum = 1.0 - n_other * MIN_OTHER

    print(f"Taxon axes: x={args.taxon_x} (idx {idx_x}), y={args.taxon_y} (idx {idx_y})")
    print(f"Background taxa: {n_other}  each gets (1-ax-ay)/{n_other}")
    print(f"Max (ax+ay): {max_sum:.4f}")

    # ── Build triangular grid ─────────────────────────────────────────────────
    vals   = np.linspace(0.0, max_sum, args.grid_n)
    starts = [(ax, ay) for ax in vals for ay in vals if ax + ay <= max_sum]
    n_starts = len(starts)
    print(f"Grid: {args.grid_n}×{args.grid_n} → {n_starts} valid starting points")

    # ── Run all grid trajectories ─────────────────────────────────────────────
    start_ax = np.array([s[0] for s in starts])
    start_ay = np.array([s[1] for s in starts])
    final_states = np.zeros((n_starts, n_taxa))
    final_ax     = np.zeros(n_starts)
    final_ay     = np.zeros(n_starts)
    # Store full trajectories for the phase portrait lines
    # (only keep x and y columns to save memory)
    traj_x_col = np.zeros((n_starts, args.n_steps + 1))
    traj_y_col = np.zeros((n_starts, args.n_steps + 1))

    t0 = time.time()
    for k, (ax, ay) in enumerate(starts):
        other = (1.0 - ax - ay) / n_other
        x0 = np.full(n_taxa, other)
        x0[idx_x] = ax
        x0[idx_y]  = ay

        # Run full trajectory, storing only the two phase-portrait columns
        x = x0.copy()
        traj_x_col[k, 0] = x[idx_x]
        traj_y_col[k, 0] = x[idx_y]
        for s in range(args.n_steps):
            mu = surrogate_predict(model_nn, x_scaler, y_scaler, x).flatten()
            x  = euler_step(x, mu, args.dt)
            traj_x_col[k, s + 1] = x[idx_x]
            traj_y_col[k, s + 1] = x[idx_y]

        final_states[k] = x
        final_ax[k] = x[idx_x]
        final_ay[k] = x[idx_y]

        if (k + 1) % 100 == 0:
            print(f"  [{k+1:>4}/{n_starts}]  elapsed {time.time()-t0:.1f}s")

    print(f"Simulated {n_starts} trajectories in {time.time()-t0:.2f} s")

    # ── Determine attractor for each trajectory ───────────────────────────────
    final_dominant_idx = np.argmax(final_states, axis=1)   # (n_starts,) int
    unique_dominants   = np.unique(final_dominant_idx)
    unique_taxa        = [taxa_ids[i] for i in unique_dominants]
    print(f"Dominant taxa found: {unique_taxa}")

    # Assign a color to each dominant taxon
    cmap20   = plt.get_cmap("tab20")
    dom_colors = {
        dom_idx: cmap20(j / max(len(unique_dominants) - 1, 1))
        for j, dom_idx in enumerate(unique_dominants)
    }

    # ── Figure 1: full_phase_portrait.png ─────────────────────────────────────
    lp_prof_colors = plt.get_cmap("Set1")(np.linspace(0, 0.8, len(lp_names)))

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    for k in range(n_starts):
        color = dom_colors[final_dominant_idx[k]]
        ax.plot(traj_x_col[k], traj_y_col[k],
                color=color, linewidth=0.5, alpha=0.35, zorder=1)
        ax.scatter(traj_x_col[k, 0], traj_y_col[k, 0],
                   color=color, s=6, alpha=0.55, zorder=2, linewidths=0)

    # LP trajectories overlay
    for p, name in enumerate(lp_names):
        traj = lp_trajs[p]
        c    = lp_prof_colors[p]
        ax.plot(traj[:, idx_x], traj[:, idx_y],
                color=c, linewidth=2.2, linestyle="-", zorder=4, label=name)
        ax.scatter(traj[0,  idx_x], traj[0,  idx_y],
                   color=c, marker="o", s=60, zorder=5)
        ax.scatter(traj[-1, idx_x], traj[-1, idx_y],
                   color=c, marker="*", s=130, zorder=5)

    ax.plot([0, max_sum], [max_sum, 0], color="grey",
            linewidth=0.7, linestyle="--", zorder=0)

    # Legend: dominant taxa as colored patches + LP profiles
    dom_patches = [
        mpatches.Patch(color=dom_colors[i], label=f"→ {taxa_ids[i]}")
        for i in unique_dominants
    ]
    lp_handles, lp_labels = ax.get_legend_handles_labels()
    ax.legend(handles=dom_patches + lp_handles,
              fontsize=7, loc="upper right", title="attractor / LP", title_fontsize=7)

    ax.set_xlabel(f"{args.taxon_x} abundance", fontsize=11)
    ax.set_ylabel(f"{args.taxon_y} abundance", fontsize=11)
    ax.set_xlim(-0.02, max_sum + 0.03)
    ax.set_ylim(-0.02, max_sum + 0.03)
    ax.set_aspect("equal")
    ax.set_title(f"Full phase portrait — {n_starts} surrogate trajectories\n"
                 f"{args.taxon_x} vs {args.taxon_y}  "
                 f"(○ start, ★ end; LP profiles in colour)", fontsize=10)

    path = args.plots_dir / "full_phase_portrait.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 2: full_basin_map.png ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

    for j, dom_idx in enumerate(unique_dominants):
        mask  = final_dominant_idx == dom_idx
        color = dom_colors[dom_idx]
        ax.scatter(start_ax[mask], start_ay[mask],
                   color=color, s=35, edgecolors="none", alpha=0.85, zorder=2,
                   label=taxa_ids[dom_idx])

    ax.plot([0, max_sum], [max_sum, 0], color="grey",
            linewidth=0.7, linestyle="--", zorder=1)

    ax.set_xlabel(f"Starting {args.taxon_x} abundance", fontsize=11)
    ax.set_ylabel(f"Starting {args.taxon_y} abundance", fontsize=11)
    ax.set_xlim(-0.01, max_sum + 0.02)
    ax.set_ylim(-0.01, max_sum + 0.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right", title="Final dominant", title_fontsize=8)
    ax.set_title(f"Basin of attraction map\n"
                 f"{args.taxon_x} vs {args.taxon_y}", fontsize=11)

    path = args.plots_dir / "full_basin_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAll figures written to {args.plots_dir}")
    for dom_idx in unique_dominants:
        count = (final_dominant_idx == dom_idx).sum()
        print(f"  {taxa_ids[dom_idx]:<6} dominates {count:>4}/{n_starts} "
              f"({100*count/n_starts:.1f}%) trajectories")


if __name__ == "__main__":
    main()
