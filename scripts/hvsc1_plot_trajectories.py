"""
Plot trajectory and landscape results from hvsc1_simulate.py /
hvsc1_ode_surrogate.py.

Taxa IDs and the phase-portrait axes are loaded from the trajectory files,
so this script works for any number of taxa.

Produces figures in plots/hvsc1/:
  trajectories.png             — LP abundance time series, one panel per profile
  phase_portrait.png           — taxon-x vs taxon-y projection of all LP trajectories
  community_growth_rate.png    — LP community growth rate vs time
  landscape_distributions.png  — per-taxon growth rate histograms
  landscape_community_gr.png   — community growth rate vs two dominant taxa
  comparison.png               — LP vs surrogate overlay (one panel per profile)
                                 [written only if hvsc1_surrogate_trajectories.npz exists]

Usage:
    python scripts/hvsc1_plot_trajectories.py \\
        --trajectories results/hvsc1_trajectories.npz \\
        --landscape results/hvsc1_landscape.npz \\
        --surrogate results/hvsc1_surrogate_trajectories.npz \\   # optional
        --phase-taxon-x 100 \\   # taxon ID string; default = most abundant at final state
        --phase-taxon-y 161      # taxon ID string; default = 2nd most abundant
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_trajectories.npz")
    p.add_argument("--landscape", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_landscape.npz")
    p.add_argument("--surrogate", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_surrogate_trajectories.npz")
    p.add_argument("--plots-dir", type=Path,
                   default=REPO_ROOT / "plots" / "hvsc1")
    p.add_argument("--phase-taxon-x", type=str, default=None,
                   help="Taxon ID for x-axis of phase portrait (default: most abundant at final)")
    p.add_argument("--phase-taxon-y", type=str, default=None,
                   help="Taxon ID for y-axis of phase portrait (default: 2nd most abundant at final)")
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    traj_data = np.load(args.trajectories, allow_pickle=True)
    land_data = np.load(args.landscape,    allow_pickle=True)

    taxa_ids      = list(traj_data["taxa_ids"])
    profile_names = list(traj_data["profile_names"])
    trajectories  = traj_data["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    gr_trajs      = traj_data["gr_trajs"]       # (n_profiles, n_steps)
    dt            = float(traj_data["dt"])
    n_steps       = int(traj_data["n_steps"])
    t_axis        = np.arange(n_steps + 1) * dt

    n_profiles = len(profile_names)
    n_taxa     = len(taxa_ids)

    print(f"Loaded {n_profiles} trajectories, {n_taxa} taxa, {n_steps} steps")

    COLORS      = plt.get_cmap("tab20")(np.linspace(0, 1.0, max(n_taxa, 2)))
    TAX_COLOR   = dict(zip(taxa_ids, COLORS))
    PROF_COLORS = plt.get_cmap("Set1")(np.linspace(0, 0.8, n_profiles))

    # Determine phase-portrait taxa from final state of "original" profile if not given
    if "original" in profile_names:
        ref_idx = profile_names.index("original")
    else:
        ref_idx = 0
    ref_final = trajectories[ref_idx, -1, :]
    sorted_by_abund = np.argsort(ref_final)[::-1]

    taxon_x = args.phase_taxon_x if args.phase_taxon_x else taxa_ids[sorted_by_abund[0]]
    taxon_y = args.phase_taxon_y if args.phase_taxon_y else taxa_ids[sorted_by_abund[1]]
    idx_x   = taxa_ids.index(taxon_x)
    idx_y   = taxa_ids.index(taxon_y)
    print(f"Phase portrait axes: {taxon_x} (x) vs {taxon_y} (y)")

    # Surrogate trajectories (optional)
    has_surrogate = args.surrogate.exists()
    if has_surrogate:
        sur_data  = np.load(args.surrogate, allow_pickle=True)
        sur_trajs = sur_data["trajectories"]
        sur_gr    = sur_data["gr_trajs"]
        print(f"Surrogate trajectories loaded from {args.surrogate}")
    else:
        print("No surrogate trajectory file found — skipping comparison plot")

    # ── Figure 1: Abundance trajectories (LP) ─────────────────────────────────
    section("Figure 1: Abundance trajectories")
    ncols = 3
    nrows = math.ceil(n_profiles / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             constrained_layout=True, sharey=True)
    axes = np.array(axes).flatten()

    for p, name in enumerate(profile_names):
        ax = axes[p]
        traj = trajectories[p]
        for j, taxon in enumerate(taxa_ids):
            ax.plot(t_axis, traj[:, j], color=TAX_COLOR[taxon],
                    linewidth=1.5, label=taxon)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Relative abundance")
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

    if n_taxa <= 20:
        axes[n_profiles - 1].legend(fontsize=7, loc="upper right",
                                    title="taxon", title_fontsize=7,
                                    ncol=max(1, n_taxa // 10))
    for ax in axes[n_profiles:]:
        ax.set_visible(False)

    fig.suptitle(f"Abundance dynamics — LP ({n_profiles} starting conditions)", fontsize=12)
    path = args.plots_dir / "trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 2: Phase portrait ───────────────────────────────────────────────
    section(f"Figure 2: Phase portrait ({taxon_x} vs {taxon_y})")
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    for p, name in enumerate(profile_names):
        traj = trajectories[p]
        ax.plot(traj[:, idx_x], traj[:, idx_y],
                color=PROF_COLORS[p], linewidth=1.5, label=name, zorder=2)
        ax.scatter(traj[0,  idx_x], traj[0,  idx_y],
                   color=PROF_COLORS[p], marker="o", s=60, zorder=3)
        ax.scatter(traj[-1, idx_x], traj[-1, idx_y],
                   color=PROF_COLORS[p], marker="*", s=120, zorder=3)

    ax.set_xlabel(f"{taxon_x} abundance")
    ax.set_ylabel(f"{taxon_y} abundance")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Phase portrait — {taxon_x} vs {taxon_y}\n(○ = start, ★ = end)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.plot([0, 1], [1, 0], color="grey", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_aspect("equal")

    path = args.plots_dir / "phase_portrait.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 3: Community growth rate ───────────────────────────────────────
    section("Figure 3: Community growth rate")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    t_gr = np.arange(n_steps) * dt + dt / 2

    for p, name in enumerate(profile_names):
        ax.plot(t_gr, gr_trajs[p], color=PROF_COLORS[p],
                linewidth=1.5, label=name)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Community growth rate")
    ax.set_title("Community growth rate during simulation", fontsize=11)
    ax.legend(fontsize=8)

    path = args.plots_dir / "community_growth_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 4: Landscape — growth rate distributions ───────────────────────
    section("Figure 4: Landscape distributions")
    mu_land = land_data["mu"]
    gr_land = land_data["community_gr"]

    n_cols_land = math.ceil(math.sqrt(n_taxa))
    n_rows_land = math.ceil(n_taxa / n_cols_land)
    fig, axes = plt.subplots(n_rows_land, n_cols_land,
                             figsize=(3.5 * n_cols_land, 3.0 * n_rows_land),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for j, taxon in enumerate(taxa_ids):
        ax = axes[j]
        ax.hist(mu_land[:, j], bins=20, color=TAX_COLOR[taxon],
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Growth rate (h⁻¹)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(taxon, fontsize=9)
        med = np.median(mu_land[:, j])
        ax.axvline(med, color="black", linestyle="--", linewidth=1,
                   label=f"med={med:.3f}")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[n_taxa:]:
        ax.set_visible(False)

    fig.suptitle("Per-taxon growth rate distribution across Dirichlet samples", fontsize=11)
    path = args.plots_dir / "landscape_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 5: Community growth rate vs dominant taxa ──────────────────────
    section(f"Figure 5: Community growth rate vs {taxon_x} and {taxon_y}")
    x_land = land_data["x"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, (idx, taxon) in zip(axes, [(idx_x, taxon_x), (idx_y, taxon_y)]):
        sc = ax.scatter(x_land[:, idx], gr_land,
                        c=x_land[:, idx], cmap="viridis",
                        s=18, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=f"{taxon} abundance")
        ax.set_xlabel(f"{taxon} abundance")
        ax.set_ylabel("Community growth rate")
        ax.set_title(f"Community growth rate vs {taxon} abundance", fontsize=10)

    fig.suptitle("Landscape: community growth rate sensitivity", fontsize=11)
    path = args.plots_dir / "landscape_community_gr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 6: LP vs Surrogate comparison ──────────────────────────────────
    if has_surrogate:
        section("Figure 6: LP vs Surrogate comparison")
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                                 constrained_layout=True, sharey=True)
        axes = np.array(axes).flatten()

        for p, name in enumerate(profile_names):
            ax = axes[p]
            lp_traj  = trajectories[p]
            sur_traj = sur_trajs[p]

            for j, taxon in enumerate(taxa_ids):
                color = TAX_COLOR[taxon]
                ax.plot(t_axis, lp_traj[:, j],
                        color=color, linewidth=2.0, linestyle="-")
                ax.plot(t_axis, sur_traj[:, j],
                        color=color, linewidth=1.2, linestyle="--")

            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Relative abundance")
            ax.set_ylim(-0.02, 1.05)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        legend_elements = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-",  label="LP"),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Surrogate"),
        ]
        if n_taxa <= 20:
            legend_elements = [
                Line2D([0], [0], color=TAX_COLOR[t], linewidth=1.5, label=t)
                for t in taxa_ids
            ] + legend_elements
        axes[n_profiles - 1].legend(handles=legend_elements, fontsize=7,
                                    loc="upper right", ncol=max(1, n_taxa // 10))
        for ax in axes[n_profiles:]:
            ax.set_visible(False)

        fig.suptitle("LP (—) vs Surrogate (- -) abundance trajectories", fontsize=12)
        path = args.plots_dir / "comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll figures written to {args.plots_dir}")


if __name__ == "__main__":
    main()
