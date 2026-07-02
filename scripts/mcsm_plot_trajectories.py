"""
Plot trajectory and landscape results from mcsm_simulate.py /
mcsm_ode_surrogate.py.

Produces figures in plots/mcsm/:
  trajectories.png         — LP abundance time series, one panel per profile
  phase_portrait.png       — I157 vs I91 projection of all LP trajectories
  community_growth_rate.png — LP community growth rate vs time
  landscape_distributions.png — per-taxon growth rate histograms
  landscape_community_gr.png  — community growth rate vs dominant taxon
  comparison.png           — LP vs surrogate overlay (one panel per profile)
                             [written only if results/mcsm_surrogate_trajectories.npz exists]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results"
PLOTS     = REPO_ROOT / "plots" / "mcsm"
PLOTS.mkdir(parents=True, exist_ok=True)

traj_data = np.load(RESULTS / "mcsm_trajectories.npz", allow_pickle=True)
land_data = np.load(RESULTS / "mcsm_landscape.npz",    allow_pickle=True)

taxa_ids      = list(traj_data["taxa_ids"])
profile_names = list(traj_data["profile_names"])
trajectories  = traj_data["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
gr_trajs      = traj_data["gr_trajs"]       # (n_profiles, n_steps)
dt            = float(traj_data["dt"])
n_steps       = int(traj_data["n_steps"])
t_axis        = np.arange(n_steps + 1) * dt

n_profiles = len(profile_names)
n_taxa     = len(taxa_ids)

COLORS      = plt.get_cmap("tab10")(np.linspace(0, 0.9, n_taxa))
TAX_COLOR   = dict(zip(taxa_ids, COLORS))
PROF_COLORS = plt.get_cmap("Set1")(np.linspace(0, 0.8, n_profiles))

# load surrogate trajectories if available
sur_path = RESULTS / "mcsm_surrogate_trajectories.npz"
has_surrogate = sur_path.exists()
if has_surrogate:
    sur_data  = np.load(sur_path, allow_pickle=True)
    sur_trajs = sur_data["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    sur_gr    = sur_data["gr_trajs"]       # (n_profiles, n_steps)
    print(f"Surrogate trajectories loaded from {sur_path}")
else:
    print("No surrogate trajectory file found — skipping comparison plot")


# ── Figure 1: Abundance trajectories (LP) ────────────────────────────────────
ncols = 3
nrows = (n_profiles + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                         constrained_layout=True, sharey=True)
axes = np.array(axes).flatten()

for p, name in enumerate(profile_names):
    ax = axes[p]
    traj = trajectories[p]
    for j, taxon in enumerate(taxa_ids):
        ax.plot(t_axis, traj[:, j], color=TAX_COLOR[taxon],
                linewidth=1.8, label=taxon)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative abundance")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

axes[n_profiles - 1].legend(fontsize=8, loc="upper right",
                             title="taxon", title_fontsize=8)
for ax in axes[n_profiles:]:
    ax.set_visible(False)

fig.suptitle("Abundance dynamics — LP (5 starting conditions)", fontsize=12)
path = PLOTS / "trajectories.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 2: Phase portrait — I157 vs I91 ───────────────────────────────────
idx157 = taxa_ids.index("I157")
idx91  = taxa_ids.index("I91")

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

for p, name in enumerate(profile_names):
    traj = trajectories[p]
    ax.plot(traj[:, idx157], traj[:, idx91],
            color=PROF_COLORS[p], linewidth=1.5, label=name, zorder=2)
    ax.scatter(traj[0,  idx157], traj[0,  idx91],
               color=PROF_COLORS[p], marker="o", s=60, zorder=3)
    ax.scatter(traj[-1, idx157], traj[-1, idx91],
               color=PROF_COLORS[p], marker="*", s=120, zorder=3)

ax.set_xlabel("I157 abundance")
ax.set_ylabel("I91 abundance")
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
ax.set_title("Phase portrait — I157 vs I91\n(○ = start, ★ = end)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.plot([0, 1], [1, 0], color="grey", linewidth=0.5, linestyle="--", zorder=1)
ax.set_aspect("equal")

path = PLOTS / "phase_portrait.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 3: Community growth rate over time ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
t_gr = np.arange(n_steps) * dt + dt / 2

for p, name in enumerate(profile_names):
    ax.plot(t_gr, gr_trajs[p], color=PROF_COLORS[p],
            linewidth=1.5, label=name)

ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--",
           label="fraction=0.5 target")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Community growth rate")
ax.set_title("Community growth rate during simulation", fontsize=11)
ax.legend(fontsize=8)

path = PLOTS / "community_growth_rate.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 4: Landscape — growth rate distributions ──────────────────────────
mu_land = land_data["mu"]
gr_land = land_data["community_gr"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
axes = axes.flatten()

for j, taxon in enumerate(taxa_ids):
    ax = axes[j]
    ax.hist(mu_land[:, j], bins=25, color=TAX_COLOR[taxon],
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Growth rate (h⁻¹)")
    ax.set_ylabel("Count")
    ax.set_title(taxon, fontsize=10)
    med = np.median(mu_land[:, j])
    ax.axvline(med, color="black", linestyle="--", linewidth=1,
               label=f"median={med:.3f}")
    ax.legend(fontsize=8)

fig.suptitle("Per-taxon growth rate distribution across Dirichlet samples",
             fontsize=11)
path = PLOTS / "landscape_distributions.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 5: Community growth rate vs dominant taxon abundance ───────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
x_land = land_data["x"]

for ax, (idx, taxon) in zip(axes, [(idx157, "I157"), (idx91, "I91")]):
    sc = ax.scatter(x_land[:, idx], gr_land,
                    c=x_land[:, idx], cmap="viridis",
                    s=18, alpha=0.7)
    plt.colorbar(sc, ax=ax, label=f"{taxon} abundance")
    ax.set_xlabel(f"{taxon} abundance")
    ax.set_ylabel("Community growth rate")
    ax.set_title(f"Community growth rate vs {taxon} abundance", fontsize=10)

fig.suptitle("Landscape: community growth rate sensitivity to dominant taxa",
             fontsize=11)
path = PLOTS / "landscape_community_gr.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 6: LP vs Surrogate comparison (if available) ──────────────────────
if has_surrogate:
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
        Line2D([0], [0], color=TAX_COLOR[t], linewidth=1.5, label=t)
        for t in taxa_ids
    ] + [
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="-",  label="LP"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Surrogate"),
    ]
    axes[n_profiles - 1].legend(handles=legend_elements, fontsize=8,
                                loc="upper right", title_fontsize=8)
    for ax in axes[n_profiles:]:
        ax.set_visible(False)

    fig.suptitle("LP (—) vs Surrogate (- -) abundance trajectories", fontsize=12)
    path = PLOTS / "comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {path}")


print("\nAll figures written to", PLOTS)
