"""
Full phase portrait via surrogate simulation from a triangular grid of
(I91, I157) starting conditions.

The remaining four taxa (I44, I89, I78, I49) share the leftover abundance
equally:  x_other = (1 - x_I91 - x_I157) / 4

Produces in plots/mcsm/:
  full_phase_portrait.png  — all trajectories in I157 vs I91 space,
                             coloured by final dominant taxon
  full_basin_map.png       — 2-D scatter of starting (I157, I91) coloured
                             by which attractor the trajectory converges to
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

# ── parameters ────────────────────────────────────────────────────────────────
CHECKPOINT = REPO_ROOT / "trained_models" / "mcsm_community_input-6_output-6_hidden-64.pt"
LP_TRAJ    = REPO_ROOT / "results" / "mcsm_trajectories.npz"
PLOTS      = REPO_ROOT / "plots" / "mcsm"
PLOTS.mkdir(parents=True, exist_ok=True)

GRID_N    = 25    # number of grid points per axis (→ ~GRID_N²/2 valid starts)
N_STEPS   = 80
DT        = 0.1
MIN_OTHER = 1e-3  # minimum abundance for each of the 4 background taxa

# ── taxa layout ───────────────────────────────────────────────────────────────
# taxa_ids = ['I44', 'I89', 'I157', 'I78', 'I91', 'I49']
TAXA_IDS  = ['I44', 'I89', 'I157', 'I78', 'I91', 'I49']
IDX_I157  = TAXA_IDS.index('I157')   # 2
IDX_I91   = TAXA_IDS.index('I91')    # 4
N_TAXA    = len(TAXA_IDS)
N_OTHER   = N_TAXA - 2               # 4 background taxa

MIN_ABUND = 1e-8


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def simulate(model_nn, x_scaler, y_scaler, x0, n_steps, dt):
    """Return trajectory array (n_steps+1, n_taxa)."""
    traj = np.full((n_steps + 1, N_TAXA), np.nan)
    traj[0] = x0
    x = x0.copy()
    for s in range(n_steps):
        mu = surrogate_predict(model_nn, x_scaler, y_scaler, x).flatten()
        x  = euler_step(x, mu, dt)
        traj[s + 1] = x
    return traj


def build_x0(i91, i157):
    """Construct abundance vector from (I91, I157) pair; distribute remainder evenly."""
    other = (1.0 - i91 - i157) / N_OTHER
    x0 = np.full(N_TAXA, other)
    x0[IDX_I91]  = i91
    x0[IDX_I157] = i157
    return x0


# ── build triangular grid ─────────────────────────────────────────────────────
# max sum of I91+I157 so each background taxon gets at least MIN_OTHER
max_sum = 1.0 - N_OTHER * MIN_OTHER

vals     = np.linspace(0.0, max_sum, GRID_N)
starts   = []
for i91 in vals:
    for i157 in vals:
        if i91 + i157 <= max_sum:
            starts.append((i91, i157))

n_starts = len(starts)
print(f"Grid: {GRID_N}×{GRID_N} → {n_starts} valid starting points")

# ── load surrogate ────────────────────────────────────────────────────────────
model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(CHECKPOINT, SurrogateNN)
print(f"Checkpoint loaded: {CHECKPOINT.name}")

# ── run all trajectories ──────────────────────────────────────────────────────
t0 = time.time()
all_trajs      = []   # (n_starts, N_STEPS+1, N_TAXA)
final_i91      = np.zeros(n_starts)
final_i157     = np.zeros(n_starts)
start_i91_arr  = np.array([s[0] for s in starts])
start_i157_arr = np.array([s[1] for s in starts])

for k, (i91, i157) in enumerate(starts):
    x0   = build_x0(i91, i157)
    traj = simulate(model_nn, x_scaler, y_scaler, x0, N_STEPS, DT)
    all_trajs.append(traj)
    final_i91[k]  = traj[-1, IDX_I91]
    final_i157[k] = traj[-1, IDX_I157]

print(f"Simulated {n_starts} trajectories in {time.time()-t0:.2f} s")


# ── determine attractor for each trajectory ───────────────────────────────────
# 1 = I91 dominant, 0 = I157 dominant (at the end of the trajectory)
i91_dominant = final_i91 > final_i157

# fraction of I91 in the I91+I157 subspace at the end (continuous, 0→1)
final_sum = final_i91 + final_i157
final_i91_frac = np.where(final_sum > 0, final_i91 / final_sum, 0.5)


# ── Figure 1: full_phase_portrait.png ─────────────────────────────────────────
# Show all grid trajectories in I157 vs I91 space; colour by attractor.
# Also overlay the 5 original LP trajectories for reference.

lp = np.load(LP_TRAJ, allow_pickle=True)
lp_trajs      = lp["trajectories"]
lp_names      = list(lp["profile_names"])
lp_prof_colors = plt.get_cmap("Set1")(np.linspace(0, 0.8, len(lp_names)))

cmap_attractor = mcolors.ListedColormap(
    [plt.get_cmap("Blues")(0.65), plt.get_cmap("Reds")(0.65)]
)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

# grid trajectories — thin, semi-transparent
for k, traj in enumerate(all_trajs):
    color = "tab:red" if i91_dominant[k] else "tab:blue"
    ax.plot(traj[:, IDX_I157], traj[:, IDX_I91],
            color=color, linewidth=0.5, alpha=0.35, zorder=1)
    # start marker
    ax.scatter(traj[0, IDX_I157], traj[0, IDX_I91],
               color=color, s=8, alpha=0.6, zorder=2, linewidths=0)

# original LP trajectories — thick, dark, named
for p, name in enumerate(lp_names):
    traj = lp_trajs[p]
    c    = lp_prof_colors[p]
    ax.plot(traj[:, IDX_I157], traj[:, IDX_I91],
            color=c, linewidth=2.2, linestyle="-", zorder=4, label=name)
    ax.scatter(traj[0,  IDX_I157], traj[0,  IDX_I91],
               color=c, marker="o", s=60, zorder=5)
    ax.scatter(traj[-1, IDX_I157], traj[-1, IDX_I91],
               color=c, marker="*", s=130, zorder=5)

# basin-boundary diagonal guide
ax.plot([0, 1], [1, 0], color="grey", linewidth=0.7, linestyle="--", zorder=0)

from matplotlib.lines import Line2D
legend_extra = [
    Line2D([0], [0], color="tab:red",  linewidth=1.2, label="Grid → I91 dominant"),
    Line2D([0], [0], color="tab:blue", linewidth=1.2, label="Grid → I157 dominant"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_extra, fontsize=8, loc="upper right")

ax.set_xlabel("I157 abundance", fontsize=11)
ax.set_ylabel("I91 abundance", fontsize=11)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
ax.set_aspect("equal")
ax.set_title(f"Full phase portrait — {n_starts} surrogate trajectories\n"
             f"(○ = start, ★ = end; LP originals shown in colour)", fontsize=11)

path = PLOTS / "full_phase_portrait.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


# ── Figure 2: full_basin_map.png ──────────────────────────────────────────────
# 2-D scatter coloured by final I91 fraction in the I91+I157 subspace.
# Gives a continuous view of the basin boundary.

fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

sc = ax.scatter(
    start_i157_arr, start_i91_arr,
    c=final_i91_frac,
    cmap="RdBu_r",     # red=I91, blue=I157
    vmin=0.0, vmax=1.0,
    s=40, edgecolors="none", alpha=0.9, zorder=2,
)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Final I91 / (I91 + I157)", fontsize=10)

ax.plot([0, max_sum], [max_sum, 0], color="grey",
        linewidth=0.7, linestyle="--", zorder=1)

ax.set_xlabel("Starting I157 abundance", fontsize=11)
ax.set_ylabel("Starting I91 abundance", fontsize=11)
ax.set_xlim(-0.01, max_sum + 0.02)
ax.set_ylim(-0.01, max_sum + 0.02)
ax.set_aspect("equal")
ax.set_title("Basin of attraction map\n"
             "(red = I91 wins, blue = I157 wins)", fontsize=11)

path = PLOTS / "full_basin_map.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")


print(f"\nAll figures written to {PLOTS}")
print(f"I91-dominant: {i91_dominant.sum()}/{n_starts}  "
      f"I157-dominant: {(~i91_dominant).sum()}/{n_starts}")
