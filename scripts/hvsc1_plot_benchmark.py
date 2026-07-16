"""
Visualise LP vs surrogate benchmark results from hvsc1_benchmark.npz.
Mirrors scripts/mcsm_plot_benchmark.py exactly (same figure layout/formatting)
so the two are directly comparable.

Produces plots/hvsc1/benchmark_walltime.png:
  Two-panel figure showing wall time comparison on a log scale.
  Panel 1 — per-trajectory wall time (LP measured vs Surrogate measured)
  Panel 2 — total wall time for N_LP and N_SUR trajectory batches
             (LP extrapolated from per-trajectory average; surrogate measured)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS   = REPO_ROOT / "results"
PLOTS     = REPO_ROOT / "plots" / "hvsc1"
PLOTS.mkdir(parents=True, exist_ok=True)

data = np.load(RESULTS / "hvsc1_benchmark.npz", allow_pickle=True)

# ── unpack ────────────────────────────────────────────────────────────────────
n_lp    = int(data["n_lp"])
n_sur   = int(data["n_sur"])
n_steps = int(data["n_steps"])
dt      = float(data["dt"])

lp_traj_times     = data["lp_traj_times"]          # (n_lp,) seconds
lp_step_times     = data["lp_step_times"]          # (n_lp, n_steps) seconds
sur_times_matched = data["sur_traj_times_matched"] # (n_lp,)
sur_times_large   = data["sur_traj_times_large"]   # (n_sur,)
sur_batch_total   = float(data["sur_batch_total"])

mean_lp_per_traj  = lp_traj_times.mean()
mean_lp_per_step  = lp_step_times.mean()
mean_sur_per_traj = sur_times_matched.mean()
mean_sur_per_step = mean_sur_per_traj / n_steps

speedup_traj = mean_lp_per_traj / mean_sur_per_traj
speedup_step = mean_lp_per_step / mean_sur_per_step

extrap_lp_n_lp  = n_lp  * mean_lp_per_traj
extrap_lp_n_sur = n_sur  * mean_lp_per_traj
sur_total_n_lp  = sur_times_matched.sum()

print(f"Mean LP   / trajectory : {mean_lp_per_traj:.2f} s")
print(f"Mean LP   / step       : {mean_lp_per_step*1000:.1f} ms")
print(f"Mean Sur  / trajectory : {mean_sur_per_traj*1000:.3f} ms")
print(f"Mean Sur  / step       : {mean_sur_per_step*1000:.4f} ms")
print(f"Speedup (per traj)     : {speedup_traj:.0f}×")
print(f"Speedup (per step)     : {speedup_step:.0f}×")


# ── colour palette ────────────────────────────────────────────────────────────
C_LP  = "#E07B54"   # warm orange
C_SUR = "#5B8DB8"   # steel blue


def _annotate_bar(ax, bar, value, unit="s", fmt=None):
    """Place a label above a bar with the value and unit."""
    if fmt is None:
        if value < 0.001:
            fmt = f"{value*1000:.3f} m{unit}"
        elif value < 1.0:
            fmt = f"{value*1000:.1f} m{unit}"
        else:
            fmt = f"{value:.2f} {unit}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.35,
        fmt,
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )


# ── Figure: two-panel comparison ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# ── Panel 1: per-trajectory wall time ─────────────────────────────────────────
ax = axes[0]
x_pos  = [0, 1]
height = [mean_lp_per_traj, mean_sur_per_traj]
labels = ["LP solver", "Surrogate"]
colors = [C_LP, C_SUR]

bars = ax.bar(x_pos, height, width=0.5, color=colors, edgecolor="white", linewidth=0.8)

# error bars for LP (std across N_LP trajectories)
ax.errorbar(0, mean_lp_per_traj, yerr=lp_traj_times.std(),
            fmt="none", color="black", capsize=5, linewidth=1.5)
ax.errorbar(1, mean_sur_per_traj, yerr=sur_times_matched.std(),
            fmt="none", color="black", capsize=5, linewidth=1.5)

for bar, val in zip(bars, height):
    _annotate_bar(ax, bar, val)

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Wall time per trajectory (s)", fontsize=11)
ax.set_title(f"Per-trajectory wall time\n({n_steps} steps × dt={dt}h = {n_steps*dt:.0f}h simulated)",
             fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda y, _: f"{y:.3f} s" if y < 1 else f"{y:.0f} s"
))
ax.set_ylim(top=max(height) * 10)

# speedup annotation
ax.text(0.97, 0.97,
        f"Speedup: {speedup_traj:.0f}×",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

# ── Panel 2: total wall time for two batch sizes ──────────────────────────────
ax = axes[1]

# groups: [n_lp trajectories, n_sur trajectories]
batch_labels = [f"{n_lp} trajectories\n(measured)", f"{n_sur} trajectories\n(extrapolated LP)"]
lp_heights   = [extrap_lp_n_lp,  extrap_lp_n_sur]
sur_heights  = [sur_total_n_lp,  sur_batch_total]

width  = 0.3
x_pos  = np.arange(len(batch_labels))

bars_lp  = ax.bar(x_pos - width/2, lp_heights,  width, color=C_LP,  edgecolor="white",
                   linewidth=0.8, label="LP solver (extrapolated)")
bars_sur = ax.bar(x_pos + width/2, sur_heights, width, color=C_SUR, edgecolor="white",
                   linewidth=0.8, label="Surrogate (measured)")

for bar, val in zip(bars_lp, lp_heights):
    label = (f"{val:.0f} s" if val < 3600
             else f"{val/3600:.1f} h")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.35,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_LP)

for bar, val in zip(bars_sur, sur_heights):
    label = (f"{val*1000:.1f} ms" if val < 1.0 else f"{val:.2f} s")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.35,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_SUR)

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(batch_labels, fontsize=10)
ax.set_ylabel("Total wall time (s)", fontsize=11)
ax.set_title("Total wall time by batch size\n(LP extrapolated from per-trajectory average)",
             fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda y, _: (f"{y*1000:.1f} ms" if y < 1
                  else f"{y:.0f} s" if y < 3600
                  else f"{y/3600:.1f} h")
))
ax.set_ylim(top=max(max(lp_heights), max(sur_heights)) * 10)

fig.suptitle("LP solver vs Surrogate — wall time comparison (hvsc1, 27-taxon community)", fontsize=13)

path = PLOTS / "benchmark_walltime.png"
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"\nSaved: {path}")
