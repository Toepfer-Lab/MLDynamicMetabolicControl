#!/usr/bin/env python
"""
iJO1366 piecewise benchmark visualisation.

Loads results/ijo1366_benchmark_piecewise_lp.npz and
       results/ijo1366_benchmark_piecewise_surrogate.npz
and writes three figures to plots/ijo1366_piecewise_benchmark/:

  objective_comparison.png         — 1×3: LP ethanol | surrogate ethanol | |error|
  timing_summary.png               — total wall time + full per-sim speedup distribution
  timing_summary_filtered.png      — same metrics recalculated after removing timeout-
                                     inflated outliers (MAD-based, upper-tail only)

Usage:
    python scripts/ijo1366_plot_piecewise_benchmark.py
    python scripts/ijo1366_plot_piecewise_benchmark.py --outdir plots/custom_dir
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import PLOT_DIR, ensure_output_dirs


# ── helpers ───────────────────────────────────────────────────────────────────

def _mad_inlier_mask(values, k=2.5):
    """True for values at most k·MAD above the median (upper-tail outlier removal)."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return values <= med + k * mad


def _set_two_line_title(ax, upper, lower, upper_fs=11, lower_fs=9, pad=22):
    """Set a subplot title with a larger upper line and a smaller lower explainer line."""
    ax.set_title(upper, fontsize=upper_fs, pad=pad)
    ax.text(0.5, 1.0, lower, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=lower_fs)


def _heatmap(ax, ackr_vals, ldh_vals, Z, title, cmap, vmin=None, vmax=None,
             cbar_label="", mark_best=False):
    mesh = ax.pcolormesh(ackr_vals, ldh_vals, Z.T,
                         cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    plt.colorbar(mesh, ax=ax, label=cbar_label)
    if mark_best:
        best = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter(ackr_vals[best[0]], ldh_vals[best[1]],
                   marker="*", s=120, color="white", zorder=5)
    ax.set_xlabel("ACKr (mmol/gDW/h)")
    ax.set_ylabel("LDH_D (mmol/gDW/h)")
    ax.set_title(title)


def plot_objective_comparison(lp, sur, outdir):
    obj_lp  = lp["objectives"]
    obj_sur = sur["objectives"]
    err     = np.abs(obj_lp - obj_sur)

    vmin = min(obj_lp.min(), obj_sur.min())
    vmax = max(obj_lp.max(), obj_sur.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        "Two Stage Ethanol Production Optimisation",
        fontsize=13,
    )

    ackr = lp["ackr_vals"]
    ldh  = lp["ldh_vals"]

    _heatmap(axes[0], ackr, ldh, obj_lp,
             "FBA  (ground truth)", "viridis", vmin, vmax,
             cbar_label="Etoh [mmol/L]", mark_best=True)
    _heatmap(axes[1], ackr, ldh, obj_sur,
             "Surrogate", "viridis", vmin, vmax,
             cbar_label="Etoh [mmol/L]", mark_best=True)
    _heatmap(axes[2], ackr, ldh, err,
             "|LP − Surrogate|", "Reds", 0.0,
             cbar_label="Absolute error [mmol/L]")

    best_lp  = np.unravel_index(np.nanargmax(obj_lp),  obj_lp.shape)
    best_sur = np.unravel_index(np.nanargmax(obj_sur), obj_sur.shape)

    _set_two_line_title(
        axes[0],
        f"FBA  —  best {obj_lp[best_lp]:.3f} mmol/L",
        f"(ACKr={ackr[best_lp[0]]:.2f}, LDH_D={ldh[best_lp[1]]:.2f})",
    )
    _set_two_line_title(
        axes[1],
        f"Surrogate  —  best {obj_sur[best_sur]:.3f} mmol/L",
        f"(ACKr={ackr[best_sur[0]]:.2f}, LDH_D={ldh[best_sur[1]]:.2f})",
    )
    _set_two_line_title(
        axes[2],
        "Relative Prediction Error",
        f"|LP − Surrogate|  —  max {err.max():.3f}, mean {err.mean():.3f} mmol/L",
    )

    path = outdir / "objective_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_timing_summary(lp, sur, outdir):
    total_lp  = float(lp["total_time"])
    total_sur = float(sur["total_time"])
    speedup   = total_lp / total_sur

    sim_lp  = lp["sim_times"].flatten()
    sim_sur = sur["sim_times"].flatten()
    per_sim_speedup = sim_lp / sim_sur

    # remove outliers from histogram (most likely due to the wait 5 seconds after solver stuck on infeasible input)
    per_sim_speedup = per_sim_speedup[per_sim_speedup < np.percentile(per_sim_speedup, 95)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("2D Optimisation — Wall-time comparison", fontsize=11)

    # ── left: total time bars ─────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(["LP", "Surrogate"], [total_lp, total_sur],
                  color=["steelblue", "darkorange"], width=0.45)
    ax.bar_label(bars,
                 labels=[f"{total_lp:.0f} s", f"{total_sur:.1f} s"],
                 padding=4, fontsize=9)
    ax.set_ylabel("Total wall time (s)")
    ax.set_title(f"Total Simulation Time: {speedup:.0f}× Speedup")
    ax.set_yscale("log")
    ax.set_ylim(1, total_lp * 3)

    # ── right: per-sim speedup distribution ──────────────────────────────────
    ax = axes[1]
    ax.hist(per_sim_speedup, bins=15, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(per_sim_speedup.mean(), color="darkorange", linestyle="--",
               label=f"mean {per_sim_speedup.mean():.0f}×")
    ax.set_xlabel("Speedup factor (LP time / surrogate time) per simulation")
    ax.set_ylabel("Count")
    ax.set_title("Per-simulation speedup distribution")
    ax.legend(fontsize=9)

    # print stats
    calls_lp  = int(lp["lp_calls"].mean())
    calls_sur = int(sur["nn_calls"].mean())
    print(f"  LP total time   : {total_lp:.1f} s  ({total_lp/60:.1f} min)")
    print(f"  Surrogate total : {total_sur:.2f} s")
    print(f"  Speedup (total) : {speedup:.1f}×")
    print(f"  Mean LP calls / sim : {calls_lp}  |  Mean NN calls / sim : {calls_sur}")
    print(f"  Per-sim speedup : {per_sim_speedup.mean():.0f}× (mean)  "
          f"{per_sim_speedup.min():.0f}× – {per_sim_speedup.max():.0f}× (range)")

    path = outdir / "timing_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_timing_summary_filtered(lp, sur, outdir, mad_k=2.5):
    """Same layout as plot_timing_summary but recalculates all metrics on inlier sims only."""
    sim_lp  = lp["sim_times"].flatten()
    sim_sur = sur["sim_times"].flatten()
    per_sim_speedup = sim_lp / sim_sur

    inlier    = _mad_inlier_mask(per_sim_speedup, mad_k)
    n_total   = len(per_sim_speedup)
    n_inlier  = int(inlier.sum())
    n_outlier = n_total - n_inlier

    sp_in        = per_sim_speedup[inlier]
    total_lp_in  = float(sim_lp[inlier].sum())
    total_sur_in = float(sim_sur[inlier].sum())
    speedup_in   = total_lp_in / total_sur_in

    med = np.median(per_sim_speedup)
    mad = np.median(np.abs(per_sim_speedup - med))
    fence = med + mad_k * mad

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        f"2D Optimisation — Wall-time comparison (n={n_inlier}/{n_total})\n"
        f"LP solver.timeout outliers removed",
        
    )

    # ── left: bar chart on inlier totals only ────────────────────────────────
    ax = axes[0]
    bars = ax.bar(["LP", "Surrogate"], [total_lp_in, total_sur_in],
                  color=["steelblue", "darkorange"], width=0.45)
    ax.bar_label(bars,
                 labels=[f"{total_lp_in:.0f} s", f"{total_sur_in:.1f} s"],
                 padding=4, fontsize=9)
    ax.set_ylabel("Total wall time (s)")
    ax.set_title(f"Conservative Total Simulation Time: {speedup_in:.0f}× Speedup", fontsize=9)
    #ax.set_yscale("log")
    ax.set_ylim(1, total_lp_in * 3)
    ax.set_yscale("log")

    # ── right: filtered speedup histogram ────────────────────────────────────
    ax = axes[1]
    ax.hist(sp_in, bins=12, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(sp_in.mean(), color="darkorange", linestyle="--",
               label=f"mean {sp_in.mean():.0f}×")
    ax.set_xlabel("Speedup factor (LP time / surrogate time) per simulation")
    ax.set_ylabel("Count")
    ax.set_title("Per-simulation speedup (timeout outliers removed)", fontsize=9)
    ax.legend(fontsize=9)

    print(f"\n  ── Filtered timing summary (n={n_inlier}/{n_total} inlier sims) ──")
    print(f"  Outlier fence      : {fence:.0f}×  (median={med:.0f}×, MAD={mad:.1f}, k={mad_k})")
    print(f"  Sims excluded      : {n_outlier}  (LP-timeout-inflated)")
    print(f"  LP total (inliers) : {total_lp_in:.1f} s  ({total_lp_in/60:.1f} min)")
    print(f"  Surr total (inliers): {total_sur_in:.2f} s")
    print(f"  Speedup (inliers)  : {speedup_in:.1f}×")
    print(f"  Per-sim speedup    : {sp_in.mean():.0f}× (mean)  "
          f"{sp_in.min():.0f}× – {sp_in.max():.0f}× (range)")

    path = outdir / "timing_summary_filtered.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lp-results", type=Path,
                   default=REPO_ROOT / "results" / "ijo1366_benchmark_piecewise_lp.npz")
    p.add_argument("--surrogate-results", type=Path,
                   default=REPO_ROOT / "results" / "ijo1366_benchmark_piecewise_surrogate.npz")
    p.add_argument("--outdir", type=Path,
                   default=PLOT_DIR / "ijo1366_piecewise_benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    args.outdir.mkdir(parents=True, exist_ok=True)

    lp  = np.load(args.lp_results)
    sur = np.load(args.surrogate_results)

    print("=" * 66)
    print("  iJO1366 piecewise benchmark — plotting")
    print(f"  LP results  : {args.lp_results}")
    print(f"  Surr results: {args.surrogate_results}")
    print(f"  Output dir  : {args.outdir}")
    print("=" * 66)

    plot_objective_comparison(lp, sur, args.outdir)
    plot_timing_summary(lp, sur, args.outdir)
    plot_timing_summary_filtered(lp, sur, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
