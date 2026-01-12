#!/usr/bin/env python
"""
Plot results from scripts/optimize_vman.py

Creates:
  1) Objective evolution curves from logged trajectories
  2) Optimized vman piecewise-constant control profile

Saves PNGs to the plots/ directory by default.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Non-interactive backend for cluster/headless runs
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Make local src importable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import PLOT_DIR, RESULTS_DIR, ensure_output_dirs  # noqa: E402



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Stem used by optimize_vman outputs (e.g. PYK_trained_model_input-1_output-4_hidden-4). "
             "If given, will load results/optimize_vman_<stem>.npz and ..._logs.npz",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to optimize_vman_*.npz (overrides --stem).",
    )
    p.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="Path to optimize_vman_*_logs.npz (overrides --stem).",
    )
    p.add_argument(
        "--max-curves",
        type=int,
        default=0,
        help="If >0, plot at most this many biomass curves (useful if logs are huge). 0 means plot all.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Try to display plots (usually not useful on a cluster). Always saves files regardless.",
    )
    p.add_argument(
    "--output-path",
    type=Path,
    default=None,
    help="Where to save the main results .npz (overrides default naming)",
    )
    p.add_argument(
        "--logs-path",
        type=Path,
        default=None,
        help="Where to save the trajectory logs .npz (overrides default naming)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    outdir = args.outdir or PLOT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    if args.stem is not None:
        results_path = RESULTS_DIR / f"optimize_vman_{args.stem}.npz"
        logs_path = RESULTS_DIR / f"optimize_vman_{args.stem}_logs.npz"
    else:
        results_path = args.results
        logs_path = args.logs

    if results_path is None or logs_path is None:
        raise ValueError("Provide either --stem OR both --results and --logs.")

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs file not found: {logs_path}")

    res = np.load(results_path, allow_pickle=True)
    logs_npz = np.load(logs_path, allow_pickle=True)

    # logs saved as np.array(dtype=object)
    logs = logs_npz["logs"].tolist()
    if not isinstance(logs, list):
        logs = list(logs)

    metadata = {}
    if "metadata" in res:
        try:
            metadata = res["metadata"].item()
        except ValueError:
            metadata = {}
    vman_id = metadata.get("vman", "vman")
    objective = "biomass"
    if "objective" in res:
        try:
            objective = str(res["objective"].item())
        except ValueError:
            objective = str(res["objective"])
    objective_label = objective.replace("_", " ").title()

    # --- Plot 1: Objective evolution during optimization (for entries that have full curves) ---
    # Some logging modes may store only biomass; handle both.
    full_entries = [
        e for e in logs
        if isinstance(e, dict) and ("t" in e) and ("objective_curve" in e)
    ]
    curve_key = "objective_curve"
    curve_label = objective_label
    if not full_entries:
        full_entries = [
            e for e in logs
            if isinstance(e, dict) and ("t" in e) and ("biomass" in e)
        ]
        curve_key = "biomass"
        curve_label = "Biomass"

    if args.max_curves and args.max_curves > 0 and len(full_entries) > args.max_curves:
        # simple downsample: take evenly spaced subset
        idxs = np.linspace(0, len(full_entries) - 1, args.max_curves).astype(int)
        full_entries = [full_entries[i] for i in idxs]

    plt.figure(figsize=(8, 5))
    for e in full_entries:
        t = np.asarray(e["t"])
        b = np.asarray(e[curve_key])
        if t.size and b.size:
            plt.plot(t, b, alpha=0.35)

    plt.xlabel("Time [h]")
    plt.ylabel(curve_label)
    plt.title(f"{curve_label} evolution during optimization\n({results_path.stem})")
    plt.grid(True)
    biomass_plot_path = outdir / f"{results_path.stem}_{curve_label.lower()}_trajectories.png"
    plt.tight_layout()
    plt.savefig(biomass_plot_path, dpi=200)

     # --- Plot 2: Optimized control profile + best objective (twin axis) ---
    opt_vman_values = np.asarray(res["opt_vman_values"], dtype=float)
    control_times = np.asarray(res["control_times"], dtype=float)

    # Ensure we have N values and N+1 time points
    if control_times.size != opt_vman_values.size + 1:
        raise ValueError(
            f"Expected control_times (N+1) and opt_vman_values (N). "
            f"Got control_times={control_times.size}, opt_vman_values={opt_vman_values.size}"
        )

    # Repeat last value so the final interval is shown up to t_end
    opt_vman_plot = np.r_[opt_vman_values, opt_vman_values[-1]]

    # Choose "best" objective curve from logs: max final value among entries with full curves
    best_entry = None
    best_final = -np.inf
    for e in logs:
        if isinstance(e, dict) and ("t" in e) and (curve_key in e):
            b = np.asarray(e[curve_key], dtype=float)
            if b.size and np.isfinite(b[-1]) and b[-1] > best_final:
                best_final = float(b[-1])
                best_entry = e

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    # Left axis: control (step)
    ax1.step(control_times, opt_vman_plot, where="post", linewidth=2, label=f"Optimized vman ({vman_id})")
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("vman (flux)")
    ax1.set_xlim(control_times[0], control_times[-1])
    ax1.grid(True)

    # Right axis: best objective (line)
    ax2 = ax1.twinx()
    if best_entry is not None:
        t_best = np.asarray(best_entry["t"], dtype=float)
        b_best = np.asarray(best_entry[curve_key], dtype=float)
        ax2.plot(
            t_best,
            b_best,
            "g",
            linewidth=2,
            label=f"{curve_label} (best, final={best_final:.3g})",
        )
    else:
        # Still create the axis, but annotate that no curve was available
        ax2.set_ylabel(curve_label)
        ax2.text(
            0.02, 0.95, f"No full {curve_label.lower()} curve found in logs",
            transform=ax2.transAxes, va="top"
        )

    if objective != "biomass":
        if "t_eval_points" in res:
            t_eval = np.asarray(res["t_eval_points"], dtype=float)
        else:
            t_eval = np.array([])

        biomass_curve = np.asarray(res["biomass"], dtype=float) if "biomass" in res else np.array([])
        glucose_curve = np.asarray(res["glucose"], dtype=float) if "glucose" in res else np.array([])

        if t_eval.size and biomass_curve.size:
            ax2.plot(
                t_eval,
                biomass_curve,
                color="tab:blue",
                linestyle="--",
                linewidth=1.5,
                label="Biomass (optimized)",
            )
        if t_eval.size and glucose_curve.size:
            ax2.plot(
                t_eval,
                glucose_curve,
                color="tab:orange",
                linestyle=":",
                linewidth=1.5,
                label="Glucose (optimized)",
            )

    if objective != "biomass" and (("biomass" in res) or ("glucose" in res)):
        ax2.set_ylabel("Concentration / objective")
    else:
        ax2.set_ylabel(curve_label)

    # Combined legend (handles from both axes)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title(f"Optimized Control + {curve_label}\n({results_path.stem})")
    plt.tight_layout()

    combo_plot_path = outdir / f"{results_path.stem}_control_plus_{curve_label.lower()}.png"
    plt.savefig(combo_plot_path, dpi=200)

    print(f"Saved {curve_label.lower()} trajectories plot to {biomass_plot_path}")
    print(f"Saved control+{curve_label.lower()} plot to {combo_plot_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
