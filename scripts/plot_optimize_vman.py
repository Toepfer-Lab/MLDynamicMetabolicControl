#!/usr/bin/env python
"""
Plot results from scripts/optimize_vman.py

Creates:
  1) Biomass evolution curves from logged trajectories
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

    # --- Plot 1: Biomass evolution during optimization (for entries that have full curves) ---
    # Some logging modes may store only final_biomass; handle both.
    full_entries = [e for e in logs if isinstance(e, dict) and ("t" in e) and ("biomass" in e)]

    if args.max_curves and args.max_curves > 0 and len(full_entries) > args.max_curves:
        # simple downsample: take evenly spaced subset
        idxs = np.linspace(0, len(full_entries) - 1, args.max_curves).astype(int)
        full_entries = [full_entries[i] for i in idxs]

    plt.figure(figsize=(8, 5))
    for e in full_entries:
        t = np.asarray(e["t"])
        b = np.asarray(e["biomass"])
        if t.size and b.size:
            plt.plot(t, b, alpha=0.35)

    plt.xlabel("Time [h]")
    plt.ylabel("Biomass")
    plt.title(f"Biomass evolution during optimization\n({results_path.stem})")
    plt.grid(True)
    biomass_plot_path = outdir / f"{results_path.stem}_biomass_trajectories.png"
    plt.tight_layout()
    plt.savefig(biomass_plot_path, dpi=200)

    # --- Plot 2: Optimized control profile ---
    opt_vman_values = np.asarray(res["opt_vman_values"])
    control_times = np.asarray(res["control_times"])

    plt.figure(figsize=(8, 4))
    # Step plot: N intervals means N values, and control_times has N+1 points
    plt.step(control_times[:-1], opt_vman_values, where="post")
    plt.xlabel("Time [h]")
    plt.ylabel("Optimized vman values")
    plt.title(f"Optimized Control Profile\n({results_path.stem})")
    plt.grid(True)
    vman_plot_path = outdir / f"{results_path.stem}_vman_profile.png"
    plt.tight_layout()
    plt.savefig(vman_plot_path, dpi=200)

    print(f"Saved biomass trajectories plot to {biomass_plot_path}")
    print(f"Saved vman profile plot to {vman_plot_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
