#!/usr/bin/env python
"""
Plot results from scripts/greedily_optimize_vman.py (greedy single-cut optimizer)

Creates:
  1) Objective evolution curves from logged trajectories
  2) Optimized vman piecewise-constant control profile (variable boundaries)

Saves PNGs to the plots/ directory by default.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import PLOT_DIR, RESULTS_DIR, ensure_output_dirs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", type=str, default=None,
                   help="If given, loads results/greedy_optimize_vman_<stem>.npz and ..._logs.npz")
    p.add_argument("--latest", type=str, default=None, metavar="BASE_STEM",
                   help="Base stem (without timestamp) to find the most recently written matching results file")
    p.add_argument("--results", type=Path, default=None, help="Path to greedy_optimize_vman_*.npz")
    p.add_argument("--logs", type=Path, default=None, help="Path to greedy_optimize_vman_*_logs.npz")
    p.add_argument("--max-curves", type=int, default=0,
                   help="If >0, plot at most this many objective curves. 0 means plot all.")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory for plots (default: plots/).")
    p.add_argument("--show", action="store_true", help="Try to display plots (still saves files).")
    return p.parse_args()


def _as_object(res, key, default=None):
    if key not in res:
        return default
    try:
        return res[key].item()
    except Exception:
        return res[key]


def _step_xy_for_variable_boundaries(boundaries, values):
    """
    Convert piecewise-constant control (boundaries, values) to x/y suitable for plt.step(..., where="post").
    boundaries: len K+1
    values: len K
    We return:
      x = boundaries
      y = [values..., values[-1]] (len K+1)
    """
    boundaries = np.asarray(boundaries, dtype=float)
    values = np.asarray(values, dtype=float)
    if boundaries.size != values.size + 1:
        raise ValueError(f"Expected boundaries (K+1) and values (K). Got {boundaries.size} vs {values.size}")
    y = np.r_[values, values[-1]]
    return boundaries, y


def main():
    args = parse_args()
    ensure_output_dirs()

    outdir = args.outdir or PLOT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    if args.latest is not None:
        # --latest accepts a full path prefix (e.g. results/greedy/greedy_optimize_vman_...)
        base_path = Path(args.latest)
        search_dir = base_path.parent if base_path.parent != Path(".") else RESULTS_DIR
        base_stem = base_path.name
        candidates = sorted(
            search_dir.glob(f"{base_stem}_????????_??????.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates = [c for c in candidates if not c.stem.endswith("_logs")]
        if not candidates:
            raise FileNotFoundError(f"No timestamped results file found matching '{base_stem}_*' in {search_dir}")
        results_path = candidates[0]
        # Support both naming conventions:
        #   new: {base}_{ts}_logs.npz  (timestamp before _logs)
        #   old: {base}_logs_{ts}.npz  (timestamp after _logs, pre-fix runs)
        new_logs = results_path.parent / f"{results_path.stem}_logs{results_path.suffix}"
        ts_suffix = results_path.stem.rsplit("_", 2)  # [..., date, time]
        old_logs = results_path.parent / f"{base_stem}_logs_{'_'.join(ts_suffix[-2:])}{results_path.suffix}"
        logs_path = new_logs if new_logs.exists() else old_logs
    elif args.stem is not None:
        results_path = RESULTS_DIR / f"greedy_optimize_vman_{args.stem}.npz"
        logs_path = RESULTS_DIR / f"greedy_optimize_vman_{args.stem}_logs.npz"
    else:
        results_path = args.results
        logs_path = args.logs

    if results_path is None or logs_path is None:
        raise ValueError("Provide one of: --latest BASE_STEM, --stem STEM, or both --results and --logs.")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs file not found: {logs_path}")

    res = np.load(results_path, allow_pickle=True)
    logs_npz = np.load(logs_path, allow_pickle=True)

    logs = logs_npz["logs"].tolist()
    if not isinstance(logs, list):
        logs = list(logs)

    metadata = {}
    if "metadata" in res:
        try:
            metadata = res["metadata"].item()
        except Exception:
            metadata = {}

    vman_id = metadata.get("vman", "vman")
    objective = str(_as_object(res, "objective", "biomass"))
    objective_label = objective.replace("_", " ").title()

    # --- Plot 1: objective evolution curves from logs ---
    full_entries = [e for e in logs if isinstance(e, dict) and ("t" in e) and ("objective_curve" in e)]
    curve_key = "objective_curve"
    curve_label = objective_label

    if not full_entries:
        # fallback to biomass curves if logged that way
        full_entries = [e for e in logs if isinstance(e, dict) and ("t" in e) and ("biomass" in e)]
        curve_key = "biomass"
        curve_label = "Biomass"

    if args.max_curves and args.max_curves > 0 and len(full_entries) > args.max_curves:
        idxs = np.linspace(0, len(full_entries) - 1, args.max_curves).astype(int)
        full_entries = [full_entries[i] for i in idxs]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(8, 5))
    for e in full_entries:
        t = np.asarray(e["t"])
        b = np.asarray(e[curve_key])
        if t.size and b.size:
            plt.plot(t, b, alpha=0.35)

    plt.xlabel("Time [h]")
    plt.ylabel(curve_label)
    plt.title(f"{curve_label} evolution during greedy optimization\n({results_path.stem})")
    plt.grid(True)
    traj_plot_path = outdir / f"{results_path.stem}_{curve_label.lower()}_trajectories_{ts}.png"
    plt.tight_layout()
    plt.savefig(traj_plot_path, dpi=200)

    # --- Plot 2: optimized control + best objective curve ---

    # Prefer projected (uniform-grid) control if present: control_times (N+1) + opt_vman_values (N)
    if "control_times" in res and "opt_vman_values" in res:
        control_times = np.asarray(res["control_times"], dtype=float)
        opt_vman_values = np.asarray(res["opt_vman_values"], dtype=float)

        if control_times.size != opt_vman_values.size + 1:
            raise ValueError(
                f"Expected control_times (N+1) and opt_vman_values (N). "
                f"Got control_times={control_times.size}, opt_vman_values={opt_vman_values.size}"
            )

        x_step = control_times
        y_step = np.r_[opt_vman_values, opt_vman_values[-1]]
        control_label = f"Optimized vman (projected, {vman_id})"

    # Otherwise plot true greedy variable-boundary control: control_boundaries_true (K+1) + vman_values_true (K)
    elif "control_boundaries_true" in res and "vman_values_true" in res:
        boundaries = np.asarray(res["control_boundaries_true"], dtype=float)
        values_true = np.asarray(res["vman_values_true"], dtype=float)

        x_step, y_step = _step_xy_for_variable_boundaries(boundaries, values_true)
        control_label = f"Optimized vman (true greedy, {vman_id})"

    else:
        raise KeyError(
            "Could not find a valid control representation. Expected either "
            "('control_times' + 'opt_vman_values') or ('control_boundaries_true' + 'vman_values_true')."
        )
    # choose best entry among logged full curves
    best_entry = None
    best_final = -np.inf
    for e in logs:
        if isinstance(e, dict) and ("t" in e) and (curve_key in e):
            b = np.asarray(e[curve_key], dtype=float)
            if b.size and np.isfinite(b[-1]) and b[-1] > best_final:
                best_final = float(b[-1])
                best_entry = e

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.step(x_step, y_step, where="post", linewidth=2, label=control_label)
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("vman (flux)")
    ax1.set_xlim(x_step[0], x_step[-1])
    ax1.grid(True)

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
        ax2.set_ylabel(curve_label)
        ax2.text(0.02, 0.95, f"No full {curve_label.lower()} curve found in logs",
                 transform=ax2.transAxes, va="top")

    # If objective != biomass, optionally overlay biomass+glucose from results (same behavior as your plotter)
    if objective != "biomass":
        t_eval = np.asarray(res["t_eval_points"], dtype=float) if "t_eval_points" in res else np.array([])
        biomass_curve = np.asarray(res["biomass"], dtype=float) if "biomass" in res else np.array([])
        glucose_curve = np.asarray(res["glucose"], dtype=float) if "glucose" in res else np.array([])

        if biomass_curve.size and np.isfinite(best_final) and best_final > 0:
            denom = np.max(biomass_curve) if biomass_curve.size else 0.0
            scale = (best_final / denom) if denom > 0 else 1.0
            biomass_scaled = biomass_curve * scale
        else:
            scale = 1.0
            biomass_scaled = biomass_curve

        if t_eval.size and biomass_scaled.size:
            ax2.plot(
                t_eval,
                biomass_scaled,
                linestyle="--",
                linewidth=1.5,
                label=f"Biomass scaled (x{scale:.2f})",
            )
        if t_eval.size and glucose_curve.size:
            ax2.plot(
                t_eval,
                glucose_curve,
                linestyle=":",
                linewidth=1.5,
                label="Glucose",
            )

        ax2.set_ylabel("Concentration / objective")
    else:
        ax2.set_ylabel(curve_label)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.title(f"Greedy Optimized Control + {curve_label}\n({results_path.stem})")
    combo_plot_path = outdir / f"{results_path.stem}_control_plus_{curve_label.lower()}_{ts}.png"
    plt.savefig(combo_plot_path, dpi=200, bbox_inches="tight")

    print(f"Saved trajectories plot to {traj_plot_path}")
    print(f"Saved control+{curve_label.lower()} plot to {combo_plot_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()