#!/usr/bin/env python
"""
Greedy multi-cut optimizer wrapped to be compatible with the existing plotting pipeline.

Outputs the SAME keys expected by scripts/plot_optimize_vman.py:
  - opt_vman_values (length N)
  - control_times (length N+1)

Internally, we find a piecewise-constant control with K intervals (K = n_cuts+1),
then "project" that control onto the equidistant N-interval grid so the existing
plot script works unchanged.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hybrid_model  # noqa: E402
from flux_config import STATE_INDEX  # noqa: E402
from runtime_utils import RESULTS_DIR, ensure_output_dirs, load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402

from greedy_optim import (  # noqa: E402
    optimize_vman_greedy_k_cuts,
    make_piecewise_constant_control,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--t-start", type=float, default=0.0, help="Optimization start time")
    parser.add_argument("--t-end", type=float, default=12.0, help="Optimization end time")
    parser.add_argument("--num-intervals", type=int, default=20, help="Number of control intervals (N)")
    parser.add_argument("--n-eval", type=int, default=200, help="Number of time points for solver outputs")
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=3,
        metavar=("glucose", "ethanol", "biomass"),
        default=[10.0, 0.0, 0.01],
        help="Initial concentrations",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic information during optimization")
    parser.add_argument(
        "--objective",
        type=str,
        default="biomass",
        choices=tuple(STATE_INDEX.keys()),
        help="Objective state to maximize (e.g., biomass or ethanol)",
    )
    parser.add_argument("--log-full-every-k", type=int, default=10, help="Log full trajectories every k evals")

    # greedy knobs (multi-cut)
    parser.add_argument("--n_cuts", type=int, default=3, help="Number of greedy cuts to insert")
    parser.add_argument("--split-grid-size", dest="split_grid_size", type=int, default=60,
                        help="Candidate split times to test (per interval)")
    parser.add_argument("--min-dt", type=float, default=1e-3, help="Exclude splits within min_dt of interval ends")
    parser.add_argument("--inner-polish-maxiter", type=int, default=80, help="Local maxiter per candidate split")
    parser.add_argument("--final-polish-maxiter", type=int, default=300, help="Final polish maxiter at end")

    parser.add_argument("--output-path", type=Path, default=None, help="Where to save main results .npz")
    parser.add_argument("--logs-path", type=Path, default=None, help="Where to save logs .npz")
    return parser.parse_args()


def project_piecewise_to_uniform_grid(t0, t1, N, boundaries, values):
    """
    Project arbitrary piecewise-constant control (boundaries, values) onto uniform grid.

    boundaries: length K+1 (e.g. [t0, t_cut1, ..., t1])
    values:     length K   (one per interval)

    Returns:
      control_times: linspace(t0,t1,N+1)
      opt_vman_values: length N, where each uniform interval uses the value at its midpoint.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    values = np.asarray(values, dtype=float)

    if boundaries.size != values.size + 1:
        raise ValueError(
            f"Expected boundaries (K+1) and values (K). Got {boundaries.size} vs {values.size}"
        )

    control_times = np.linspace(t0, t1, N + 1)

    # midpoints of each uniform interval
    mids = 0.5 * (control_times[:-1] + control_times[1:])

    # piecewise control function
    vfun = make_piecewise_constant_control(boundaries, values)

    opt_vman_values = np.array([vfun(tm) for tm in mids], dtype=float)
    return control_times, opt_vman_values


def main():
    args = parse_args()
    ensure_output_dirs()

    model, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    feasible_range = metadata.get("feasible_range")
    lower, upper = feasible_range[0], feasible_range[1]
    if lower is None or upper is None:
        raise ValueError("feasible_range must be present in checkpoint metadata (lower, upper).")

    t_eval_points = np.linspace(args.t_start, args.t_end, args.n_eval)
    z0 = np.array(args.initial_state, dtype=float)

    print(f"Initial conditions set to: {args.initial_state}")
    print(f"vman bounds: [{lower}, {upper}]")

    # ---- run greedy multi-cut optimizer ----
    greedy_res, logs = optimize_vman_greedy_k_cuts(
        model=model,
        hybrid_ode=hybrid_model.hybrid_ode,
        z0=z0,
        t_span=(args.t_start, args.t_end),
        t_eval_points=t_eval_points,
        value_bounds=(lower, upper),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        objective=args.objective,
        n_cuts=args.n_cuts,
        split_grid_size=args.split_grid_size,
        min_dt=args.min_dt,
        inner_polish_maxiter=args.inner_polish_maxiter,
        final_polish_maxiter=args.final_polish_maxiter,
        seed=0,
        verbose=args.verbose,
        log_full_every_k=args.log_full_every_k,
    )

    boundaries_true = np.asarray(greedy_res["boundaries"], dtype=float)
    values_true = np.asarray(greedy_res["vman_values"], dtype=float)

    print(f"Greedy result: cuts_used={greedy_res.get('n_cuts_used')}  intervals={values_true.size}")
    print(f"Boundaries (true): {boundaries_true}")
    print(f"Values (true): {values_true}")

    # ---- project onto uniform grid for existing plot script ----
    control_times, opt_vman_values = project_piecewise_to_uniform_grid(
        args.t_start, args.t_end, args.num_intervals, boundaries_true, values_true
    )

    # Simulate using the *uniform-grid* control (matches what plot_optimize_vman.py will show)
    def vman_t_uniform(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = int(np.clip(idx, 0, args.num_intervals - 1))
        return float(opt_vman_values[idx])

    sol_opt = solve_ivp(
        fun=lambda t, z: hybrid_model.hybrid_ode(t, z, vman_t_uniform, model, x_scaler, y_scaler),
        t_span=(args.t_start, args.t_end),
        y0=z0,
        t_eval=t_eval_points,
        method="RK45",
    )

    output_path = args.output_path or (RESULTS_DIR / f"optimize_vman_{args.checkpoint.stem}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obj_idx = STATE_INDEX[args.objective]
    objective_curve = sol_opt.y[obj_idx, :] if sol_opt.success else np.array([])
    biomass_curve = sol_opt.y[STATE_INDEX["biomass"], :] if sol_opt.success else np.array([])
    glucose_curve = sol_opt.y[STATE_INDEX["glucose"], :] if sol_opt.success else np.array([])

    np.savez_compressed(
        output_path,
        # --- keys required by existing plot script ---
        opt_vman_values=opt_vman_values,
        control_times=control_times,
        t_eval_points=t_eval_points,
        objective=args.objective,
        objective_curve=objective_curve,
        biomass=biomass_curve,
        glucose=glucose_curve,
        solver_success=sol_opt.success,
        metadata=metadata,

        # --- extra info (won't break plot script) ---
        optimizer="greedy_k_cuts",
        control_boundaries_true=boundaries_true,
        vman_values_true=values_true,
        n_cuts_requested=int(greedy_res.get("n_cuts_requested", args.n_cuts)),
        n_cuts_used=int(greedy_res.get("n_cuts_used", values_true.size - 1)),
        best_score=float(greedy_res.get("best_score", np.nan)),
        greedy_params=dict(
            n_cuts=args.n_cuts,
            split_grid_size=args.split_grid_size,
            min_dt=args.min_dt,
            inner_polish_maxiter=args.inner_polish_maxiter,
            final_polish_maxiter=args.final_polish_maxiter,
        ),
    )

    logs_path = args.logs_path or (RESULTS_DIR / f"optimize_vman_{args.checkpoint.stem}_logs.npz")
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logs_path, logs=np.array(logs, dtype=object))

    print(f"Stored trajectory logs to {logs_path}")
    print(f"Results saved to {output_path}")
    print(f"Solver success: {sol_opt.success}")


if __name__ == "__main__":
    main()