#!/usr/bin/env python
"""
Optimize the manipulated flux trajectory (vman) to maximize biomass.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# Make local src importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hybrid_model  # noqa: E402
from flux_config import STATE_INDEX  # noqa: E402
from optim import piecewise_constant_control, optimize_vman  # noqa: E402
from runtime_utils import RESULTS_DIR, ensure_output_dirs, load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help="Optimization start time",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=12.0,
        help="Optimization end time",
    )
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=20,
        help="Number of control intervals (N)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=200,
        help="Number of time points for solver outputs",
    )
    parser.add_argument(
        "--bounds-lower",
        type=float,
        default=None,
        help="Lower bound for vman (overrides metadata)",
    )
    parser.add_argument(
        "--bounds-upper",
        type=float,
        default=None,
        help="Upper bound for vman (overrides metadata)",
    )
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=3,
        metavar=("glucose", "ethanol", "biomass"),
        default=[10.0, 0.0, 0.01],
        help="Initial concentrations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic information during optimization",
    )
    parser.add_argument(
        "--log-full-every-k",
        type=int,
        default=10,
        help="Log full trajectories every k evaluations",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save the main results .npz (overrides default naming)",
    )
    parser.add_argument(
        "--logs-path",
        type=Path,
        default=None,
        help="Where to save the logs .npz (overrides default naming)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    model, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    feasible_range = metadata.get("feasible_range")

    lower = args.bounds_lower if args.bounds_lower is not None else (feasible_range[0] if feasible_range is not None else None)
    upper = args.bounds_upper if args.bounds_upper is not None else (feasible_range[1] if feasible_range is not None else None)
    if lower is None or upper is None:
        raise ValueError("Bounds must be provided either via --bounds-lower/--bounds-upper or stored in metadata.")

    rxn_bounds = [(lower, upper)] * args.num_intervals
    # Solver output sampling grid; separate from control_times (decision grid).
    t_eval_points = np.linspace(args.t_start, args.t_end, args.n_eval)

    result, logs = optimize_vman(
        model=model,
        hybrid_ode=hybrid_model.hybrid_ode,
        z0=args.initial_state,
        t_span=(args.t_start, args.t_end),
        N=args.num_intervals,
        t_eval_points=t_eval_points,
        bounds=rxn_bounds,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        log_full_every_k=args.log_full_every_k,
        verbose=args.verbose,
        global_maxiter=80,
        global_popsize=6,
        topk_polish=3,
        polish_maxiter=500,
        seed=0,
    )


    opt_vman_values = result.x
    print(f"optimal vman values: {opt_vman_values}")

    # Control grid used to construct the optimized piecewise-constant profile.
    control_times = np.linspace(args.t_start, args.t_end, args.num_intervals + 1)
    vman_t_opt = piecewise_constant_control(control_times, opt_vman_values)

    sol_opt = solve_ivp(
        fun=lambda t, z: hybrid_model.hybrid_ode(t, z, vman_t_opt, model, x_scaler, y_scaler),
        t_span=(args.t_start, args.t_end),
        y0=np.array(args.initial_state, dtype=float),
        t_eval=t_eval_points,
        method="RK45",
    )

    output_path = args.output_path or (RESULTS_DIR / f"optimize_vman_{args.checkpoint.stem}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
    output_path,
    opt_vman_values=opt_vman_values,
    control_times=control_times,
    t_eval_points=t_eval_points,
    biomass=sol_opt.y[STATE_INDEX["biomass"], :] if sol_opt.success else np.array([]),
    
    solver_success=sol_opt.success,
    metadata=metadata,
)

    logs_path = args.logs_path or (RESULTS_DIR / f"optimize_vman_{args.checkpoint.stem}_logs.npz")
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(logs_path, logs=np.array(logs, dtype=object))
    print(f"Stored trajectory logs to {logs_path}")
    print("Optimizer message:", result.message)
    print("nfev:", result.nfev, "nit:", result.nit)
    print(f"Optimization success: {result.success}, final objective {-result.fun:.4f}")
    final_biomass = -result.fun
    print(f"Optimal final biomass: {final_biomass:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
