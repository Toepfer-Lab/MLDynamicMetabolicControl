#!/usr/bin/env python
"""
Simulate the hybrid ODE using a trained surrogate model.
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
    sys.path.append(str(SRC_DIR))

import hybrid_model  # noqa: E402
from flux_config import STATE_INDEX  # noqa: E402
from optim import piecewise_constant_control  # noqa: E402
from runtime_utils import RESULTS_DIR, ensure_output_dirs, load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--t-start", type=float, default=0.0, help="Simulation start time")
    parser.add_argument("--t-end", type=float, default=12.0, help="Simulation end time")
    parser.add_argument("--num-points", type=int, default=200, help="Number of time evaluation points")
    parser.add_argument(
        "--vman-value",
        type=float,
        default=5.0,
        help="Constant control value if no control file is provided",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        default=None,
        help="Optional CSV/TSV with two columns: time, value for piecewise-constant control",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz file. Defaults to results/hybrid_simulation_<checkpoint_stem>.npz",
    )
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=3,
        metavar=("glucose", "ethanol", "biomass"),
        default=[10.0, 0.0, 0.01],
        help="Initial concentrations",
    )
    return parser.parse_args()


def load_control_from_file(path: Path):
    data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Control file must have exactly two columns: time, value")
    times = data[:, 0]
    values = data[:, 1]
    if not np.all(np.diff(times) >= 0):
        raise ValueError("Control times must be non-decreasing")
    return times, values


def main():
    args = parse_args()
    ensure_output_dirs()

    model, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)

    t_eval = np.linspace(args.t_start, args.t_end, args.num_points)
    t_span = (args.t_start, args.t_end)

    if args.control_file:
        control_times, control_values = load_control_from_file(args.control_file)
        control_fn = piecewise_constant_control(control_times, control_values)
    else:
        control_times = np.array([args.t_start, args.t_end])
        control_values = np.array([args.vman_value, args.vman_value])
        control_fn = piecewise_constant_control(control_times, control_values)

    sol = solve_ivp(
        fun=lambda t, z: hybrid_model.hybrid_ode(t, z, control_fn, model, x_scaler, y_scaler),
        t_span=t_span,
        y0=np.array(args.initial_state, dtype=float),
        t_eval=t_eval,
        method="RK45",
    )

    output_path = args.output or RESULTS_DIR / f"hybrid_simulation_{args.checkpoint.stem}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        t=sol.t,
        y=sol.y,
        control_times=control_times,
        control_values=control_values,
        metadata=metadata,
    )

    final_biomass = sol.y[STATE_INDEX["biomass"], -1] if sol.success else float("nan")
    print(f"Simulation success: {sol.success}")
    print(f"Final biomass: {final_biomass:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
