#!/usr/bin/env python
"""
ECC2comp surrogate benchmark — sweep over constant ACKr.

Mirrors ecc2comp_benchmark_lp.py exactly — same 1-D grid, same ODE
structure (src/hybrid_model.py's h(z) rate scaling), same initial
conditions — but replaces model.optimize() with the trained NN
(trained_models/ACKr_trained_model_input-1_output-4_hidden-4.pt).

The surrogate is called at every ODE step, exactly as the LP is in the LP
benchmark, so nn_calls == lp_calls for an equal per-call comparison (same
discipline used by ijo1366_benchmark_surrogate.py).

Results saved to results/ecc2comp_benchmark_surrogate.npz.

Usage:
    python scripts/ecc2comp_benchmark_surrogate.py
    python scripts/ecc2comp_benchmark_surrogate.py --grid-n 3   # quick smoke test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flux_config import FLUX_INDEX  # noqa: E402
from hybrid_model import h  # noqa: E402
from runtime_utils import (RESULTS_DIR, ensure_output_dirs,  # noqa: E402
                           load_surrogate_checkpoint, surrogate_predict)
from surrogateNN import SurrogateNN  # noqa: E402


# ── ODE / simulation ─────────────────────────────────────────────────────────

def run_simulation(model_nn, x_scaler, y_scaler, ackr_val, glc0, etoh0, bm0, t_end):
    """
    One surrogate hybrid-ODE simulation with ACKr pinned to a constant.

    The surrogate is called at every ODE step (same call structure as the LP
    benchmark) so that nn_calls == lp_calls for an equal comparison.

    Returns (objective, wall_time_s, nn_calls, ode_evals, success).
    """
    X_point  = np.array([[ackr_val]], dtype=np.float32)
    nn_calls = [0]

    def rhs(t, z):
        z = np.maximum(z, 0)
        glucose, ethanol, biomass = z
        Y_pred = surrogate_predict(model_nn, x_scaler, y_scaler, X_point)[0]
        nn_calls[0] += 1
        v_glc  = Y_pred[FLUX_INDEX["glc"]]
        v_etoh = Y_pred[FLUX_INDEX["etoh"]]
        v_bio  = Y_pred[FLUX_INDEX["biomass"]]
        rate = biomass * h(z)
        dzdt = np.array([rate * v_glc, rate * v_etoh, rate * v_bio])
        dzdt[z <= 0] = np.maximum(dzdt[z <= 0], 0)
        return dzdt

    t0 = time.time()
    result = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=[glc0, etoh0, bm0],
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=0.1,
    )
    elapsed   = time.time() - t0
    bm_final  = result.y[2, -1] if result.success else bm0
    objective = bm_final - bm0
    return objective, elapsed, nn_calls[0], result.nfev, result.success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--t-end",     type=float, default=12.0)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--etoh0",     type=float, default=0.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "ACKr_trained_model_input-1_output-4_hidden-4.pt")
    p.add_argument("--output",    type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ecc2comp_benchmark_surrogate.npz"

    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    ackr_lo, ackr_hi = metadata["feasible_range"]
    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    n_total = args.grid_n

    print("=" * 66)
    print("  ECC2comp surrogate benchmark — 1-D grid over ACKr")
    print(f"  Grid       : {args.grid_n} simulations")
    print(f"  ACKr range : [{ackr_lo:.3f}, {ackr_hi:.3f}]")
    print(f"  T_end      : {args.t_end} h  |  Glc0={args.glc0}  Etoh0={args.etoh0}  Bm0={args.bm0}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output     : {output_path}")
    print("=" * 66)

    objectives = np.full(args.grid_n, np.nan)
    sim_times  = np.zeros(args.grid_n)
    nn_calls   = np.zeros(args.grid_n, dtype=int)
    ode_evals  = np.zeros(args.grid_n, dtype=int)

    t_total_start = time.time()
    for i, ackr in enumerate(ackr_vals):
        obj, elapsed, n_nn, n_ode, ok = run_simulation(
            model_nn, x_scaler, y_scaler, ackr, args.glc0, args.etoh0, args.bm0, args.t_end
        )
        objectives[i] = obj
        sim_times[i]  = elapsed
        nn_calls[i]   = n_nn
        ode_evals[i]  = n_ode
        print(f"  [{i+1:>3}/{n_total}]  ACKr={ackr:+.3f}  "
              f"obj={obj:.4f}  {1000*elapsed:.3f}ms/sim  nn_calls={n_nn}")

    total_time = time.time() - t_total_start
    best_idx   = int(np.nanargmax(objectives))

    print(f"\n── ECC2comp surrogate benchmark summary ────────────────────────")
    print(f"  Total wall time    : {total_time:.4f} s")
    print(f"  Mean time / sim    : {1000*sim_times.mean():.3f} ms")
    print(f"  Mean NN calls / sim: {nn_calls.mean():.0f}")
    print(f"  Mean NN time / call: {(1000*sim_times/np.maximum(nn_calls,1)).mean():.4f} ms")
    print(f"  Mean ODE evals/sim : {ode_evals.mean():.0f}")
    print(f"  Best objective     : {objectives[best_idx]:.4f} gDW/L  (ACKr={ackr_vals[best_idx]:.3f})")

    np.savez_compressed(
        output_path,
        ackr_vals=ackr_vals,
        objectives=objectives,
        sim_times=sim_times,
        nn_calls=nn_calls,
        ode_evals=ode_evals,
        total_time=total_time,
        glc0=args.glc0, etoh0=args.etoh0, bm0=args.bm0,
        t_end=args.t_end,
        grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
