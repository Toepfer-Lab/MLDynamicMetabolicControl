#!/usr/bin/env python
"""
iJO1366 surrogate benchmark — grid search over constant (ACKr, LDH_D).

Mirrors ijo1366_benchmark_lp.py exactly — same N×N grid, same ODE structure,
same initial conditions — but replaces model.optimize() with the trained NN.

The surrogate is called at every ODE step, exactly as the LP is in the LP
benchmark. Although the inputs (ACKr, LDH_D) are constant during a simulation
so the output is numerically identical each time, calling it at every step
benchmarks the per-call overhead of the NN vs the LP on equal terms.

EX_glc__D_e is intentionally absent from surrogate outputs; glucose dynamics
are handled directly via Michaelis-Menten kinetics (as designed in Stage 1).

Results saved to results/ijo1366_benchmark_surrogate.npz.

Usage:
    python scripts/ijo1366_benchmark_surrogate.py
    python scripts/ijo1366_benchmark_surrogate.py --grid-n 3   # quick smoke test
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

from flux_config_ijo1366 import INPUT_FLUX_IDS, OUTPUT_INDEX
from runtime_utils import (RESULTS_DIR, ensure_output_dirs,
                           load_surrogate_checkpoint, surrogate_predict)
from surrogateNN import SurrogateNN

VMAX_GLC = 10.0   # mmol/gDW/h
KM_GLC   = 0.01   # mM


# ── ODE / simulation ─────────────────────────────────────────────────────────

def run_simulation(model_nn, x_scaler, y_scaler, ackr_val, ldh_d_val,
                   bm0, glc0, t_end):
    """
    One surrogate dFBA simulation with ACKr and LDH_D pinned to constants.

    The surrogate is called at every ODE step (same call structure as the LP
    benchmark) so that nn_calls == lp_calls for an equal comparison.

    Returns (objective, wall_time_s, nn_calls, ode_evals, success).
    """
    X_point  = np.array([[ackr_val, ldh_d_val]], dtype=np.float32)
    nn_calls = [0]

    def rhs(t, y):
        bm, glc = y
        glc = max(glc, 0.0)
        if glc < 1e-9:
            return [0.0, 0.0]
        mm_rate = VMAX_GLC * glc / (KM_GLC + glc)
        Y_pred   = surrogate_predict(model_nn, x_scaler, y_scaler, X_point)[0]
        nn_calls[0] += 1
        mu = float(Y_pred[OUTPUT_INDEX["bio"]])
        return [mu * bm, -mm_rate * bm]

    t0 = time.time()
    result = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=[bm0, glc0],
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=0.1,
    )
    elapsed   = time.time() - t0
    bm_final  = result.y[0, -1] if result.success else bm0
    objective = bm_final - bm0
    return objective, elapsed, nn_calls[0], result.nfev, result.success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--t-end",     type=float, default=10.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--data-path",  type=Path,
                   default=REPO_ROOT / "data" / "ijo1366_anaerobic.npz")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models" / "ijo1366_anaerobic_hidden-16.pt")
    p.add_argument("--output",     type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ijo1366_benchmark_surrogate.npz"

    # Grid bounds from training data (same domain as LP benchmark)
    data = np.load(args.data_path)
    feasible_range = data["feasible_range"]
    ackr_lo, ackr_hi = feasible_range[INPUT_FLUX_IDS.index("ACKr")]
    ldh_lo,  ldh_hi  = feasible_range[INPUT_FLUX_IDS.index("LDH_D")]

    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    ldh_vals  = np.linspace(ldh_lo,  ldh_hi,  args.grid_n)
    n_total   = args.grid_n ** 2

    print("=" * 66)
    print("  iJO1366 surrogate benchmark — grid search")
    print(f"  Grid       : {args.grid_n}×{args.grid_n} = {n_total} simulations")
    print(f"  ACKr range : [{ackr_lo:.3f}, {ackr_hi:.3f}]")
    print(f"  LDH_D range: [{ldh_lo:.3f},  {ldh_hi:.3f}]")
    print(f"  T_end      : {args.t_end} h  |  BM0={args.bm0}  Glc0={args.glc0}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output     : {output_path}")
    print("=" * 66)

    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )

    objectives = np.full((args.grid_n, args.grid_n), np.nan)
    sim_times  = np.zeros((args.grid_n, args.grid_n))
    nn_calls   = np.zeros((args.grid_n, args.grid_n), dtype=int)
    ode_evals  = np.zeros((args.grid_n, args.grid_n), dtype=int)

    t_total_start = time.time()
    sim_idx = 0

    for i, ackr in enumerate(ackr_vals):
        for j, ldh in enumerate(ldh_vals):
            obj, elapsed, n_nn, n_ode, ok = run_simulation(
                model_nn, x_scaler, y_scaler, ackr, ldh,
                args.bm0, args.glc0, args.t_end
            )
            objectives[i, j] = obj
            sim_times[i, j]  = elapsed
            nn_calls[i, j]   = n_nn
            ode_evals[i, j]  = n_ode
            sim_idx += 1
            if sim_idx % 10 == 0 or sim_idx == n_total:
                pct = 100 * sim_idx / n_total
                print(f"  [{sim_idx:>3}/{n_total}] ({pct:.0f}%)  "
                      f"ACKr={ackr:+.2f}  LDH_D={ldh:+.2f}  "
                      f"obj={obj:.4f}  {1000*elapsed:.2f}ms/sim")

    total_time = time.time() - t_total_start
    best_idx   = np.unravel_index(np.nanargmax(objectives), objectives.shape)

    print(f"\n── Surrogate benchmark summary ─────────────────────────────────")
    print(f"  Total wall time    : {total_time:.3f} s")
    print(f"  Mean time / sim    : {1000*sim_times.mean():.2f} ms")
    print(f"  Mean NN calls / sim: {nn_calls.mean():.0f}")
    print(f"  Mean ODE evals/sim : {ode_evals.mean():.0f}")
    print(f"  Best objective     : {objectives[best_idx]:.4f} gDW/L  "
          f"(ACKr={ackr_vals[best_idx[0]]:.3f}, "
          f"LDH_D={ldh_vals[best_idx[1]]:.3f})")

    np.savez_compressed(
        output_path,
        ackr_vals=ackr_vals,
        ldh_vals=ldh_vals,
        objectives=objectives,
        sim_times=sim_times,
        nn_calls=nn_calls,
        ode_evals=ode_evals,
        total_time=total_time,
        bm0=args.bm0,
        t_end=args.t_end,
        grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
