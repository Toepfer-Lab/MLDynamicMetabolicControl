#!/usr/bin/env python
"""
iJO1366 surrogate piecewise-constant benchmark.

Mirrors ijo1366_benchmark_piecewise_lp.py exactly — same two-phase structure,
same grid, same initial conditions — but replaces model.optimize() with the
trained surrogate at every ODE step.

Phase 1 growth-optimal values are loaded from results/ijo1366_benchmark_piecewise_lp.npz
(ackr1 and ldh1 scalars saved there), creating a direct DVC dependency on the LP piecewise stage.

State: [BM (gDW/L), Glc (mM), Etoh (mmol/L)]

  dBM/dt   = mu · BM        (mu from surrogate, OUTPUT_INDEX["bio"])
  dGlc/dt  = -mm_rate · BM  (Michaelis-Menten; glucose not a surrogate output)
  dEtoh/dt = v_etoh · BM    (v_etoh from surrogate, OUTPUT_INDEX["etoh"])

Results saved to results/ijo1366_benchmark_piecewise_surrogate.npz.

Usage:
    python scripts/ijo1366_benchmark_piecewise_surrogate.py
    python scripts/ijo1366_benchmark_piecewise_surrogate.py --grid-n 3
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


# ── ODE ──────────────────────────────────────────────────────────────────────

def make_surrogate_rhs(model_nn, x_scaler, y_scaler, ackr_val, ldh_val, nn_calls):
    """
    ODE RHS for one phase with (ackr_val, ldh_val) fixed.
    The surrogate is called at every ODE step (same structure as LP benchmark).
    nn_calls is shared across both phases so the total is accumulated.
    """
    X_point = np.array([[ackr_val, ldh_val]], dtype=np.float32)

    def rhs(t, y):
        bm, glc, etoh = y
        glc = max(glc, 0.0)
        if glc < 1e-9:
            return [0.0, 0.0, 0.0]
        mm_rate   = VMAX_GLC * glc / (KM_GLC + glc)
        Y_pred    = surrogate_predict(model_nn, x_scaler, y_scaler, X_point)[0]
        nn_calls[0] += 1
        mu        = float(Y_pred[OUTPUT_INDEX["bio"]])
        flux_etoh = float(Y_pred[OUTPUT_INDEX["etoh"]])
        return [mu * bm, -mm_rate * bm, flux_etoh * bm]

    return rhs


def run_simulation(model_nn, x_scaler, y_scaler,
                   ackr1, ldh1, ackr2, ldh2, tau, bm0, glc0, t_end):
    """
    Two-phase surrogate dFBA:
      Phase 1 [0, τ]:       (ackr1, ldh1) — growth
      Phase 2 [τ, t_end]:   (ackr2, ldh2) — production

    Returns (etoh_final, wall_time_s, nn_calls, ode_evals, success).
    """
    nn_calls = [0]
    rhs1 = make_surrogate_rhs(model_nn, x_scaler, y_scaler, ackr1, ldh1, nn_calls)
    rhs2 = make_surrogate_rhs(model_nn, x_scaler, y_scaler, ackr2, ldh2, nn_calls)

    t0 = time.time()
    r1 = solve_ivp(rhs1, t_span=(0.0, tau), y0=[bm0, glc0, 0.0],
                   method="RK45", rtol=1e-4, atol=1e-6, max_step=0.1)

    if not r1.success:
        return 0.0, time.time() - t0, nn_calls[0], r1.nfev, False

    r2 = solve_ivp(rhs2, t_span=(tau, t_end), y0=r1.y[:, -1],
                   method="RK45", rtol=1e-4, atol=1e-6, max_step=0.1)

    elapsed    = time.time() - t0
    etoh_final = r2.y[2, -1] if r2.success else 0.0
    return etoh_final, elapsed, nn_calls[0], r1.nfev + r2.nfev, r2.success


# ── helpers ───────────────────────────────────────────────────────────────────

def load_phase1_values(piecewise_lp_path):
    """Load phase 1 values saved by the piecewise LP benchmark (ackr1, ldh1 scalars)."""
    d = np.load(piecewise_lp_path)
    return float(d["ackr1"]), float(d["ldh1"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--tau",       type=float, default=5.0)
    p.add_argument("--t-end",     type=float, default=10.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--piecewise-lp-results", type=Path,
                   default=REPO_ROOT / "results" / "ijo1366_benchmark_piecewise_lp.npz",
                   help="Piecewise LP benchmark results (provides ackr1, ldh1 scalars)")
    p.add_argument("--data-path", type=Path,
                   default=REPO_ROOT / "data" / "ijo1366_anaerobic.npz")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models" / "ijo1366_anaerobic_hidden-16.pt")
    p.add_argument("--output",    type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ijo1366_benchmark_piecewise_surrogate.npz"

    ackr1, ldh1 = load_phase1_values(args.piecewise_lp_results)

    data = np.load(args.data_path)
    feasible_range = data["feasible_range"]
    ackr_lo, ackr_hi = feasible_range[INPUT_FLUX_IDS.index("ACKr")]
    ldh_lo,  ldh_hi  = feasible_range[INPUT_FLUX_IDS.index("LDH_D")]

    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    ldh_vals  = np.linspace(ldh_lo,  ldh_hi,  args.grid_n)
    n_total   = args.grid_n ** 2

    print("=" * 66)
    print("  iJO1366 surrogate piecewise benchmark")
    print(f"  Phase 1  : ACKr={ackr1:.3f}  LDH_D={ldh1:.3f}  "
          f"[0 → {args.tau}h]  (growth-optimal from constant benchmark)")
    print(f"  Phase 2  : {args.grid_n}×{args.grid_n} grid  "
          f"[{args.tau} → {args.t_end}h]  (ethanol production)")
    print(f"  Objective: Etoh(T_end) [mmol/L]")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output   : {output_path}")
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

    for i, ackr2 in enumerate(ackr_vals):
        for j, ldh2 in enumerate(ldh_vals):
            obj, elapsed, n_nn, n_ode, ok = run_simulation(
                model_nn, x_scaler, y_scaler,
                ackr1, ldh1, ackr2, ldh2,
                args.tau, args.bm0, args.glc0, args.t_end
            )
            objectives[i, j] = obj
            sim_times[i, j]  = elapsed
            nn_calls[i, j]   = n_nn
            ode_evals[i, j]  = n_ode
            sim_idx += 1
            if sim_idx % 10 == 0 or sim_idx == n_total:
                pct = 100 * sim_idx / n_total
                print(f"  [{sim_idx:>3}/{n_total}] ({pct:.0f}%)  "
                      f"ACKr2={ackr2:+.2f}  LDH_D2={ldh2:+.2f}  "
                      f"etoh={obj:.4f}  {1000*elapsed:.2f}ms/sim")

    total_time = time.time() - t_total_start
    best_idx   = np.unravel_index(np.nanargmax(objectives), objectives.shape)

    print(f"\n── Surrogate piecewise summary ──────────────────────────────────")
    print(f"  Total wall time    : {total_time:.3f} s")
    print(f"  Mean time / sim    : {1000*sim_times.mean():.2f} ms")
    print(f"  Mean NN calls / sim: {nn_calls.mean():.0f}")
    print(f"  Mean ODE evals/sim : {ode_evals.mean():.0f}")
    print(f"  Best ethanol       : {objectives[best_idx]:.4f} mmol/L  "
          f"(ACKr2={ackr_vals[best_idx[0]]:.3f}, "
          f"LDH_D2={ldh_vals[best_idx[1]]:.3f})")

    np.savez_compressed(
        output_path,
        ackr_vals=ackr_vals, ldh_vals=ldh_vals,
        objectives=objectives, sim_times=sim_times,
        nn_calls=nn_calls, ode_evals=ode_evals,
        total_time=total_time,
        ackr1=ackr1, ldh1=ldh1, tau=args.tau,
        bm0=args.bm0, glc0=args.glc0, t_end=args.t_end, grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
