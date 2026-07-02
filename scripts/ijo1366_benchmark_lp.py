#!/usr/bin/env python
"""
iJO1366 LP benchmark — grid search over constant (ACKr, LDH_D).

For each point on an N×N grid of (ACKr, LDH_D) values, runs a full dFBA
simulation using model.optimize() at each ODE step (as in the SOA approach).
Glucose uptake is governed by Michaelis-Menten kinetics; ACKr and LDH_D are
held constant for the duration of each simulation.

Objective: total biomass produced = BM(T_end) - BM0
           (= integral of mu·BM dt, equivalent to maximising BM(T_end) for fixed BM0)

Results saved to results/ijo1366_benchmark_lp.npz for comparison with
ijo1366_benchmark_surrogate.py (same grid, same ODE, same ICs).

Usage:
    python scripts/ijo1366_benchmark_lp.py
    python scripts/ijo1366_benchmark_lp.py --grid-n 3   # quick smoke test (9 sims)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from cobra.io import read_sbml_model
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flux_config_ijo1366 import INPUT_FLUX_IDS
from runtime_utils import RESULTS_DIR, ensure_output_dirs

GLC_RXN  = "EX_glc__D_e"
VMAX_GLC = 10.0   # mmol/gDW/h
KM_GLC   = 0.01   # mM


# ── medium / model setup ─────────────────────────────────────────────────────

def configure_medium(model, condition="anaerobic"):
    medium = dict(model.medium)
    for glc_id in ("EX_glc__D_e", "EX_glc_D_e"):
        if glc_id in {r.id for r in model.exchanges}:
            medium[glc_id] = 10.0
            break
    if condition == "anaerobic" and "EX_o2_e" in medium:
        medium["EX_o2_e"] = 0.0
    model.medium = medium
    biomass_rxns = [r for r in model.reactions if "BIOMASS" in r.id]
    core_biomass = [r for r in biomass_rxns if "core" in r.id.lower()]
    model.objective = (core_biomass or biomass_rxns)[0].id


# ── ODE / simulation ─────────────────────────────────────────────────────────

def run_simulation(model, ackr_val, ldh_d_val, bm0, glc0, t_end):
    """
    One dFBA simulation with ACKr and LDH_D pinned to constants.

    ACKr and LDH_D are fixed in an outer `with model:` context (once per
    simulation). The inner `with model:` per ODE step only changes the glucose
    bound, reducing per-step overhead from 3 bound changes to 1.

    Returns (objective, wall_time_s, n_lp_calls, ode_evals, success).
    """
    lp_calls = [0]

    def rhs(t, y):
        bm, glc = y
        glc = max(glc, 0.0)
        if glc < 1e-9:
            return [0.0, 0.0]
        mm_rate = VMAX_GLC * glc / (KM_GLC + glc)
        with model:
            model.reactions.get_by_id(GLC_RXN).lower_bound = -mm_rate
            try:
                model.solver.configuration.timeout = 5
            except Exception:
                pass
            sol = model.optimize()
            lp_calls[0] += 1
            if sol.status != "optimal":
                return [0.0, 0.0]
            mu       = sol.objective_value
            flux_glc = sol.fluxes[GLC_RXN]
        return [mu * bm, flux_glc * bm]

    with model:
        model.reactions.get_by_id("ACKr").bounds  = (ackr_val,  ackr_val)
        model.reactions.get_by_id("LDH_D").bounds = (ldh_d_val, ldh_d_val)
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
    return objective, elapsed, lp_calls[0], result.nfev, result.success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--t-end",     type=float, default=10.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--condition", default="anaerobic")
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "iJO1366.xml")
    p.add_argument("--data-path",  type=Path,
                   default=REPO_ROOT / "data" / "ijo1366_anaerobic.npz")
    p.add_argument("--output",     type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ijo1366_benchmark_lp.npz"

    # Grid bounds from training data (same domain as surrogate was trained on)
    data = np.load(args.data_path)
    feasible_range = data["feasible_range"]   # shape (n_inputs, 2)
    ackr_lo, ackr_hi = feasible_range[INPUT_FLUX_IDS.index("ACKr")]
    ldh_lo,  ldh_hi  = feasible_range[INPUT_FLUX_IDS.index("LDH_D")]

    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    ldh_vals  = np.linspace(ldh_lo,  ldh_hi,  args.grid_n)
    n_total   = args.grid_n ** 2

    print("=" * 66)
    print("  iJO1366 LP benchmark — grid search")
    print(f"  Grid       : {args.grid_n}×{args.grid_n} = {n_total} simulations")
    print(f"  ACKr range : [{ackr_lo:.3f}, {ackr_hi:.3f}]")
    print(f"  LDH_D range: [{ldh_lo:.3f},  {ldh_hi:.3f}]")
    print(f"  T_end      : {args.t_end} h  |  BM0={args.bm0}  Glc0={args.glc0}")
    print(f"  Output     : {output_path}")
    print("=" * 66)

    print(f"\nLoading model from {args.model_path} ...")
    model = read_sbml_model(str(args.model_path))
    configure_medium(model, args.condition)

    objectives = np.full((args.grid_n, args.grid_n), np.nan)
    sim_times  = np.zeros((args.grid_n, args.grid_n))
    lp_calls   = np.zeros((args.grid_n, args.grid_n), dtype=int)
    ode_evals  = np.zeros((args.grid_n, args.grid_n), dtype=int)

    t_total_start = time.time()
    sim_idx = 0

    for i, ackr in enumerate(ackr_vals):
        for j, ldh in enumerate(ldh_vals):
            obj, elapsed, n_lp, n_ode, ok = run_simulation(
                model, ackr, ldh, args.bm0, args.glc0, args.t_end
            )
            objectives[i, j] = obj
            sim_times[i, j]  = elapsed
            lp_calls[i, j]   = n_lp
            ode_evals[i, j]  = n_ode
            sim_idx += 1
            if sim_idx % 10 == 0 or sim_idx == n_total:
                pct = 100 * sim_idx / n_total
                elapsed_total = time.time() - t_total_start
                eta = (elapsed_total / sim_idx) * (n_total - sim_idx)
                print(f"  [{sim_idx:>3}/{n_total}] ({pct:.0f}%)  "
                      f"ACKr={ackr:+.2f}  LDH_D={ldh:+.2f}  "
                      f"obj={obj:.4f}  {elapsed:.2f}s/sim  "
                      f"ETA {eta:.0f}s")

    total_time = time.time() - t_total_start
    best_idx   = np.unravel_index(np.nanargmax(objectives), objectives.shape)

    print(f"\n── LP benchmark summary ────────────────────────────────────────")
    print(f"  Total wall time    : {total_time:.1f} s  ({total_time/60:.1f} min)")
    print(f"  Mean time / sim    : {sim_times.mean():.2f} s")
    print(f"  Mean LP calls / sim: {lp_calls.mean():.0f}")
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
        lp_calls=lp_calls,
        ode_evals=ode_evals,
        total_time=total_time,
        bm0=args.bm0,
        t_end=args.t_end,
        grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
