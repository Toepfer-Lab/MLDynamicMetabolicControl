#!/usr/bin/env python
"""
iJO1366 LP piecewise-constant benchmark.

Two-phase control:
  Phase 1 [0, τ):       growth-optimal (ACKr, LDH_D) from constant LP results
  Phase 2 [τ, T_end]:   grid search over (ACKr, LDH_D) — production phase

Objective: ethanol produced = Etoh(T_end)

The LP is re-solved at every ODE step throughout both phases. The control
change at τ means the LP problem is genuinely different between phases, so
the LP solves serve a real purpose here (not just repeated identical calls).

State: [BM (gDW/L), Glc (mM), Etoh (mmol/L)]

  dBM/dt   = mu · BM        (mu from LP objective)
  dGlc/dt  = v_glc · BM     (v_glc from LP, bounded by MM kinetics)
  dEtoh/dt = v_etoh · BM    (v_etoh from LP)

Phase 1 growth-optimal values are loaded from results/ijo1366_benchmark_lp.npz.
Results saved to results/ijo1366_benchmark_piecewise_lp.npz.

Usage:
    python scripts/ijo1366_benchmark_piecewise_lp.py
    python scripts/ijo1366_benchmark_piecewise_lp.py --grid-n 3  # smoke test
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
ETOH_RXN = "EX_etoh_e"
VMAX_GLC = 10.0   # mmol/gDW/h
KM_GLC   = 0.01   # mM


# ── model setup ──────────────────────────────────────────────────────────────

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


# ── ODE ──────────────────────────────────────────────────────────────────────

def make_lp_rhs(model, lp_calls):
    """
    ODE RHS that solves the LP at each call using the model's current bounds.
    ACKr and LDH_D are fixed by the caller's outer `with model:` context;
    this RHS only changes the glucose bound per step.
    """
    def rhs(t, y):
        bm, glc, etoh = y
        glc = max(glc, 0.0)
        if glc < 1e-9:
            return [0.0, 0.0, 0.0]
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
                return [0.0, 0.0, 0.0]
            mu        = sol.objective_value
            flux_glc  = sol.fluxes[GLC_RXN]
            flux_etoh = sol.fluxes.get(ETOH_RXN, 0.0)
        return [mu * bm, flux_glc * bm, flux_etoh * bm]
    return rhs


def run_simulation(model, ackr1, ldh1, ackr2, ldh2, tau, bm0, glc0, t_end):
    """
    Two-phase dFBA:
      Phase 1 [0, τ]:       (ackr1, ldh1) — growth
      Phase 2 [τ, t_end]:   (ackr2, ldh2) — production

    lp_calls counter is shared across both phases.
    Returns (etoh_final, wall_time_s, lp_calls, ode_evals, success).
    """
    lp_calls = [0]
    rhs = make_lp_rhs(model, lp_calls)

    # Phase 1: set ACKr/LDH_D once for the entire growth phase
    with model:
        model.reactions.get_by_id("ACKr").bounds  = (ackr1, ackr1)
        model.reactions.get_by_id("LDH_D").bounds = (ldh1,  ldh1)
        t0 = time.time()
        r1 = solve_ivp(rhs, t_span=(0.0, tau), y0=[bm0, glc0, 0.0],
                       method="RK45", rtol=1e-4, atol=1e-6, max_step=0.1)

    if not r1.success:
        return 0.0, time.time() - t0, lp_calls[0], r1.nfev, False

    # Phase 2: switch to production-phase control
    with model:
        model.reactions.get_by_id("ACKr").bounds  = (ackr2, ackr2)
        model.reactions.get_by_id("LDH_D").bounds = (ldh2,  ldh2)
        r2 = solve_ivp(rhs, t_span=(tau, t_end), y0=r1.y[:, -1],
                       method="RK45", rtol=1e-4, atol=1e-6, max_step=0.1)

    elapsed    = time.time() - t0
    etoh_final = r2.y[2, -1] if r2.success else 0.0
    return etoh_final, elapsed, lp_calls[0], r1.nfev + r2.nfev, r2.success


# ── helpers ───────────────────────────────────────────────────────────────────

def load_phase1_values(lp_results_path):
    """Return (ACKr, LDH_D) of the biomass-maximising point from the constant benchmark."""
    d = np.load(lp_results_path)
    best = np.unravel_index(np.nanargmax(d["objectives"]), d["objectives"].shape)
    return float(d["ackr_vals"][best[0]]), float(d["ldh_vals"][best[1]])


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--tau",       type=float, default=5.0,
                   help="Phase switch time in hours (default: 5.0)")
    p.add_argument("--t-end",     type=float, default=10.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--condition", default="anaerobic")
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "iJO1366.xml")
    p.add_argument("--lp-results", type=Path,
                   default=REPO_ROOT / "results" / "ijo1366_benchmark_lp.npz",
                   help="Constant LP benchmark results (source of phase 1 values)")
    p.add_argument("--data-path", type=Path,
                   default=REPO_ROOT / "data" / "ijo1366_anaerobic.npz")
    p.add_argument("--output",    type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ijo1366_benchmark_piecewise_lp.npz"

    ackr1, ldh1 = load_phase1_values(args.lp_results)

    data = np.load(args.data_path)
    feasible_range = data["feasible_range"]
    ackr_lo, ackr_hi = feasible_range[INPUT_FLUX_IDS.index("ACKr")]
    ldh_lo,  ldh_hi  = feasible_range[INPUT_FLUX_IDS.index("LDH_D")]

    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    ldh_vals  = np.linspace(ldh_lo,  ldh_hi,  args.grid_n)
    n_total   = args.grid_n ** 2

    print("=" * 66)
    print("  iJO1366 LP piecewise benchmark")
    print(f"  Phase 1  : ACKr={ackr1:.3f}  LDH_D={ldh1:.3f}  "
          f"[0 → {args.tau}h]  (growth-optimal from constant benchmark)")
    print(f"  Phase 2  : {args.grid_n}×{args.grid_n} grid  "
          f"[{args.tau} → {args.t_end}h]  (ethanol production)")
    print(f"  Objective: Etoh(T_end) [mmol/L]")
    print(f"  Output   : {output_path}")
    print("=" * 66)

    model = read_sbml_model(str(args.model_path))
    configure_medium(model, args.condition)

    objectives = np.full((args.grid_n, args.grid_n), np.nan)
    sim_times  = np.zeros((args.grid_n, args.grid_n))
    lp_calls   = np.zeros((args.grid_n, args.grid_n), dtype=int)
    ode_evals  = np.zeros((args.grid_n, args.grid_n), dtype=int)

    t_total_start = time.time()
    sim_idx = 0

    for i, ackr2 in enumerate(ackr_vals):
        for j, ldh2 in enumerate(ldh_vals):
            obj, elapsed, n_lp, n_ode, ok = run_simulation(
                model, ackr1, ldh1, ackr2, ldh2,
                args.tau, args.bm0, args.glc0, args.t_end
            )
            objectives[i, j] = obj
            sim_times[i, j]  = elapsed
            lp_calls[i, j]   = n_lp
            ode_evals[i, j]  = n_ode
            sim_idx += 1
            if sim_idx % 10 == 0 or sim_idx == n_total:
                pct = 100 * sim_idx / n_total
                eta = ((time.time() - t_total_start) / sim_idx) * (n_total - sim_idx)
                print(f"  [{sim_idx:>3}/{n_total}] ({pct:.0f}%)  "
                      f"ACKr2={ackr2:+.2f}  LDH_D2={ldh2:+.2f}  "
                      f"etoh={obj:.4f}  {elapsed:.2f}s/sim  ETA {eta:.0f}s")

    total_time = time.time() - t_total_start
    best_idx   = np.unravel_index(np.nanargmax(objectives), objectives.shape)

    print(f"\n── LP piecewise summary ─────────────────────────────────────────")
    print(f"  Total wall time    : {total_time:.1f} s  ({total_time/60:.1f} min)")
    print(f"  Mean time / sim    : {sim_times.mean():.2f} s")
    print(f"  Mean LP calls / sim: {lp_calls.mean():.0f}")
    print(f"  Mean ODE evals/sim : {ode_evals.mean():.0f}")
    print(f"  Best ethanol       : {objectives[best_idx]:.4f} mmol/L  "
          f"(ACKr2={ackr_vals[best_idx[0]]:.3f}, "
          f"LDH_D2={ldh_vals[best_idx[1]]:.3f})")

    np.savez_compressed(
        output_path,
        ackr_vals=ackr_vals, ldh_vals=ldh_vals,
        objectives=objectives, sim_times=sim_times,
        lp_calls=lp_calls, ode_evals=ode_evals,
        total_time=total_time,
        ackr1=ackr1, ldh1=ldh1, tau=args.tau,
        bm0=args.bm0, glc0=args.glc0, t_end=args.t_end, grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
