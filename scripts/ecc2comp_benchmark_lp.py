#!/usr/bin/env python
"""
ECC2comp LP benchmark — sweep over constant ACKr.

For each point on a 1-D grid of ACKr values, runs a full hybrid-ODE
simulation using model.optimize() at each ODE step, with the same
Michaelis-Menten/ethanol-inhibition rate scaling h(z) used by the trained
surrogate's hybrid_ode (src/hybrid_model.py) — only the flux-response step
differs (real LP solve here vs. NN forward pass in
ecc2comp_benchmark_surrogate.py). ACKr is held constant for the duration
of each simulation, matching the original ER&A single-manipulated-flux
reproduction at this model scale.

This is the smallest-scale point on the cross-scale complexity comparison
(scripts/complexity_comparison.py) — no benchmark previously existed here.

Objective: total biomass produced = BM(T_end) - BM0.

Results saved to results/ecc2comp_benchmark_lp.npz for comparison with
ecc2comp_benchmark_surrogate.py (same grid, same ODE, same ICs).

Usage:
    python scripts/ecc2comp_benchmark_lp.py
    python scripts/ecc2comp_benchmark_lp.py --grid-n 3   # quick smoke test
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

from hybrid_model import h  # noqa: E402
from runtime_utils import RESULTS_DIR, ensure_output_dirs  # noqa: E402

ACKR_RXN = "ACKr"


# ── medium / model setup ─────────────────────────────────────────────────────
# Mirrors configure_medium() in scripts/generate_fba_data.py, used to build
# the training data these surrogate checkpoints were trained on.

def configure_medium(model):
    model.reactions.get_by_id("EX_glyc_e").bounds = (0, 1000)
    model.reactions.get_by_id("EX_succ_e").bounds = (0, 1000)
    model.objective = "EX_Biomass"
    medium = model.medium
    medium["EX_glc__D_e"] = 10.0
    medium["EX_o2_e"] = 0.0
    medium["EX_glyc_e"] = 0.0
    medium["EX_succ_e"] = 0.0
    medium["EX_ac_e"] = 0.0
    model.medium = medium


# ── ODE / simulation ─────────────────────────────────────────────────────────

def run_simulation(model, ackr_val, glc0, etoh0, bm0, t_end):
    """
    One hybrid-ODE simulation with ACKr pinned to a constant, real LP solve
    at every ODE step (matching hybrid_model.hybrid_ode's rate structure,
    with model.optimize() in place of the surrogate).

    Returns (objective, wall_time_s, n_lp_calls, ode_evals, success).
    """
    lp_calls = [0]

    def rhs(t, z):
        z = np.maximum(z, 0)
        glucose, ethanol, biomass = z
        sol = model.optimize()
        lp_calls[0] += 1
        if sol.status != "optimal":
            return [0.0, 0.0, 0.0]
        v_glc  = sol.fluxes["EX_glc__D_e"]
        v_etoh = sol.fluxes["EX_etoh_e"]
        v_bio  = sol.fluxes["EX_Biomass"]
        rate = biomass * h(z)
        dzdt = np.array([rate * v_glc, rate * v_etoh, rate * v_bio])
        dzdt[z <= 0] = np.maximum(dzdt[z <= 0], 0)
        return dzdt

    with model:
        model.reactions.get_by_id(ACKR_RXN).bounds = (ackr_val, ackr_val)
        try:
            model.solver.configuration.timeout = 5
        except Exception:
            pass
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
    return objective, elapsed, lp_calls[0], result.nfev, result.success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-n",    type=int,   default=10)
    p.add_argument("--t-end",     type=float, default=12.0)
    p.add_argument("--glc0",      type=float, default=10.0)
    p.add_argument("--etoh0",     type=float, default=0.0)
    p.add_argument("--bm0",       type=float, default=0.01)
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "ECC2comp_configured")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "ACKr_trained_model_input-1_output-4_hidden-4.pt",
                   help="Only used to source the ACKr feasible_range (same domain "
                        "as ecc2comp_benchmark_surrogate.py); no LP solves depend on it.")
    p.add_argument("--output",    type=Path, default=None)
    return p.parse_args()


def main():
    import torch  # local import: only needed to read checkpoint metadata

    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or RESULTS_DIR / "ecc2comp_benchmark_lp.npz"

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ackr_lo, ackr_hi = payload["metadata"]["feasible_range"]
    ackr_vals = np.linspace(ackr_lo, ackr_hi, args.grid_n)
    n_total = args.grid_n

    print("=" * 66)
    print("  ECC2comp LP benchmark — 1-D grid over ACKr")
    print(f"  Grid       : {args.grid_n} simulations")
    print(f"  ACKr range : [{ackr_lo:.3f}, {ackr_hi:.3f}]")
    print(f"  T_end      : {args.t_end} h  |  Glc0={args.glc0}  Etoh0={args.etoh0}  Bm0={args.bm0}")
    print(f"  Output     : {output_path}")
    print("=" * 66)

    print(f"\nLoading model from {args.model_path} ...")
    model = read_sbml_model(str(args.model_path))
    configure_medium(model)

    objectives = np.full(args.grid_n, np.nan)
    sim_times  = np.zeros(args.grid_n)
    lp_calls   = np.zeros(args.grid_n, dtype=int)
    ode_evals  = np.zeros(args.grid_n, dtype=int)

    t_total_start = time.time()
    for i, ackr in enumerate(ackr_vals):
        obj, elapsed, n_lp, n_ode, ok = run_simulation(
            model, ackr, args.glc0, args.etoh0, args.bm0, args.t_end
        )
        objectives[i] = obj
        sim_times[i]  = elapsed
        lp_calls[i]   = n_lp
        ode_evals[i]  = n_ode
        print(f"  [{i+1:>3}/{n_total}]  ACKr={ackr:+.3f}  "
              f"obj={obj:.4f}  {elapsed:.3f}s/sim  lp_calls={n_lp}")

    total_time = time.time() - t_total_start
    best_idx   = int(np.nanargmax(objectives))

    print(f"\n── ECC2comp LP benchmark summary ───────────────────────────────")
    print(f"  Total wall time    : {total_time:.2f} s")
    print(f"  Mean time / sim    : {sim_times.mean():.3f} s  ({sim_times.mean()*1000:.1f} ms)")
    print(f"  Mean LP calls / sim: {lp_calls.mean():.0f}")
    print(f"  Mean LP time / call: {(sim_times/np.maximum(lp_calls,1)).mean()*1000:.3f} ms")
    print(f"  Mean ODE evals/sim : {ode_evals.mean():.0f}")
    print(f"  Best objective     : {objectives[best_idx]:.4f} gDW/L  (ACKr={ackr_vals[best_idx]:.3f})")

    np.savez_compressed(
        output_path,
        ackr_vals=ackr_vals,
        objectives=objectives,
        sim_times=sim_times,
        lp_calls=lp_calls,
        ode_evals=ode_evals,
        total_time=total_time,
        glc0=args.glc0, etoh0=args.etoh0, bm0=args.bm0,
        t_end=args.t_end,
        grid_n=args.grid_n,
    )
    print(f"  Saved to           : {output_path}")


if __name__ == "__main__":
    main()
