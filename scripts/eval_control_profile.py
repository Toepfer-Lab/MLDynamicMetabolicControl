#!/usr/bin/env python3
"""
Compare best constant vman control vs a saved optimized piecewise profile.

What it does:
  1) Finds the best CONSTANT vman value (maximizing final biomass at t_end)
     via coarse grid + optional golden-section refinement.
  2) Loads an optimized profile from optimize_vman_*.npz (opt_vman_values + control_times)
  3) Simulates both and plots biomass trajectories.

Example:
  python scripts/compare_best_constant_vs_opt.py \
    --checkpoint trained_models/ACKr_trained_model_input-1_output-4_hidden-4.pt \
    --opt-results results/optimize_vman_ACKr_hidden-4_tend-12.0_N-20.npz \
    --lb -32 --ub 12 \
    --grid 41 \
    --refine \
    --also-const -8 -16 0 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hybrid_model  # noqa: E402
from flux_config import STATE_INDEX  # noqa: E402
from runtime_utils import load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402


def piecewise_constant_control(control_times, values):
    values = np.asarray(values, dtype=float)

    def control(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return control


def simulate_final_biomass(model, x_scaler, y_scaler, z0, t_span, t_eval, vman_t):
    rhs = lambda t, z: hybrid_model.hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)
    sol = solve_ivp(rhs, t_span, np.asarray(z0, dtype=float), t_eval=t_eval, method="RK45")
    if not sol.success:
        return None, sol
    B = sol.y[STATE_INDEX["biomass"], :]
    return float(B[-1]), sol


def golden_section_max(f, a, b, tol=1e-3, max_iter=60):
    """
    Maximize f on [a,b] using golden-section search (unimodal assumption locally).
    Returns (x_best, f_best).
    """
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c)
    fd = f(d)

    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc is None:
            fc = -np.inf
        if fd is None:
            fd = -np.inf

        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
        it += 1

    x_best = (a + b) / 2
    f_best = f(x_best)
    return x_best, f_best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--opt-results", type=Path, required=True)

    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=12.0)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--initial-state", type=float, nargs=3, default=[10.0, 0.0, 0.01],
                   metavar=("glucose", "ethanol", "biomass"))

    # constant-search bounds
    p.add_argument("--lb", type=float, required=True, help="Lower bound for constant vman search")
    p.add_argument("--ub", type=float, required=True, help="Upper bound for constant vman search")
    p.add_argument("--grid", type=int, default=41, help="Coarse grid points for constant search")

    p.add_argument("--refine", action="store_true",
                   help="Polish best constant with golden-section search in a small bracket.")
    p.add_argument("--refine-width", type=float, default=4.0,
                   help="Half-width around best coarse v to refine (clipped to [lb,ub]).")
    p.add_argument("--refine-tol", type=float, default=1e-3)

    p.add_argument("--also-const", type=float, nargs="*", default=None,
                   help="Extra constant values to simulate/plot for reference (e.g. -8).")

    p.add_argument("--outdir", type=Path, default=Path("plots/baselines"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    t_span = (args.t_start, args.t_end)
    t_eval = np.linspace(args.t_start, args.t_end, args.n_eval)

    # --- load optimized profile
    res = np.load(args.opt_results, allow_pickle=True)
    opt_vals = np.asarray(res["opt_vman_values"], dtype=float)
    control_times = np.asarray(res["control_times"], dtype=float)
    vman_opt = piecewise_constant_control(control_times, opt_vals)

    opt_final, sol_opt = simulate_final_biomass(
        model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_opt
    )

    # --- coarse constant search
    grid_vs = np.linspace(args.lb, args.ub, args.grid)
    best_v = None
    best_B = -np.inf
    best_sol = None

    def eval_const(v):
        vman_t = (lambda t, vv=float(v): vv)
        finalB, _ = simulate_final_biomass(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_t)
        return finalB

    for v in grid_vs:
        Bv = eval_const(v)
        if Bv is None:
            continue
        if Bv > best_B:
            best_B = Bv
            best_v = float(v)

    # --- optional refine around best_v
    if args.refine and best_v is not None:
        a = max(args.lb, best_v - args.refine_width)
        b = min(args.ub, best_v + args.refine_width)

        def f(v):
            return eval_const(v)

        v_ref, B_ref = golden_section_max(f, a, b, tol=args.refine_tol)
        if B_ref is not None and B_ref > best_B:
            best_v, best_B = float(v_ref), float(B_ref)

    # simulate best constant (full trajectory)
    vman_best = (lambda t, vv=float(best_v): vv)
    best_final, sol_best = simulate_final_biomass(
        model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_best
    )

    # --- print summary
    print("\n=== Best constant vs optimized ===")
    print(f"Best constant v*: {best_v:.6g}   final biomass: {best_final:.6g}")
    if opt_final is None or not sol_opt.success:
        print("Optimized profile: FAILED")
    else:
        print(f"Optimized profile final biomass: {opt_final:.6g}")

    # --- plot biomass trajectories
    plt.figure(figsize=(9, 5))

    if sol_opt is not None and sol_opt.success:
        plt.plot(sol_opt.t, sol_opt.y[STATE_INDEX["biomass"], :], label=f"optimized (final={opt_final:.4g})")

    if sol_best is not None and sol_best.success:
        plt.plot(sol_best.t, sol_best.y[STATE_INDEX["biomass"], :], label=f"best constant v={best_v:.4g} (final={best_final:.4g})")

    # extra constants
    if args.also_const:
        for v in args.also_const:
            vman_t = (lambda t, vv=float(v): vv)
            Bv, solv = simulate_final_biomass(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_t)
            if solv.success:
                plt.plot(solv.t, solv.y[STATE_INDEX["biomass"], :], alpha=0.6, label=f"const v={v:g} (final={Bv:.4g})")

    plt.xlabel("Time [h]")
    plt.ylabel("Biomass")
    plt.title("Best constant control vs optimized profile")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    outpath = args.outdir / f"best_constant_vs_opt_{args.opt_results.stem}.png"
    plt.savefig(outpath, dpi=200)
    print(f"\nSaved plot to {outpath}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()