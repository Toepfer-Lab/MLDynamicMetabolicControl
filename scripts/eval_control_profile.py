#!/usr/bin/env python3
"""
Compare best constant vman control vs a saved optimized piecewise profile,
and visualize what the hybrid model is doing internally.

Adds:
  - vman(t) trace
  - states z(t): glucose, ethanol, biomass
  - h(z) and rate = biomass * h(z)
  - NN-predicted exchange fluxes: glc, etoh, co2, biomass
  - effective rates (ODE derivatives): rate*v_glc, rate*v_etoh, rate*v_bio

Example:
python scripts/eval_control_profile.py --checkpoint trained_models/ACKr_trained_model_input-1_output-4_hidden-4.pt     --opt-results results/optimize_vman_ACKr_hidden-4_tend-12.0_N-20.npz     --lb -32 --ub 12     --grid 41     -
-refine     --also-const -8 
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
from flux_config import STATE_INDEX, FLUX_INDEX  # noqa: E402
from runtime_utils import load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402


# ---------------------------
# control helpers
# ---------------------------
def piecewise_constant_control(control_times, values):
    values = np.asarray(values, dtype=float)

    def control(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return control


# ---------------------------
# simulation helpers
# ---------------------------
def simulate(model, x_scaler, y_scaler, z0, t_span, t_eval, vman_t):
    """Run solve_ivp and return solution."""
    rhs = lambda t, z: hybrid_model.hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)
    sol = solve_ivp(rhs, t_span, np.asarray(z0, dtype=float), t_eval=t_eval, method="RK45")
    return sol


def final_biomass_from_sol(sol):
    if sol is None or (not sol.success):
        return None
    return float(sol.y[STATE_INDEX["biomass"], -1])


def golden_section_max(f, a, b, tol=1e-3, max_iter=60):
    """Maximize f on [a,b] using golden-section search."""
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


def compute_h_components(glucose, ethanol):
    """
    Return (glc_term, etoh_term, h).
    glc_term = glucose/(glucose+ks)
    etoh_term = 1/(1+ethanol/ki)
    h = glc_term * etoh_term
    """
    ks = 2.964e-4
    ki = 25.0
    glc_term = glucose / (glucose + ks)
    etoh_term = 1.0 / (1.0 + ethanol / ki)
    return glc_term, etoh_term, glc_term * etoh_term


def log_hybrid_internals(sol, vman_t, model, x_scaler, y_scaler):
    """
    Evaluate hybrid internals on the solution time grid:
      - vman(t)
      - NN-predicted vext(t) based on vman(t)
      - h(z), components, rate=biomass*h
      - effective rates: rate*v_glc, rate*v_etoh, rate*v_bio
    """
    if sol is None or (not sol.success):
        return None

    t = sol.t
    z = sol.y  # shape: (n_states, n_time)
    glucose = z[STATE_INDEX["glucose"], :]
    ethanol = z[STATE_INDEX["ethanol"], :]
    biomass = z[STATE_INDEX["biomass"], :]

    # vman(t)
    vman = np.array([float(vman_t(tt)) for tt in t], dtype=float)

    # h(z) and components
    glc_term, etoh_term, hval = compute_h_components(glucose, ethanol)
    rate = biomass * hval

    # NN-predicted exchange fluxes as function of vman only
    # (mirrors hybrid_ode logic)
    vman_scaled = x_scaler.transform(vman.reshape(-1, 1))
    # torch-free inference (safe + fast): call model directly with torch
    import torch  # local import to keep script standalone
    with torch.no_grad():
        vext_scaled = model(torch.tensor(vman_scaled, dtype=torch.float32)).numpy()
    vext = y_scaler.inverse_transform(vext_scaled)  # shape: (T, n_flux)

    # pull named fluxes (must exist in FLUX_INDEX)
    v_etoh = vext[:, FLUX_INDEX["etoh"]]
    v_glc  = vext[:, FLUX_INDEX["glc"]]
    v_co2  = vext[:, FLUX_INDEX["co2"]]
    v_bio  = vext[:, FLUX_INDEX["biomass"]]

    # effective rates / derivatives per your ODE definition
    dglc_dt = rate * v_glc
    detoh_dt = rate * v_etoh
    dbio_dt = rate * v_bio

    return {
        "t": t,
        "glucose": glucose,
        "ethanol": ethanol,
        "biomass": biomass,
        "vman": vman,
        "glc_term": glc_term,
        "etoh_term": etoh_term,
        "h": hval,
        "rate": rate,
        "v_glc": v_glc,
        "v_etoh": v_etoh,
        "v_co2": v_co2,
        "v_bio": v_bio,
        "dglc_dt": dglc_dt,
        "detoh_dt": detoh_dt,
        "dbio_dt": dbio_dt,
    }


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--opt-results", type=Path, required=True)

    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=12.0)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument(
        "--initial-state",
        type=float,
        nargs=3,
        default=[10.0, 0.0, 0.01],
        metavar=("glucose", "ethanol", "biomass"),
    )

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


# ---------------------------
# plotting
# ---------------------------
def plot_runs(runs, outpath: Path, title: str):
    """
    runs: list of dicts, each with keys:
      - name (label)
      - final (final biomass)
      - log (dict from log_hybrid_internals)
    """
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    ax0, ax1, ax2, ax3, ax4 = axes

    # 1) vman(t)
    for r in runs:
        log = r["log"]
        ax0.plot(log["t"], log["vman"], label=r["name"])
    ax0.set_ylabel("vman")
    ax0.grid(True)
    ax0.legend(loc="best")

    # 2) states
    for r in runs:
        log = r["log"]
        ax1.plot(log["t"], log["glucose"], label=f"{r['name']} | glc")
        ax1.plot(log["t"], log["ethanol"], label=f"{r['name']} | etoh", linestyle="--")
        ax1.plot(log["t"], log["biomass"], label=f"{r['name']} | bio", linestyle=":")
    ax1.set_ylabel("States")
    ax1.grid(True)
    ax1.legend(loc="best", ncol=2)

    # 3) rate factors
    for r in runs:
        log = r["log"]
        ax2.plot(log["t"], log["h"], label=f"{r['name']} | h(z)")
        ax2.plot(log["t"], log["rate"], label=f"{r['name']} | rate=b*h", linestyle="--")
        # optionally show components as faint lines
        ax2.plot(log["t"], log["glc_term"], alpha=0.35, label=f"{r['name']} | glc_term")
        ax2.plot(log["t"], log["etoh_term"], alpha=0.35, label=f"{r['name']} | etoh_term")
    ax2.set_ylabel("Rate factors")
    ax2.grid(True)
    ax2.legend(loc="best", ncol=2)

    # 4) NN-predicted exchange fluxes
    for r in runs:
        log = r["log"]
        ax3.plot(log["t"], log["v_glc"], label=f"{r['name']} | v_glc")
        ax3.plot(log["t"], log["v_etoh"], label=f"{r['name']} | v_etoh", linestyle="--")
        ax3.plot(log["t"], log["v_co2"], label=f"{r['name']} | v_co2", linestyle=":")
        ax3.plot(log["t"], log["v_bio"], label=f"{r['name']} | v_bio", linestyle="-.")

    ax3.set_ylabel("Exchange fluxes\n(NN outputs)")
    ax3.grid(True)
    ax3.legend(loc="best", ncol=2)

    # 5) effective rates (ODE derivatives)
    for r in runs:
        log = r["log"]
        ax4.plot(log["t"], log["dglc_dt"], label=f"{r['name']} | dglc/dt")
        ax4.plot(log["t"], log["detoh_dt"], label=f"{r['name']} | detoh/dt", linestyle="--")
        ax4.plot(log["t"], log["dbio_dt"], label=f"{r['name']} | dbio/dt", linestyle=":")
    ax4.set_xlabel("Time [h]")
    ax4.set_ylabel("Effective rates")
    ax4.grid(True)
    ax4.legend(loc="best", ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=200)
    print(f"\nSaved plot to {outpath}")


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

    sol_opt = simulate(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_opt)
    opt_final = final_biomass_from_sol(sol_opt)

    # --- coarse constant search
    grid_vs = np.linspace(args.lb, args.ub, args.grid)
    best_v = None
    best_B = -np.inf

    def eval_const(v):
        vman_t = (lambda t, vv=float(v): vv)
        sol = simulate(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_t)
        return final_biomass_from_sol(sol)

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

    # simulate best constant
    vman_best = (lambda t, vv=float(best_v): vv)
    sol_best = simulate(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_best)
    best_final = final_biomass_from_sol(sol_best)

    print("\n=== Best constant vs optimized ===")
    print(f"Best constant v*: {best_v:.6g}   final biomass: {best_final:.6g}")
    if opt_final is None or not sol_opt.success:
        print("Optimized profile: FAILED")
    else:
        print(f"Optimized profile final biomass: {opt_final:.6g}")

    runs = []

    # optimized
    if sol_opt is not None and sol_opt.success:
        log_opt = log_hybrid_internals(sol_opt, vman_opt, model, x_scaler, y_scaler)
        runs.append({
            "name": f"optimized (final={opt_final:.3g})",
            "final": opt_final,
            "log": log_opt,
        })

    # best constant
    if sol_best is not None and sol_best.success:
        log_best = log_hybrid_internals(sol_best, vman_best, model, x_scaler, y_scaler)
        runs.append({
            "name": f"best const v={best_v:.3g} (final={best_final:.3g})",
            "final": best_final,
            "log": log_best,
        })

    # extra constants
    if args.also_const:
        for v in args.also_const:
            vman_t = (lambda t, vv=float(v): vv)
            sol = simulate(model, x_scaler, y_scaler, args.initial_state, t_span, t_eval, vman_t)
            final = final_biomass_from_sol(sol)
            if sol.success:
                log = log_hybrid_internals(sol, vman_t, model, x_scaler, y_scaler)
                runs.append({
                    "name": f"const v={v:g} (final={final:.3g})",
                    "final": final,
                    "log": log,
                })

    outpath = args.outdir / f"hybrid_diagnostics_{args.opt_results.stem}.png"
    plot_runs(
        runs,
        outpath,
        title="Hybrid model diagnostics: control, states, h(z), exchange fluxes, and effective rates",
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()