"""
Community abundance dynamics — landscape sweep and simulation loop for HvSC1.

Generalised version of mcsm_simulate.py that accepts CLI arguments for
model path, output paths, and simulation parameters.  The community's own
default exchange bounds are used as the medium (no external CSV).

Two tasks in one job:

  Task 1 — Landscape sweep (N_LANDSCAPE Dirichlet samples):
    For each sampled abundance vector, run cooperative_tradeoff and record
    per-taxon growth rates.

  Task 2 — Simulation loop from multiple starting points:
    Euler / replicator update:  x_new = x * (1 + mu * dt), renormalise.

Results saved to:
    --output-landscape   (default: results/hvsc1_landscape.npz)
    --output-trajectories (default: results/hvsc1_trajectories.npz)
"""

import argparse
import time
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from micom import load_pickle

cobra.Configuration.solver = "cplex"

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_ABUND = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "hvsc1_comm.pickle")
    p.add_argument("--output-landscape", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_landscape.npz")
    p.add_argument("--output-trajectories", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_trajectories.npz")
    p.add_argument("--n-landscape", type=int,   default=200)
    p.add_argument("--n-steps",     type=int,   default=80)
    p.add_argument("--dt",          type=float, default=0.1)
    p.add_argument("--fraction",    type=float, default=0.5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--convergence-threshold", type=float, default=0.99,
                   help="Stop LP calls when dominant taxon exceeds this abundance. "
                        "Prevents numeric LP failures at near-monoculture states.")
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, default_medium):
    """Re-apply model default medium and unlock internal exchange cross-feeding."""
    comm.medium = default_medium
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def solve(comm, abundance_vec, taxa_ids, default_medium, fraction):
    comm.set_abundance(pd.Series(dict(zip(taxa_ids, abundance_vec))))
    apply_medium_and_unlock(comm, default_medium)
    t0 = time.time()
    sol = comm.cooperative_tradeoff(fraction=fraction, pfba=False, fluxes=False)
    elapsed = time.time() - t0
    mu = sol.members.reindex(taxa_ids)["growth_rate"].values.astype(float)
    return mu, sol.status, float(sol.growth_rate), elapsed


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def main():
    args = parse_args()
    args.output_landscape.parent.mkdir(parents=True, exist_ok=True)
    args.output_trajectories.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading model")
    t0 = time.time()
    comm = load_pickle(str(args.model_path))
    print(f"  Loaded in {time.time()-t0:.1f} s")

    taxa_ids = list(comm.taxa)
    n_taxa   = len(taxa_ids)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    # Capture default medium BEFORE any set_abundance call
    default_medium = dict(comm.medium)
    print(f"  Default medium components: {len(default_medium)}")
    apply_medium_and_unlock(comm, default_medium)

    # Capture original abundances BEFORE landscape sweep mutates comm.taxonomy
    orig_abund = (comm.taxonomy
                  .set_index("id")["abundance"]
                  .reindex(taxa_ids).values.astype(float))
    orig_abund /= orig_abund.sum()
    print(f"  Original abundance from model taxonomy:")
    for t, v in zip(taxa_ids, orig_abund):
        print(f"    {t:<6} {v:.6f}")

    # ── 2. Landscape sweep ────────────────────────────────────────────────────
    section(f"2. Landscape sweep — {args.n_landscape} Dirichlet samples")
    rng     = np.random.default_rng(args.seed)
    samples = rng.dirichlet(np.ones(n_taxa), size=args.n_landscape)

    landscape_x  = np.zeros((args.n_landscape, n_taxa))
    landscape_mu = np.zeros((args.n_landscape, n_taxa))
    landscape_gr = np.zeros(args.n_landscape)
    landscape_ok = np.ones(args.n_landscape, dtype=bool)

    t_sweep_start = time.time()
    for i, x in enumerate(samples):
        mu, status, comm_gr, elapsed = solve(
            comm, x, taxa_ids, default_medium, args.fraction
        )
        landscape_x[i]  = x
        landscape_mu[i] = mu
        landscape_gr[i] = comm_gr
        if status != "optimal":
            landscape_ok[i] = False
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:>3}/{args.n_landscape}]  status={status}  "
                  f"comm_gr={comm_gr:.4f}  time={elapsed:.1f}s")

    sweep_time = time.time() - t_sweep_start
    n_ok = landscape_ok.sum()
    print(f"\n  Sweep done in {sweep_time:.0f} s  ({n_ok}/{args.n_landscape} optimal)")
    print(f"  Community growth rate: min={landscape_gr[landscape_ok].min():.4f}  "
          f"max={landscape_gr[landscape_ok].max():.4f}  "
          f"mean={landscape_gr[landscape_ok].mean():.4f}")
    print(f"\n  Per-taxon growth rate range across all samples:")
    for j, t in enumerate(taxa_ids):
        lo = landscape_mu[landscape_ok, j].min()
        hi = landscape_mu[landscape_ok, j].max()
        print(f"    {t:<6}  [{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f}")

    np.savez_compressed(
        args.output_landscape,
        taxa_ids=taxa_ids, x=landscape_x, mu=landscape_mu,
        community_gr=landscape_gr, optimal=landscape_ok,
        fraction=args.fraction, n_samples=args.n_landscape, seed=args.seed,
    )
    print(f"\n  Saved: {args.output_landscape}")

    # ── 3. Simulation loop ────────────────────────────────────────────────────
    section(f"3. Simulation — {args.n_steps} steps × dt={args.dt}h = "
            f"{args.n_steps * args.dt:.1f}h total")

    start_profiles = {
        "original"    : orig_abund,
        "dirichlet_a" : rng.dirichlet(np.ones(n_taxa)),
        "dirichlet_b" : rng.dirichlet(np.ones(n_taxa)),
        "dirichlet_c" : rng.dirichlet(np.ones(n_taxa)),
        "dirichlet_d" : rng.dirichlet(np.ones(n_taxa)),
    }

    all_trajectories = {}
    all_mu_trajs     = {}
    all_gr_trajs     = {}
    all_times        = {}

    print(f"\n  Initial abundances (top-5 taxa by original abundance shown):")
    top5 = np.argsort(orig_abund)[::-1][:5]
    header = f"  {'profile':<16}" + "".join(f"  {taxa_ids[j]:<8}" for j in top5)
    print(header)
    for name, x0 in start_profiles.items():
        row = f"  {name:<16}" + "".join(f"  {x0[j]:<8.4f}" for j in top5)
        print(row)

    for name, x0 in start_profiles.items():
        print(f"\n  ── Trajectory: {name} ──")
        traj    = np.full((args.n_steps + 1, n_taxa), np.nan)
        mu_traj = np.full((args.n_steps,     n_taxa), np.nan)
        gr_traj = np.full((args.n_steps,),           np.nan)
        traj[0] = x0.copy()
        x = x0.copy()
        t_sim = time.time()

        last_mu = np.zeros(n_taxa)   # fallback growth rates for non-optimal solves
        last_gr = 0.0
        for step in range(args.n_steps):
            # Stop LP calls once a taxon dominates — prevents numeric LP failures
            # when 26 taxa are simultaneously near MIN_ABUND.
            if np.max(x) >= args.convergence_threshold:
                dominant = taxa_ids[np.argmax(x)]
                print(f"    step {step+1:>3}  converged: dominant={dominant} "
                      f"({x.max():.4f}) — filling remaining {args.n_steps - step} steps")
                for rem in range(step, args.n_steps):
                    traj[rem + 1] = x
                    mu_traj[rem]  = last_mu
                    gr_traj[rem]  = last_gr
                break

            mu, status, comm_gr, elapsed = solve(
                comm, x, taxa_ids, default_medium, args.fraction
            )
            if status != "optimal":
                # Graceful fallback: reuse last valid growth rates rather than
                # stopping. Prints a warning so non-optimal steps are visible.
                print(f"    step {step+1:>3}  WARNING status={status} — "
                      f"using last valid mu (dominant={taxa_ids[np.argmax(x)]} "
                      f"{x.max():.3f})")
                mu      = last_mu.copy()
                comm_gr = last_gr
            else:
                last_mu = mu.copy()
                last_gr = comm_gr

            mu_traj[step] = mu
            gr_traj[step] = comm_gr
            x = euler_step(x, mu, args.dt)
            traj[step + 1] = x

            if (step + 1) % 10 == 0:
                dominant = taxa_ids[np.argmax(x)]
                print(f"    step {step+1:>3}  t={(step+1)*args.dt:5.1f}h  "
                      f"comm_gr={comm_gr:.4f}  dominant={dominant} ({x.max():.3f})")

        sim_elapsed = time.time() - t_sim
        all_trajectories[name] = traj
        all_mu_trajs[name]     = mu_traj
        all_gr_trajs[name]     = gr_traj
        all_times[name]        = sim_elapsed

        final_x  = traj[~np.isnan(traj).any(axis=1)][-1]
        dominant = taxa_ids[np.argmax(final_x)]
        print(f"  Final state (t={args.n_steps*args.dt:.1f}h):  "
              f"dominant={dominant} ({final_x.max():.4f})")
        print(f"  Wall time: {sim_elapsed:.0f} s")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    section("4. Saving trajectory results")
    np.savez_compressed(
        args.output_trajectories,
        taxa_ids      = taxa_ids,
        profile_names = list(all_trajectories.keys()),
        trajectories  = np.stack(list(all_trajectories.values())),
        mu_trajs      = np.stack(list(all_mu_trajs.values())),
        gr_trajs      = np.stack(list(all_gr_trajs.values())),
        dt=args.dt, n_steps=args.n_steps, fraction=args.fraction, seed=args.seed,
    )
    print(f"  Saved: {args.output_trajectories}")

    section("5. Summary")
    print(f"  Landscape: {args.n_landscape} samples, {n_ok} optimal, {sweep_time:.0f}s")
    for name, t_wall in all_times.items():
        final_x = all_trajectories[name]
        final_x = final_x[~np.isnan(final_x).any(axis=1)][-1]
        dominant = taxa_ids[np.argmax(final_x)]
        print(f"  {name:<16}  wall={t_wall:.0f}s  "
              f"final dominant={dominant} ({final_x.max():.3f})")


if __name__ == "__main__":
    main()
