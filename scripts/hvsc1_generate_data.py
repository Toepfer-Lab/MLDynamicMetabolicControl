"""
Generate training data for the HvSC1 community surrogate.

For each of N Dirichlet-sampled abundance vectors, cooperative_tradeoff is
solved once.  The community's own default exchange bounds are used as the
medium (minimal medium — no external CSV).  Infeasible solves are dropped.

Output npz keys (compatible with train_surrogate.py):
    X              (N_optimal, n_taxa)  — abundance vectors
    Y              (N_optimal, n_taxa)  — per-taxon growth rates
    feasible_range (n_taxa, 2)          — observed input ranges
    taxa_ids, community_gr, fraction, n_samples, seed

Usage:
    python scripts/hvsc1_generate_data.py \\
        --n-samples 20000 \\
        --fraction 0.5 \\
        --model-path model/hvsc1_comm.pickle \\
        --output data/hvsc1_training.npz
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
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples",  type=int,   default=20000)
    p.add_argument("--fraction",   type=float, default=0.5)
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "hvsc1_comm.pickle")
    p.add_argument("--output",     type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz")
    p.add_argument("--seed",       type=int,   default=SEED)
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, default_medium):
    """Re-apply the model's default medium and unlock cross-feeding."""
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


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading model")
    t0 = time.time()
    comm = load_pickle(str(args.model_path))
    print(f"  Loaded in {time.time()-t0:.1f} s")

    taxa_ids = list(comm.taxa)
    n_taxa   = len(taxa_ids)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    # Capture default medium before any mutation
    default_medium = dict(comm.medium)
    print(f"  Default medium components: {len(default_medium)}")

    # Unlock internal exchanges once here (also repeated inside solve)
    apply_medium_and_unlock(comm, default_medium)

    section(f"2. Sampling {args.n_samples} Dirichlet abundance vectors")
    rng     = np.random.default_rng(args.seed)
    samples = rng.dirichlet(np.ones(n_taxa), size=args.n_samples)

    X_buf  = np.zeros((args.n_samples, n_taxa))
    Y_buf  = np.zeros((args.n_samples, n_taxa))
    gr_buf = np.zeros(args.n_samples)
    ok_buf = np.zeros(args.n_samples, dtype=bool)

    t_sweep = time.time()
    pad = len(str(args.n_samples))

    for i, x in enumerate(samples):
        mu, status, comm_gr, elapsed = solve(
            comm, x, taxa_ids, default_medium, args.fraction
        )
        X_buf[i]  = x
        Y_buf[i]  = mu
        gr_buf[i] = comm_gr
        ok_buf[i] = (status == "optimal")

        if (i + 1) % 500 == 0 or i == 0 or i == args.n_samples - 1:
            n_ok_so_far = ok_buf[:i + 1].sum()
            print(f"  [{i+1:>{pad}}/{args.n_samples}]  "
                  f"last_status={status}  comm_gr={comm_gr:.4f}  "
                  f"time={elapsed:.1f}s  ok={n_ok_so_far}/{i+1}")

    sweep_time = time.time() - t_sweep
    n_ok = ok_buf.sum()
    print(f"\n  Sweep done in {sweep_time:.0f} s  ({sweep_time/3600:.2f} h)")
    print(f"  Optimal rows : {n_ok}/{args.n_samples}  "
          f"Dropped: {args.n_samples - n_ok}")

    X  = X_buf[ok_buf]
    Y  = Y_buf[ok_buf]
    gr = gr_buf[ok_buf]

    section("3. Data summary")
    print(f"  Training samples : {X.shape[0]}  (X: {X.shape}, Y: {Y.shape})")
    print(f"\n  Per-taxon growth rate range across optimal samples:")
    for j, t in enumerate(taxa_ids):
        lo, hi = Y[:, j].min(), Y[:, j].max()
        print(f"    {t:<6}  [{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f}")

    feasible_range = np.stack([X.min(axis=0), X.max(axis=0)], axis=1)

    section("4. Saving")
    np.savez_compressed(
        args.output,
        X=X,
        Y=Y,
        feasible_range=feasible_range,
        taxa_ids=taxa_ids,
        community_gr=gr,
        fraction=args.fraction,
        n_samples=args.n_samples,
        n_optimal=n_ok,
        seed=args.seed,
    )
    print(f"  X shape        : {X.shape}")
    print(f"  Y shape        : {Y.shape}")
    print(f"  feasible_range : {feasible_range.shape}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
