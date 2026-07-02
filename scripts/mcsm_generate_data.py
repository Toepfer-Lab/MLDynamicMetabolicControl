"""
Generate training data for the MCSM surrogate.

For each of N Dirichlet-sampled abundance vectors, run cooperative_tradeoff
and record per-taxon growth rates. Infeasible solves are dropped and a warning
is printed. The output npz uses the same keys (X, Y, feasible_range) that
train_surrogate.py expects.

Usage:
    python scripts/mcsm_generate_data.py \\
        --n-samples 2000 \\
        --fraction 0.5 \\
        --model-path model/dcom.pickle \\
        --medium-path data/Completed_maize_leaf_medium.csv \\
        --output data/mcsm_training.npz
"""

import argparse
import time
from pathlib import Path

import cobra
import micom
import numpy as np
import pandas as pd
from micom import load_pickle

cobra.Configuration.solver = "cplex"

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_ABUND = 1e-8
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--fraction", type=float, default=0.5,
                   help="cooperative_tradeoff fraction parameter")
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "dcom.pickle")
    p.add_argument("--medium-path", type=Path,
                   default=REPO_ROOT / "data" / "Completed_maize_leaf_medium.csv")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "data" / "mcsm_training.npz")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, medium_dict):
    comm.medium = medium_dict
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def solve(comm, abundance_vec, taxa_ids, medium_dict, fraction):
    comm.set_abundance(pd.Series(dict(zip(taxa_ids, abundance_vec))))
    apply_medium_and_unlock(comm, medium_dict)
    t0 = time.time()
    sol = comm.cooperative_tradeoff(fraction=fraction, pfba=False, fluxes=False)
    elapsed = time.time() - t0
    mu = sol.members.reindex(taxa_ids)["growth_rate"].values.astype(float)
    return mu, sol.status, float(sol.growth_rate), elapsed


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading model and applying medium")
    t0 = time.time()
    comm = load_pickle(str(args.model_path))
    print(f"  Loaded in {time.time()-t0:.1f} s")

    medium_df = pd.read_csv(args.medium_path, index_col=0)
    medium_dict = dict(zip(medium_df["reaction"], medium_df["flux"]))
    apply_medium_and_unlock(comm, medium_dict)

    taxa_ids = list(comm.taxa)
    n_taxa = len(taxa_ids)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")
    print(f"  Medium components accepted: {len(comm.medium)}")

    section(f"2. Sampling {args.n_samples} Dirichlet abundance vectors")
    rng = np.random.default_rng(args.seed)
    samples = rng.dirichlet(np.ones(n_taxa), size=args.n_samples)

    X_all = np.zeros((args.n_samples, n_taxa))
    Y_all = np.zeros((args.n_samples, n_taxa))
    gr_all = np.zeros(args.n_samples)
    ok_mask = np.ones(args.n_samples, dtype=bool)

    t_sweep = time.time()
    for i, x in enumerate(samples):
        mu, status, comm_gr, elapsed = solve(
            comm, x, taxa_ids, medium_dict, args.fraction
        )
        X_all[i] = x
        Y_all[i] = mu
        gr_all[i] = comm_gr
        if status != "optimal":
            ok_mask[i] = False
        if (i + 1) % 100 == 0 or i == 0 or i == args.n_samples - 1:
            n_ok = ok_mask[: i + 1].sum()
            print(f"  [{i+1:>{len(str(args.n_samples))}}/{args.n_samples}]  "
                  f"status={status}  comm_gr={comm_gr:.4f}  "
                  f"time={elapsed:.1f}s  ok={n_ok}")

    sweep_time = time.time() - t_sweep
    n_ok = ok_mask.sum()
    n_infeasible = args.n_samples - n_ok
    print(f"\n  Sweep done in {sweep_time:.0f} s")
    print(f"  Optimal: {n_ok}/{args.n_samples}  Infeasible: {n_infeasible}")

    X = X_all[ok_mask]
    Y = Y_all[ok_mask]
    gr = gr_all[ok_mask]

    section("3. Data summary")
    print(f"  Training samples: {X.shape[0]}")
    print(f"\n  Per-taxon growth rate range (optimal samples only):")
    for j, t in enumerate(taxa_ids):
        lo, hi = Y[:, j].min(), Y[:, j].max()
        print(f"    {t:<12}  [{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f}")
    print(f"\n  Community growth rate: min={gr.min():.4f}  "
          f"max={gr.max():.4f}  mean={gr.mean():.4f}")

    # feasible_range: (n_inputs, 2) — observed min/max per input dimension
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
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  feasible_range shape: {feasible_range.shape}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
