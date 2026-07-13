"""
Generate training data for the MCSM surrogate.

For each of N Dirichlet-sampled abundance vectors, cooperative_tradeoff is
solved TWICE: once with the full curated medium (gln_flag=1) and once with
the glutamine exchange set to zero (gln_flag=0).  The binary flag is appended
as the 7th input column, so the surrogate learns both regimes from a single
model.  Infeasible solves are dropped individually.

Output npz keys (compatible with train_surrogate.py):
    X              (2*N_optimal, n_taxa+1)  — abundances + gln_flag
    Y              (2*N_optimal, n_taxa)    — per-taxon growth rates
    feasible_range (n_taxa+1, 2)            — observed input ranges
    taxa_ids, community_gr, gln_exchange, fraction, n_samples, seed

Usage:
    python scripts/mcsm_generate_data.py \\
        --n-samples 5000 \\
        --fraction 0.5 \\
        --gln-exchange EX_gln__L_m \\
        --model-path model/dcom.pickle \\
        --medium-path data/Completed_maize_leaf_medium.csv \\
        --output data/mcsm_training.npz
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
    p.add_argument("--n-samples",    type=int,  default=5000)
    p.add_argument("--fraction",     type=float, default=0.5)
    p.add_argument("--gln-exchange", type=str,  default="EX_gln__L_m",
                   help="Community exchange reaction ID for glutamine")
    p.add_argument("--model-path",   type=Path,
                   default=REPO_ROOT / "model" / "dcom.pickle")
    p.add_argument("--medium-path",  type=Path,
                   default=REPO_ROOT / "data" / "Completed_maize_leaf_medium.csv")
    p.add_argument("--output",       type=Path,
                   default=REPO_ROOT / "data" / "mcsm_training.npz")
    p.add_argument("--seed",         type=int,  default=SEED)
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, medium_dict):
    """Set community medium and remove artificial cap on cross-feeding."""
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

    section("1. Loading model and medium")
    t0 = time.time()
    comm = load_pickle(str(args.model_path))
    print(f"  Loaded in {time.time()-t0:.1f} s")

    medium_df = pd.read_csv(args.medium_path, index_col=0)
    medium_full  = dict(zip(medium_df["reaction"], medium_df["flux"]))
    medium_no_gln = {k: v for k, v in medium_full.items()}
    medium_no_gln[args.gln_exchange] = 0.0

    gln_present = medium_full.get(args.gln_exchange, 0.0)
    if gln_present == 0.0:
        print(f"  WARNING: {args.gln_exchange} not found or already 0 in medium CSV")
    else:
        print(f"  {args.gln_exchange} flux in full medium: {gln_present:.4f}")

    taxa_ids = list(comm.taxa)
    n_taxa   = len(taxa_ids)
    n_inputs = n_taxa + 1    # 6 abundances + gln_flag
    print(f"  Taxa ({n_taxa}): {taxa_ids}")
    print(f"  Input dimensions: {n_inputs} (taxa abundances + gln_flag)")

    section(f"2. Sampling {args.n_samples} Dirichlet abundance vectors")
    rng     = np.random.default_rng(args.seed)
    samples = rng.dirichlet(np.ones(n_taxa), size=args.n_samples)

    # pre-allocate at maximum possible size (2 solves × N samples)
    X_buf  = np.zeros((2 * args.n_samples, n_inputs))
    Y_buf  = np.zeros((2 * args.n_samples, n_taxa))
    gr_buf = np.zeros(2 * args.n_samples)
    ok_buf = np.zeros(2 * args.n_samples, dtype=bool)
    row    = 0

    t_sweep = time.time()
    pad = len(str(args.n_samples))

    for i, x in enumerate(samples):
        for gln_flag, medium in [(1, medium_full), (0, medium_no_gln)]:
            mu, status, comm_gr, elapsed = solve(
                comm, x, taxa_ids, medium, args.fraction
            )
            X_buf[row, :n_taxa] = x
            X_buf[row, n_taxa]  = gln_flag
            Y_buf[row]  = mu
            gr_buf[row] = comm_gr
            ok_buf[row] = (status == "optimal")
            row += 1

        if (i + 1) % 100 == 0 or i == 0 or i == args.n_samples - 1:
            n_ok_so_far = ok_buf[:row].sum()
            print(f"  [{i+1:>{pad}}/{args.n_samples}]  "
                  f"last_status={status}  comm_gr={comm_gr:.4f}  "
                  f"time={elapsed:.1f}s  ok_rows={n_ok_so_far}/{row}")

    sweep_time = time.time() - t_sweep
    n_rows = row
    n_ok   = ok_buf[:n_rows].sum()
    print(f"\n  Sweep done in {sweep_time:.0f} s  "
          f"({sweep_time/3600:.2f} h)")
    print(f"  Total LP calls : {n_rows}  ({args.n_samples} samples × 2 flags)")
    print(f"  Optimal rows   : {n_ok}/{n_rows}  "
          f"Dropped: {n_rows - n_ok}")

    X  = X_buf[:n_rows][ok_buf[:n_rows]]
    Y  = Y_buf[:n_rows][ok_buf[:n_rows]]
    gr = gr_buf[:n_rows][ok_buf[:n_rows]]

    section("3. Data summary")
    print(f"  Training samples : {X.shape[0]}  (X: {X.shape}, Y: {Y.shape})")

    gln1_mask = X[:, n_taxa] == 1
    gln0_mask = X[:, n_taxa] == 0
    print(f"  gln_flag=1 rows  : {gln1_mask.sum()}")
    print(f"  gln_flag=0 rows  : {gln0_mask.sum()}")

    print(f"\n  Per-taxon growth rate range  [gln=1]  vs  [gln=0]:")
    for j, t in enumerate(taxa_ids):
        lo1, hi1 = Y[gln1_mask, j].min(), Y[gln1_mask, j].max()
        lo0, hi0 = Y[gln0_mask, j].min(), Y[gln0_mask, j].max()
        print(f"    {t:<12}  gln=1 [{lo1:.3f}, {hi1:.3f}]  "
              f"gln=0 [{lo0:.3f}, {hi0:.3f}]")

    # feasible_range: (n_inputs, 2) — observed min/max per input column
    # Force the flag column to [0, 1] regardless of what was feasible
    feasible_range = np.stack([X.min(axis=0), X.max(axis=0)], axis=1)
    feasible_range[n_taxa] = [0.0, 1.0]

    section("4. Saving")
    np.savez_compressed(
        args.output,
        X=X,
        Y=Y,
        feasible_range=feasible_range,
        taxa_ids=taxa_ids,
        community_gr=gr,
        gln_exchange=args.gln_exchange,
        fraction=args.fraction,
        n_samples=args.n_samples,
        n_optimal=n_ok,
        seed=args.seed,
    )
    print(f"  X shape            : {X.shape}")
    print(f"  Y shape            : {Y.shape}")
    print(f"  feasible_range     : {feasible_range.shape}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
