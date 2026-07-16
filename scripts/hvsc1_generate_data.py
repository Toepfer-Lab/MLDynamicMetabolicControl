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
import sys
import time
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from micom import load_pickle

cobra.Configuration.solver = "cplex"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hvsc1_sampling import (  # noqa: E402
    DEFAULT_MIXTURE,
    DEFAULT_DOMINANT_BANDS,
    parse_bands,
    parse_weights,
    draw_samples,
)

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
    p.add_argument("--sampling-mode", type=str, default="naive",
                   choices=["naive", "low-alpha", "one-dominant", "co-dominant", "mixture"],
                   help="naive: original Dirichlet(1,...,1) (default, unchanged "
                        "behavior). low-alpha: sparse/multi-modal via low-concentration "
                        "Dirichlet. one-dominant: one taxon elevated to a value drawn "
                        "from --dominant-bands, remainder Dirichlet among the rest. "
                        "co-dominant: 2 (occasionally 3) taxa elevated to comparable "
                        "levels with a small gap -- targets close-competition states "
                        "the other modes don't cover. mixture: blend of all of the "
                        "above per --mixture-weights.")
    p.add_argument("--mixture-weights", type=str, default=DEFAULT_MIXTURE,
                   help="Comma-separated mode=fraction pairs, only used when "
                        "--sampling-mode mixture. Fractions need not sum to exactly 1 "
                        "(remainder goes to the last listed mode).")
    p.add_argument("--alpha-lo", type=float, default=0.01,
                   help="Low end of the log-uniform concentration range for low-alpha sampling")
    p.add_argument("--alpha-hi", type=float, default=0.5,
                   help="High end of the log-uniform concentration range for low-alpha sampling")
    p.add_argument("--dominant-bands", type=str, default=DEFAULT_DOMINANT_BANDS,
                   help="Comma-separated lo-hi bands for the dominant taxon's abundance "
                        "in one-dominant sampling; samples are spread evenly across "
                        "bands and across all taxa as the chosen dominant.")
    p.add_argument("--co-dom-level-lo", type=float, default=0.15,
                   help="Low end of the base elevated level for co-dominant sampling")
    p.add_argument("--co-dom-level-hi", type=float, default=0.45,
                   help="High end of the base elevated level for co-dominant sampling")
    p.add_argument("--co-dom-max-gap", type=float, default=0.1,
                   help="Max spread between co-dominant candidates' levels around the base")
    p.add_argument("--co-dom-triple-prob", type=float, default=0.15,
                   help="Probability a co-dominant draw uses 3 competing taxa instead of 2")
    p.add_argument("--candidates-file", type=Path, default=None,
                   help="If set, evaluate a fixed externally-supplied set of abundance "
                        "vectors (an npz with an 'X' array, e.g. produced by "
                        "hvsc1_al_generate_pool.py) instead of drawing new samples via "
                        "--sampling-mode. Mutually exclusive in effect with --sampling-mode "
                        "(the latter is ignored when this is set).")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Only used with --candidates-file: this task's index into an "
                        "--n-shards-way deterministic split of the candidates file's rows "
                        "(np.array_split), matching the SLURM array task id.")
    p.add_argument("--n-shards", type=int, default=None,
                   help="Only used with --candidates-file: total number of shards the "
                        "candidates file is split into (must match the SLURM --array size).")
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

    sampling_mode_out = args.sampling_mode
    if args.candidates_file is not None:
        section(f"2. Loading fixed candidates from {args.candidates_file}")
        cand = np.load(args.candidates_file, allow_pickle=True)
        if "taxa_ids" in cand:
            cand_taxa_ids = [str(t) for t in cand["taxa_ids"]]
            if cand_taxa_ids != taxa_ids:
                raise SystemExit(
                    f"taxa_ids mismatch: candidates file has {cand_taxa_ids}, "
                    f"model has {taxa_ids}. Refusing to evaluate candidates from a "
                    "different community build/taxon ordering."
                )
        samples = cand["X"]
        if args.n_shards is not None:
            if args.shard_index is None:
                raise SystemExit("--n-shards requires --shard-index")
            samples = np.array_split(samples, args.n_shards)[args.shard_index]
        n_samples = len(samples)
        # Reuse the candidates' own recorded sampling_mode (e.g. "mixture") so
        # downstream shard-merging/bookkeeping stays meaningful -- this batch
        # was not drawn with args.sampling_mode at all.
        if "sampling_mode" in cand:
            sampling_mode_out = str(cand["sampling_mode"])
        print(f"  Candidates in this shard: {n_samples}")
    else:
        section(f"2. Sampling {args.n_samples} abundance vectors (mode={args.sampling_mode})")
        rng = np.random.default_rng(args.seed)
        samples = draw_samples(rng, n_taxa, args)
        n_samples = args.n_samples

    X_buf  = np.zeros((n_samples, n_taxa))
    Y_buf  = np.zeros((n_samples, n_taxa))
    gr_buf = np.zeros(n_samples)
    ok_buf = np.zeros(n_samples, dtype=bool)

    t_sweep = time.time()
    pad = len(str(n_samples))

    for i, x in enumerate(samples):
        mu, status, comm_gr, elapsed = solve(
            comm, x, taxa_ids, default_medium, args.fraction
        )
        X_buf[i]  = x
        Y_buf[i]  = mu
        gr_buf[i] = comm_gr
        ok_buf[i] = (status == "optimal")

        if (i + 1) % 500 == 0 or i == 0 or i == n_samples - 1:
            n_ok_so_far = ok_buf[:i + 1].sum()
            print(f"  [{i+1:>{pad}}/{n_samples}]  "
                  f"last_status={status}  comm_gr={comm_gr:.4f}  "
                  f"time={elapsed:.1f}s  ok={n_ok_so_far}/{i+1}")

    sweep_time = time.time() - t_sweep
    n_ok = ok_buf.sum()
    print(f"\n  Sweep done in {sweep_time:.0f} s  ({sweep_time/3600:.2f} h)")
    print(f"  Optimal rows : {n_ok}/{n_samples}  "
          f"Dropped: {n_samples - n_ok}")

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
        n_samples=n_samples,
        n_optimal=n_ok,
        seed=args.seed,
        sampling_mode=sampling_mode_out,
    )
    print(f"  X shape        : {X.shape}")
    print(f"  Y shape        : {Y.shape}")
    print(f"  feasible_range : {feasible_range.shape}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
