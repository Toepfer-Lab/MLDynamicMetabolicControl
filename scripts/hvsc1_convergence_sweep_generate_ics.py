"""
Generate a large pool of starting abundance vectors ("initial conditions",
ICs) for scripts/hvsc1_convergence_sweep.py -- a browsing tool that runs many
short, convergence-truncated LP trajectories to explore the 27D initial-
condition space more broadly than the handful of fixed profiles in
hvsc1_simulate.py.

Uses the SAME corner-aware mixture sampler already validated for training-
data generation (src/hvsc1_sampling.py's sample_mixture), not naive
Dirichlet -- naive Dirichlet at n_taxa=27 essentially never reaches the
near-monoculture/near-tie starting states that are the interesting cases to
browse (see results/calculations_log.md). Pure numpy, no CPLEX needed --
kept separate from the LP-solving sweep script so the pool can be generated/
inspected without cluster CPLEX modules.

Usage:
    python scripts/hvsc1_convergence_sweep_generate_ics.py --n 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hvsc1_sampling import (  # noqa: E402
    DEFAULT_MIXTURE,
    DEFAULT_DOMINANT_BANDS,
    parse_bands,
    parse_weights,
    sample_mixture,
)


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300,
                   help="Number of starting configurations to generate")
    p.add_argument("--training-data", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz",
                   help="Used only to recover taxa_ids in the correct column order")
    p.add_argument("--mixture-weights", type=str, default=DEFAULT_MIXTURE)
    p.add_argument("--alpha-lo", type=float, default=0.01)
    p.add_argument("--alpha-hi", type=float, default=0.5)
    p.add_argument("--dominant-bands", type=str, default=DEFAULT_DOMINANT_BANDS)
    p.add_argument("--co-dom-level-lo", type=float, default=0.15)
    p.add_argument("--co-dom-level-hi", type=float, default=0.45)
    p.add_argument("--co-dom-max-gap", type=float, default=0.1)
    p.add_argument("--co-dom-triple-prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=10_000,
                   help="Fresh seed block, distinct from data-gen (1000/2000/9000), "
                        "ensemble-train (7000), and AL-pool (8000) seed ranges.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_convergence_sweep_ics.npz")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Recovering taxa ordering")
    d = np.load(args.training_data, allow_pickle=True)
    taxa_ids = [str(t) for t in d["taxa_ids"]]
    n_taxa = len(taxa_ids)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    section(f"2. Generating {args.n} starting configurations (mode=mixture)")
    rng = np.random.default_rng(args.seed)
    weights = parse_weights(args.mixture_weights)
    bands = parse_bands(args.dominant_bands)
    X = sample_mixture(
        rng, n_taxa, args.n, weights,
        args.alpha_lo, args.alpha_hi, bands,
        args.co_dom_level_lo, args.co_dom_level_hi,
        args.co_dom_max_gap, args.co_dom_triple_prob,
    )
    row_sums = X.sum(axis=1)
    print(f"  X shape: {X.shape}   row-sum check: "
          f"min={row_sums.min():.6f} max={row_sums.max():.6f}")
    max_abundance = X.max(axis=1)
    print(f"  max-abundance: min={max_abundance.min():.4f} "
          f"max={max_abundance.max():.4f} mean={max_abundance.mean():.4f}")

    section("3. Saving")
    np.savez_compressed(
        args.output,
        X=X,
        taxa_ids=taxa_ids,
        n=args.n,
        seed=args.seed,
        sampling_mode="mixture",
        mixture_weights=args.mixture_weights,
    )
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
