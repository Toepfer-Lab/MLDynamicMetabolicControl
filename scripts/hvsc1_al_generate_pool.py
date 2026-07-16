"""
Active-learning candidate pool generation + Query-by-Committee variance
scoring + top-N selection.

Generates a pool of unlabeled abundance vectors using the SAME validated
mixture-mode sampler already used for hvsc1's corner-coverage topup (not a
fresh naive-Dirichlet pool, which would recreate the exact corner blind spot
this whole effort exists to fix -- see results/calculations_log.md), scores
every candidate by the prediction variance across a QBC ensemble (see
hvsc1_al_ensemble_train.py), and writes out the top --n-select
highest-variance candidates for LP evaluation (see
hvsc1_al_eval_candidates_array.sh). Fully vectorized over the pool -- the
only Python-level loop is over ensemble members (O(n_members), not
O(pool_size)).

Stage A of an active-learning round -- pure torch/numpy, no CPLEX/micom
needed.

Usage:
    python scripts/hvsc1_al_generate_pool.py \\
        --ensemble-dir trained_models/hvsc1_al_ensemble/round01 \\
        --pool-size 100000 --n-select 8000 \\
        --round-id round01
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
from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ensemble-dir", type=Path, required=True,
                   help="Directory of member*.pt checkpoints from hvsc1_al_ensemble_train.py")
    p.add_argument("--training-data", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz",
                   help="Used only to recover taxa_ids for pool generation + output metadata")
    p.add_argument("--pool-size", type=int, default=100_000,
                   help="Can be far smaller than a blind-search pool: corner coverage here "
                        "is structural (one-dominant/co-dominant always place a taxon in a "
                        "high band), not probabilistic, so the pool only needs enough "
                        "candidates to rank within an already corner-rich set.")
    p.add_argument("--n-select", type=int, default=8_000,
                   help="Top-N highest-variance candidates to keep for LP evaluation")
    p.add_argument("--sampling-mode", type=str, default="mixture", choices=["mixture"],
                   help="Pool sampling mode. Only 'mixture' is supported here: generating "
                        "the pool via naive/uniform Dirichlet would recreate the exact "
                        "corner blind spot this pipeline exists to fix.")
    p.add_argument("--mixture-weights", type=str, default=DEFAULT_MIXTURE)
    p.add_argument("--alpha-lo", type=float, default=0.01)
    p.add_argument("--alpha-hi", type=float, default=0.5)
    p.add_argument("--dominant-bands", type=str, default=DEFAULT_DOMINANT_BANDS)
    p.add_argument("--co-dom-level-lo", type=float, default=0.15)
    p.add_argument("--co-dom-level-hi", type=float, default=0.45)
    p.add_argument("--co-dom-max-gap", type=float, default=0.1)
    p.add_argument("--co-dom-triple-prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=8000,
                   help="Fresh seed block for pool generation, distinct from data-gen "
                        "(1000=full, 2000=corner_topup, 9000=reserved_corner_test) and "
                        "ensemble-train (7000) seed ranges.")
    p.add_argument("--round-id", type=str, default="round01")
    p.add_argument("--output", type=Path, default=None,
                   help="Default: data/hvsc1_al_candidates/<round-id>_candidates.npz")
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        REPO_ROOT / "data" / "hvsc1_al_candidates" / f"{args.round_id}_candidates.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading ensemble members")
    member_paths = sorted(args.ensemble_dir.glob("member*.pt"))
    if not member_paths:
        raise SystemExit(f"No member*.pt checkpoints found in {args.ensemble_dir}")
    members = [load_surrogate_checkpoint(p, SurrogateNN) for p in member_paths]
    print(f"  Loaded {len(members)} members from {args.ensemble_dir}")
    for p in member_paths:
        print(f"    {p.name}")

    section("2. Recovering taxa ordering")
    d = np.load(args.training_data, allow_pickle=True)
    taxa_ids = [str(t) for t in d["taxa_ids"]]
    n_taxa = len(taxa_ids)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    section(f"3. Generating pool of {args.pool_size} candidates (mode={args.sampling_mode})")
    rng = np.random.default_rng(args.seed)
    weights = parse_weights(args.mixture_weights)
    bands = parse_bands(args.dominant_bands)
    X_pool = sample_mixture(
        rng, n_taxa, args.pool_size, weights,
        args.alpha_lo, args.alpha_hi, bands,
        args.co_dom_level_lo, args.co_dom_level_hi,
        args.co_dom_max_gap, args.co_dom_triple_prob,
    )
    row_sums = X_pool.sum(axis=1)
    print(f"  Pool shape: {X_pool.shape}   "
          f"row-sum check: min={row_sums.min():.6f} max={row_sums.max():.6f}")

    section("4. Scoring pool by ensemble prediction variance (vectorized)")
    preds = np.stack([
        surrogate_predict(model_nn, x_scaler, y_scaler, X_pool)
        for model_nn, x_scaler, y_scaler, _ in members
    ], axis=0)  # (n_members, pool_size, n_taxa)
    variance = preds.var(axis=0)   # (pool_size, n_taxa) -- no per-candidate loop
    score = variance.mean(axis=1)  # (pool_size,) scalar disagreement score
    print(f"  score stats: mean={score.mean():.6f}  std={score.std():.6f}  "
          f"min={score.min():.6f}  max={score.max():.6f}")

    section(f"5. Selecting top {args.n_select} highest-variance candidates")
    n_select = min(args.n_select, args.pool_size)
    if n_select >= args.pool_size:
        top_idx = np.arange(args.pool_size)
    else:
        top_idx = np.argpartition(-score, n_select)[:n_select]
    top_idx = top_idx[np.argsort(-score[top_idx])]  # sorted descending, for inspectability
    X_selected = X_pool[top_idx]
    score_selected = score[top_idx]
    print(f"  Selected score stats: mean={score_selected.mean():.6f}  "
          f"min={score_selected.min():.6f}  max={score_selected.max():.6f}")

    section("6. Saving")
    np.savez_compressed(
        output,
        X=X_selected,
        score=score_selected,
        taxa_ids=taxa_ids,
        pool_size=args.pool_size,
        n_select=n_select,
        sampling_mode=args.sampling_mode,
        mixture_weights=args.mixture_weights,
        ensemble_dir=str(args.ensemble_dir),
        seed=args.seed,
        round_id=args.round_id,
        pool_score_mean=float(score.mean()),
        pool_score_std=float(score.std()),
        pool_score_max=float(score.max()),
    )
    print(f"  X shape: {X_selected.shape}")
    print(f"  Saved: {output}")


if __name__ == "__main__":
    main()
