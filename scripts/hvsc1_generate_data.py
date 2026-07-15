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


DEFAULT_MIXTURE = "naive=0.10,low-alpha=0.20,one-dominant=0.45,co-dominant=0.25"
DEFAULT_DOMINANT_BANDS = "0.1-0.3,0.3-0.5,0.5-0.7,0.7-0.9,0.9-0.999"


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
    return p.parse_args()


def parse_bands(spec):
    """'0.1-0.3,0.3-0.5' -> [(0.1,0.3), (0.3,0.5)]"""
    bands = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        bands.append((float(lo), float(hi)))
    return bands


def parse_weights(spec):
    """'naive=0.1,low-alpha=0.2' -> {'naive': 0.1, 'low-alpha': 0.2}"""
    weights = {}
    for part in spec.split(","):
        mode, frac = part.split("=")
        weights[mode.strip()] = float(frac)
    return weights


# ── Sampling strategies ───────────────────────────────────────────────────────
# See the module docstring / results/calculations_log.md for the reasoning
# behind this mixture -- summary: naive Dirichlet(1,...,1) concentrates almost
# all its mass near the simplex centroid in high dimensions (P(any coordinate
# > 0.5) = 0.5^(n_taxa-1), astronomically small at n_taxa=27), so it never
# samples the near-monoculture states real trajectories actually converge to.
# The modes below deliberately construct that missing coverage.

def sample_naive(rng, n_taxa, n):
    return rng.dirichlet(np.ones(n_taxa), size=n)


def sample_low_alpha(rng, n_taxa, n, alpha_lo=0.01, alpha_hi=0.5):
    """Generic sparse/multi-modal coverage, not tied to a specific designed
    dominant taxon. alpha_lo needs to reach ~0.01-0.05 before P(max>0.9)
    becomes non-negligible at n_taxa=27 (verified by direct simulation)."""
    alphas = np.exp(rng.uniform(np.log(alpha_lo), np.log(alpha_hi), size=n))
    return np.stack([rng.dirichlet(np.full(n_taxa, a)) for a in alphas])


def sample_one_dominant(rng, n_taxa, n, bands=None):
    """One taxon's abundance drawn from a band in `bands`, remainder split via
    Dirichlet among the other n_taxa-1. Samples are spread evenly across bands
    and across all n_taxa possible dominant taxa, so every taxon gets covered
    -- not just the ones seen in today's reference trajectories."""
    if bands is None:
        bands = parse_bands(DEFAULT_DOMINANT_BANDS)
    n_bands = len(bands)
    X = np.zeros((n, n_taxa))
    band_idx = rng.integers(0, n_bands, size=n)
    dom_idx = rng.integers(0, n_taxa, size=n)
    for i in range(n):
        lo, hi = bands[band_idx[i]]
        dom_val = rng.uniform(lo, hi)
        rest = rng.dirichlet(np.ones(n_taxa - 1)) * (1.0 - dom_val)
        mask = np.arange(n_taxa) != dom_idx[i]
        X[i, dom_idx[i]] = dom_val
        X[i, mask] = rest
    return X


def sample_co_dominant(rng, n_taxa, n, level_lo=0.15, level_hi=0.45,
                       max_gap=0.1, triple_prob=0.15):
    """2 (occasionally 3) taxa elevated to comparable-but-not-identical levels
    -- the category single-dominant sampling misses entirely. Directly targets
    close-competition states like the observed 1234/1432/1391 near-tie."""
    X = np.zeros((n, n_taxa))
    for i in range(n):
        n_dom = 3 if rng.random() < triple_prob else 2
        dom_idxs = rng.choice(n_taxa, size=n_dom, replace=False)
        base_level = rng.uniform(level_lo, level_hi)
        gaps = rng.uniform(-max_gap / 2, max_gap / 2, size=n_dom)
        dom_vals = np.clip(base_level + gaps, 0.01, 0.9)
        # Rescale (preserving relative gaps) if the dominant candidates alone
        # would leave no room for the remainder -- clipping to 0.9 alone isn't
        # enough since 2-3 candidates near the top of the level range can
        # still sum past 1.0.
        max_dom_total = 0.95
        total_dom = dom_vals.sum()
        if total_dom > max_dom_total:
            dom_vals = dom_vals * (max_dom_total / total_dom)
            total_dom = max_dom_total
        remainder = 1.0 - total_dom
        rest_idxs = np.setdiff1d(np.arange(n_taxa), dom_idxs)
        rest = rng.dirichlet(np.ones(len(rest_idxs))) * remainder
        X[i, dom_idxs] = dom_vals
        X[i, rest_idxs] = rest
    return X


def sample_mixture(rng, n_taxa, n, weights, alpha_lo, alpha_hi, bands,
                   co_dom_level_lo, co_dom_level_hi, co_dom_max_gap, co_dom_triple_prob):
    modes = list(weights.keys())
    counts = {}
    remaining = n
    for m in modes[:-1]:
        counts[m] = int(round(weights[m] * n))
        remaining -= counts[m]
    counts[modes[-1]] = remaining

    parts = []
    for mode, cnt in counts.items():
        if cnt <= 0:
            continue
        if mode == "naive":
            parts.append(sample_naive(rng, n_taxa, cnt))
        elif mode == "low-alpha":
            parts.append(sample_low_alpha(rng, n_taxa, cnt, alpha_lo, alpha_hi))
        elif mode == "one-dominant":
            parts.append(sample_one_dominant(rng, n_taxa, cnt, bands))
        elif mode == "co-dominant":
            parts.append(sample_co_dominant(rng, n_taxa, cnt, co_dom_level_lo,
                                            co_dom_level_hi, co_dom_max_gap,
                                            co_dom_triple_prob))
        else:
            raise ValueError(f"Unknown mixture component: {mode}")
        print(f"    mixture component '{mode}': {cnt} samples")
    X = np.vstack(parts)
    rng.shuffle(X)  # avoid block structure in row order
    return X


def draw_samples(rng, n_taxa, args):
    if args.sampling_mode == "naive":
        return sample_naive(rng, n_taxa, args.n_samples)
    elif args.sampling_mode == "low-alpha":
        return sample_low_alpha(rng, n_taxa, args.n_samples, args.alpha_lo, args.alpha_hi)
    elif args.sampling_mode == "one-dominant":
        return sample_one_dominant(rng, n_taxa, args.n_samples, parse_bands(args.dominant_bands))
    elif args.sampling_mode == "co-dominant":
        return sample_co_dominant(rng, n_taxa, args.n_samples, args.co_dom_level_lo,
                                  args.co_dom_level_hi, args.co_dom_max_gap,
                                  args.co_dom_triple_prob)
    elif args.sampling_mode == "mixture":
        return sample_mixture(rng, n_taxa, args.n_samples, parse_weights(args.mixture_weights),
                              args.alpha_lo, args.alpha_hi, parse_bands(args.dominant_bands),
                              args.co_dom_level_lo, args.co_dom_level_hi,
                              args.co_dom_max_gap, args.co_dom_triple_prob)
    else:
        raise ValueError(f"Unknown sampling mode: {args.sampling_mode}")


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

    section(f"2. Sampling {args.n_samples} abundance vectors (mode={args.sampling_mode})")
    rng     = np.random.default_rng(args.seed)
    samples = draw_samples(rng, n_taxa, args)

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
        sampling_mode=args.sampling_mode,
    )
    print(f"  X shape        : {X.shape}")
    print(f"  Y shape        : {Y.shape}")
    print(f"  feasible_range : {feasible_range.shape}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
