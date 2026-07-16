"""
Dirichlet-based sampling strategies for hvsc1 abundance vectors.

Extracted from scripts/hvsc1_generate_data.py so these pure-numpy functions
can be imported without pulling in cobra/micom (which requires a working
CPLEX install on the Python path, since hvsc1_generate_data.py sets
cobra.Configuration.solver = "cplex" at module level on import).

See scripts/hvsc1_generate_data.py's module docstring / results/calculations_log.md
for the reasoning behind the mixture -- summary: naive Dirichlet(1,...,1)
concentrates almost all its mass near the simplex centroid in high dimensions
(P(any coordinate > 0.5) = 0.5^(n_taxa-1), astronomically small at n_taxa=27),
so it never samples the near-monoculture states real trajectories actually
converge to. The modes below deliberately construct that missing coverage.
"""

import numpy as np

DEFAULT_MIXTURE = "naive=0.10,low-alpha=0.20,one-dominant=0.45,co-dominant=0.25"
DEFAULT_DOMINANT_BANDS = "0.1-0.3,0.3-0.5,0.5-0.7,0.7-0.9,0.9-0.999"


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
    """Dispatch on args.sampling_mode. `args` is any object/namespace exposing
    the sampling-mode CLI attributes (n_samples, alpha_lo, alpha_hi,
    dominant_bands, co_dom_level_lo, co_dom_level_hi, co_dom_max_gap,
    co_dom_triple_prob, mixture_weights)."""
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
