"""
Select a stratified subsample of Use-case B's landscape sweep for LP validation.

Bins convergence_time and agreement_fraction independently into an
--n-bins-convergence x --n-bins-agreement grid using EQUAL-WIDTH edges over
each metric's observed range (not quantile edges: agreement_fraction is
heavily skewed toward 1.0 -- most starting points are robustly stable -- so
quantile edges collapse almost entirely inside that mode and under-resolve
the rare, scientifically-interesting low-agreement/boundary-like tail;
equal-width bins guarantee that tail gets its own dedicated cell(s)
regardless of how few points populate it). Picks one IC per occupied cell up
to --n-select, falling back to random draws from unselected cells/points if
fewer than --n-select cells are occupied. This gives a subsample spanning the
FULL range of both metrics -- not a sequential/first-N sample, which would be
biased toward whatever the pool's generation order happens to favor.

Writes two files:
  --output                 -- an ICs file directly consumable by
                               hvsc1_convergence_sweep.py (schema: X, taxa_ids,
                               n, seed, sampling_mode)
  --selected-indices-output -- the chosen rows' global_indices into the
                               original landscape_sweep.npz, so results can be
                               joined back to the surrogate-predicted values.

Usage:
    python scripts/hvsc1_usecase_b_select_validation_ics.py
"""

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--landscape-npz", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b" / "landscape_sweep.npz")
    p.add_argument("--n-select", type=int, default=40)
    p.add_argument("--n-bins-convergence", type=int, default=6)
    p.add_argument("--n-bins-agreement", type=int, default=6)
    p.add_argument("--seed", type=int, default=14_800)
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b_validation_ics.npz")
    p.add_argument("--selected-indices-output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b_validation_selected_indices.npz")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    section("1. Loading landscape sweep")
    d = np.load(args.landscape_npz, allow_pickle=True)
    taxa_ids = [str(t) for t in d["taxa_ids"]]
    x0 = d["x0"]
    convergence_time = d["convergence_time"].astype(float)
    agreement_fraction = d["agreement_fraction"].astype(float)
    global_indices = d["global_indices"]
    n = len(convergence_time)
    print(f"  N = {n}")

    section(f"2. Equal-width binning ({args.n_bins_convergence} x {args.n_bins_agreement})")
    conv_edges = np.linspace(convergence_time.min(), convergence_time.max(),
                             args.n_bins_convergence + 1)
    agree_edges = np.linspace(agreement_fraction.min(), agreement_fraction.max(),
                              args.n_bins_agreement + 1)
    conv_edges = np.unique(conv_edges)
    agree_edges = np.unique(agree_edges)
    conv_bin = np.clip(np.digitize(convergence_time, conv_edges[1:-1]), 0, len(conv_edges) - 2)
    agree_bin = np.clip(np.digitize(agreement_fraction, agree_edges[1:-1]), 0, len(agree_edges) - 2)
    cell_id = conv_bin * (len(agree_edges) - 1) + agree_bin
    unique_cells = np.unique(cell_id)
    print(f"  Occupied cells: {len(unique_cells)} "
          f"(of {(len(conv_edges)-1)*(len(agree_edges)-1)} possible)")

    section("3. Selecting one IC per occupied cell")
    selected = []
    for c in rng.permutation(unique_cells):
        members = np.where(cell_id == c)[0]
        selected.append(int(rng.choice(members)))
        if len(selected) >= args.n_select:
            break
    if len(selected) < args.n_select:
        remaining = args.n_select - len(selected)
        pool = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
        selected.extend(int(i) for i in extra)
        print(f"  Only {len(unique_cells)} occupied cells; topped up with "
              f"{len(extra)} random draws to reach {len(selected)}")
    selected = np.array(selected[:args.n_select])
    print(f"  Selected {len(selected)} ICs "
          f"(convergence_time range [{convergence_time[selected].min():.0f},"
          f"{convergence_time[selected].max():.0f}], "
          f"agreement_fraction range [{agreement_fraction[selected].min():.2f},"
          f"{agreement_fraction[selected].max():.2f}])")

    section("4. Saving")
    np.savez_compressed(
        args.output,
        X=x0[selected], taxa_ids=taxa_ids, n=len(selected),
        seed=args.seed, sampling_mode="usecase_b_stratified_validation",
    )
    print(f"  Saved: {args.output}")
    np.savez_compressed(
        args.selected_indices_output,
        global_indices=global_indices[selected],
        landscape_npz=str(args.landscape_npz),
    )
    print(f"  Saved: {args.selected_indices_output}")


if __name__ == "__main__":
    main()
