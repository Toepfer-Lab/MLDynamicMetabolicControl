"""
Merge hvsc1 training-data shards (produced by hvsc1_generate_data_array.sh)
into the single data/hvsc1_training.npz the rest of the hvsc1 pipeline
expects (train_surrogate.py only reads X, Y, feasible_range from it).

Validates that taxa_ids and fraction agree across every shard before
merging (a mismatch would silently corrupt column semantics), concatenates
X/Y/community_gr, and RECOMPUTES feasible_range as min/max over the full
concatenated X — never reused from a single shard's narrower range, since
train_surrogate.py treats this as the global input-normalization range.

Usage:
    python scripts/hvsc1_merge_shards.py --shard-dir data/hvsc1_shards/full --expected-shards 8
    python scripts/hvsc1_merge_shards.py --shard-dir data/hvsc1_shards/pilot --expected-shards 3
    # Combine multiple batches (e.g. the original naive run + a corner-coverage
    # topup) into one merged dataset in a single call:
    python scripts/hvsc1_merge_shards.py \\
        --shard-dir data/hvsc1_shards/full data/hvsc1_shards/corner_topup \\
        --expected-shards 24
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

SHARD_RE = re.compile(r"hvsc1_training_shard(\d+)_seed(\d+)\.npz$")


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", type=Path, required=True, nargs="+",
                   help="One or more directories containing "
                        "hvsc1_training_shard##_seed###.npz files. Pass multiple "
                        "to combine separately-generated batches (e.g. the "
                        "original naive run and a corner-coverage topup) in one merge.")
    p.add_argument("--expected-shards", type=int, default=None,
                   help="If set, error out (unless --allow-partial) when fewer "
                        "shard files than this are found")
    p.add_argument("--allow-partial", action="store_true",
                   help="Proceed even if fewer than --expected-shards files are found")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz")
    return p.parse_args()


def main():
    args = parse_args()

    section("1. Discovering shards")
    shard_files = []
    for shard_dir in args.shard_dir:
        dir_files = sorted(shard_dir.glob("hvsc1_training_shard*_seed*.npz"))
        if not dir_files:
            sys.exit(f"No shard files found in {shard_dir}")

        found_task_ids = set()
        for f in dir_files:
            m = SHARD_RE.search(f.name)
            if not m:
                sys.exit(f"Shard filename doesn't match the expected pattern: {f}")
            found_task_ids.add(int(m.group(1)))
        print(f"  Found {len(dir_files)} shard file(s) in {shard_dir}")
        for f in dir_files:
            print(f"    {f.name}")

        if args.expected_shards is not None and len(dir_files) < args.expected_shards:
            missing = sorted(set(range(args.expected_shards)) - found_task_ids)
            msg = (f"Expected {args.expected_shards} shards in {shard_dir}, "
                   f"found {len(dir_files)}. Missing task indices: {missing}")
            if not args.allow_partial:
                sys.exit(
                    msg + "\nResubmit the missing indices, e.g.:\n"
                    f"  sbatch --array={','.join(map(str, missing))} "
                    f"--export=SHARD_SUBDIR={shard_dir.name} "
                    f"scripts/hvsc1_generate_data_array.sh\n"
                    "Or pass --allow-partial to merge anyway."
                )
            print(f"  WARNING: {msg} (proceeding anyway, --allow-partial set)")
        shard_files.extend(dir_files)

    section("2. Loading and validating shards")
    shards = []
    for f in shard_files:
        d = np.load(f, allow_pickle=True)
        shards.append({
            "file": f.name,
            "X": d["X"], "Y": d["Y"],
            "community_gr": d["community_gr"],
            "taxa_ids": list(d["taxa_ids"]),
            "fraction": float(d["fraction"]),
            "seed": int(d["seed"]),
            "n_samples": int(d["n_samples"]),
            "n_optimal": int(d["n_optimal"]),
        })
        print(f"  {f.name}: X={d['X'].shape}  n_optimal={int(d['n_optimal'])}/{int(d['n_samples'])}  "
              f"seed={int(d['seed'])}")

    ref_taxa_ids = shards[0]["taxa_ids"]
    ref_fraction = shards[0]["fraction"]
    for s in shards[1:]:
        if s["taxa_ids"] != ref_taxa_ids:
            sys.exit(f"taxa_ids mismatch in {s['file']}: does not match {shards[0]['file']}. "
                     "Refusing to merge shards from different community builds/orderings.")
        if abs(s["fraction"] - ref_fraction) > 1e-9:
            sys.exit(f"fraction mismatch in {s['file']}: {s['fraction']} != {ref_fraction} "
                     f"(from {shards[0]['file']}). Refusing to merge incomparable rows.")
    print(f"  All shards agree on taxa_ids ({len(ref_taxa_ids)} taxa) and fraction ({ref_fraction}).")

    section("3. Merging")
    X = np.vstack([s["X"] for s in shards])
    Y = np.vstack([s["Y"] for s in shards])
    community_gr = np.concatenate([s["community_gr"] for s in shards])

    total_n_optimal = sum(s["n_optimal"] for s in shards)
    total_n_samples = sum(s["n_samples"] for s in shards)
    if total_n_optimal != X.shape[0]:
        sys.exit(f"Internal consistency check failed: sum(n_optimal)={total_n_optimal} "
                 f"!= X.shape[0]={X.shape[0]}")

    feasible_range = np.stack([X.min(axis=0), X.max(axis=0)], axis=1)

    print(f"  Merged X shape       : {X.shape}")
    print(f"  Merged Y shape       : {Y.shape}")
    print(f"  Total attempted      : {total_n_samples}")
    print(f"  Total optimal        : {total_n_optimal}  "
          f"({100*total_n_optimal/total_n_samples:.1f}%)")
    print(f"  feasible_range       : recomputed over full merged X (not any single shard)")

    section("4. Per-taxon growth rate summary (sanity check)")
    for j, t in enumerate(ref_taxa_ids):
        lo, hi = Y[:, j].min(), Y[:, j].max()
        print(f"    {t:<6}  [{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f}")

    section("5. Saving merged dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X, Y=Y,
        feasible_range=feasible_range,
        taxa_ids=ref_taxa_ids,
        community_gr=community_gr,
        fraction=ref_fraction,
        n_samples=total_n_samples,
        n_optimal=total_n_optimal,
        shard_seeds=np.array([s["seed"] for s in shards]),
        shard_files=np.array([s["file"] for s in shards]),
    )
    print(f"  Saved: {args.output}")
    print(f"\n  NOTE: if this pipeline is ever run via `dvc repro`, remember to `dvc commit "
          f"hvsc1_generate_data` after this manual merge -- the stage's output has "
          f"cache: false, persist: true, so dvc.lock won't otherwise know this file changed "
          f"and a later `dvc repro` could silently regenerate (and overwrite) it via the "
          f"single serial script.")


if __name__ == "__main__":
    main()
