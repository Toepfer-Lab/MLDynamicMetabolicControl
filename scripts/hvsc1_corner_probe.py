"""
Corner probe: test the surrogate's behavior on synthetic near-corner
(near-monoculture) abundance vectors it likely never saw during training.

For each of a set of taxa observed as final/near-tied dominants in the
reference LP trajectories, and a ladder of abundance levels, constructs
a vector with that taxon at the given level and the remainder split evenly
among the other taxa. Checks whether argmax(mu_pred) matches the intended
dominant taxon, and whether the error is large enough to flip which taxon
stays dominant after a single Euler step (x*(1+mu*dt)).

Zero-cost (pure inference on an already-trained checkpoint, no cluster job).
Run BEFORE and AFTER the corner-coverage topup (see hvsc1_generate_data.py
--sampling-mode) for a clean before/after comparison.

Usage:
    python scripts/hvsc1_corner_probe.py
    python scripts/hvsc1_corner_probe.py --checkpoint trained_models/hvsc1_community_input-27_output-27_hidden-128.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402

DOMINANT_TAXA = ["1234", "1432", "1391", "364", "946", "1056", "1334", "504", "352"]
LEVELS = [0.3, 0.5, 0.7, 0.9, 0.99]

# Close-competition pairs: taxa observed competing near-simultaneously in the
# real "original" trajectory's first step (1234, 1432, 1391 all had mu*dt~2.97).
# Unlike the single-dominant test above (one taxon overwhelming, trivially easy
# even for a badly-extrapolating monotonic function), this tests whether the
# surrogate can correctly and consistently RANK two closely-matched candidates
# -- the actual failure mode observed, not a proxy for it.
CLOSE_PAIRS = [("1234", "1432"), ("1234", "1391"), ("1432", "1391")]
CLOSE_LEVELS = [0.1, 0.15, 0.2, 0.3]
CLOSE_GAP = 0.02  # A gets level+gap/2, B gets level-gap/2 -- A should win


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--training-data", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz",
                   help="Used only to recover taxa_ids in the correct column order")
    p.add_argument("--dt", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    section("1. Loading checkpoint and taxa ordering")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    d = np.load(args.training_data, allow_pickle=True)
    taxa_ids = [str(t) for t in d["taxa_ids"]]
    n_taxa = len(taxa_ids)
    print(f"  Checkpoint : {args.checkpoint.name}")
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    section("2. Probing synthetic near-corner vectors")
    results = []
    for dom in DOMINANT_TAXA:
        if dom not in taxa_ids:
            print(f"  WARNING: {dom} not in taxa_ids, skipping")
            continue
        dom_idx = taxa_ids.index(dom)
        for level in LEVELS:
            x = np.full(n_taxa, (1.0 - level) / (n_taxa - 1))
            x[dom_idx] = level
            mu_pred = surrogate_predict(model_nn, x_scaler, y_scaler, x[None, :])[0]
            pred_idx = int(np.argmax(mu_pred))
            pred_taxon = taxa_ids[pred_idx]
            match = pred_taxon == dom

            x_next = x * (1.0 + mu_pred * args.dt)
            x_next = np.maximum(x_next, 1e-8)
            x_next /= x_next.sum()
            stays_dominant = taxa_ids[int(np.argmax(x_next))] == dom

            results.append(dict(
                dominant=dom, level=level, pred_top=pred_taxon, match=match,
                mu_dom=float(mu_pred[dom_idx]), mu_top=float(mu_pred[pred_idx]),
                stays_dominant_after_step=stays_dominant,
            ))
            flag = "OK       " if match else "MISMATCH "
            print(f"  dom={dom:<6} level={level:<5} pred_top={pred_taxon:<6} "
                  f"mu_dom={mu_pred[dom_idx]:8.3f}  mu_top={mu_pred[pred_idx]:8.3f}  "
                  f"[{flag}]  stays_dominant_after_1_step={stays_dominant}")

    section("3. Summary")
    n = len(results)
    n_match = sum(r["match"] for r in results)
    n_stays = sum(r["stays_dominant_after_step"] for r in results)
    print(f"  argmax matches intended dominant taxon : {n_match}/{n} ({100*n_match/n:.1f}%)")
    print(f"  Still dominant after 1 Euler step       : {n_stays}/{n} ({100*n_stays/n:.1f}%)")

    print("\n  Per-level breakdown:")
    for level in LEVELS:
        subset = [r for r in results if r["level"] == level]
        m = sum(r["match"] for r in subset)
        s = sum(r["stays_dominant_after_step"] for r in subset)
        print(f"    level={level:<5} argmax_match={m}/{len(subset)}   "
              f"stays_dominant={s}/{len(subset)}")

    print("\n  Per-taxon breakdown:")
    for dom in DOMINANT_TAXA:
        subset = [r for r in results if r["dominant"] == dom]
        if not subset:
            continue
        m = sum(r["match"] for r in subset)
        print(f"    dominant={dom:<6} argmax_match={m}/{len(subset)}")

    section("4. Close-competition probe (the actual observed failure mode)")
    print("  Two candidates elevated to comparable levels with a small enforced\n"
          "  gap (A = level+gap/2, B = level-gap/2); A should win. Tests ranking\n"
          "  of near-tied competitors, not single-taxon dominance.")
    close_results = []
    for a, b in CLOSE_PAIRS:
        if a not in taxa_ids or b not in taxa_ids:
            print(f"  WARNING: {a} or {b} not in taxa_ids, skipping")
            continue
        a_idx, b_idx = taxa_ids.index(a), taxa_ids.index(b)
        for level in CLOSE_LEVELS:
            a_val = level + CLOSE_GAP / 2
            b_val = level - CLOSE_GAP / 2
            remainder = 1.0 - a_val - b_val
            x = np.full(n_taxa, remainder / (n_taxa - 2))
            x[a_idx] = a_val
            x[b_idx] = b_val
            mu_pred = surrogate_predict(model_nn, x_scaler, y_scaler, x[None, :])[0]
            correct = mu_pred[a_idx] > mu_pred[b_idx]
            gap_pred = float(mu_pred[a_idx] - mu_pred[b_idx])
            close_results.append(dict(a=a, b=b, level=level, correct=correct,
                                       gap_pred=gap_pred))
            flag = "OK       " if correct else "MISMATCH "
            print(f"  A={a:<6}(in={a_val:.3f}) vs B={b:<6}(in={b_val:.3f})  "
                  f"mu_A={mu_pred[a_idx]:8.3f}  mu_B={mu_pred[b_idx]:8.3f}  "
                  f"pred_gap={gap_pred:+7.3f}  [{flag}]")

    n_close = len(close_results)
    n_close_ok = sum(r["correct"] for r in close_results)
    print(f"\n  Close-competition ranking correct: {n_close_ok}/{n_close} "
          f"({100*n_close_ok/max(n_close,1):.1f}%)")


if __name__ == "__main__":
    main()
