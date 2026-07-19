"""
Validate Use-case A's sensitivity sweep against real LP ground truth.

Loads the stratified validation candidates (written by
hvsc1_sensitivity_anchor.py) and the matching LP-solved shard (written by
scripts/hvsc1_al_eval_candidates_array.sh via hvsc1_generate_data.py
--candidates-file), matches rows by exact value equality (the LP shard's X is
an order-preserving subsequence of the candidates, since hvsc1_generate_data.py
just iterates its input rows in order), predicts with the surrogate at the
matched rows, and reports global + per-taxon-output MAE/R2.

Usage:
    python scripts/hvsc1_usecase_a_validate_compare.py \\
        --profile original --time-step 30
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--profile", type=str, default="original")
    p.add_argument("--time-step", type=int, default=0)
    p.add_argument("--candidates", type=Path, default=None,
                   help="Default: results/hvsc1_usecase_a/validation_candidates_{profile}_t{time_step}.npz")
    p.add_argument("--lp-shard", type=Path, default=None,
                   help="Default: data/hvsc1_shards/usecase_a_validation/hvsc1_training_shard00_seed13000.npz")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "hvsc1_usecase_a")
    p.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "plots" / "hvsc1_usecase_a")
    return p.parse_args()


def main():
    args = parse_args()
    candidates_path = args.candidates or (
        args.output_dir / f"validation_candidates_{args.profile}_t{args.time_step}.npz"
    )
    lp_shard_path = args.lp_shard or (
        REPO_ROOT / "data" / "hvsc1_shards" / "usecase_a_validation"
        / "hvsc1_training_shard00_seed13000.npz"
    )

    section("1. Loading candidates and LP shard")
    cand = np.load(candidates_path, allow_pickle=True)
    lp = np.load(lp_shard_path, allow_pickle=True)
    X_cand = cand["X"]
    X_lp, Y_lp = lp["X"], lp["Y"]
    n_optimal, n_samples = int(lp["n_optimal"]), int(lp["n_samples"])
    print(f"  Candidates : {candidates_path}  ({len(X_cand)} rows)")
    print(f"  LP shard   : {lp_shard_path}  (optimal {n_optimal}/{n_samples})")

    section("2. Matching LP rows back to candidates (exact value equality)")
    # hvsc1_generate_data.py iterates its input rows in order and only drops
    # non-optimal ones, so X_lp is an order-preserving subsequence of X_cand.
    matched_idx = []
    j = 0
    for i in range(len(X_lp)):
        while j < len(X_cand) and not np.allclose(X_cand[j], X_lp[i], atol=1e-10):
            j += 1
        if j >= len(X_cand):
            raise SystemExit(f"Could not match LP row {i} back to any candidate -- "
                              "candidates/LP shard mismatch.")
        matched_idx.append(j)
        j += 1
    print(f"  Matched {len(matched_idx)}/{len(X_lp)} LP rows to candidates in order.")

    section("3. Surrogate prediction at matched points")
    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    Y_pred = surrogate_predict(model_nn, x_scaler, y_scaler, X_lp)

    mae = float(np.mean(np.abs(Y_pred - Y_lp)))
    r2 = float(r2_score(Y_lp, Y_pred))
    per_taxon_mae = np.mean(np.abs(Y_pred - Y_lp), axis=0)
    taxa_ids = [str(t) for t in lp["taxa_ids"]]

    print(f"  Global MAE : {mae:.4f}")
    print(f"  Global R2  : {r2:.4f}")
    print(f"  {'Taxon':<8}{'MAE':>10}")
    for t, m in sorted(zip(taxa_ids, per_taxon_mae), key=lambda tm: -tm[1]):
        print(f"  {t:<8}{m:>10.4f}")

    section("4. Saving results")
    out_path = args.output_dir / f"validation_comparison_{args.profile}_t{args.time_step}.npz"
    np.savez_compressed(
        out_path, X=X_lp, Y_true=Y_lp, Y_pred=Y_pred, mae=mae, r2=r2,
        per_taxon_mae=per_taxon_mae, taxa_ids=taxa_ids,
    )
    print(f"  Saved: {out_path}")

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(Y_lp.flatten(), Y_pred.flatten(), s=10, alpha=0.5, color="#5B8DB8")
    lims = [min(Y_lp.min(), Y_pred.min()), max(Y_lp.max(), Y_pred.max())]
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("True growth rate (LP)")
    ax.set_ylabel("Predicted growth rate (surrogate)")
    ax.set_title(f"Use-case A oracle validation -- anchor: {args.profile} @ t={args.time_step}\n"
                 f"MAE={mae:.3f}  R2={r2:.4f}  (n={len(X_lp)})")
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plots_dir / f"validation_scatter_{args.profile}_t{args.time_step}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path}")


if __name__ == "__main__":
    main()
