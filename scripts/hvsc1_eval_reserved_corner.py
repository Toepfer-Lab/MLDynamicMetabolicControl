"""
Evaluate the surrogate (and, optionally, a QBC deep ensemble) against the
untouched data/hvsc1_shards/reserved_corner_test/ shards -- a mixture-mode,
NEVER-merged-into-training held-out set (seeds 9000-9003), the one dataset
in the repo that is reliably corner-rich AND untouched by any round of
training.

This is the trustworthy "did this active-learning round actually help"
signal. hvsc1_eval_stratified.py only stratifies the ordinary IID train/test
split -- its corner buckets stay data-starved regardless of how much AL
improves the model, since that split is drawn from whatever the training
set currently contains. Ensemble prediction variance shrinking during
active learning is a different signal again: it only means the committee
agrees more (which can also happen from overfitting), not that accuracy on
truly unseen corner data has improved.

Buckets by max-abundance using the SAME BUCKETS/MAE/R2 logic as
hvsc1_eval_stratified.py (duplicated here intentionally, not imported --
this script must never write/merge into the IID-split evaluation path).

When --ensemble-dir is given, also runs a committee-diagnostic check: since
reserved_corner_test has true LP-solved Y and no ensemble member has ever
trained on it, we can directly test whether the ensemble's per-point
prediction VARIANCE (the same score hvsc1_al_generate_pool.py uses to rank
unlabeled candidates) actually correlates with real prediction ERROR here.
If it doesn't, the committee has converged to near-identical functions and
variance-based active-learning selection would be picking arbitrary points,
not genuinely hard ones -- worth knowing before spending any LP budget on
Stage B's candidate batch.

STANDING INVARIANT: never pass data/hvsc1_shards/reserved_corner_test/ to
scripts/hvsc1_merge_shards.py --shard-dir, at any round -- doing so would
destroy the only clean corner-aware held-out set in the repo.

Usage:
    python scripts/hvsc1_eval_reserved_corner.py
    python scripts/hvsc1_eval_reserved_corner.py \
        --checkpoint trained_models/hvsc1_community_input-27_output-27_hidden-128.pt \
        --ensemble-dir trained_models/hvsc1_al_ensemble/round01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict, ensemble_predict  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402

BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
MIN_N_FOR_R2 = 30
N_VARIANCE_QUANTILES = 4


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt",
                   help="Single production checkpoint to evaluate")
    p.add_argument("--ensemble-dir", type=Path, default=None,
                   help="If set, also evaluate the deep-ensemble average of member*.pt "
                        "checkpoints in this directory, side by side with --checkpoint, "
                        "and run the variance-vs-error committee diagnostic")
    p.add_argument("--reserved-dir", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_shards" / "reserved_corner_test",
                   help="Held-out shard directory -- never pass this to hvsc1_merge_shards.py")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to write per-bucket metrics as JSON (for "
                        "programmatic consumers like hvsc1_al_capacity_sweep.py)")
    return p.parse_args()


def load_reserved_shards(reserved_dir):
    shard_files = sorted(reserved_dir.glob("hvsc1_training_shard*_seed*.npz"))
    if not shard_files:
        raise SystemExit(f"No shard files found in {reserved_dir}")
    Xs, Ys = [], []
    for f in shard_files:
        d = np.load(f, allow_pickle=True)
        Xs.append(d["X"])
        Ys.append(d["Y"])
        print(f"  {f.name}: X={d['X'].shape}")
    return np.vstack(Xs), np.vstack(Ys)


def bucket_metrics(Y_pred, Y_true, max_abundance, label):
    """Print the per-bucket MAE/R2 table and return it as a JSON-serializable dict."""
    print(f"\n  --- {label} ---")
    print(f"  {'Bucket':<14}{'n':>6}  {'MAE':>10}  {'R2':>10}")
    print(f"  {'-'*46}")
    global_mae = float(np.mean(np.abs(Y_pred - Y_true)))
    global_r2 = float(r2_score(Y_true, Y_pred))
    buckets_out = []
    for lo, hi in BUCKETS:
        mask = (max_abundance >= lo) & (max_abundance < hi if hi < 1.0 else max_abundance <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.1f},{hi:.1f})    {n:>6}  {'--':>10}  {'--':>10}")
            buckets_out.append({"lo": lo, "hi": hi, "n": n, "mae": None, "r2": None})
            continue
        mae = float(np.mean(np.abs(Y_pred[mask] - Y_true[mask])))
        r2 = float(r2_score(Y_true[mask], Y_pred[mask])) if n >= MIN_N_FOR_R2 else None
        r2_str = f"{r2:.4f}" if r2 is not None else f"(n<{MIN_N_FOR_R2})"
        print(f"  [{lo:.1f},{hi:.1f})    {n:>6}  {mae:>10.4f}  {r2_str:>10}")
        buckets_out.append({"lo": lo, "hi": hi, "n": n, "mae": mae, "r2": r2})
    print(f"  {'-'*46}")
    print(f"  {'GLOBAL':<14}{len(max_abundance):>6}  {global_mae:>10.4f}  {global_r2:>10.4f}")
    return {"label": label, "buckets": buckets_out,
            "global_mae": global_mae, "global_r2": global_r2,
            "n_total": int(len(max_abundance))}


def committee_diagnostic(members, X, Y):
    """Test whether ensemble prediction VARIANCE (the same score
    hvsc1_al_generate_pool.py uses to rank unlabeled candidates) actually
    correlates with real prediction ERROR on this held-out, ground-truth set.

    Returns (report_dict, score, error) where score/error are per-point
    arrays (for callers that want to plot/inspect further).
    """
    raw = ensemble_predict(members, X, reduce=False)          # (n_members, N, 27)
    Y_pred_ens = raw.mean(axis=0)                              # (N, 27)
    score = raw.var(axis=0).mean(axis=1)                       # (N,) -- same formula as the pool scorer
    error = np.mean(np.abs(Y_pred_ens - Y), axis=1)            # (N,) per-point MAE

    rho, pval = spearmanr(score, error)

    edges = np.quantile(score, np.linspace(0, 1, N_VARIANCE_QUANTILES + 1))
    quantile_idx = np.clip(np.digitize(score, edges[1:-1]), 0, N_VARIANCE_QUANTILES - 1)

    print(f"\n  --- Committee diagnostic: variance vs. actual error ---")
    print(f"  Spearman rho(variance, error) = {rho:.4f}  (p={pval:.2e})")
    print(f"  {'Quantile':<12}{'n':>6}  {'score range':>24}  {'mean |error|':>14}")
    print(f"  {'-'*58}")
    quantiles_out = []
    for q in range(N_VARIANCE_QUANTILES):
        mask = quantile_idx == q
        n = int(mask.sum())
        mean_err = float(error[mask].mean()) if n else None
        lo_s, hi_s = edges[q], edges[q + 1]
        label = f"Q{q+1}{' (lowest var)' if q == 0 else ''}{' (highest var)' if q == N_VARIANCE_QUANTILES - 1 else ''}"
        mean_err_str = f"{mean_err:.4f}" if mean_err is not None else "--"
        print(f"  {label:<12}{n:>6}  [{lo_s:.2e},{hi_s:.2e}]{'':>3}  {mean_err_str:>14}")
        quantiles_out.append({"quantile": q + 1, "n": n, "score_lo": float(lo_s),
                              "score_hi": float(hi_s), "mean_abs_error": mean_err})
    print(f"  {'-'*58}")

    ratio = None
    if quantiles_out[0]["mean_abs_error"] and quantiles_out[-1]["mean_abs_error"]:
        ratio = quantiles_out[-1]["mean_abs_error"] / quantiles_out[0]["mean_abs_error"]
        print(f"  Highest-variance / lowest-variance quantile error ratio: {ratio:.2f}x")
        if ratio > 1.5 or rho > 0.3:
            print("  -> Variance tracks real error here: committee disagreement looks like "
                  "a genuine difficulty signal. Safe to trust pool-selection on unlabeled candidates.")
        else:
            print("  -> Variance does NOT clearly track real error here: possible low-diversity "
                  "committee. Consider architecture variation or smaller per-member bootstrap "
                  "fraction before spending LP budget on the pool's top-N picks.")

    return ({"spearman_rho": float(rho), "spearman_pvalue": float(pval),
            "quantiles": quantiles_out, "high_low_ratio": ratio},
           score, error)


def main():
    args = parse_args()

    section(f"1. Loading held-out reserved-corner-test shards from {args.reserved_dir}")
    X, Y = load_reserved_shards(args.reserved_dir)
    max_abundance = X.max(axis=1)
    print(f"  Total rows   : {X.shape[0]}")
    print(f"  max-abundance: min={max_abundance.min():.4f} "
          f"max={max_abundance.max():.4f} mean={max_abundance.mean():.4f}")

    section("2. Single checkpoint")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"  Checkpoint: {args.checkpoint.name}")
    Y_pred_single = surrogate_predict(model_nn, x_scaler, y_scaler, X)
    single_metrics = bucket_metrics(Y_pred_single, Y, max_abundance,
                                    f"single checkpoint ({args.checkpoint.name})")
    output_payload = {"checkpoint": str(args.checkpoint), "single": single_metrics}

    if args.ensemble_dir is not None:
        section(f"3. Deep ensemble ({args.ensemble_dir})")
        member_paths = sorted(args.ensemble_dir.glob("member*.pt"))
        if not member_paths:
            print(f"  WARNING: no member*.pt found in {args.ensemble_dir}, skipping")
        else:
            members = [load_surrogate_checkpoint(p, SurrogateNN) for p in member_paths]
            print(f"  Loaded {len(members)} members")
            Y_pred_ens = ensemble_predict(members, X)
            ens_metrics = bucket_metrics(Y_pred_ens, Y, max_abundance,
                                         f"deep ensemble ({len(members)} members)")
            output_payload["ensemble_dir"] = str(args.ensemble_dir)
            output_payload["ensemble"] = ens_metrics

            section("4. Committee diagnostic (variance vs. actual error)")
            diag_report, _, _ = committee_diagnostic(members, X, Y)
            output_payload["committee_diagnostic"] = diag_report

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2))
        print(f"\n  Saved metrics JSON: {args.output}")


if __name__ == "__main__":
    main()
