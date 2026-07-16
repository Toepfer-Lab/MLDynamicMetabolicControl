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

STANDING INVARIANT: never pass data/hvsc1_shards/reserved_corner_test/ to
scripts/hvsc1_merge_shards.py --shard-dir, at any round -- doing so would
destroy the only clean corner-aware held-out set in the repo.

Usage:
    python scripts/hvsc1_eval_reserved_corner.py
    python scripts/hvsc1_eval_reserved_corner.py \\
        --checkpoint trained_models/hvsc1_community_input-27_output-27_hidden-128.pt \\
        --ensemble-dir trained_models/hvsc1_al_ensemble/round01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict, ensemble_predict  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402

BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
MIN_N_FOR_R2 = 30


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
                        "checkpoints in this directory, side by side with --checkpoint")
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

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2))
        print(f"\n  Saved metrics JSON: {args.output}")


if __name__ == "__main__":
    main()
