"""
Stratified evaluation: bucket the held-out test set by max-abundance
(distance from a simplex corner) and report per-bucket MAE/R^2, instead
of one global number.

Why: train_surrogate.py's train/test split (surrogateNN.ML_data_prep) is a
plain IID random split, so a global R^2 is computed almost entirely within
whatever region dominates the training distribution. For hvsc1's original
naive-Dirichlet dataset that region is the mid-simplex (max abundance never
exceeds ~0.5), so the global R^2=0.988 said nothing about the near-corner
region where real trajectories actually spend most of their time. This
script reproduces the EXACT train/test split train_surrogate.py used (same
data_path from the checkpoint's own metadata, same ML_data_prep seed/ratios)
so bucketed numbers are directly comparable to the headline metric, then
recovers unscaled abundances via the fitted x_scaler to bucket by
max-abundance.

Usage:
    python scripts/hvsc1_eval_stratified.py
    python scripts/hvsc1_eval_stratified.py --checkpoint trained_models/hvsc1_community_input-27_output-27_hidden-128.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint  # noqa: E402
from surrogateNN import SurrogateNN, ML_data_prep  # noqa: E402

BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
MIN_N_FOR_R2 = 30


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--data-path", type=Path, default=None,
                   help="Override the data path; defaults to the checkpoint's "
                        "own saved metadata['data_path']")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    section("1. Loading checkpoint and reproducing its train/test split")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    data_path = args.data_path or Path(metadata["data_path"])
    print(f"  Checkpoint : {args.checkpoint.name}")
    print(f"  Data path  : {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X, Y = data["X"], data["Y"]
    print(f"  Full dataset: X={X.shape}  Y={Y.shape}")

    # Re-run the identical prep pipeline train_surrogate.py used -- same
    # random_state=42 baked into ML_data_prep, same test_size/val_size ->
    # byte-for-byte the same X_test/Y_test tensors used for the headline metric.
    _, _, X_train, Y_train, X_val, Y_val, X_test, Y_test = ML_data_prep(
        X, Y, test_size=args.test_size, val_size=args.val_size
    )
    print(f"  Recovered test split: X_test={tuple(X_test.shape)}  "
          f"Y_test={tuple(Y_test.shape)}")

    section("2. Recovering unscaled abundances for bucketing")
    X_test_raw = x_scaler.inverse_transform(X_test.numpy())
    max_abundance = X_test_raw.max(axis=1)
    print(f"  max-abundance in test set: min={max_abundance.min():.4f} "
          f"max={max_abundance.max():.4f} mean={max_abundance.mean():.4f}")

    section("3. Predicting on the test set")
    model_nn.eval()
    with torch.no_grad():
        Y_pred_scaled = model_nn(X_test).numpy()
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    Y_true = y_scaler.inverse_transform(Y_test.numpy())

    section("4. Per-bucket metrics")
    print(f"  {'Bucket':<14}{'n':>6}  {'MAE':>10}  {'R2':>10}")
    print(f"  {'-'*46}")
    global_mae = np.mean(np.abs(Y_pred - Y_true))
    global_r2 = r2_score(Y_true, Y_pred)
    for lo, hi in BUCKETS:
        mask = (max_abundance >= lo) & (max_abundance < hi if hi < 1.0 else max_abundance <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.1f},{hi:.1f})    {n:>6}  {'--':>10}  {'--':>10}")
            continue
        mae = float(np.mean(np.abs(Y_pred[mask] - Y_true[mask])))
        if n >= MIN_N_FOR_R2:
            r2 = r2_score(Y_true[mask], Y_pred[mask])
            r2_str = f"{r2:.4f}"
        else:
            r2_str = f"(n<{MIN_N_FOR_R2})"
        print(f"  [{lo:.1f},{hi:.1f})    {n:>6}  {mae:>10.4f}  {r2_str:>10}")
    print(f"  {'-'*46}")
    print(f"  {'GLOBAL':<14}{len(max_abundance):>6}  {global_mae:>10.4f}  {global_r2:>10.4f}")

    metrics_path = args.checkpoint.with_name(args.checkpoint.stem + "_metrics.json")
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            headline = json.load(f)
        print(f"\n  Headline metrics.json (for comparison): "
              f"R2={headline.get('r2_score'):.4f}  test_mse={headline.get('test_mse'):.4f}")


if __name__ == "__main__":
    main()
