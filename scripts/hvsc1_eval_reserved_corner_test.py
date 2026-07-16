"""
Evaluate a trained hvsc1 surrogate checkpoint against the wholly-reserved
near-corner test set (data/hvsc1_reserved_corner_test.npz), which was
generated with its own seed range (9000+) and never merged into
data/hvsc1_training.npz. Unlike hvsc1_eval_stratified.py (which reproduces
the IID train/test split of the training data itself), every row here is
guaranteed unseen during training -- this is the leakage-free claim.

Usage:
    python scripts/hvsc1_eval_reserved_corner_test.py
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
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--reserved-data", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_reserved_corner_test.npz")
    return p.parse_args()


def main():
    args = parse_args()

    section("1. Loading checkpoint and reserved corner test set")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"  Checkpoint     : {args.checkpoint.name}")
    print(f"  Checkpoint data: {metadata['data_path']}")
    print(f"  Reserved data  : {args.reserved_data}")

    data = np.load(args.reserved_data, allow_pickle=True)
    X, Y = data["X"], data["Y"]
    print(f"  Reserved set: X={X.shape}  Y={Y.shape}  "
          f"(entirely unseen -- own seed range, never merged into training data)")

    section("2. Bucketing by max-abundance")
    max_abundance = X.max(axis=1)
    print(f"  max-abundance: min={max_abundance.min():.4f} "
          f"max={max_abundance.max():.4f} mean={max_abundance.mean():.4f}")

    section("3. Predicting")
    X_scaled = x_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    model_nn.eval()
    with torch.no_grad():
        Y_pred_scaled = model_nn(X_t).numpy()
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    Y_true = Y

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


if __name__ == "__main__":
    main()
