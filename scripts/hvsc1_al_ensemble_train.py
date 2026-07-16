"""
Train a bootstrap ensemble of SurrogateNN models on the current hvsc1
training set, for use as a Query-by-Committee ensemble in the active-learning
pipeline (round-based data generation; see scripts/hvsc1_al_generate_pool.py).

Each of --n-members models is trained on an independent bootstrap resample
(sampling with replacement) of the same dataset, with a distinct random seed
for weight initialization -- both are sources of committee diversity. Reuses
SurrogateNN/ML_data_prep/train_model/save_surrogate_checkpoint unchanged, so
every member is a normal, independently-loadable checkpoint (no new format).

Stage A of an active-learning round -- pure torch/numpy, no CPLEX/micom
needed, can run interactively or as a plain (non-array) sbatch job.

Usage:
    python scripts/hvsc1_al_ensemble_train.py \\
        --data-path data/hvsc1_training.npz \\
        --n-members 5 \\
        --round-id round01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surrogateNN import SurrogateNN, ML_data_prep, train_model  # noqa: E402
from runtime_utils import save_surrogate_checkpoint  # noqa: E402


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz")
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Default matches hvsc1/params.yaml's current production "
                        "architecture; override after a capacity sweep picks a new value.")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--base-seed", type=int, default=7000,
                   help="Seed block for this ensemble's bootstrap resampling + weight "
                        "init (member i uses base_seed+i). Default 7000 is a fresh block "
                        "distinct from data-generation seed ranges already in use "
                        "(1000=full, 2000=corner_topup, 9000=reserved_corner_test).")
    p.add_argument("--round-id", type=str, default="round01")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: trained_models/hvsc1_al_ensemble/<round-id>/")
    p.add_argument("--verbose", action="store_true", help="Print per-epoch training progress")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (
        REPO_ROOT / "trained_models" / "hvsc1_al_ensemble" / args.round_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    section("1. Loading data")
    data = np.load(args.data_path, allow_pickle=True)
    X, Y = data["X"], data["Y"]
    n_rows = X.shape[0]
    print(f"  Data path : {args.data_path}")
    print(f"  X shape   : {X.shape}   Y shape: {Y.shape}")

    section(f"2. Training {args.n_members} ensemble members (bootstrap resample + distinct init)")
    loss_fn = nn.MSELoss()
    member_metrics = []
    for i in range(args.n_members):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, n_rows, size=n_rows)
        X_i, Y_i = X[boot_idx], Y[boot_idx]

        torch.manual_seed(seed)
        x_scaler, y_scaler, X_train, Y_train, X_val, Y_val, X_test, Y_test = ML_data_prep(
            X_i, Y_i, test_size=args.test_size, val_size=args.val_size
        )
        model = SurrogateNN(
            input_dim=X_train.shape[1], hidden_dim=args.hidden_dim,
            output_dim=Y_train.shape[1], n_layers=args.n_layers,
        )
        model, train_losses, val_losses = train_model(
            model, X_train, Y_train, X_val, Y_val, loss_fn=loss_fn,
            lr=args.lr, epochs=args.epochs, patience=args.patience, verbose=args.verbose,
        )

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test)
            test_mse = loss_fn(pred_test, Y_test).item()
        Y_test_true = y_scaler.inverse_transform(Y_test.numpy())
        pred_test_true = y_scaler.inverse_transform(pred_test.numpy())
        r2 = r2_score(Y_test_true.flatten(), pred_test_true.flatten())

        path = output_dir / f"member{i:02d}.pt"
        metadata = {
            "round_id": args.round_id,
            "member_idx": i,
            "bootstrap_seed": seed,
            "data_path": str(args.data_path),
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "n_bootstrap_rows": n_rows,
        }
        save_surrogate_checkpoint(model, x_scaler, y_scaler, metadata, path)
        member_metrics.append({
            "member": i, "seed": seed, "test_mse": test_mse, "r2": r2, "path": str(path),
        })
        print(f"  member {i:02d}  seed={seed}  test_mse={test_mse:.4f}  "
              f"r2={r2:.4f}  -> {path.name}")

    section("3. Ensemble summary")
    mse_vals = [m["test_mse"] for m in member_metrics]
    r2_vals = [m["r2"] for m in member_metrics]
    print(f"  test_mse: mean={np.mean(mse_vals):.4f}  std={np.std(mse_vals):.4f}")
    print(f"  r2      : mean={np.mean(r2_vals):.4f}  std={np.std(r2_vals):.4f}")

    summary_path = output_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps({
        "round_id": args.round_id,
        "data_path": str(args.data_path),
        "n_members": args.n_members,
        "base_seed": args.base_seed,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "members": member_metrics,
    }, indent=2))
    print(f"  Saved ensemble summary: {summary_path}")


if __name__ == "__main__":
    main()
