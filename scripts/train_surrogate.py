#!/usr/bin/env python
"""
Train the surrogate neural network on generated FBA data and save a reusable checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Make local src importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import surrogateNN  # noqa: E402
from runtime_utils import (  # noqa: E402
    DATA_DIR,
    MODEL_DIR,
    ensure_output_dirs,
    save_surrogate_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vman", default="PYK", help="Reaction ID associated with this dataset (for naming only)")
    parser.add_argument(
        "--condition",
        default="anaerobic",
        help="Condition label to include in output naming (e.g. aerobic/anaerobic)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to .npz file produced by generate_fba_data.py",
    )
    parser.add_argument("--hidden-dim", type=int, default=4, help="Number of hidden neurons")
    parser.add_argument("--epochs", type=int, default=5000, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of training data used for validation")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional filename for the saved checkpoint (default builds from vman/condition/hidden dim)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    data_path = args.data_path or DATA_DIR / f"fba_data_{args.vman}_{args.condition}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    X, Y = data["X"], data["Y"]
    feasible_range = data["feasible_range"]
    flux_order = data["flux_order"] if "flux_order" in data else None
    flux_labels = data["flux_labels"] if "flux_labels" in data else None

    x_scaler, y_scaler, X_train, Y_train, X_val, Y_val, X_test, Y_test = surrogateNN.ML_data_prep(
        X, Y, test_size=args.test_size, val_size=args.val_size
    )

    model = surrogateNN.SurrogateNN(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        hidden_dim=args.hidden_dim,
    )
    loss_fn = nn.MSELoss()

    model, train_losses, val_losses = surrogateNN.train_model(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        loss_fn=loss_fn,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        verbose=args.verbose,
    )

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        loss_test = loss_fn(pred_test, Y_test).item()

    # Undo scaling for metrics
    Y_test_true = y_scaler.inverse_transform(Y_test.numpy())
    pred_test_true = y_scaler.inverse_transform(pred_test.numpy())
    r2 = r2_score(Y_test_true.flatten(), pred_test_true.flatten())

    base_name = args.model_name or f"{args.vman}_{args.condition}_input-{X_train.shape[1]}_output-{Y_train.shape[1]}_hidden-{args.hidden_dim}"
    checkpoint_path = MODEL_DIR / f"{base_name}.pt"
    metadata = {
        "vman": args.vman,
        "condition": args.condition,
        "feasible_range": feasible_range.tolist(),
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "data_path": str(data_path),
    }
    if flux_order is not None:
        metadata["flux_order"] = flux_order.tolist()
    if flux_labels is not None:
        metadata["flux_labels"] = flux_labels.tolist()
    save_surrogate_checkpoint(model, x_scaler, y_scaler, metadata, checkpoint_path)

    metrics_path = MODEL_DIR / f"{checkpoint_path.stem}_metrics.json"
    metrics = {
        "train_loss_final": float(train_losses[-1]),
        "val_loss_final": float(val_losses[-1]),
        "test_mse": float(loss_test),
        "r2_score": float(r2),
        "checkpoint": str(checkpoint_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model checkpoint to {checkpoint_path}")
    print(f"Test MSE: {loss_test:.4f} | R2: {r2:.4f}")


if __name__ == "__main__":
    main()
