"""
Capacity sweep for the hvsc1 surrogate: grid over --hidden-dim/--n-layers,
each trained via the existing scripts/train_surrogate.py (unchanged), then
scored against the untouched reserved_corner_test/ set via
scripts/hvsc1_eval_reserved_corner.py -- so "bigger is better" is checked
against real held-out corner accuracy, not just val/test MSE on the IID
split the training data itself was drawn from.

Run once per active-learning round (Stage D), after Stage C merges new data
in and before the final production retrain -- the winning
(hidden_dim, n_layers) becomes the architecture for both the production
checkpoint and the next round's ensemble members (see
hvsc1_al_ensemble_train.py --hidden-dim/--n-layers), so ensemble and
production capacity stay in sync as the dataset grows.

Usage:
    python scripts/hvsc1_al_capacity_sweep.py \\
        --data-path data/hvsc1_training.npz --round-id round01
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_training.npz")
    p.add_argument("--hidden-dims", type=str, default="128,256,384",
                   help="Comma-separated hidden_dim values to sweep")
    p.add_argument("--n-layers-list", type=str, default="2,3",
                   help="Comma-separated n_layers values to sweep")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--round-id", type=str, default="round01")
    p.add_argument("--reserved-dir", type=Path,
                   default=REPO_ROOT / "data" / "hvsc1_shards" / "reserved_corner_test")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Interpreter to invoke train_surrogate.py/"
                        "hvsc1_eval_reserved_corner.py with")
    return p.parse_args()


def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


def corner_bucket_mae(metrics_json_path):
    """Extract the [0.9, 1.0) bucket MAE from a hvsc1_eval_reserved_corner.py
    --output JSON file's 'single' checkpoint section."""
    if not metrics_json_path.exists():
        return None
    payload = json.loads(metrics_json_path.read_text())
    for bucket in payload.get("single", {}).get("buckets", []):
        if bucket["lo"] == 0.9 and bucket["hi"] == 1.0:
            return bucket["mae"]
    return None


def main():
    args = parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    n_layers_list = [int(x) for x in args.n_layers_list.split(",")]

    section(f"1. Sweeping {len(hidden_dims)}x{len(n_layers_list)} architecture grid "
            f"({len(hidden_dims) * len(n_layers_list)} configs)")
    results = []
    for hidden_dim in hidden_dims:
        for n_layers in n_layers_list:
            model_name = f"hvsc1_al_sweep_{args.round_id}_hidden-{hidden_dim}_layers-{n_layers}"
            print(f"\n  --- hidden_dim={hidden_dim}  n_layers={n_layers} ---")

            run([
                args.python, REPO_ROOT / "scripts" / "train_surrogate.py",
                "--vman", "hvsc1", "--condition", "community",
                "--data-path", args.data_path,
                "--hidden-dim", hidden_dim, "--n-layers", n_layers,
                "--epochs", args.epochs, "--patience", args.patience, "--lr", args.lr,
                "--model-name", model_name,
            ])
            checkpoint = REPO_ROOT / "trained_models" / f"{model_name}.pt"
            metrics_path = checkpoint.with_name(checkpoint.stem + "_metrics.json")
            train_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

            eval_json = REPO_ROOT / "results" / f"{model_name}_reserved_corner.json"
            run([
                args.python, REPO_ROOT / "scripts" / "hvsc1_eval_reserved_corner.py",
                "--checkpoint", checkpoint,
                "--reserved-dir", args.reserved_dir,
                "--output", eval_json,
            ])
            mae_09_10 = corner_bucket_mae(eval_json)

            results.append({
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "checkpoint": str(checkpoint),
                "test_mse": train_metrics.get("test_mse"),
                "r2_score": train_metrics.get("r2_score"),
                "corner_0.9_1.0_mae": mae_09_10,
            })

    section("2. Comparison table")
    print(f"  {'hidden_dim':>10}  {'n_layers':>8}  {'test_mse':>10}  "
          f"{'r2':>8}  {'corner[0.9,1.0)_MAE':>20}")
    for r in results:
        mse_str = f"{r['test_mse']:.4f}" if r["test_mse"] is not None else "--"
        r2_str = f"{r['r2_score']:.4f}" if r["r2_score"] is not None else "--"
        mae_str = f"{r['corner_0.9_1.0_mae']:.4f}" if r["corner_0.9_1.0_mae"] is not None else "--"
        print(f"  {r['hidden_dim']:>10}  {r['n_layers']:>8}  {mse_str:>10}  "
              f"{r2_str:>8}  {mae_str:>20}")

    valid = [r for r in results if r["corner_0.9_1.0_mae"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["corner_0.9_1.0_mae"])
        print(f"\n  Best by corner[0.9,1.0) MAE: hidden_dim={best['hidden_dim']} "
              f"n_layers={best['n_layers']} (MAE={best['corner_0.9_1.0_mae']:.4f}) "
              f"-> {best['checkpoint']}")
    else:
        best = None
        scored = [r for r in results if r["test_mse"] is not None]
        if scored:
            best = min(scored, key=lambda r: r["test_mse"])
            print(f"\n  Corner bucket had too few points in every config; falling back to "
                  f"best test_mse: hidden_dim={best['hidden_dim']} n_layers={best['n_layers']} "
                  f"-> {best['checkpoint']}")

    summary_path = REPO_ROOT / "results" / f"hvsc1_al_capacity_sweep_{args.round_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(
        {"round_id": args.round_id, "results": results, "best": best}, indent=2
    ))
    print(f"\n  Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
