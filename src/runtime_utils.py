"""
Utility helpers for CLI scripts and cluster runs.
"""

from pathlib import Path
import sys
import torch

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOT_DIR = PROJECT_ROOT / "plots"


def ensure_repo_on_path():
    """
    Make sure ``src`` is importable when scripts are launched via sbatch.
    """
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))


def ensure_output_dirs():
    """
    Create common output folders if they do not exist.
    """
    for path in (DATA_DIR, MODEL_DIR, RESULTS_DIR, PLOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_surrogate_checkpoint(model, x_scaler, y_scaler, metadata, path):
    """
    Persist model weights plus scalers so downstream jobs can reload everything.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "model_hyperparams": {
            "input_dim": getattr(model, "input_dim", None) or model.net[0].in_features,
            "hidden_dim": getattr(model, "hidden_dim", None) or model.net[0].out_features,
            "output_dim": getattr(model, "output_dim", None) or model.net[-1].out_features,
        },
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def load_surrogate_checkpoint(path, model_cls):
    """
    Load a surrogate checkpoint and rebuild the model and scalers.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hyper = payload.get("model_hyperparams", {})
    model = model_cls(
        input_dim=hyper.get("input_dim", 1),
        hidden_dim=hyper.get("hidden_dim", 5),
        output_dim=hyper.get("output_dim", 3),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    metadata = payload.get("metadata", {})
    return model, payload["x_scaler"], payload["y_scaler"], metadata

