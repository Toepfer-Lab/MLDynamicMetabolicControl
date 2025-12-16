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
    payload = torch.load(path, map_location="cpu", weights_only=False)

    # Case 1: new-format payload dict with hyperparams + scalers + metadata
    if isinstance(payload, dict) and ("model_state" in payload or "state_dict" in payload or "model_state_dict" in payload):
        hyper = payload.get("model_hyperparams", {})
        model = model_cls(
            input_dim=hyper.get("input_dim", 1),
            hidden_dim=hyper.get("hidden_dim", 5),
            output_dim=hyper.get("output_dim", 3),
        )

        state = payload.get("model_state") or payload.get("state_dict") or payload.get("model_state_dict")
        model.load_state_dict(state)
        model.eval()

        metadata = payload.get("metadata", {})
        x_scaler = payload.get("x_scaler", None)
        y_scaler = payload.get("y_scaler", None)

        if x_scaler is None or y_scaler is None:
            raise KeyError(
                "Checkpoint is missing x_scaler/y_scaler. "
                "Re-train with train_surrogate.py or re-save checkpoint using save_surrogate_checkpoint()."
            )

        return model, x_scaler, y_scaler, metadata

    # Case 2: weights-only checkpoint (raw state_dict)
    if isinstance(payload, dict):
        # This is a best-effort fallback; we don't know hyperparams or scalers.
        raise KeyError(
            "This checkpoint looks like a weights-only state_dict (no model_state/x_scaler/y_scaler). "
            "Please re-save it with save_surrogate_checkpoint() or retrain using train_surrogate.py."
        )

    raise TypeError(f"Unrecognized checkpoint format: {type(payload)}")


