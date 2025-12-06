"""
predictor_numpy.py

Small helper module:
- loads model/logreg_model.json
- exposes get_model() and predict_proba_one()
"""

import json
import os
import numpy as np

def get_model(model_path: str):
    """Load model JSON from given path and return dict."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model JSON not found at: {model_path}")

    with open(model_path, "r") as f:
        saved = json.load(f)

    mean = np.array(saved["mean"], dtype=float)
    std = np.array(saved["std"], dtype=float)
    std[std == 0] = 1.0

    return {
        "feature_names": saved["feature_names"],
        "weights": np.array(saved["weights"], dtype=float),
        "bias": float(saved["bias"]),
        "mean": mean,
        "std": std,
    }


def predict_proba_one(input_row: dict, model_path: str) -> float:
    """
    input_row: dict feature_name -> value (floats)
    model_path: path to logreg_model.json
    returns probability in percent (0..100)
    """
    model = get_model(model_path)
    fnames = model["feature_names"]

    x = np.array([float(input_row[name]) for name in fnames], dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    z = x_std.dot(model["weights"]) + model["bias"]
    p = 1.0 / (1.0 + np.exp(-z))
    return float(p * 100.0)
