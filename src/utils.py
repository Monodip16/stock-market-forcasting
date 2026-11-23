"""
Utility Functions for Stock Market Prediction Project.

This module provides helper utilities for:
- Reproducibility (random seed management across NumPy, Python, and TensorFlow)
- Metric computations (Mean Absolute Error and Root Mean Squared Error)
- Environment setup (Matplotlib temporary cache, TensorFlow log verbosity)
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure safe temporary directories for matplotlib to avoid permission warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds across Python standard library, NumPy, and TensorFlow
    to ensure full determinism and reproducibility in model training.

    Parameters:
    -----------
    seed : int
        The integer seed to use (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """
    Compute key forecasting evaluation metrics:
    - Mean Absolute Error (MAE): Measures average magnitude of errors in identical units.
    - Root Mean Squared Error (RMSE): Penalizes larger errors more heavily.
    - Mean Absolute Percentage Error (MAPE): Relative error expressed as a percentage.

    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        Ground truth actual prices.
    y_pred : np.ndarray or pd.Series
        Model predicted prices.

    Returns:
    --------
    dict[str, float]
        Dictionary containing MAE, RMSE, and MAPE metrics rounded to 4 decimals.
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    y_p = np.asarray(y_pred, dtype=float).flatten()

    # Drop any NaN pairs if present
    valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p))
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    if len(y_t) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Compute MAPE safely avoiding zero-division
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + epsilon))) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
    }


def ensure_directory(path: str) -> None:
    """Ensure that the parent directory of a file path exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
