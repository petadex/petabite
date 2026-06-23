"""Activity prediction metrics for regression and classification.

Regression (continuous activity score): RMSE, R², Spearman ρ.
Classification (active/inactive or activity bins): accuracy, macro F1.
Both are reported after each eval round so the active-learning loop can track
model improvement as wet-lab data accumulates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, R^2, and Spearman correlation."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rho = spearmanr(y_true, y_pred).correlation
    return {
        "rmse": rmse,
        "r2": float(r2) if r2 == r2 else 0.0,
        "spearman": float(rho) if rho == rho else 0.0,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro F1."""
    accuracy = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": accuracy, "f1": f1}
