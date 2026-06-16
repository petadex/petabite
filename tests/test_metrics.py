import numpy as np

from petabite.trainer_module.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    m = regression_metrics(y_true=y, y_pred=y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert "spearman" in m


def test_classification_metrics_perfect():
    y = np.array([0, 1, 1, 0])
    m = classification_metrics(y_true=y, y_pred=y)
    assert m["accuracy"] == 1.0
    assert "f1" in m
