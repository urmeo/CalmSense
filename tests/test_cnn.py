"""The 1D-CNN learns a signal and generalizes to held-out data.

torch is an optional heavy dependency; if it is not installed these tests skip
cleanly rather than erroring at collection (which would abort the whole run).
"""

import numpy as np
import pytest


def test_cnn_generalizes_to_held_out_split():
    pytest.importorskip("torch")
    from src.models.dl.cnn_1d import CNN1DClassifier

    rng = np.random.RandomState(0)
    n, c, length = 160, 5, 256
    X = rng.randn(n, c, length).astype("float32")
    y = (rng.randn(n) > 0).astype(int)
    X[y == 1, 0, :] += 2.0  # class signal in channel 0

    X_tr, y_tr = X[:120], y[:120]
    X_te, y_te = X[120:], y[120:]

    model = CNN1DClassifier(in_channels=c, max_epochs=25, random_state=0)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)

    assert proba.shape == (len(y_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    # accuracy on data the model never saw
    assert (model.predict(X_te) == y_te).mean() > 0.7


def test_cnn_loso_reports_pooled_accuracy(monkeypatch):
    from scripts.run_experiment import cnn_loso
    from src.models.dl import cnn_1d

    class ConstantClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(cnn_1d, "CNN1DClassifier", ConstantClassifier)
    # Unequal subject sizes distinguish pooled accuracy from the subject mean.
    result = cnn_loso(
        np.zeros((6, 1, 8)), np.array([0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 1, 1])
    )
    assert result["accuracy_mean"] == 0.5
    assert result["pooled_accuracy"] == pytest.approx(1 / 3)
