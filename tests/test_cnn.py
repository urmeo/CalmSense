"""The 1D-CNN learns a signal and separates training from validation subjects."""

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


def test_cnn_subject_holdout_normalizes_and_weights_training_only(monkeypatch):
    pytest.importorskip("torch")
    from src.models.dl import cnn_1d

    groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
    y = np.tile([0, 0, 0, 0, 1, 1], 4)
    rng = np.random.RandomState(1)
    X = rng.randn(24, 1, 128).astype("float32")
    X += np.repeat([0, 10, 100, 1000], 6)[:, None, None]
    model = cnn_1d.CNN1DClassifier(
        in_channels=1, max_epochs=1, batch_size=8, random_state=3, device="cpu"
    )
    train, validation = model._validation_split(y, groups)
    assert set(groups[train]).isdisjoint(groups[validation])
    np.testing.assert_array_equal(model._validation_split(y, groups)[0], train)
    seen_weights = []
    original = cnn_1d.compute_class_weight

    def weights(class_weight, *, classes, y):
        seen_weights.append(y.copy())
        return original(class_weight, classes=classes, y=y)

    monkeypatch.setattr(cnn_1d, "compute_class_weight", weights)
    model.fit(X, y, groups=groups)

    np.testing.assert_allclose(model._mean, X[train].mean(axis=(0, 2), keepdims=True))
    np.testing.assert_allclose(model._std, X[train].std(axis=(0, 2), keepdims=True) + 1e-8)
    assert not np.allclose(model._mean, X.mean(axis=(0, 2), keepdims=True))
    np.testing.assert_array_equal(seen_weights[0], y[train])
    assert model.predict_proba(np.empty((0, 1, 128), dtype="float32")).shape == (0, 2)


def test_cnn_rejects_unfitted_and_nonfinite_predictions():
    pytest.importorskip("torch")
    from sklearn.exceptions import NotFittedError

    from src.models.dl.cnn_1d import CNN1DClassifier

    model = CNN1DClassifier(in_channels=1, max_epochs=1)
    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 1, 128)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.full((8, 1, 128), np.nan), np.tile([0, 1], 4))


def test_cnn_retains_singleton_minimum_length_training_batch():
    pytest.importorskip("torch")
    from src.models.dl.cnn_1d import CNN1DClassifier

    rng = np.random.RandomState(2)
    X = rng.randn(8, 1, 61).astype("float32")
    y = np.tile([0, 1], 4)
    model = CNN1DClassifier(
        in_channels=1, max_epochs=1, batch_size=5, val_fraction=0.25, random_state=2, device="cpu"
    )
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (8, 2) and np.isfinite(proba).all()


@pytest.mark.parametrize("failure", ["overflow", "invalid_scale", "invalid_network"])
def test_cnn_rejects_nonfinite_normalization_or_inference(failure):
    torch = pytest.importorskip("torch")
    from src.models.dl.cnn_1d import CNN1DClassifier

    class FixedNetwork(torch.nn.Module):
        def forward(self, X):
            fill = float("nan") if failure == "invalid_network" else 0.0
            return torch.full((len(X), 2), fill)

    model = CNN1DClassifier(in_channels=1, device="cpu")
    model.classes_ = np.array([0, 1])
    model.model = FixedNetwork()
    model._mean = np.zeros((1, 1, 1), dtype="float32")
    model._std = np.full((1, 1, 1), 1e-8, dtype="float32")
    X = np.zeros((2, 1, 61), dtype="float32")
    if failure == "overflow":
        X.fill(1e38)  # Finite before scaling; float32 cannot represent the normalized values.
    elif failure == "invalid_scale":
        model._std.fill(np.inf)
    with pytest.raises(ValueError, match="CNN (normalization|inference)"):
        model.predict_proba(X)
