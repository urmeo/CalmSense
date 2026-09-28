"""Shared experiment inputs and subset runs preserve benchmark boundaries."""

import sys

import numpy as np
import pandas as pd
import pytest

from scripts import calibration, run_experiment


@pytest.mark.parametrize("synthetic,n_subjects", [(False, 6), (True, 6), (True, 8)])
def test_binary_input_loading_preserves_features_and_synthetic_settings(
    monkeypatch, synthetic, n_subjects
):
    from src import synthetic as generator

    frame = pd.DataFrame(
        {"label": [1, 2, 3], "subject_id": ["A", "B", "C"], "signal": [1, np.inf, 3]}
    )
    raw = np.arange(6).reshape(3, 1, 2)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return frame, raw, None

    monkeypatch.setattr(generator, "features", generate)
    monkeypatch.setattr(run_experiment, "load_cached", lambda: (frame, raw))
    X, y, groups, columns, task_raw = run_experiment.load_binary_task(synthetic, n_subjects)
    np.testing.assert_array_equal(X, [[1], [np.nan]])
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_array_equal(groups, ["A", "B"])
    np.testing.assert_array_equal(task_raw, raw[:2])
    assert columns == ["signal"]
    assert calls == (
        [{"n_subjects": n_subjects, "block_sec": 150, "seed": 42}] if synthetic else []
    )


def test_binary_input_loading_requires_cache(monkeypatch):
    monkeypatch.setattr(run_experiment, "load_cached", lambda: None)
    with pytest.raises(SystemExit, match="No cached features"):
        run_experiment.load_binary_task()


def test_subset_run_preserves_full_cache_and_output_locations(tmp_path, monkeypatch):
    cache = tmp_path / "features.parquet"
    cache.write_bytes(b"full benchmark cache")
    paths = [tmp_path / name for name in ("results", "figures", "models")]
    for attr, path in zip(("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"), paths):
        path.mkdir()
        (path / "snapshot").write_bytes(b"full benchmark artifact")
        monkeypatch.setattr(run_experiment, attr, path)
    labels = np.tile([1, 1, 2, 2], 12)
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(["S2", "S3"], 24),
            "label": labels,
            "signal": labels.astype(float),
            "noise": np.random.RandomState(42).normal(size=len(labels)),
        }
    )

    class Dataset:
        def build(self, subjects=None, cache=True):
            assert subjects == ["S2", "S3"]
            if cache:
                (tmp_path / "features.parquet").write_bytes(b"subset cache")
            return frame, np.zeros((len(labels), 1, 8)), labels

    monkeypatch.setattr(run_experiment, "WindowedDataset", Dataset)
    monkeypatch.setattr(run_experiment, "set_seed", lambda _: None)
    monkeypatch.setattr(run_experiment, "CLASSIFIERS", ["lr"])
    monkeypatch.setattr(run_experiment, "TASKS", {"binary": run_experiment.TASKS["binary"]})
    monkeypatch.setattr(run_experiment, "shap_analysis", lambda *args: pd.DataFrame())
    for plot in ("model_comparison", "confusion", "per_subject", "embedding", "gap"):
        monkeypatch.setattr(
            run_experiment, f"plot_{plot}", lambda *args: args[-1].write_bytes(b"subset figure")
        )
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--subjects", "S2", "S3", "--no-cnn"])
    run_experiment.run()
    assert cache.read_bytes() == b"full benchmark cache"
    for path in paths:
        assert {p.name for p in path.iterdir()} == {"snapshot", "subset"}
        assert (path / "snapshot").read_bytes() == b"full benchmark artifact"
    assert (paths[0] / "subset" / "metrics.json").is_file()
    assert (paths[1] / "subset" / "binary_confusion.png").read_bytes() == b"subset figure"
    assert (paths[2] / "subset" / "stress_classifier.joblib").is_file()
    assert (paths[2] / "subset" / "stress_classifier.joblib.sha256").is_file()


@pytest.mark.parametrize("n_groups", [1, 2, 6])
def test_global_calibrator_uses_only_inner_training_folds(monkeypatch, n_groups):
    groups = np.repeat(np.arange(n_groups), 4)
    y = np.tile([0, 1], len(groups) // 2)
    X = np.arange(len(y)).reshape(-1, 1)
    fitted = []

    class Pipeline:
        classes_ = [0, 1]
        named_steps = {"clf": object()}

        def fit(self, train_X, train_y, **kwargs):
            self.train_groups = set(groups[train_X[:, 0]])
            fitted.append(self.train_groups)

        def predict_proba(self, test_X):
            assert self.train_groups.isdisjoint(groups[test_X[:, 0]])
            positive = test_X[:, 0] / len(y)
            return np.column_stack([1 - positive, positive])

    def fit_calibrator(raw, labels, method):
        np.testing.assert_array_equal(raw, X[:, 0] / len(y))
        np.testing.assert_array_equal(labels, y)
        assert method == "isotonic"
        return "calibrator"

    monkeypatch.setattr(calibration, "_fit_calibrator", fit_calibrator)
    result = calibration._global_calibrator(Pipeline, X, y, groups, "isotonic")
    assert result == (None if n_groups == 1 else "calibrator")
    assert len(fitted) == (0 if n_groups == 1 else min(5, n_groups))
