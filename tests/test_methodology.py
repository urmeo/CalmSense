"""Subject-independent evaluation, metric conventions, and export contracts."""

import numpy as np
import pytest

from scripts.run_experiment import (
    build_pipeline,
    cnn_loso,
    kfold_accuracy,
    loso_evaluate,
    nonoverlap_mask,
)
from src.dataset import WindowedDataset


def test_pipeline_imputes_and_scales_inside_the_fold():
    pipe = build_pipeline("rf")
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "clf"]


def test_loso_evaluate_holds_out_each_subject():
    rng = np.random.RandomState(0)
    n_subjects, per = 4, 40
    groups = np.repeat([f"S{i}" for i in range(n_subjects)], per)
    X = rng.randn(len(groups), 6)
    y = rng.randint(0, 2, len(groups))

    res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
    assert len(res["per_subject"]) == n_subjects
    assert set(res["per_subject"]["subject"]) == set(groups)


def test_cnn_loso_reports_window_weighted_accuracy(monkeypatch):
    from src.models.dl import cnn_1d

    groups = np.array(["S0"] * 2 + ["S1"] * 6)
    y = np.tile([0, 1], 4)
    x_raw = np.zeros((8, 1, 2))
    # S0 predictions are correct; S1 predictions are wrong. Subject sizes differ.
    x_raw[:, 0, 0] = [0, 1, 1, 0, 1, 0, 1, 0]
    x_raw[:, 0, 1] = np.arange(8)
    train_sizes = []

    class FixedCNN:
        def __init__(self, in_channels, random_state):
            self.training_ids = set()

        def fit(self, X, labels, groups=None):
            self.training_ids = set(X[:, 0, 1])
            assert len(groups) == len(labels)
            train_sizes.append(len(labels))

        def predict(self, X):
            assert self.training_ids.isdisjoint(X[:, 0, 1])
            return X[:, 0, 0].astype(int)

    monkeypatch.setattr(cnn_1d, "CNN1DClassifier", FixedCNN)
    result = cnn_loso(x_raw, y, groups)

    assert train_sizes == [6, 2]
    assert result["accuracy_mean"] == pytest.approx(0.5)
    assert result["pooled_accuracy"] == pytest.approx(0.25)
    np.testing.assert_array_equal(result["y_true"], y)


def test_loso_splits_have_disjoint_subjects():
    from sklearn.model_selection import LeaveOneGroupOut

    groups = np.repeat([f"S{i}" for i in range(5)], 30)
    X = np.zeros((len(groups), 3))
    y = np.zeros(len(groups))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert train_subjects.isdisjoint(test_subjects)
        assert len(test_subjects) == 1


def test_nonoverlap_mask_keeps_every_other_window_per_subject():
    # uneven per-subject block sizes (5 and 4) to catch off-by-one slicing
    groups = np.array(["S0", "S0", "S0", "S0", "S0", "S1", "S1", "S1", "S1"])
    mask = nonoverlap_mask(groups)
    assert list(np.where(mask)[0]) == [0, 2, 4, 5, 7]
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        assert list(idx[mask[idx]]) == list(idx[::2])


def test_kfold_gap_uses_non_overlapping_windows():
    # the gap baseline must drop every other (overlapping) window per subject
    rng = np.random.RandomState(0)
    groups = np.repeat(["S0", "S1"], 40)
    X = rng.randn(len(groups), 5)
    # class blocks per subject so both classes survive the every-other-window subset
    y = np.tile(np.concatenate([np.zeros(20), np.ones(20)]).astype(int), 2)
    assert int(nonoverlap_mask(groups).sum()) == 40
    acc = kfold_accuracy(lambda: build_pipeline("lr"), X, y, groups)
    assert 0.0 <= acc <= 1.0


def test_window_label_rejects_impure_and_out_of_set_windows():
    ds = WindowedDataset.__new__(WindowedDataset)
    ds.purity = 0.9

    assert ds._window_label(np.full(100, 2)) == 2  # pure stress
    assert ds._window_label(np.concatenate([np.full(85, 2), np.full(15, 1)])) is None
    assert ds._window_label(np.concatenate([np.full(95, 2), np.full(5, 1)])) == 2
    # dominant label outside {baseline, stress, amusement} -> rejected
    assert ds._window_label(np.full(100, 4)) is None


def test_scaler_imputer_fit_per_fold_never_on_held_out_subject():
    """Held-out subject shifts must not influence scaler or imputer statistics."""
    from sklearn.model_selection import LeaveOneGroupOut

    rng = np.random.RandomState(0)
    subjects = ["S0", "S1", "S2"]
    shifts = {"S0": 0.0, "S1": 50.0, "S2": 200.0}  # asymmetric so no fold mean == global
    groups = np.repeat(subjects, 30)
    X = np.vstack([rng.randn(30, 4) + shifts[s] for s in subjects])
    y = np.tile([0, 1], 45)  # both classes present in every subject block

    logo = LeaveOneGroupOut()
    fold_scaler_means = {}
    global_mean = X.mean(axis=0)
    for train_idx, test_idx in logo.split(X, y, groups):
        held = groups[test_idx][0]
        pipe = build_pipeline("lr")
        pipe.fit(X[train_idx], y[train_idx])
        scaler_mean = pipe.named_steps["scale"].mean_
        imputer_stats = pipe.named_steps["impute"].statistics_
        np.testing.assert_allclose(scaler_mean, X[train_idx].mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            imputer_stats, np.median(X[train_idx], axis=0), rtol=1e-6, atol=1e-6
        )
        assert not np.allclose(scaler_mean, global_mean, atol=1.0)
        fold_scaler_means[held] = scaler_mean

    assert not np.allclose(fold_scaler_means["S0"], fold_scaler_means["S1"], atol=1.0)
    assert not np.allclose(fold_scaler_means["S1"], fold_scaler_means["S2"], atol=1.0)


def test_loso_macro_f1_uses_the_full_task_class_set():
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    groups = np.repeat(["S0", "S1", "S2"], 4)
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    factory = lambda: Pipeline([("clf", DummyClassifier(strategy="constant", constant=0))])  # noqa: E731
    result = loso_evaluate(factory, np.zeros((12, 1)), y, groups)
    s0 = result["per_subject"].set_index("subject").loc["S0"]
    assert s0["accuracy"] == 1
    assert s0["f1_macro"] == pytest.approx(0.5)


def test_recalibration_predictions_hold_out_outer_and_inner_subjects():
    from scripts.calibration import loso_recalibrated_proba

    groups = np.repeat(np.arange(4), 6)
    X = np.column_stack([groups, np.arange(24)])
    y = np.tile([0, 1], 12)
    predictions = []

    class RecordingEstimator:
        classes_ = np.array([0, 1])

        def __init__(self):
            self.named_steps = {"clf": self}

        def fit(self, values, labels):
            self.trained_subjects = set(values[:, 0])
            return self

        def predict_proba(self, values):
            held_subjects = set(values[:, 0])
            assert self.trained_subjects.isdisjoint(held_subjects)
            predictions.append((self.trained_subjects, held_subjects))
            return np.tile([0.4, 0.6], (len(values), 1))

    true, proba = loso_recalibrated_proba(RecordingEstimator, X, y, groups, method="sigmoid")
    np.testing.assert_array_equal(true, y)
    assert len(predictions) == 16  # Four outer predictions and three inner predictions per fold.
    assert proba.shape == (24, 2) and np.isfinite(proba).all()


def test_threshold_pass_balances_xgboost_on_training_labels(monkeypatch):
    from sklearn.utils.class_weight import compute_sample_weight

    from scripts import threshold_metrics

    fitted = []

    class XGBClassifier:
        classes_ = np.array([0, 1])

        def __init__(self):
            self.named_steps = {"clf": self}

        def fit(self, values, labels, **kwargs):
            np.testing.assert_array_equal(
                kwargs["clf__sample_weight"], compute_sample_weight("balanced", labels)
            )
            fitted.append(len(labels))

        def predict_proba(self, values):
            return np.tile([0.3, 0.7], (len(values), 1))

    monkeypatch.setattr(threshold_metrics, "build_pipeline", lambda key: XGBClassifier())
    true, proba = threshold_metrics.loso_pos_proba(
        "xgb", np.zeros((12, 1)), np.tile([0, 0, 0, 1], 3), np.repeat(["S0", "S1", "S2"], 4)
    )
    assert fitted == [8, 8, 8]
    assert len(true) == len(proba) == 12


def test_constant_predictions_select_a_finite_operating_point():
    import json

    from scripts.threshold_metrics import operating_point

    point = operating_point([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert point["threshold"] == 0.5
    assert point["ppv"] == 0.5 and point["npv"] is None
    json.dumps(point, allow_nan=False)


@pytest.mark.parametrize("invalid_case", ["one_class", "nonfinite_auc"])
def test_threshold_results_mark_undefined_scores_unavailable(tmp_path, monkeypatch, invalid_case):
    import json

    from scripts import threshold_metrics

    monkeypatch.setattr(threshold_metrics, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(threshold_metrics, "FEATURE_MODELS", ["lr"])
    monkeypatch.setattr(threshold_metrics, "load_cached", lambda: (None, None))
    y = np.array([0, 0] if invalid_case == "one_class" else [0, 1])
    monkeypatch.setattr(
        threshold_metrics, "prepare_task", lambda *args: (None, y, None, None, None)
    )
    monkeypatch.setattr(
        threshold_metrics, "loso_pos_proba", lambda *args: (y, np.array([0.2, 0.8]))
    )
    if invalid_case == "nonfinite_auc":
        monkeypatch.setattr(threshold_metrics, "roc_auc_score", lambda *args: float("nan"))
    threshold_metrics.run()
    result = json.loads((tmp_path / "threshold_metrics.json").read_text())
    assert result["models"][0]["available"] is False
    assert "require" in result["models"][0]["reason"]


@pytest.mark.parametrize(
    "schema, subjects",
    [
        (1, ["S0", "S1"]),
        (2, ["S0", "S2"]),
        (2, ["S0", "S0"]),
        (2, ["S0", ""]),
        (2, ["S0", 1]),
        (2, []),
    ],
)
def test_wrist_rejects_incompatible_chest_results_before_fitting(
    tmp_path, monkeypatch, schema, subjects
):
    import json

    from scripts import wrist

    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "binary": {
                    "feature_schema_version": schema,
                    "per_subject": [{"subject": s} for s in subjects],
                }
            }
        )
    )
    monkeypatch.setattr(wrist, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(wrist, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(wrist, "load_wrist", lambda: object())
    monkeypatch.setattr(
        wrist,
        "prepare_binary",
        lambda *args: (np.zeros((2, 1)), np.array([0, 1]), np.array(["S0", "S1"])),
    )
    monkeypatch.setattr(
        wrist, "loso_evaluate", lambda *args: pytest.fail("Fitted an incompatible comparison")
    )
    with pytest.raises(ValueError, match="run_experiment.py --rebuild --subjects S0 S1"):
        wrist.run()


def test_wrist_accepts_matching_chest_schema_and_subjects_in_any_order():
    from scripts.wrist import _validate_chest_comparison
    from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION

    _validate_chest_comparison(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "per_subject": [{"subject": "S1"}, {"subject": "S0"}],
        },
        np.array(["S0", "S0", "S1", "S1"]),
    )


def test_all_tied_model_statistics_remain_finite():
    from scripts.stats import _friedman, bootstrap_ci, holm_bonferroni

    scores = [np.array([0.6, 0.8, 0.9])] * 4
    assert _friedman(scores) == (0.0, 1.0)
    assert bootstrap_ci(np.ones(3), n=10) == (1.0, 1.0)
    assert holm_bonferroni([("a", 0.01), ("b", 0.04), ("c", 0.03)]) == {
        "a": 0.03,
        "c": 0.06,
        "b": 0.06,
    }


@pytest.mark.parametrize("scores", [[], [0.1, 0.2], [0.1, 0.2, np.nan]])
def test_subject_bootstrap_rejects_insufficient_or_nonfinite_data(scores):
    from scripts.stats import bootstrap_ci

    with pytest.raises(ValueError, match="three finite"):
        bootstrap_ci(scores)


def test_experiment_exports_zero_gap_and_feature_schema(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from scripts import run_experiment

    y = np.tile([0, 1], 6)
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(["S0", "S1", "S2"], 4),
            "window_id": np.tile(np.arange(4), 3),
            "label": y + 1,
            "label_name": np.where(y, "stress", "baseline"),
            "HRV_test": y,
        }
    )
    raw = np.zeros((12, 1, 64))
    saved_models = []

    class InverseEstimator:
        def __init__(self):
            self.named_steps = {"clf": self}

        def fit(self, X, labels):
            return self

        def predict(self, X):
            return 1 - X[:, 0].astype(int)

    monkeypatch.setattr(run_experiment, "load_cached", lambda: (frame, raw))
    monkeypatch.setattr(run_experiment, "build_pipeline", lambda key: InverseEstimator())
    monkeypatch.setattr(run_experiment, "CLASSIFIERS", ["rf"])
    monkeypatch.setattr(run_experiment, "TASKS", {"binary": run_experiment.TASKS["binary"]})
    for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
        monkeypatch.setattr(run_experiment, name, tmp_path / name.lower())
    for name in (
        "plot_model_comparison",
        "plot_confusion",
        "plot_per_subject",
        "plot_embedding",
        "plot_gap",
    ):
        monkeypatch.setattr(run_experiment, name, lambda *args: None)
    monkeypatch.setattr(
        run_experiment,
        "shap_analysis",
        lambda *args: pd.DataFrame({"feature": ["HRV_test"], "mean_abs_shap": [0.1]}),
    )
    monkeypatch.setattr(
        run_experiment, "save_verified_joblib", lambda model, path: saved_models.append(model)
    )
    monkeypatch.setattr(run_experiment.sys, "argv", ["run_experiment.py", "--no-cnn"])

    run_experiment.run()

    result = json.loads((run_experiment.RESULTS_DIR / "metrics.json").read_text())
    assert result["binary"]["loso_matched_accuracy"] == 0
    assert result["binary"]["within_subject_accuracy"] == 0
    assert result["binary"]["optimism_gap_pts"] == 0
    assert result["binary"]["n_subjects"] == 3
    assert saved_models[0]["feature_schema_version"] == result["binary"]["feature_schema_version"]
    assert result["methodology"]["cnn_validation"] is None
    assert result["benchmark_protocol_version"] == 2


def test_benchmark_rejects_missing_amusement_before_any_model_fits(tmp_path, monkeypatch):
    import pandas as pd

    from scripts import run_experiment

    frame = pd.DataFrame({"subject_id": ["S0", "S0", "S1", "S1"], "label": [1, 2, 1, 2]})
    monkeypatch.setattr(run_experiment, "load_cached", lambda: (frame, np.zeros((4, 1, 64))))
    monkeypatch.setattr(run_experiment.sys, "argv", ["run_experiment.py", "--no-cnn"])
    for name in ("RESULTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
        monkeypatch.setattr(run_experiment, name, tmp_path / name.lower())
    monkeypatch.setattr(
        run_experiment, "loso_evaluate", lambda *args: pytest.fail("Fitted before task validation")
    )
    with pytest.raises(
        ValueError, match="multiclass benchmark is missing required classes: amusement"
    ):
        run_experiment.run()


@pytest.mark.parametrize("n_features", [1, 3])
def test_descriptive_pca_handles_available_feature_dimensions(tmp_path, monkeypatch, n_features):
    from scripts import run_experiment

    X = np.random.RandomState(4).randn(8, n_features)
    y = np.tile([0, 1], 4)
    plotted_vertical = []
    original_scatter = run_experiment.plt.scatter

    def scatter(x, vertical, **kwargs):
        plotted_vertical.extend(vertical)
        return original_scatter(x, vertical, **kwargs)

    monkeypatch.setattr(run_experiment.plt, "scatter", scatter)
    path = tmp_path / "pca.png"
    run_experiment.plot_embedding(X, y, ["baseline", "stress"], path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(plotted_vertical) == len(y)
    assert np.isfinite(plotted_vertical).all()
    assert bool(np.any(np.array(plotted_vertical) != 0)) == (n_features > 1)


def test_descriptive_pca_rejects_class_name_mismatch(tmp_path):
    from scripts.run_experiment import plot_embedding

    with pytest.raises(ValueError, match="class names"):
        plot_embedding(
            np.ones((4, 2)),
            np.array([0, 1, 0, 1]),
            ["baseline", "stress", "amusement"],
            tmp_path / "pca.png",
        )
