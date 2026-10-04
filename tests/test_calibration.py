import numpy as np
import pytest

from src.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    net_benefit,
    normalize_binary_calibration,
    reliability_curve,
)


def test_binary_brier_is_independent_of_probability_representation():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.3])
    assert brier_score(y, p) == pytest.approx(0.0375)
    assert brier_score(y, np.column_stack([1 - p, p])) == pytest.approx(0.0375)


def test_multiclass_brier_remains_summed():
    y = np.array([0, 2])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    assert brier_score(y, p) == pytest.approx(0.06)


def test_legacy_calibration_conversion_preserves_source_and_is_idempotent():
    original = {
        "loso": {"ece": 0.07, "brier": 0.136},
        "gap_significance": {
            "mean_brier_gap": 0.06,
            "ci95": [0.02, 0.1],
            "wilcoxon_p": 0.001,
            "per_subject": {"S2": {"loso": 0.2, "within": 0.1}},
        },
        "provenance": {"git_sha": "historical", "generated_at": "original"},
    }
    out = normalize_binary_calibration(original)
    assert out["loso"] == {"ece": 0.07, "brier": 0.068}
    assert out["gap_significance"]["mean_brier_gap"] == 0.03
    assert out["gap_significance"]["ci95"] == [0.01, 0.05]
    assert out["gap_significance"]["per_subject"]["S2"] == {"loso": 0.1, "within": 0.05}
    assert out["gap_significance"]["wilcoxon_p"] == original["gap_significance"]["wilcoxon_p"]
    assert out["provenance"] == original["provenance"]
    assert original["loso"]["brier"] == 0.136
    assert normalize_binary_calibration(out) == out


def test_unknown_brier_scale_is_rejected():
    with pytest.raises(ValueError, match="Unknown binary Brier definition"):
        normalize_binary_calibration({"brier_definition": "unknown"})


def test_net_benefit_at_impossible_threshold_is_zero():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.3, 0.8, 0.2])
    nb = net_benefit(y, p, np.array([0.5, 1.0]))
    assert np.all(np.isfinite(nb))
    assert nb[1] == 0.0


def test_perfect_calibration_has_zero_ece():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    proba = np.full(10, 0.5)
    assert expected_calibration_error(y, proba, n_bins=5) < 1e-9


def test_overconfident_model_has_large_ece():
    y = np.array([1, 0] * 50)
    proba = np.full(100, 0.99)
    ece = expected_calibration_error(y, proba, n_bins=10)
    assert abs(ece - 0.49) < 0.05


def test_brier_rewards_confident_correct():
    y = np.array([1, 0])
    confident = brier_score(y, np.array([0.9, 0.1]))
    unsure = brier_score(y, np.array([0.5, 0.5]))
    assert confident < unsure


def test_brier_perfect_is_zero():
    y = np.array([1, 0, 1])
    assert brier_score(y, np.array([1.0, 0.0, 1.0])) < 1e-12


def test_reliability_bins_sum_to_n():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    proba = rng.rand(200)
    rows = reliability_curve(y, proba, n_bins=10)
    assert sum(r["count"] for r in rows) == 200


def test_mce_at_least_ece():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 300)
    proba = rng.rand(300)
    assert maximum_calibration_error(y, proba) >= expected_calibration_error(y, proba) - 1e-9


def test_net_benefit_treat_none_baseline():
    y = np.array([1, 0, 1, 0])
    nb = net_benefit(y, np.zeros(4), np.array([0.2, 0.5]))
    assert np.allclose(nb, 0.0)


def test_gap_significance_detects_consistent_gap():
    from scripts.calibration import gap_significance

    loso = {f"S{i}": 0.20 for i in range(12)}
    within = {f"S{i}": 0.10 for i in range(12)}
    sig = gap_significance(loso, within)
    assert sig["n_subjects"] == 12
    assert abs(sig["mean_brier_gap"] - 0.10) < 1e-9
    assert sig["ci95"][0] > 0
    assert sig["wilcoxon_p"] < 0.05


@pytest.mark.parametrize(
    "y, proba",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([0], [np.nan]),
        ([1], [1.1]),
        ([2], [0.3]),
        ([0.5], [0.5]),
        ([0], [[0.2, 0.2]]),
        ([0], [[1.0]]),
        ([[0]], [0.1]),
    ],
)
def test_calibration_rejects_invalid_probability_contract(y, proba):
    for metric in (brier_score, expected_calibration_error, reliability_curve):
        with pytest.raises(ValueError):
            metric(y, proba)


@pytest.mark.parametrize("bins", [0, -1, 2.5, True])
def test_calibration_requires_positive_integer_bin_count(bins):
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0, 1], [0.2, 0.8], bins)


def test_binary_reliability_ties_match_two_column_predictions():
    y = [0, 0, 1]
    p = np.array([0.5, 0.5, 0.5])
    assert reliability_curve(y, p) == reliability_curve(y, np.column_stack([1 - p, p]))


@pytest.mark.parametrize("thresholds", [[-0.1], [1.1], [np.nan], [[0.5]]])
def test_net_benefit_rejects_invalid_thresholds(thresholds):
    with pytest.raises(ValueError, match="Thresholds"):
        net_benefit([0, 1], [0.1, 0.9], np.array(thresholds))


def test_single_class_loso_probabilities_keep_baseline_stress_columns():
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    from scripts.calibration import loso_proba

    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])
    groups = np.array(["S0", "S0", "S1", "S1"])
    factory = lambda: Pipeline([("clf", DummyClassifier())])  # noqa: E731
    true, proba, subjects = loso_proba(factory, X, y, groups)
    np.testing.assert_array_equal(true, y)
    np.testing.assert_array_equal(subjects, groups)
    np.testing.assert_array_equal(proba, [[0, 1], [0, 1], [1, 0], [1, 0]])


@pytest.mark.parametrize("label", [0, 1])
def test_sigmoid_single_class_calibrator_returns_class_constant(label):
    from scripts.calibration import _apply_calibrator, _fit_calibrator

    model = _fit_calibrator(np.array([0.2, 0.4, 0.8]), np.full(3, label), "sigmoid")
    np.testing.assert_array_equal(
        _apply_calibrator(model, np.array([0.1, 0.9]), "sigmoid"), [label, label]
    )


def test_gap_significance_rejects_missing_or_unpaired_subject_scores():
    from scripts.calibration import gap_significance

    with pytest.raises(ValueError, match="three subjects"):
        gap_significance({}, {})
    with pytest.raises(ValueError, match="three subjects"):
        gap_significance({"S0": 0.1, "S1": 0.2, "S2": 0.3}, {"S0": 0.1, "S1": 0.2, "S3": 0.3})


@pytest.mark.parametrize(
    "classes, probabilities",
    [
        ([0], [[np.nan]]),
        ([0], [[0.5]]),
        ([0, 1], [[0.2, 0.9]]),
        ([0, 1], [[-0.1, 1.1]]),
        ([0, 1], [[0.2]]),
        ([0, 0], [[0.5, 0.5]]),
    ],
)
def test_estimator_probability_contract_rejects_invalid_folds(classes, probabilities):
    from scripts.calibration import _pos_proba

    class Estimator:
        classes_ = classes

        def predict_proba(self, X):
            return probabilities

    with pytest.raises(ValueError, match="(probabilities|classes)"):
        _pos_proba(Estimator(), np.zeros((1, 1)))


def test_estimator_positive_probabilities_follow_reversed_class_order():
    from scripts.calibration import _pos_proba

    class Estimator:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return [[0.8, 0.2], [0.3, 0.7]]

    np.testing.assert_array_equal(_pos_proba(Estimator(), np.zeros((2, 1))), [0.8, 0.3])
