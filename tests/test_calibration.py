"""Calibration metrics behave correctly on known inputs."""

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
    # At threshold 1.0 nobody is flagged, so net benefit must be 0, not NaN.
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.3, 0.8, 0.2])
    nb = net_benefit(y, p, np.array([0.5, 1.0]))
    assert np.all(np.isfinite(nb))
    assert nb[1] == 0.0


def test_perfect_calibration_has_zero_ece():
    # confidence exactly matches accuracy in every bin
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    proba = np.full(10, 0.5)
    assert expected_calibration_error(y, proba, n_bins=5) < 1e-9


def test_overconfident_model_has_large_ece():
    # always says 0.99 but only half right
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
    # a useless model flagging nobody yields zero benefit
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
    assert sig["ci95"][0] > 0  # gap is consistently positive
    assert sig["wilcoxon_p"] < 0.05
