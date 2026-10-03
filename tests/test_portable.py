"""The shared cross-dataset feature space is consistent and robust."""

import numpy as np
import pytest

from src.portable import portable_features


def test_portable_features_keys_and_finite():
    rng = np.random.RandomState(0)
    feats = portable_features(
        eda=rng.rand(240) + 5,
        temp=rng.rand(240) + 33,
        acc_mag=rng.rand(1920) + 1,
        hr=rng.rand(70) * 10 + 70,
        eda_fs=4,
        temp_fs=4,
    )
    assert len(feats) == 18
    assert any(k.startswith("EDA_") for k in feats)
    assert any(k.startswith("HR_") for k in feats)
    assert np.isfinite(feats["EDA_mean"]) and np.isfinite(feats["HR_std"])


def test_portable_features_handle_empty():
    feats = portable_features(
        eda=np.array([]),
        temp=np.array([1.0]),
        acc_mag=np.array([]),
        hr=np.array([]),
        eda_fs=4,
        temp_fs=4,
    )
    assert len(feats) == 18
    assert np.isnan(feats["EDA_mean"])


def test_same_columns_both_datasets():
    rng = np.random.RandomState(1)
    a = portable_features(
        rng.rand(240), rng.rand(240), rng.rand(1920), rng.rand(70), eda_fs=4, temp_fs=4
    )
    b = portable_features(
        rng.rand(480), rng.rand(480), rng.rand(480), rng.rand(60), eda_fs=8, temp_fs=8
    )
    assert set(a) == set(b)  # device-agnostic, identical feature names


@pytest.mark.parametrize("missing", [False, True])
def test_slopes_match_physical_units_across_sampling_rates(missing):
    for fs in (4, 8):
        t = np.arange(60 * fs) / fs
        eda = 3.0 + 0.25 * t
        temp = 35.0 - 0.01 * t
        if missing:
            eda[[1, 7, 20, 90]] = np.nan
            temp[[4, 8, 35]] = np.inf
        features = portable_features(eda, temp, [], [], eda_fs=fs, temp_fs=fs)
        assert features["EDA_slope"] == pytest.approx(0.25)
        assert features["TEMP_slope"] == pytest.approx(-0.01)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, True, None, "invalid"])
@pytest.mark.parametrize("channel", ["eda_fs", "temp_fs"])
def test_invalid_sampling_rates_are_rejected(rate, channel):
    rates = {"eda_fs": 4, "temp_fs": 4, channel: rate}
    with pytest.raises(ValueError, match="finite and positive"):
        portable_features([1, 2], [35, 36], [], [], **rates)


def test_sampling_rates_must_be_explicit():
    with pytest.raises(TypeError, match="eda_fs"):
        portable_features([1, 2], [35, 36], [], [])
