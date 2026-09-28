"""The shared cross-dataset feature space is consistent and robust."""

import json

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


@pytest.mark.parametrize("rate", [4, 8])
def test_slopes_use_seconds_and_preserve_missing_sample_gaps(rate):
    t = np.arange(60 * rate) / rate
    eda = 4 + 0.02 * t
    temp = 33 - 0.005 * t
    eda[20 * rate : 40 * rate] = np.nan
    temp[10 * rate : 30 * rate] = np.inf
    features = portable_features(eda, temp, np.ones(10), np.ones(10), eda_fs=rate, temp_fs=rate)
    assert features["EDA_slope"] == pytest.approx(0.02)
    assert features["TEMP_slope"] == pytest.approx(-0.005)


@pytest.mark.parametrize("field", ["eda_fs", "temp_fs"])
@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_portable_features_reject_invalid_sample_rates(field, rate):
    rates = {"eda_fs": 4, "temp_fs": 4, field: rate}
    with pytest.raises(ValueError, match="finite and positive"):
        portable_features([], [], [], [], **rates)


def test_cross_dataset_ignores_legacy_slope_caches(tmp_path, monkeypatch):
    import pandas as pd

    from scripts import cross_dataset

    processed = tmp_path / "processed"
    processed.mkdir()
    for name in ("portable_wesad.parquet", "portable_noneeg.parquet"):
        (processed / name).write_bytes(b"legacy cache must not be read or replaced")
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(cross_dataset, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cross_dataset, "FIGURES_DIR", tmp_path / "figures")
    frame = pd.DataFrame(
        {"subject": ["synthetic_a", "synthetic_b"], "label": [0, 1], "EDA_slope": [0.02, 0.02]}
    )
    monkeypatch.setattr(cross_dataset, "wesad_portable", lambda: frame)
    monkeypatch.setattr(cross_dataset.non_eeg, "build", lambda: frame)
    scores = {"accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5}
    monkeypatch.setattr(cross_dataset, "within", lambda *args: scores)
    monkeypatch.setattr(cross_dataset, "transfer", lambda *args: scores)

    cross_dataset.run()

    for name in ("wesad", "noneeg"):
        assert (processed / f"portable_{name}_v2.parquet").is_file()
        assert (processed / f"portable_{name}.parquet").read_bytes().startswith(b"legacy cache")
    output = json.loads((tmp_path / "results" / "cross_dataset.json").read_text())
    assert output["feature_schema_version"] == 2
    assert output["slope_time_unit"] == "second"
