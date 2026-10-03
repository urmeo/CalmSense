"""The shared cross-dataset feature space is consistent and robust."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.datasets import non_eeg
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


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, True, np.bool_(True), None, "invalid"])
@pytest.mark.parametrize("channel", ["eda_fs", "temp_fs"])
def test_invalid_sampling_rates_are_rejected(rate, channel):
    rates = {"eda_fs": 4, "temp_fs": 4, channel: rate}
    with pytest.raises(ValueError, match="finite and positive"):
        portable_features([1, 2], [35, 36], [], [], **rates)


def test_sampling_rates_must_be_explicit():
    with pytest.raises(TypeError, match="eda_fs"):
        portable_features([1, 2], [35, 36], [], [])


def test_noneeg_windows_align_annotations_sensor_channels_and_hr(tmp_path, monkeypatch):
    monkeypatch.setattr(non_eeg, "DATA_DIR", tmp_path)
    (tmp_path / "Subject1_AccTempEDA.hea").touch()
    sensor_time = np.arange(180 * 8) / 8
    sensor = np.column_stack(
        [
            np.zeros((len(sensor_time), 2)),
            np.ones(len(sensor_time)),
            35 - 0.01 * sensor_time,
            3 + 0.25 * sensor_time,
        ]
    )
    hr = np.column_stack([np.zeros(180), 70 + np.arange(180)])
    annotations = SimpleNamespace(
        sample=np.array([0, 480, 960]), aux_note=["Relax", "CognitiveStress", "PhysicalStress"]
    )

    def read(record):
        return (
            SimpleNamespace(fs=1, p_signal=hr)
            if record.endswith("SpO2HR")
            else SimpleNamespace(fs=8, p_signal=sensor)
        )

    monkeypatch.setitem(
        sys.modules, "wfdb", SimpleNamespace(rdrecord=read, rdann=lambda *args: annotations)
    )
    frame = non_eeg.build(["Subject1"])
    assert frame["label"].tolist() == [0, 1]
    np.testing.assert_allclose(frame["EDA_slope"], 0.25)
    np.testing.assert_allclose(frame["TEMP_slope"], -0.01)
    np.testing.assert_allclose(frame["HR_mean"], [99.5, 159.5])


def test_noneeg_rejects_annotations_beyond_available_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(non_eeg, "DATA_DIR", tmp_path)
    (tmp_path / "Subject1_AccTempEDA.hea").touch()
    annotations = SimpleNamespace(sample=np.array([0, 900]), aux_note=["Relax", "CognitiveStress"])
    monkeypatch.setitem(
        sys.modules,
        "wfdb",
        SimpleNamespace(
            rdann=lambda *args: annotations,
            rdrecord=lambda record: SimpleNamespace(
                fs=1 if record.endswith("SpO2HR") else 8,
                p_signal=np.ones(
                    (
                        60 if record.endswith("SpO2HR") else 480,
                        2 if record.endswith("SpO2HR") else 5,
                    )
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="outside the sensor record"):
        non_eeg.build(["Subject1"])
