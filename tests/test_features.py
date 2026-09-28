"""Feature extractors produce correct values on known signals."""

import numpy as np
import pytest

from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.preprocessing.ecg_processor import ECGProcessor


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("sampling_rate", [4, 700])
def test_slopes_keep_original_sample_times(sampling_rate, missing):
    from src.features.eda_features import EDAFeatureExtractor
    from src.features.temperature_features import TemperatureFeatureExtractor

    values = 25.0 + 0.1 * np.arange(30) / sampling_rate
    if missing:
        values[5:15] = np.nan
    temperature = TemperatureFeatureExtractor(sampling_rate).extract_all(values)
    tonic = EDAFeatureExtractor(sampling_rate).extract_tonic_features(values)
    assert temperature["TEMP_slope"] == pytest.approx(0.1)
    assert tonic["SCL_slope"] == pytest.approx(0.1)


def test_entropy_preserves_reference_values_and_degenerate_inputs():
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    extractor = HRVNonlinearExtractor()
    rr = 800 + 30 * np.random.RandomState(42).randn(120)
    assert extractor.compute_sample_entropy(rr) == pytest.approx(1.934860312862147)
    assert extractor.compute_approximate_entropy(rr) == pytest.approx(0.6134612324264683)
    constant = np.full(120, 800.0)
    assert np.isnan(extractor.compute_sample_entropy(constant))
    assert extractor.compute_approximate_entropy(constant) == 0.0
    for compute in (extractor.compute_sample_entropy, extractor.compute_approximate_entropy):
        assert np.isnan(compute(rr[:3]))


def test_missing_eda_preserves_feature_order_and_zero_event_counts():
    from src.features.eda_features import EDAFeatureExtractor

    extractor = EDAFeatureExtractor()
    features = extractor.extract_all({})
    assert list(features) == [
        "SCL_mean",
        "SCL_std",
        "SCL_slope",
        "SCL_min",
        "SCL_max",
        "SCR_count",
        "SCR_rate",
        "SCR_amplitude_mean",
        "SCR_amplitude_max",
        "SCR_rise_time_mean",
        "SCR_recovery_time_mean",
        "SCR_AUC",
        "EDA_mean",
        "EDA_range",
        "EDA_kurtosis",
    ]
    assert all(
        value == 0.0 if key.startswith("SCR_") else np.isnan(value)
        for key, value in features.items()
    )
    features["SCL_mean"] = 3.0
    assert np.isnan(extractor.extract_all({})["SCL_mean"])


def test_compute_failure_preserves_earlier_values_and_remaining_nans(monkeypatch):
    extractor = HRVTimeDomainExtractor()

    def fail(_):
        raise RuntimeError("synthetic compute failure")

    monkeypatch.setattr(extractor, "compute_sdnn", fail)
    features = extractor.extract_all(np.full(60, 800.0))
    assert features["MeanNN"] == 800.0
    assert all(np.isnan(value) for key, value in features.items() if key != "MeanNN")


def test_rmssd_constant_rr_is_zero():
    rr = np.full(60, 800.0)  # constant heartbeat
    features = HRVTimeDomainExtractor().extract_all(rr)
    assert features["RMSSD"] == 0.0
    assert abs(features["MeanNN"] - 800.0) < 1e-6


def test_frequency_features_nan_below_min_rr():
    from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor

    feats = HRVFrequencyDomainExtractor().extract_all(np.full(5, 800.0))  # < 30 required
    assert all(np.isnan(v) for v in feats.values())


def test_nonlinear_features_finite_with_enough_rr():
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    rng = np.random.RandomState(0)
    rr = 800 + 30 * rng.randn(120)  # > 50 required, physiological variation
    feats = HRVNonlinearExtractor().extract_all(rr)
    for key in ("SD1", "SD2", "SampEn"):
        assert np.isfinite(feats[key]), f"{key} should be finite on a well-formed RR series"


def test_hrv_matches_known_sequence():
    rr = np.tile([800.0, 820.0], 30)  # alternating RR, 60 beats
    f = HRVTimeDomainExtractor().extract_all(rr)
    assert abs(f["MeanNN"] - 810.0) < 1e-6
    assert abs(f["RMSSD"] - 20.0) < 1e-6  # every successive diff is 20 ms
    assert abs(f["SDNN"] - np.std(rr, ddof=1)) < 1e-6


def test_too_few_rr_returns_nan():
    # below the minimum RR count HRV is undefined
    features = HRVTimeDomainExtractor().extract_all(np.array([800.0, 810.0]))
    assert np.isnan(features["SDNN"])


def test_rpeaks_recover_known_rate():
    fs = 700
    t = np.arange(0, 30, 1 / fs)
    # 1 Hz synthetic beats -> ~60 BPM
    ecg = np.sin(2 * np.pi * 1.0 * t) ** 21
    peaks = ECGProcessor(sampling_rate=fs).detect_r_peaks(ecg)
    rate = len(peaks) / 30
    assert 0.8 < rate < 1.2
