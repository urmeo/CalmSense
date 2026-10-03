"""Feature extractors produce correct values on known signals."""

import numpy as np
import pytest

from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.preprocessing.ecg_processor import ECGProcessor


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


def test_triangular_index_uses_fixed_rr_bins_and_handles_constant_intervals():
    extractor = HRVTimeDomainExtractor()
    assert extractor.compute_hrvti(np.full(60, 800.0)) == 1.0
    # The values straddle fixed 7.8125 ms bins despite a total range below one bin width.
    assert extractor.compute_hrvti(np.tile([800.0, 807.0], 30)) == 2.0


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


def test_poincare_geometry_and_cvi_use_the_paired_ellipse_axes():
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    extractor = HRVNonlinearExtractor()
    trending = extractor.compute_poincare(np.array([800.0, 820.0, 840.0]))
    assert trending["SD1"] == 0.0
    assert trending["SD2"] == pytest.approx(20.0)
    features = extractor.compute_poincare(np.array([800.0, 810.0, 790.0, 820.0]))
    assert features["SD1"] == pytest.approx(np.sqrt(950 / 3))
    assert features["SD2"] == pytest.approx(np.sqrt(50 / 3))
    assert features["CVI"] == pytest.approx(np.log10(16 * np.sqrt(950 / 3) * np.sqrt(50 / 3)))


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


def test_tonic_and_temperature_slopes_preserve_missing_sample_times():
    from src.features.eda_features import EDAFeatureExtractor
    from src.features.temperature_features import TemperatureFeatureExtractor

    time = np.arange(40) / 4.0
    tonic, temperature = 3 + 0.25 * time, 35 - 0.01 * time
    tonic[[1, 4, 7, 15]] = np.nan
    temperature[[2, 6, 14, 29]] = np.inf
    assert EDAFeatureExtractor(4).extract_tonic_features(tonic)["SCL_slope"] == pytest.approx(0.25)
    assert TemperatureFeatureExtractor(4).extract_all(temperature)["TEMP_slope"] == pytest.approx(
        -0.01
    )


def test_rr_interpolation_uses_its_declared_sampling_rate():
    from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor

    time, interpolated = HRVFrequencyDomainExtractor(interpolation_rate=4)._interpolate_rr(
        np.full(40, 810.0)
    )
    np.testing.assert_allclose(np.diff(time), 0.25)
    np.testing.assert_allclose(interpolated, 810.0)


def test_band_power_includes_unsampled_endpoints():
    from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor

    extractor = HRVFrequencyDomainExtractor()
    freqs = np.arange(0, 2.01, 0.05)
    density = np.full(len(freqs), 2.0)
    assert extractor.compute_vlf_power(freqs, density) == pytest.approx(2 * (0.04 - 0.0033))
    assert extractor.compute_lf_power(freqs, density) == pytest.approx(2 * (0.15 - 0.04))
    assert extractor.compute_hf_power(freqs, density) == pytest.approx(2 * (0.40 - 0.15))


def test_lomb_density_integrates_to_rr_variance_and_handles_constant_rr():
    from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor

    extractor = HRVFrequencyDomainExtractor()
    rr = 800 + 30 * np.sin(np.arange(100) * 0.6)
    freqs, density = extractor.compute_psd(rr, method="lomb")
    assert np.trapezoid(density, freqs) == pytest.approx(np.var(rr))
    features = extractor.extract_all(np.full(60, 800.0), method="lomb")
    assert features["Total_power"] == 0.0
    assert np.isnan(features["LF_peak_freq"])
    with pytest.raises(ValueError, match="PSD method"):
        extractor.compute_psd(rr, method="typo")


@pytest.mark.parametrize("rr", [np.full(20, 800.0), np.tile([800.0, 820.0], 10)])
def test_sample_entropy_is_zero_when_every_matching_pattern_extends(rr):
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    assert HRVNonlinearExtractor().compute_sample_entropy(rr) == 0.0


def test_sample_entropy_reports_zero_continuation_probability():
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    # [800, 800] appears twice, followed by different nonmatching next intervals.
    rr = np.array([800, 800, 900, 800, 800, 1000])
    assert np.isinf(HRVNonlinearExtractor().compute_sample_entropy(rr, m=2, r=0.01))


def test_recurrence_determinism_uses_the_requested_embedding():
    from src.features.hrv_nonlinear import HRVNonlinearExtractor

    extractor = HRVNonlinearExtractor()
    rr = np.tile([800.0, 820.0], 3)
    # Five two-sample vectors: eight recurrent off-diagonal points, six in lines >=2.
    assert extractor.compute_rqa_determinism(rr, embedding_dim=2) == pytest.approx(6 / 8)
    assert extractor.compute_rqa_determinism(rr, embedding_dim=2, time_delay=2) == pytest.approx(
        1.0
    )
    assert np.isnan(extractor.compute_rqa_determinism(rr, embedding_dim=4, time_delay=2))
    assert np.isnan(extractor.compute_dfa(np.full(100, 800.0))).all()


def test_flat_signals_have_no_detected_respiration_or_motion_frequency():
    from src.features.accelerometer_features import AccelerometerFeatureExtractor
    from src.features.respiration_features import RespirationFeatureExtractor

    respiration = RespirationFeatureExtractor(10).extract_all(np.ones(200))
    assert np.isnan(respiration["RESP_rate"])
    assert respiration["RESP_amplitude"] == 0.0
    assert respiration["RESP_apnea_index"] == 100.0
    movement = AccelerometerFeatureExtractor(32).extract_from_magnitude(np.ones(320))
    assert np.isnan(movement["ACC_peak_freq"])
    assert movement["ACC_zero_crossings"] == 0.0


def test_missing_accelerometer_samples_do_not_compress_event_time():
    from src.features.accelerometer_features import AccelerometerFeatureExtractor

    magnitude = np.tile([0.0, 2.0], 50)
    magnitude[20:30] = np.nan
    features = AccelerometerFeatureExtractor(10).extract_from_magnitude(magnitude)
    assert features["ACC_zero_crossings"] == pytest.approx(88 / 10)
    assert np.isnan(features["ACC_peak_freq"])


def test_ectopic_correction_excludes_nonfinite_and_nonpositive_intervals():
    processor = ECGProcessor()
    _, valid = processor.remove_ectopic_beats(np.array([800.0, np.nan, -10.0, 810.0, 800.0]))
    np.testing.assert_array_equal(valid, [True, False, False, True, True])
    corrected = processor.interpolate_artifacts([800.0, np.nan, -10.0, 810.0, 800.0], valid)
    assert np.isfinite(corrected).all()
    assert np.isnan(processor.interpolate_artifacts([4000, 5000], [False, False])).all()


@pytest.mark.parametrize("peaks", [[700, 350], [350, 350], [0, np.nan], [-1, 350]])
def test_rr_extraction_rejects_invalid_peak_order_or_values(peaks):
    with pytest.raises(ValueError, match="strictly increasing"):
        ECGProcessor().extract_rr_intervals(np.asarray(peaks))


def test_ecg_empty_or_constant_input_has_no_peaks():
    processor = ECGProcessor()
    assert processor.detect_r_peaks(np.array([])).size == 0
    assert processor.detect_r_peaks(np.ones(100)).size == 0
    with pytest.raises(ValueError, match="Nyquist"):
        processor.bandpass_filter(np.ones(100), low=10, high=5)


@pytest.mark.parametrize(
    "name",
    [
        "EDAFeatureExtractor",
        "TemperatureFeatureExtractor",
        "RespirationFeatureExtractor",
        "AccelerometerFeatureExtractor",
    ],
)
@pytest.mark.parametrize("rate", [0, -1, np.inf, np.nan, True])
def test_feature_extractors_reject_invalid_sampling_rates(name, rate):
    from src import features

    with pytest.raises(ValueError, match="finite and positive"):
        getattr(features, name)(sampling_rate=rate)


@pytest.mark.parametrize(
    "parameters",
    [
        {"window_sec": 0},
        {"overlap": 1},
        {"overlap": -0.1},
        {"purity": 0},
        {"purity": np.nan},
        {"window_sec": 0.0001},
    ],
)
def test_window_configuration_is_rejected_before_loading_data(parameters):
    from src.dataset import WindowedDataset
    from src.dataset_wrist import WristDataset

    for builder in (WindowedDataset, WristDataset):
        with pytest.raises(ValueError):
            builder(**parameters)


def test_feature_schema_does_not_compute_dummy_signals(monkeypatch):
    from src.features.feature_pipeline import FeatureExtractionPipeline

    pipeline = FeatureExtractionPipeline()
    monkeypatch.setattr(
        pipeline,
        "extract_window_features",
        lambda _: pytest.fail("Schema discovery must not compute signals"),
    )
    names = pipeline.get_feature_names()
    assert names == list(pipeline.get_feature_descriptions())
    names.clear()
    assert pipeline.get_feature_count() == 60
    assert list(pipeline.extract_all_features([])) == [
        "subject_id",
        "window_id",
        "label",
        *pipeline.get_feature_names(),
    ]


def test_synthetic_subjects_skip_unavailable_wesad_ids(tmp_path, monkeypatch):
    from src import synthetic
    from src.config import VALID_SUBJECTS
    from src.data.loader import WESADLoader

    monkeypatch.setattr(synthetic, "_subject", lambda seed, block_sec: {"seed": seed})
    data_path = synthetic.write_dataset(tmp_path, n_subjects=15, block_sec=1)
    assert WESADLoader(data_path).subjects == VALID_SUBJECTS
    assert not (data_path / "S12").exists()
    with pytest.raises(ValueError, match="real-data caches"):
        synthetic.features(cache=True)


def test_wesad_loader_rejects_misaligned_and_malformed_channels(tmp_path):
    import pickle

    from src.data.loader import WESADLoader

    subject = tmp_path / "S2"
    subject.mkdir()
    record = {
        "signal": {"chest": {"ECG": np.zeros((700, 1))}, "wrist": {"EDA": np.zeros((4, 1))}},
        "label": np.ones(700),
    }
    path = subject / "S2.pkl"
    path.write_bytes(pickle.dumps(record))
    loader = WESADLoader(tmp_path)
    assert len(loader.load_subject("S2")["label"]) == 700
    record["signal"]["chest"]["ECG"] = np.zeros((650, 1))
    path.write_bytes(pickle.dumps(record))
    with pytest.raises(ValueError, match="duration"):
        loader.load_subject("S2")
    record["signal"]["chest"]["ECG"] = np.zeros((700, 3))
    path.write_bytes(pickle.dumps(record))
    with pytest.raises(ValueError, match="signal shape"):
        loader.load_subject("S2")


def test_cached_chest_features_require_current_protocol_and_aligned_rows(tmp_path, monkeypatch):
    import pandas as pd

    from src import dataset
    from src.config import VALID_SUBJECTS
    from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION

    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", tmp_path)
    columns = dataset.FeatureExtractionPipeline().get_feature_names()
    frame = pd.DataFrame(
        {"subject_id": ["S2"], "window_id": [0], "label": [1], **{name: [0.0] for name in columns}}
    )
    frame.to_parquet(tmp_path / "features.parquet", index=False)
    np.savez(tmp_path / "raw_windows.npz", x=np.zeros((1, 5, 1024)))
    assert dataset.load_cached() is None
    frame.attrs.update(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_parameters": {
                "fs": 700.0,
                "window_samples": 42000,
                "step": 21000,
                "purity": 0.9,
                "cnn_length": 1024,
                "subjects": sorted(VALID_SUBJECTS),
            },
        }
    )
    builder = dataset.WindowedDataset.__new__(dataset.WindowedDataset)
    builder._save(frame, np.zeros((1, 5, 1024)))
    assert dataset.load_cached()[1].shape == (1, 5, 1024)
    frame["window_id"] = 700
    frame.to_parquet(tmp_path / "features.parquet", index=False)
    with pytest.raises(ValueError, match="not aligned"):
        dataset.load_cached()
    frame.attrs["dataset_parameters"]["purity"] = 0.7
    frame.to_parquet(tmp_path / "features.parquet", index=False)
    assert dataset.load_cached() is None
