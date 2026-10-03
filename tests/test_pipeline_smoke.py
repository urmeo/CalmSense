"""The full pipeline runs end-to-end on synthetic data, no WESAD download needed."""

import weakref

import numpy as np
import pytest

from scripts import calibration
from scripts.run_experiment import build_pipeline, loso_evaluate, prepare_task
from src.dataset import WindowedDataset
from src.dataset_wrist import WristDataset
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION, FeatureExtractionPipeline
from src.synthetic import features


@pytest.fixture(scope="module")
def synth():
    df, x_raw, _ = features(n_subjects=4, block_sec=100, seed=3)
    return df, x_raw


def test_pipeline_builds_features(synth):
    df, _ = synth
    assert len(df) > 0
    assert df["subject_id"].nunique() == 4
    feat = [c for c in df.columns if c not in ("subject_id", "window_id", "label", "label_name")]
    assert len(feat) >= 50


def test_loso_runs_end_to_end(synth):
    df, x_raw = synth
    X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
    res = loso_evaluate(lambda: build_pipeline("lr"), X, y, groups)
    assert 0.0 <= res["accuracy_mean"] <= 1.0
    assert len(res["per_subject"]) == 4


def test_calibration_outputs_are_valid(synth):
    df, x_raw = synth
    X, y, groups, _, _ = prepare_task(df, x_raw, [1, 2])
    out = calibration.compute(X, y, groups, model="lr", n_bins=10)
    for key in ("loso", "within_subject", "recalibrated_isotonic"):
        assert 0.0 <= out[key]["ece"] <= 1.0
        assert 0.0 <= out[key]["brier"] <= 2.0
    dc = out["decision_curve"]
    assert len(dc["thresholds"]) == len(dc["net_benefit_uncalibrated"])


@pytest.mark.parametrize("fs", [350, 1000])
def test_chest_dataset_rejects_rates_that_reinterpret_native_wesad(fs, tmp_path):
    with pytest.raises(ValueError, match="WESAD chest signals and labels are sampled at 700"):
        WindowedDataset(fs=fs, data_path=tmp_path / "missing")


@pytest.mark.parametrize("device", ["chest", "wrist"])
def test_subject_recordings_are_released_before_loading_the_next(device):
    builder_type = WindowedDataset if device == "chest" else WristDataset
    builder = builder_type.__new__(builder_type)
    builder.features = FeatureExtractionPipeline(
        {group: group == "temperature" for group in FeatureExtractionPipeline.DEFAULT_CONFIG}
    )
    builder.purity = 0.9
    builder.fs, builder.window_samples, builder.step, builder.cnn_length = 700, 42000, 21000, 16
    builder.window_sec, builder.overlap = 60.0, 0.5
    recordings = []
    seen = []

    def process(subject):
        assert all(reference() is None for reference in recordings)
        seen.append(subject)
        if subject == "S4":
            return ([], [], []) if device == "chest" else ([], [])
        label = 1 if subject == "S2" else 2
        recording = np.full(1000, 35.0 + label / 10)
        recordings.append(weakref.ref(recording))
        windows = [
            {
                "temperature": recording[:240],
                "subject_id": subject,
                "window_id": label * 21000,
                "label": label,
            }
        ]
        return (
            (windows, [np.full((5, 16), label)], [label])
            if device == "chest"
            else (windows, [label])
        )

    builder._process_subject = process
    built = builder.build(subjects=["S2", "S4", "S3"], cache=False)
    frame = built[0] if device == "chest" else built
    assert seen == ["S2", "S4", "S3"]
    assert all(reference() is None for reference in recordings)
    assert frame["subject_id"].tolist() == ["S2", "S3"]
    assert frame["window_id"].tolist() == [21000, 42000]
    assert frame["label_name"].tolist() == ["baseline", "stress"]
    np.testing.assert_allclose(frame["TEMP_mean"], [35.1, 35.2])
    assert frame.attrs["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert frame.attrs["dataset_parameters"]["subjects"] == ["S2", "S3", "S4"]
    if device == "chest":
        _, raw, labels = built
        np.testing.assert_array_equal(labels, frame["label"])
        np.testing.assert_array_equal(raw[:, 0, 0], labels)


@pytest.mark.parametrize("device", ["chest", "wrist"])
def test_dataset_without_retained_windows_keeps_its_empty_schema(device):
    builder_type = WindowedDataset if device == "chest" else WristDataset
    builder = builder_type.__new__(builder_type)
    builder.features = FeatureExtractionPipeline()
    builder.purity = 0.9
    builder.fs, builder.window_samples, builder.step, builder.cnn_length = 700, 42000, 21000, 16
    builder.window_sec, builder.overlap = 60.0, 0.5
    builder._process_subject = lambda _: ([], [], []) if device == "chest" else ([], [])
    built = builder.build(subjects=["S2"], cache=False)
    frame = built[0] if device == "chest" else built
    assert frame.empty
    assert list(frame) == [
        "subject_id",
        "window_id",
        "label",
        *builder.features.get_feature_names(),
        "label_name",
    ]
    if device == "chest":
        _, raw, labels = built
        assert raw.shape == (0, 5, 16)
        assert labels.size == 0
