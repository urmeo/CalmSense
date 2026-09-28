"""The full pipeline runs end-to-end on synthetic data, no WESAD download needed."""

import pytest

from scripts import calibration
from scripts.run_experiment import build_pipeline, loso_evaluate, prepare_task
from src.synthetic import features


@pytest.mark.parametrize("wrist", [False, True])
def test_dataset_releases_subject_signals_before_loading_next(wrist):
    import weakref
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.dataset import WindowedDataset
    from src.dataset_wrist import WristDataset

    cls = WristDataset if wrist else WindowedDataset
    dataset = cls.__new__(cls)
    dataset.loader = SimpleNamespace(subjects=["S2", "S3"])
    dataset.cnn_length = 8
    signals = []

    def process(subject):
        assert all(ref() is None for ref in signals)
        signal = np.arange(6, dtype=float)
        signals.append(weakref.ref(signal))
        windows = [
            {"signal": signal[:3], "subject_id": subject, "label": 1},
            {"signal": signal[3:], "subject_id": subject, "label": 2},
        ]
        return (windows, [1, 2]) if wrist else (windows, [np.zeros((5, 8))] * 2, [1, 2])

    def extract(windows, show_progress):
        return pd.DataFrame(
            [
                {"subject_id": w["subject_id"], "label": w["label"], "mean": w["signal"].mean()}
                for w in windows
            ]
        )

    dataset._process_subject = process
    dataset.features = SimpleNamespace(extract_all_features=extract)
    output = dataset.build(cache=False)
    frame = output if wrist else output[0]
    assert frame["subject_id"].tolist() == ["S2", "S2", "S3", "S3"]
    assert frame["mean"].tolist() == [1, 4, 1, 4]
    assert frame["label_name"].tolist() == ["baseline", "stress"] * 2
    assert all(ref() is None for ref in signals)
    if not wrist:
        assert output[1].shape == (4, 5, 8)
        np.testing.assert_array_equal(output[2], [1, 2, 1, 2])


def test_model_checksum_tracks_replaced_snapshot(tmp_path):
    import hashlib

    import joblib

    from scripts.run_experiment import _save_model

    path = tmp_path / "stress_classifier.joblib"
    for version in (1, 2):
        _save_model({"version": version}, path)
        assert joblib.load(path) == {"version": version}
        digest, filename = path.with_suffix(".joblib.sha256").read_text().split()
        assert filename == path.name
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


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


@pytest.mark.parametrize("demo_defaults", [False, True])
def test_synthetic_tuning_preserves_recorded_results(tmp_path, monkeypatch, demo_defaults):
    import json

    from scripts import tuning
    from src import synthetic

    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    figures.mkdir()
    result = results / "tuning.json"
    figure = figures / "tuning.png"
    result.write_text("recorded tuning")
    figure.write_bytes(b"recorded figure")
    metrics = {"binary": {"models": [{"model": "Random Forest", "accuracy_mean": 0.91}]}}
    (results / "metrics.json").write_text(json.dumps(metrics))
    if demo_defaults:
        (results / "demo").mkdir()
        metrics["binary"]["models"][0]["accuracy_mean"] = 0.61
        (results / "demo" / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(tuning, "RESULTS_DIR", results)
    monkeypatch.setattr(tuning, "FIGURES_DIR", figures)
    monkeypatch.setattr(synthetic, "features", lambda **kwargs: (None, None, None))
    monkeypatch.setattr(tuning, "prepare_task", lambda *args: (None,) * 5)
    monkeypatch.setattr(tuning, "compute", lambda *args: {"Random Forest": {"accuracy_mean": 0.75}})
    plotted_defaults = []
    plot = tuning._plot

    def record_plot(tuned, defaults, path):
        plotted_defaults.append(defaults)
        plot(tuned, defaults, path)

    monkeypatch.setattr(tuning, "_plot", record_plot)

    tuning.run(synthetic=True)

    assert result.read_text() == "recorded tuning"
    assert figure.read_bytes() == b"recorded figure"
    output = json.loads((results / "demo" / "tuning.json").read_text())
    assert output["Random Forest"]["accuracy_mean"] == 0.75
    assert "provenance" in output
    assert plotted_defaults == ([{"Random Forest": 0.61}] if demo_defaults else [])
    assert (figures / "demo" / "tuning.png").is_file() == demo_defaults
