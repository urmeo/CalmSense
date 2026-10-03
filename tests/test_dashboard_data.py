"""The dashboard exports matched comparisons without modifying research sources."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_dashboard_data, export_signals
from src.calibration import BINARY_BRIER_DEFINITION
from src.config import RESULTS_DIR


def _read_module(path):
    text = path.read_text(encoding="utf-8")
    prefix, suffix = "const data = ", ";\n\nexport default data;\n"
    assert text.startswith(prefix) and text.endswith(suffix)
    return json.loads(text[len(prefix) : -len(suffix)])


def test_export_preserves_matched_metrics_and_normalizes_historical_brier(tmp_path, monkeypatch):
    result_files = list(RESULTS_DIR.glob("*.json"))
    originals = {path: path.read_bytes() for path in result_files}
    output = tmp_path / "dashboard" / "results.ts"
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    build_dashboard_data.run()
    exported = _read_module(output)
    metrics = json.loads(originals[RESULTS_DIR / "metrics.json"])
    calibration = json.loads(originals[RESULTS_DIR / "calibration.json"])
    for task in ("binary", "multiclass"):
        data = exported[task]
        assert data["loso_matched_accuracy"] == metrics[task]["loso_matched_accuracy"]
        assert (
            round((data["within_subject_accuracy"] - data["loso_matched_accuracy"]) * 100, 1)
            == data["optimism_gap_pts"]
        )
    assert exported["calibration"]["loso_matched"] == {
        **calibration["loso_matched"],
        "brier": calibration["loso_matched"]["brier"] / 2,
    }
    assert exported["calibration"]["loso"]["brier"] == pytest.approx(
        calibration["loso"]["brier"] / 2
    )
    assert exported["calibration"]["brier_definition"] == BINARY_BRIER_DEFINITION
    assert exported["personalization"]["brier_definition"] == BINARY_BRIER_DEFINITION
    assert all(path.read_bytes() == before for path, before in originals.items())


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), -float("inf")])
def test_export_rejects_nonfinite_values_without_replacing_snapshot(
    tmp_path, monkeypatch, invalid_number
):
    output = tmp_path / "results.ts"
    before = "const data = {};\n\nexport default data;\n"
    output.write_text(before, encoding="utf-8")
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(
        build_dashboard_data,
        "_load_json",
        lambda name: {"binary": {"n_windows": invalid_number}} if name == "metrics.json" else None,
    )
    monkeypatch.setattr(build_dashboard_data, "_load_csv", lambda name: None)

    with pytest.raises(ValueError, match="Out of range float values"):
        build_dashboard_data.run()

    assert output.read_text(encoding="utf-8") == before


def test_signal_export_creates_output_folder_and_preserves_aligned_samples(tmp_path, monkeypatch):
    samples = np.arange(6, dtype=float)
    recording = {
        "chest": {
            "ECG": samples,
            "EDA": samples + 10,
            "Temp": samples + 20,
            "ACC": np.column_stack((samples + 30, samples + 40, samples + 50)),
        },
        "label": np.repeat([1, 2, 3], 2),
    }
    loader = SimpleNamespace(load_subject=lambda subject: recording)
    output = tmp_path / "dashboard" / "signals.ts"
    monkeypatch.setattr(export_signals, "DASHBOARD_SIGNALS", output)
    monkeypatch.setattr(export_signals, "WESADLoader", lambda: loader)
    monkeypatch.setattr(export_signals, "FS", SimpleNamespace(CHEST=2))
    monkeypatch.setattr(export_signals, "SUBJECTS", ["S2"])
    monkeypatch.setattr(export_signals, "SECONDS", 1)
    monkeypatch.setattr(export_signals, "OUT_FS", 2)

    export_signals.run()

    data = _read_module(output)["S2"]
    assert data["time"] == [0, 0.5, 1, 1.5, 2, 2.5]
    assert data["conditions"] == [
        "Baseline",
        "Baseline",
        "Stress",
        "Stress",
        "Amusement",
        "Amusement",
    ]
    for name, offset in {
        "ecg": 0,
        "eda": 10,
        "temp": 20,
        "accX": 30,
        "accY": 40,
        "accZ": 50,
    }.items():
        assert data[name] == (samples + offset).tolist()
