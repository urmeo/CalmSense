"""The dashboard exports matched comparisons without modifying research sources."""

import json
from pathlib import Path

import pytest

from scripts import build_dashboard_data
from src.calibration import BINARY_BRIER_DEFINITION


def test_export_preserves_matched_metrics_and_normalizes_historical_brier(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parent.parent
    result_files = list((root / "results").glob("*.json"))
    originals = {path: path.read_bytes() for path in result_files}
    output = tmp_path / "results.json"
    monkeypatch.setattr(build_dashboard_data, "FRONTEND", output)
    build_dashboard_data.run()
    exported = json.loads(output.read_text())
    metrics = json.loads(originals[root / "results/metrics.json"])
    calibration = json.loads(originals[root / "results/calibration.json"])
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
