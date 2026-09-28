"""Signal exports retain condition boundaries; provenance needs deliberate replacement."""

import json

import numpy as np
import pytest

from scripts import build_dashboard_data, stamp_provenance
from scripts.export_signals import _slice


@pytest.mark.parametrize("labels", [[1, 1, 0, 0, 1, 1], [0, 0, 0], [1, 1, 1]])
def test_signal_slice_rejects_short_or_disjoint_condition_runs(labels):
    labels = np.asarray(labels)
    assert _slice(np.arange(len(labels)), labels, 1, 4) is None


@pytest.mark.parametrize("length,want", [(8, 4), (9, 4), (8, 3), (9, 3), (4, 4)])
def test_signal_slice_preserves_center_of_contiguous_condition(length, want):
    labels = np.asarray([0, 0] + [1] * length + [2, 2])
    signal = np.arange(len(labels))
    start = 2 + length // 2 - want // 2
    np.testing.assert_array_equal(_slice(signal, labels, 1, want), signal[start : start + want])


def test_signal_slice_selects_one_sufficient_condition_run():
    labels = np.asarray([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2])
    result = _slice(np.arange(len(labels)), labels, 1, 4)
    np.testing.assert_array_equal(result, [5, 6, 7, 8])
    assert np.all(labels[result] == 1)


def test_provenance_preserves_existing_lineage_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(stamp_provenance, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "results" / "provenance.json"
    path.parent.mkdir()
    original = '{"git_sha": "old-source", "analysis_reruns": {"retained": true}}\n'
    path.write_text(original)
    with pytest.raises(SystemExit, match="full.*reproduction"):
        stamp_provenance.run()
    assert path.read_text() == original


@pytest.mark.parametrize("replace", [False, True])
def test_provenance_can_create_or_explicitly_replace_stamp(tmp_path, monkeypatch, replace):
    monkeypatch.setattr(stamp_provenance, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        stamp_provenance, "provenance", lambda: {"git_sha": "new-source", "generated_at": "now"}
    )
    monkeypatch.setattr(stamp_provenance, "_package_versions", lambda: {"numpy": "fixture"})
    path = tmp_path / "results" / "provenance.json"
    if replace:
        path.parent.mkdir()
        path.write_text('{"git_sha": "old-source"}')
    stamp_provenance.run(overwrite=replace)
    result = json.loads(path.read_text())
    assert result["git_sha"] == "new-source"
    assert result["packages"] == {"numpy": "fixture"}
    assert result["data"]["n_subjects"] == len(stamp_provenance.WESAD_SHA256)


def test_dashboard_export_matches_committed_snapshot(tmp_path, monkeypatch):
    snapshot = build_dashboard_data.FRONTEND.read_bytes()
    target = tmp_path / "results.json"
    monkeypatch.setattr(build_dashboard_data, "FRONTEND", target)
    build_dashboard_data.run()
    assert target.read_bytes() == snapshot
    json.dumps(json.loads(target.read_text()), allow_nan=False)


def test_dashboard_export_keeps_optional_sections_optional(tmp_path, monkeypatch):
    expected = json.loads(build_dashboard_data.FRONTEND.read_text())["binary"]
    metrics = json.loads((build_dashboard_data.RESULTS_DIR / "metrics.json").read_text())
    (tmp_path / "metrics.json").write_text(json.dumps({"binary": metrics["binary"]}))
    target = tmp_path / "dashboard.json"
    monkeypatch.setattr(build_dashboard_data, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build_dashboard_data, "FRONTEND", target)
    build_dashboard_data.run()
    assert json.loads(target.read_text()) == {"binary": expected}
