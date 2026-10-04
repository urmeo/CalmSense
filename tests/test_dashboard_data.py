import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_dashboard_data, export_signals, stamp_provenance, update_readme_tables
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


def test_signal_clips_do_not_span_separate_condition_blocks():
    labels = np.array([1, 1, 2, 1, 1])
    assert export_signals._slice(np.arange(5), labels, label=1, want=3) is None
    labels = np.array([0, 1, 1, 1, 1, 1, 0])
    clip = export_signals._slice(np.arange(7), labels, label=1, want=3)
    assert clip.tolist() == [2, 3, 4]
    assert np.all(labels[clip] == 1)


def test_signal_clips_reject_misaligned_samples():
    with pytest.raises(ValueError, match="lengths differ"):
        export_signals._slice(np.arange(4), np.ones(5), label=1, want=3)


def test_export_rejects_partial_benchmark_without_replacing_snapshot(tmp_path, monkeypatch):
    output = tmp_path / "results.ts"
    output.write_text("existing snapshot", encoding="utf-8")
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(
        build_dashboard_data, "_load_json", lambda name: {} if name == "metrics.json" else None
    )
    monkeypatch.setattr(build_dashboard_data, "_load_csv", lambda name: None)
    with pytest.raises(ValueError, match="non-empty model comparison"):
        build_dashboard_data.run()
    assert output.read_text(encoding="utf-8") == "existing snapshot"


def test_export_copies_protocol_metadata_from_the_benchmark_run(tmp_path, monkeypatch):
    load_json = build_dashboard_data._load_json

    def current_run(name):
        value = load_json(name)
        if name == "metrics.json":
            value["benchmark_protocol_version"] = 2
        return value

    output = tmp_path / "results.ts"
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", current_run)
    build_dashboard_data.run()
    assert _read_module(output)["benchmark_protocol_version"] == 2


def test_export_copies_recorded_subject_counts(tmp_path, monkeypatch):
    load_json = build_dashboard_data._load_json

    def subset_run(name):
        value = load_json(name)
        if name == "metrics.json":
            value["binary"]["n_subjects"] = 3
            value["multiclass"]["n_subjects"] = 4
        return value

    output = tmp_path / "results.ts"
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", subset_run)
    build_dashboard_data.run()
    exported = _read_module(output)
    assert exported["binary"]["n_subjects"] == 3
    assert exported["multiclass"]["n_subjects"] == 4


@pytest.mark.parametrize("count", [None, 0, 1, 2.5, "15"])
def test_export_rejects_invalid_subject_counts(tmp_path, monkeypatch, count):
    load_json = build_dashboard_data._load_json

    def invalid_run(name):
        value = load_json(name)
        if name == "metrics.json":
            value["binary"]["n_subjects"] = count
        return value

    output = tmp_path / "results.ts"
    output.write_text("existing snapshot", encoding="utf-8")
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", invalid_run)
    with pytest.raises(ValueError, match="n_subjects must be an integer >= 2"):
        build_dashboard_data.run()
    assert output.read_text(encoding="utf-8") == "existing snapshot"


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("n_windows",), True, "positive integer"),
        (("n_features",), False, "positive integer"),
        (("classes",), ["stress", "baseline"], "classes must be"),
        (("models", 0, "model"), "", "nonempty strings"),
        (("models", 1, "model"), "Logistic Regression", "must be unique"),
        (("models", 0, "accuracy_mean"), "0.9", "numeric value"),
        (("models", 0, "f1_macro_mean"), 1.2, "numeric value"),
        (("models", 0, "balanced_accuracy"), -0.1, "numeric value"),
        (("models", 0, "accuracy_std"), None, "numeric value"),
        (("loso_accuracy",), 0.1, "disagrees with"),
        (("optimism_gap_pts",), 4.0, "must match"),
        (("loso_matched_accuracy",), None, "must match"),
    ],
)
def test_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot(
    tmp_path, monkeypatch, field_path, value, message
):
    original_load = build_dashboard_data._load_json

    def malformed_run(name):
        data = original_load(name)
        if name == "metrics.json":
            target = data["binary"]
            for field in field_path[:-1]:
                target = target[field]
            target[field_path[-1]] = value
        return data

    output = tmp_path / "results.ts"
    output.write_text("existing snapshot")
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", malformed_run)
    with pytest.raises(ValueError, match=message):
        build_dashboard_data.run()
    assert output.read_text() == "existing snapshot"


def test_export_accepts_absent_matched_comparison_without_inventing_scores(tmp_path, monkeypatch):
    original_load = build_dashboard_data._load_json

    def unmatched_run(name):
        data = original_load(name)
        if name == "metrics.json":
            for key in ("loso_matched_accuracy", "within_subject_accuracy", "optimism_gap_pts"):
                data["binary"][key] = None
        return data

    output = tmp_path / "results.ts"
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", unmatched_run)
    build_dashboard_data.run()
    data = _read_module(output)["binary"]
    assert data["loso_matched_accuracy"] is None
    assert data["within_subject_accuracy"] is None
    assert data["optimism_gap_pts"] is None


def test_environment_snapshot_fingerprints_results_and_excludes_itself(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "metrics.json").write_bytes(b'{"binary": {}}')
    (tmp_path / "provenance.json").write_text("old snapshot", encoding="utf-8")
    monkeypatch.setattr(stamp_provenance, "RESULTS_DIR", tmp_path)
    stamp_provenance.run()
    snapshot = json.loads((tmp_path / "provenance.json").read_text())
    assert snapshot["provenance_kind"] == "environment_snapshot"
    assert snapshot["data"]["n_reference_subjects"] == 15
    assert "n_subjects" not in snapshot["data"]
    assert snapshot["result_sha256"] == {
        "metrics.json": hashlib.sha256(b'{"binary": {}}').hexdigest()
    }
    assert (tmp_path / "metrics.json").read_bytes() == b'{"binary": {}}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_readme_metrics_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="README metrics must be finite"):
        update_readme_tables._f(value)


def test_personalization_table_rejects_an_unknown_brier_scale(tmp_path, monkeypatch):
    (tmp_path / "personalization.json").write_text(
        json.dumps({"brier_definition": "unknown_scale"}), encoding="utf-8"
    )
    monkeypatch.setattr(update_readme_tables, "RESULTS", tmp_path)
    with pytest.raises(ValueError, match="positive-class MSE"):
        update_readme_tables._personalization_table()


def test_new_primary_protocol_does_not_certify_saved_ancillary_runs(tmp_path, monkeypatch):
    loader = build_dashboard_data._load_json

    def current_primary(name):
        value = loader(name)
        if name == "metrics.json":
            value["benchmark_protocol_version"] = 2
        return value

    output = tmp_path / "results.ts"
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", current_primary)
    build_dashboard_data.run()
    assert _read_module(output)["unverified_sections"] == [
        "ablation",
        "calibration",
        "cross_dataset",
        "personalization",
        "shap",
        "stats",
        "wrist",
    ]


@pytest.mark.parametrize("fault", [None, "checksum", "model", "scope", "path"])
def test_shap_link_requires_the_recorded_benchmark_artifact(tmp_path, monkeypatch, fault):
    from src.utils import sha256_file

    loader = build_dashboard_data._load_json

    def current_primary(name):
        value = loader(name)
        if name == "metrics.json":
            artifact = {
                "model": "XGBoost",
                "scope": "full_data_binary_fit",
                "path": "shap_top_features.csv",
                "sha256": sha256_file(RESULTS_DIR / "shap_top_features.csv"),
            }
            if fault:
                artifact["sha256" if fault == "checksum" else fault] = "incorrect"
            value["benchmark_protocol_version"] = 2
            value["artifacts"] = {"shap": artifact}
        return value

    output = tmp_path / "results.ts"
    output.write_text("previous snapshot")
    monkeypatch.setattr(build_dashboard_data, "DASHBOARD_RESULTS", output)
    monkeypatch.setattr(build_dashboard_data, "_load_json", current_primary)
    if fault:
        with pytest.raises(ValueError, match="SHAP snapshot does not match"):
            build_dashboard_data.run()
        assert output.read_text() == "previous snapshot"
    else:
        build_dashboard_data.run()
        assert "shap" not in _read_module(output)["unverified_sections"]
