import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
import json
from types import SimpleNamespace
import numpy as np
from scripts import (
    build_dashboard_data,
    export_signals,
    stamp_provenance,
    update_readme_tables,
)
from src.calibration import BINARY_BRIER_DEFINITION
from src.config import RESULTS_DIR


def _read_module(path):
    text = path.read_text(encoding="utf-8")
    prefix, suffix = ("const data = ", ";\n\nexport default data;\n")
    assert text.startswith(prefix) and text.endswith(suffix)
    return json.loads(text[len(prefix) : -len(suffix)])


class DashboardDataTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_export_preserves_matched_metrics_and_normalizes_historical_brier(self):
        tmp_path = self.tmp_path
        result_files = list(RESULTS_DIR.glob("*.json"))
        originals = {path: path.read_bytes() for path in result_files}
        output = tmp_path / "dashboard" / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        build_dashboard_data.run()
        exported = _read_module(output)
        metrics = json.loads(originals[RESULTS_DIR / "metrics.json"])
        calibration = json.loads(originals[RESULTS_DIR / "calibration.json"])
        for task in ("binary", "multiclass"):
            data = exported[task]
            self.assertEqual(
                data["loso_matched_accuracy"], metrics[task]["loso_matched_accuracy"]
            )
            self.assertEqual(
                round(
                    (data["within_subject_accuracy"] - data["loso_matched_accuracy"])
                    * 100,
                    1,
                ),
                data["optimism_gap_pts"],
            )
        self.assertEqual(
            exported["calibration"]["loso_matched"],
            {
                **calibration["loso_matched"],
                "brier": calibration["loso_matched"]["brier"] / 2,
            },
        )
        np.testing.assert_allclose(
            exported["calibration"]["loso"]["brier"],
            calibration["loso"]["brier"] / 2,
            rtol=1e-06,
            atol=1e-12,
        )
        self.assertEqual(
            exported["calibration"]["brier_definition"], BINARY_BRIER_DEFINITION
        )
        self.assertEqual(
            exported["personalization"]["brier_definition"], BINARY_BRIER_DEFINITION
        )
        self.assertTrue(
            all((path.read_bytes() == before for path, before in originals.items()))
        )

    def _case_export_rejects_nonfinite_values_without_replacing_snapshot(
        self, invalid_number
    ):
        tmp_path = self.tmp_path
        output = tmp_path / "results.ts"
        before = "const data = {};\n\nexport default data;\n"
        output.write_text(before, encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(
                build_dashboard_data,
                "_load_json",
                lambda name: (
                    {"binary": {"n_windows": invalid_number}}
                    if name == "metrics.json"
                    else None
                ),
            )
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_csv", lambda name: None)
        )
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), before)

    def test_signal_export_creates_output_folder_and_preserves_aligned_samples(self):
        tmp_path = self.tmp_path
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
        self.stack.enter_context(
            patch.object(export_signals, "DASHBOARD_SIGNALS", output)
        )
        self.stack.enter_context(
            patch.object(export_signals, "WESADLoader", lambda: loader)
        )
        self.stack.enter_context(
            patch.object(export_signals, "FS", SimpleNamespace(CHEST=2))
        )
        self.stack.enter_context(patch.object(export_signals, "SUBJECTS", ["S2"]))
        self.stack.enter_context(patch.object(export_signals, "SECONDS", 1))
        self.stack.enter_context(patch.object(export_signals, "OUT_FS", 2))
        export_signals.run()
        data = _read_module(output)["S2"]
        self.assertEqual(data["time"], [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertEqual(
            data["conditions"],
            ["Baseline", "Baseline", "Stress", "Stress", "Amusement", "Amusement"],
        )
        for name, offset in {
            "ecg": 0,
            "eda": 10,
            "temp": 20,
            "accX": 30,
            "accY": 40,
            "accZ": 50,
        }.items():
            self.assertEqual(data[name], (samples + offset).tolist())

    def test_signal_clips_do_not_span_separate_condition_blocks(self):
        labels = np.array([1, 1, 2, 1, 1])
        self.assertIs(
            export_signals._slice(np.arange(5), labels, label=1, want=3), None
        )
        labels = np.array([0, 1, 1, 1, 1, 1, 0])
        clip = export_signals._slice(np.arange(7), labels, label=1, want=3)
        self.assertEqual(clip.tolist(), [2, 3, 4])
        self.assertTrue(np.all(labels[clip] == 1))

    def test_signal_clips_reject_misaligned_samples(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            export_signals._slice(np.arange(4), np.ones(5), label=1, want=3)

    def test_export_rejects_partial_benchmark_without_replacing_snapshot(self):
        tmp_path = self.tmp_path
        output = tmp_path / "results.ts"
        output.write_text("existing snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(
                build_dashboard_data,
                "_load_json",
                lambda name: {} if name == "metrics.json" else None,
            )
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_csv", lambda name: None)
        )
        with self.assertRaisesRegex(ValueError, "non-empty model comparison"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing snapshot")

    def test_export_copies_protocol_metadata_from_the_benchmark_run(self):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def current_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["benchmark_protocol_version"] = 2
            return value

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", current_run)
        )
        build_dashboard_data.run()
        self.assertEqual(_read_module(output)["benchmark_protocol_version"], 2)

    def test_export_copies_recorded_subject_counts(self):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def subset_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["binary"]["n_subjects"] = 3
                value["multiclass"]["n_subjects"] = 4
            return value

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", subset_run)
        )
        build_dashboard_data.run()
        exported = _read_module(output)
        self.assertEqual(exported["binary"]["n_subjects"], 3)
        self.assertEqual(exported["multiclass"]["n_subjects"], 4)

    def _case_export_rejects_invalid_subject_counts(self, count):
        tmp_path = self.tmp_path
        load_json = build_dashboard_data._load_json

        def invalid_run(name):
            value = load_json(name)
            if name == "metrics.json":
                value["binary"]["n_subjects"] = count
            return value

        output = tmp_path / "results.ts"
        output.write_text("existing snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", invalid_run)
        )
        with self.assertRaisesRegex(ValueError, "n_subjects must be an integer >= 2"):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(encoding="utf-8"), "existing snapshot")

    def _case_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot(
        self, field_path, value, message
    ):
        tmp_path = self.tmp_path
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
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", malformed_run)
        )
        with self.assertRaisesRegex(ValueError, message):
            build_dashboard_data.run()
        self.assertEqual(output.read_text(), "existing snapshot")

    def test_export_accepts_absent_matched_comparison_without_inventing_scores(self):
        tmp_path = self.tmp_path
        original_load = build_dashboard_data._load_json

        def unmatched_run(name):
            data = original_load(name)
            if name == "metrics.json":
                for key in (
                    "loso_matched_accuracy",
                    "within_subject_accuracy",
                    "optimism_gap_pts",
                ):
                    data["binary"][key] = None
            return data

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", unmatched_run)
        )
        build_dashboard_data.run()
        data = _read_module(output)["binary"]
        self.assertIs(data["loso_matched_accuracy"], None)
        self.assertIs(data["within_subject_accuracy"], None)
        self.assertIs(data["optimism_gap_pts"], None)

    def test_environment_snapshot_fingerprints_results_and_excludes_itself(self):
        tmp_path = self.tmp_path
        import hashlib

        (tmp_path / "metrics.json").write_bytes(b'{"binary": {}}')
        (tmp_path / "provenance.json").write_text("old snapshot", encoding="utf-8")
        self.stack.enter_context(
            patch.object(stamp_provenance, "RESULTS_DIR", tmp_path)
        )
        stamp_provenance.run()
        snapshot = json.loads((tmp_path / "provenance.json").read_text())
        self.assertEqual(snapshot["provenance_kind"], "environment_snapshot")
        self.assertEqual(snapshot["data"]["n_reference_subjects"], 15)
        self.assertNotIn("n_subjects", snapshot["data"])
        self.assertEqual(
            snapshot["result_sha256"],
            {"metrics.json": hashlib.sha256(b'{"binary": {}}').hexdigest()},
        )
        self.assertEqual((tmp_path / "metrics.json").read_bytes(), b'{"binary": {}}')

    def _case_readme_metrics_reject_nonfinite_values(self, value):
        with self.assertRaisesRegex(ValueError, "README metrics must be finite"):
            update_readme_tables._f(value)

    def test_personalization_table_rejects_an_unknown_brier_scale(self):
        tmp_path = self.tmp_path
        (tmp_path / "personalization.json").write_text(
            json.dumps({"brier_definition": "unknown_scale"}), encoding="utf-8"
        )
        self.stack.enter_context(
            patch.object(update_readme_tables, "RESULTS", tmp_path)
        )
        with self.assertRaisesRegex(ValueError, "positive-class MSE"):
            update_readme_tables._personalization_table()

    def test_new_primary_protocol_does_not_certify_saved_ancillary_runs(self):
        tmp_path = self.tmp_path
        loader = build_dashboard_data._load_json

        def current_primary(name):
            value = loader(name)
            if name == "metrics.json":
                value["benchmark_protocol_version"] = 2
            return value

        output = tmp_path / "results.ts"
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", current_primary)
        )
        build_dashboard_data.run()
        self.assertEqual(
            _read_module(output)["unverified_sections"],
            [
                "ablation",
                "calibration",
                "cross_dataset",
                "personalization",
                "shap",
                "stats",
                "wrist",
            ],
        )

    def _case_shap_link_requires_the_recorded_benchmark_artifact(self, fault):
        tmp_path = self.tmp_path
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
        self.stack.enter_context(
            patch.object(build_dashboard_data, "DASHBOARD_RESULTS", output)
        )
        self.stack.enter_context(
            patch.object(build_dashboard_data, "_load_json", current_primary)
        )
        if fault:
            with self.assertRaisesRegex(ValueError, "SHAP snapshot does not match"):
                build_dashboard_data.run()
            self.assertEqual(output.read_text(), "previous snapshot")
        else:
            build_dashboard_data.run()
            self.assertNotIn("shap", _read_module(output)["unverified_sections"])


for index, invalid_number in enumerate([float("nan"), float("inf"), -float("inf")]):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_nonfinite_values_without_replacing_snapshot_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_nonfinite_values_without_replacing_snapshot,
            invalid_number=invalid_number,
        ),
    )
for index, count in enumerate([None, 0, 1, 2.5, "15"]):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_invalid_subject_counts_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_invalid_subject_counts, count=count
        ),
    )
for index, (field_path, value, message) in enumerate(
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
    ]
):
    setattr(
        DashboardDataTests,
        f"test_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot_{index}",
        partialmethod(
            DashboardDataTests._case_export_rejects_malformed_or_inconsistent_scores_before_replacing_snapshot,
            field_path=field_path,
            value=value,
            message=message,
        ),
    )
for index, value in enumerate([float("nan"), float("inf"), -float("inf")]):
    setattr(
        DashboardDataTests,
        f"test_readme_metrics_reject_nonfinite_values_{index}",
        partialmethod(
            DashboardDataTests._case_readme_metrics_reject_nonfinite_values, value=value
        ),
    )
for index, fault in enumerate([None, "checksum", "model", "scope", "path"]):
    setattr(
        DashboardDataTests,
        f"test_shap_link_requires_the_recorded_benchmark_artifact_{index}",
        partialmethod(
            DashboardDataTests._case_shap_link_requires_the_recorded_benchmark_artifact,
            fault=fault,
        ),
    )
