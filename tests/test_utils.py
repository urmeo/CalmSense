"""Provenance stamping, artifact verification, and effect-size helpers."""

import hashlib
import math

import joblib
import pytest

from src.utils import (
    load_verified_joblib,
    paired_effect_size,
    provenance,
    save_verified_joblib,
    sha256_file,
)


def test_provenance_has_sha_and_timestamp():
    p = provenance()
    assert set(p) == {"git_sha", "working_tree_dirty", "feature_schema_version", "generated_at"}
    assert isinstance(p["git_sha"], str) and len(p["git_sha"]) >= 7
    assert "T" in p["generated_at"]  # ISO-8601
    assert isinstance(p["working_tree_dirty"], bool)
    assert p["feature_schema_version"] == 2


def test_paired_effect_size_matches_hand_calc():
    a = [0.9, 0.8, 0.95, 0.7]
    b = [0.85, 0.78, 0.90, 0.72]
    es = paired_effect_size(a, b)
    assert es["n"] == 4
    # mean diff 0.025 over sd of diffs -> positive, small-sample g < d
    assert es["cohens_d"] > 0
    assert abs(es["hedges_g"]) < abs(es["cohens_d"])


def test_paired_effect_size_zero_when_identical():
    es = paired_effect_size([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert es["cohens_d"] == 0.0 and es["hedges_g"] == 0.0


def test_paired_hedges_correction_uses_difference_degrees_of_freedom():
    es = paired_effect_size([1.0, 2.0, 3.0, 4.0], [0.0] * 4)
    d = 2.5 / math.sqrt(5 / 3)
    assert es["cohens_d"] == pytest.approx(d)
    # With four pairs, df=3 and the exact correction is sqrt(pi/6).
    assert es["hedges_g"] == pytest.approx(d * math.sqrt(math.pi / 6))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([], []),
        ([1, 2], [0, 0]),
        ([1, 2, 3], [0]),
        ([[1, 2, 3]], [[0, 0, 0]]),
        ([1, 2, float("nan")], [0, 0, 0]),
    ],
)
def test_effect_size_rejects_unaligned_or_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        paired_effect_size(a, b)


def test_constant_nonzero_difference_has_undefined_effect_size():
    es = paired_effect_size([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert es == {"cohens_d": None, "hedges_g": None, "n": 3}


def test_file_sha256_streams_without_read_bytes(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "recording.bin"
    contents = b"recording" * 100_000
    path.write_bytes(contents)
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("Hashing buffered the whole file")
    )
    assert sha256_file(path) == hashlib.sha256(contents).hexdigest()


def test_saved_model_and_checksum_stay_in_sync(tmp_path):
    path = tmp_path / "models" / "stress_classifier.joblib"
    first = {"features": ["ECG"], "classes": ["baseline", "stress"]}
    save_verified_joblib(first, path)
    first_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_verified_joblib(path) == first

    updated = {"features": ["ECG", "EDA"], "classes": ["baseline", "stress"]}
    save_verified_joblib(updated, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest != first_digest
    assert path.with_name(path.name + ".sha256").read_text() == f"{digest}  {path.name}\n"
    assert load_verified_joblib(path) == updated


def test_tampered_model_is_rejected_before_unpickling(tmp_path, monkeypatch):
    path = tmp_path / "stress_classifier.joblib"
    save_verified_joblib({"features": ["ECG"]}, path)
    path.write_bytes(b"tampered model")

    def refuse_unpickling(*args, **kwargs):
        pytest.fail("Tampered artifact reached joblib.load")

    monkeypatch.setattr(joblib, "load", refuse_unpickling)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_joblib(path)


def test_model_load_uses_verified_snapshot_if_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    trusted = {"trusted": True}
    save_verified_joblib(trusted, path)
    original_load = joblib.load

    def replace_then_load(snapshot):
        path.write_bytes(b"untrusted replacement")
        return original_load(snapshot)

    monkeypatch.setattr(joblib, "load", replace_then_load)
    assert load_verified_joblib(path) == trusted


def test_failed_model_serialization_keeps_existing_verified_model(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    save_verified_joblib({"version": 1}, path)
    original_bytes = path.read_bytes()
    checksum = path.with_name(path.name + ".sha256")
    original_checksum = checksum.read_bytes()

    def failed_dump(bundle, destination):
        destination.write_bytes(b"partial serialization")
        raise OSError("serialization failed")

    monkeypatch.setattr(joblib, "dump", failed_dump)
    with pytest.raises(OSError, match="serialization failed"):
        save_verified_joblib({"version": 2}, path)
    assert path.read_bytes() == original_bytes
    assert checksum.read_bytes() == original_checksum
    assert load_verified_joblib(path) == {"version": 1}


@pytest.mark.parametrize("contents", ["", "not a hash", "0" * 63])
def test_invalid_model_sidecar_fails_before_loading(tmp_path, contents):
    path = tmp_path / "model.joblib"
    save_verified_joblib({}, path)
    path.with_name(path.name + ".sha256").write_text(contents)
    with pytest.raises(ValueError, match="Invalid SHA-256 sidecar"):
        load_verified_joblib(path)
