"""Provenance stamping, artifact verification, and effect-size helpers."""

import hashlib

import joblib
import pytest

from src.utils import load_verified_joblib, paired_effect_size, provenance, save_verified_joblib


def test_provenance_has_sha_and_timestamp():
    p = provenance()
    assert set(p) == {"git_sha", "generated_at"}
    assert isinstance(p["git_sha"], str) and len(p["git_sha"]) >= 7
    assert "T" in p["generated_at"]  # ISO-8601


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
