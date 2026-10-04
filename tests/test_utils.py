import hashlib
import math
from pathlib import Path

import joblib
import pytest

from src.utils import (
    atomic_write_text,
    load_verified_joblib,
    paired_effect_size,
    provenance,
    save_verified_joblib,
    sha256_file,
    write_json,
)


def test_provenance_has_sha_and_timestamp():
    p = provenance()
    assert set(p) == {"git_sha", "working_tree_dirty", "feature_schema_version", "generated_at"}
    assert isinstance(p["git_sha"], str) and len(p["git_sha"]) >= 7
    assert "T" in p["generated_at"]
    assert isinstance(p["working_tree_dirty"], bool)
    assert p["feature_schema_version"] == 2


def test_paired_effect_size_matches_hand_calc():
    a = [0.9, 0.8, 0.95, 0.7]
    b = [0.85, 0.78, 0.90, 0.72]
    es = paired_effect_size(a, b)
    assert es["n"] == 4
    assert es["cohens_d"] > 0
    assert abs(es["hedges_g"]) < abs(es["cohens_d"])


def test_paired_effect_size_zero_when_identical():
    es = paired_effect_size([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert es["cohens_d"] == 0.0 and es["hedges_g"] == 0.0


def test_paired_hedges_correction_uses_difference_degrees_of_freedom():
    es = paired_effect_size([1.0, 2.0, 3.0, 4.0], [0.0] * 4)
    d = 2.5 / math.sqrt(5 / 3)
    assert es["cohens_d"] == pytest.approx(d)
    # Exact correction: sqrt(pi/6) at df=3.
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


@pytest.mark.parametrize("scale", [1e-200, 1e-17, 1.0, 1e150])
def test_paired_effect_size_is_unit_invariant_without_variance_underflow(scale):
    expected = paired_effect_size([1.0, 2.0, 3.0], [0.0] * 3)
    scaled = paired_effect_size([scale, 2 * scale, 3 * scale], [0.0] * 3)
    assert scaled["cohens_d"] == pytest.approx(expected["cohens_d"])
    assert scaled["hedges_g"] == pytest.approx(expected["hedges_g"])


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


@pytest.mark.parametrize("existing", [False, True])
def test_failed_model_sidecar_publication_restores_previous_pair(tmp_path, monkeypatch, existing):
    path = tmp_path / "model.joblib"
    sidecar = path.with_name(path.name + ".sha256")
    before = None
    if existing:
        save_verified_joblib({"version": 1}, path)
        before = path.read_bytes(), sidecar.read_bytes()
    replace = Path.replace

    def fail_sidecar(source, destination):
        if source.name == sidecar.name:
            raise OSError("sidecar publication failed")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_sidecar)
    with pytest.raises(OSError, match="sidecar publication failed"):
        save_verified_joblib({"version": 2}, path)
    if existing:
        assert (path.read_bytes(), sidecar.read_bytes()) == before
        assert load_verified_joblib(path) == {"version": 1}
    else:
        assert not path.exists() and not sidecar.exists()
    assert set(tmp_path.iterdir()) == ({path, sidecar} if existing else set())


@pytest.mark.parametrize("contents", ["", "not a hash", "0" * 63])
def test_invalid_model_sidecar_fails_before_loading(tmp_path, contents):
    path = tmp_path / "model.joblib"
    save_verified_joblib({}, path)
    path.with_name(path.name + ".sha256").write_text(contents)
    with pytest.raises(ValueError, match="Invalid SHA-256 sidecar"):
        load_verified_joblib(path)


def test_atomic_text_creates_utf8_output_and_preserves_permissions(tmp_path):
    path = tmp_path / "nested" / "report.txt"
    atomic_write_text(path, "Temperature: 30°C\n")
    assert path.read_text(encoding="utf-8") == "Temperature: 30°C\n"
    path.chmod(0o640)
    atomic_write_text(path, "updated\n")
    assert path.read_text() == "updated\n"
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_output_io_failure_keeps_previous_snapshot_and_cleans_temporary_files(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "results.json"
    path.write_text("previous snapshot")
    original_write = Path.write_text

    def partial_write(destination, text, **kwargs):
        original_write(destination, "partial", **kwargs)
        raise OSError("disk failure")

    def failed_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(
        Path,
        "write_text" if failure == "write" else "replace",
        partial_write if failure == "write" else failed_replace,
    )
    with pytest.raises(OSError, match="disk failure"):
        write_json(path, {"score": 0.9})
    assert path.read_text() == "previous snapshot"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_invalid_json_cannot_replace_previous_results(tmp_path, value):
    path = tmp_path / "results.json"
    path.write_text("previous snapshot")
    with pytest.raises((TypeError, ValueError)):
        write_json(path, {"score": value})
    assert path.read_text() == "previous snapshot"
    assert list(tmp_path.iterdir()) == [path]
