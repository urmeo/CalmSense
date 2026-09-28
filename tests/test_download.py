"""Archive extraction refuses path traversal and oversized archives."""

import hashlib
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import download_data
from scripts.download_data import _download, _safe_extract


def test_safe_extract_unpacks_benign_zip(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a/b.txt", "hello")
    out = tmp_path / "out"
    _safe_extract(archive, out)
    assert (out / "a" / "b.txt").read_text() == "hello"


def test_safe_extract_blocks_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "pwned")
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(archive, tmp_path / "out")


def test_safe_extract_blocks_zip_bomb(tmp_path):
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data.bin", b"x" * 4096)
    with pytest.raises(RuntimeError, match="size cap"):
        _safe_extract(archive, tmp_path / "out", max_bytes=1024)


def test_download_refuses_non_https(tmp_path):
    with pytest.raises(ValueError, match="non-HTTPS"):
        _download("http://example.com/x.zip", tmp_path / "x.zip")


def _wesad_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
    contents = {"S2": b"subject two", "S3": b"subject three"}
    monkeypatch.setattr(
        download_data,
        "WESAD_SHA256",
        {sid: hashlib.sha256(data).hexdigest() for sid, data in contents.items()},
    )
    for sid, data in contents.items():
        path = tmp_path / "WESAD" / sid / f"{sid}.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    return tmp_path / "WESAD"


def test_wesad_verification_streams_files(tmp_path, monkeypatch):
    _wesad_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("whole-file read")))
    download_data.verify_wesad()


@pytest.mark.parametrize("problem", ["missing", "corrupt"])
def test_existing_wesad_requires_all_valid_subjects(tmp_path, monkeypatch, problem):
    target = _wesad_fixture(tmp_path, monkeypatch)
    damaged = target / "S3" / "S3.pkl"
    if problem == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"incomplete")
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    download = Mock(side_effect=AssertionError("must preserve existing data"))
    monkeypatch.setattr(download_data, "_download", download)
    with pytest.raises(SystemExit, match="S3: (missing|checksum mismatch)"):
        download_data.download_wesad()
    assert {p: p.read_bytes() for p in target.rglob("*") if p.is_file()} == before
    download.assert_not_called()


def _noneeg_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "NONEEG_DIR", tmp_path)
    target = tmp_path / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    target.mkdir()
    for subject in range(1, 21):
        for record, extensions in [
            ("AccTempEDA", ("hea", "dat", "atr")),
            ("SpO2HR", ("hea", "dat")),
        ]:
            for extension in extensions:
                (target / f"Subject{subject}_{record}.{extension}").write_bytes(b"fixture")
    return target


@pytest.mark.parametrize(
    "missing",
    ["all", "Subject20_AccTempEDA.hea", "Subject20_SpO2HR.dat", "Subject20_AccTempEDA.atr"],
)
def test_existing_noneeg_requires_complete_records(tmp_path, monkeypatch, missing):
    target = _noneeg_fixture(tmp_path, monkeypatch)
    if missing == "all":
        for path in target.iterdir():
            path.unlink()
    else:
        (target / missing).unlink()
    sentinel = target / "unrelated.txt"
    sentinel.write_text("preserve")
    download = Mock(side_effect=AssertionError("must preserve existing data"))
    monkeypatch.setattr(download_data, "_download", download)
    with pytest.raises(SystemExit, match="Non-EEG.*incomplete"):
        download_data.download_noneeg()
    assert sentinel.read_text() == "preserve"
    download.assert_not_called()


def test_existing_noneeg_rejects_empty_data_file(tmp_path, monkeypatch):
    target = _noneeg_fixture(tmp_path, monkeypatch)
    (target / "Subject20_SpO2HR.dat").write_bytes(b"")
    with pytest.raises(SystemExit, match="Subject20_SpO2HR.dat"):
        download_data.download_noneeg()


def test_existing_complete_noneeg_does_not_download(tmp_path, monkeypatch):
    _noneeg_fixture(tmp_path, monkeypatch)
    download = Mock(side_effect=AssertionError("already complete"))
    monkeypatch.setattr(download_data, "_download", download)
    download_data.download_noneeg()
    download.assert_not_called()


@pytest.mark.parametrize("complete", [False, True])
def test_new_noneeg_download_checks_extracted_records(tmp_path, monkeypatch, complete):
    monkeypatch.setattr(download_data, "NONEEG_DIR", tmp_path)

    def download(url, path):
        with zipfile.ZipFile(path, "w") as archive:
            if complete:
                target = _noneeg_fixture(tmp_path, monkeypatch)
                for source in target.iterdir():
                    archive.write(source, str(source.relative_to(tmp_path)))
                    source.unlink()
                target.rmdir()
            else:
                archive.writestr("unrelated.txt", "preserve")

    monkeypatch.setattr(download_data, "_download", download)
    if complete:
        download_data.download_noneeg()
        assert len(list(tmp_path.rglob("*.hea"))) == 40
    else:
        with pytest.raises(SystemExit, match="Non-EEG.*incomplete"):
            download_data.download_noneeg()
        assert (tmp_path / "unrelated.txt").read_text() == "preserve"


def test_wesad_download_uses_larger_bounded_extraction_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "_download", lambda url, path: path.touch())
    verify = Mock()
    monkeypatch.setattr(download_data, "verify_wesad", verify)
    extract = Mock()
    monkeypatch.setattr(download_data, "_safe_extract", extract)
    download_data.download_wesad()
    assert extract.call_args.kwargs["max_bytes"] == 32 * 1024**3
    verify.assert_called_once()


def test_wesad_size_allowance_does_not_relax_default_cap(tmp_path, monkeypatch):
    archive = Mock()
    archive.__enter__ = Mock(return_value=archive)
    archive.__exit__ = Mock(return_value=False)
    member = zipfile.ZipInfo("WESAD/S2/S2.pkl")
    member.file_size = 17_578_304_666
    archive.infolist.return_value = [member]
    monkeypatch.setattr(zipfile, "ZipFile", Mock(return_value=archive))
    with pytest.raises(RuntimeError, match="size cap"):
        _safe_extract(tmp_path / "wesad.zip", tmp_path / "out")
    _safe_extract(tmp_path / "wesad.zip", tmp_path / "out", max_bytes=32 * 1024**3)
    archive.extractall.assert_called_once()
    member.file_size = 32 * 1024**3 + 1
    with pytest.raises(RuntimeError, match="size cap"):
        _safe_extract(tmp_path / "wesad.zip", tmp_path / "out", max_bytes=32 * 1024**3)


def test_reproduce_verifies_complete_datasets_before_overwriting_benchmarks():
    result = subprocess.run(
        ["make", "-n", "reproduce"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = result.stdout.splitlines()
    assert commands[:3] == [
        "python scripts/download_data.py --verify-wesad",
        "python scripts/download_data.py --verify-noneeg",
        "python scripts/run_experiment.py",
    ]
    assert commands[-1] == "python scripts/stamp_provenance.py --overwrite"
