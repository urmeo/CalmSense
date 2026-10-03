"""Archive extraction refuses path traversal and oversized archives."""

import hashlib
import io
import zipfile

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


@pytest.mark.parametrize("missing_subject", [None, "S2", "S3"])
def test_existing_wesad_must_be_complete_and_verified(tmp_path, monkeypatch, missing_subject):
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
    reference = {}
    for sid in ("S2", "S3"):
        payload = sid.encode()
        reference[sid] = hashlib.sha256(payload).hexdigest()
        if sid == missing_subject:
            continue
        subject = tmp_path / "WESAD" / sid
        subject.mkdir(parents=True)
        (subject / f"{sid}.pkl").write_bytes(payload)
    monkeypatch.setattr(download_data, "WESAD_SHA256", reference)
    monkeypatch.setattr(
        download_data, "_download", lambda *args: pytest.fail("Unexpected download")
    )
    if missing_subject:
        with pytest.raises(SystemExit, match=f"{missing_subject}: missing"):
            download_data.download_wesad()
    else:
        download_data.download_wesad()


class DownloadResponse(io.BytesIO):
    def __init__(self, payload, *, length=None, url="https://example.com/data.zip"):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}
        self.url = url

    def geturl(self):
        return self.url


def test_download_streams_to_complete_file(tmp_path, monkeypatch):
    response = DownloadResponse(b"verified archive")
    calls = []

    def open_response(url, *, timeout):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(download_data, "urlopen", open_response)
    dest = tmp_path / "nested" / "data.zip"
    _download("https://example.com/data.zip", str(dest))
    assert dest.read_bytes() == b"verified archive"
    assert calls == [("https://example.com/data.zip", download_data.DOWNLOAD_TIMEOUT)]
    assert not dest.with_suffix(".zip.part").exists()


@pytest.mark.parametrize("redirect", [False, True])
def test_download_failure_preserves_previous_file(tmp_path, monkeypatch, redirect):
    response = DownloadResponse(
        b"short body",
        length=100,
        url="http://example.com/data.zip" if redirect else "https://example.com/data.zip",
    )
    monkeypatch.setattr(download_data, "urlopen", lambda *args, **kwargs: response)
    dest = tmp_path / "data.zip"
    dest.write_bytes(b"previous archive")
    error = ValueError if redirect else RuntimeError
    with pytest.raises(error, match="non-HTTPS|incomplete"):
        _download("https://example.com/data.zip", dest)
    assert dest.read_bytes() == b"previous archive"
    assert not dest.with_suffix(".zip.part").exists()


@pytest.mark.parametrize("problem", [None, "missing", "changed", "manifest"])
def test_existing_noneeg_requires_complete_publisher_manifest(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(download_data, "NONEEG_DIR", tmp_path)
    names = ("Subject1_AccTempEDA.dat", "Subject1_SpO2HR.hea")
    monkeypatch.setattr(download_data, "NONEEG_FILES", names)
    target = tmp_path / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    target.mkdir()
    references = []
    for name in names:
        payload = name.encode()
        (target / name).write_bytes(payload)
        references.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    (target / "SHA256SUMS.txt").write_text("\n".join(references) + "\n")
    if problem == "missing":
        (target / names[1]).unlink()
    elif problem == "changed":
        (target / names[1]).write_bytes(b"changed")
    elif problem == "manifest":
        (target / "SHA256SUMS.txt").write_text("invalid digest\n")
    monkeypatch.setattr(
        download_data, "_download", lambda *args: pytest.fail("Unexpected download")
    )
    if problem:
        with pytest.raises(SystemExit, match="integrity check failed"):
            download_data.download_noneeg()
    else:
        download_data.download_noneeg()


@pytest.mark.parametrize("dataset", ["wesad", "noneeg"])
def test_dataset_is_installed_only_after_integrity_verification_and_retry_succeeds(
    tmp_path, monkeypatch, dataset
):
    payload = b"verified recording"
    digest = hashlib.sha256(payload).hexdigest()
    valid = False
    if dataset == "wesad":
        monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
        monkeypatch.setattr(download_data, "WESAD_SHA256", {"S2": digest})
        target = tmp_path / "WESAD"
        file_name = "S2/S2.pkl"
        download = download_data.download_wesad
    else:
        monkeypatch.setattr(download_data, "NONEEG_DIR", tmp_path)
        monkeypatch.setattr(download_data, "NONEEG_FILES", ("record.dat",))
        target = tmp_path / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
        file_name = "record.dat"
        download = download_data.download_noneeg

    def download_archive(url, destination):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(f"{target.name}/{file_name}", payload if valid else b"corrupt")
            if dataset == "noneeg":
                archive.writestr(f"{target.name}/SHA256SUMS.txt", f"{digest}  {file_name}\n")

    monkeypatch.setattr(download_data, "_download", download_archive)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        download()
    assert not target.exists()
    assert [p.suffix for p in tmp_path.iterdir()] == [".zip"]
    valid = True
    download()
    assert (target / file_name).read_bytes() == payload
    assert list(tmp_path.iterdir()) == [target]
