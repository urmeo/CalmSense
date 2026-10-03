"""Portable caches must never silently reuse sample-based slope features."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import cross_dataset
from src.portable import PORTABLE_FEATURE_COLUMNS


@pytest.fixture
def features():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in PORTABLE_FEATURE_COLUMNS})
    frame["subject"] = ["S2", "S3"]
    frame["label"] = [0, 1]
    return frame


def test_legacy_cache_is_ignored_and_v2_cache_is_verified(tmp_path, monkeypatch, features):
    directory = tmp_path / "processed"
    directory.mkdir()
    legacy = directory / "portable_wesad.parquet"
    legacy.write_bytes(b"legacy sample-based slopes")
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", directory)
    builder = Mock(return_value=features)

    frame, metadata = cross_dataset._load_or_build_cache("wesad", builder)
    builder.assert_called_once_with()
    assert legacy.read_bytes() == b"legacy sample-based slopes"
    assert metadata["schema_version"] == 2
    assert metadata["sampling_rates_hz"] == {"eda": 4, "temp": 4}
    assert metadata["slope_units"] == "per_second"
    assert metadata["finite_sample_timestamps"] == "original_sample_positions"
    builder.reset_mock()
    cached, again = cross_dataset._load_or_build_cache("wesad", builder)
    builder.assert_not_called()
    pd.testing.assert_frame_equal(frame, cached)
    assert metadata == again


def test_cache_creates_parent_directories(tmp_path, monkeypatch, features):
    directory = tmp_path / "nested" / "processed"
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", directory)
    _, metadata = cross_dataset._load_or_build_cache("noneeg", lambda: features)
    assert (directory / "portable_noneeg_v2.parquet").exists()
    assert metadata["sampling_rates_hz"] == {"eda": 8, "temp": 8}


@pytest.mark.parametrize("fault", ["missing_metadata", "old_schema", "bad_checksum"])
def test_unverified_v2_cache_is_rejected(tmp_path, monkeypatch, features, fault):
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
    cross_dataset._load_or_build_cache("wesad", lambda: features)
    sidecar = tmp_path / "portable_wesad_v2.json"
    if fault == "missing_metadata":
        sidecar.unlink()
    else:
        metadata = json.loads(sidecar.read_text())
        if fault == "old_schema":
            metadata["schema_version"] = 1
        else:
            metadata["cache_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(metadata))
    builder = Mock(return_value=features)
    with pytest.raises(ValueError, match="metadata|schema|checksum"):
        cross_dataset._load_or_build_cache("wesad", builder)
    builder.assert_not_called()


@pytest.mark.parametrize("field", ["extractor_sha256", "extraction_packages"])
def test_cache_rejects_changed_extractor_or_environment_and_can_rebuild(
    tmp_path, monkeypatch, features, field
):
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
    _, old = cross_dataset._load_or_build_cache("wesad", lambda: features)
    schema = cross_dataset._cache_schema("wesad")
    schema[field] = {"changed": "new version"}
    monkeypatch.setattr(cross_dataset, "_cache_schema", lambda dataset: schema)
    builder = Mock(return_value=features)
    with pytest.raises(ValueError, match="schema mismatch"):
        cross_dataset._load_or_build_cache("wesad", builder)
    builder.assert_not_called()
    _, new = cross_dataset._load_or_build_cache("wesad", builder, rebuild=True)
    builder.assert_called_once_with()
    assert new[field] != old[field]


def test_cross_dataset_labels_allow_integer_valued_float_storage(features):
    features["label"] = features["label"].astype(float)
    _, y, _ = cross_dataset._xy(features, list(PORTABLE_FEATURE_COLUMNS))
    assert y.dtype.kind in "iu"
    assert y.tolist() == [0, 1]


@pytest.mark.parametrize("columns", [[], ["label"], ["subject"], ["eda_mean", "eda_mean"]])
def test_cross_dataset_rejects_metadata_as_features(features, columns):
    with pytest.raises(ValueError, match="feature columns"):
        cross_dataset._xy(features, columns)


def test_failed_rebuild_preserves_verified_cache(tmp_path, monkeypatch, features):
    monkeypatch.setattr(cross_dataset, "PROCESSED_DATA_DIR", tmp_path)
    cross_dataset._load_or_build_cache("wesad", lambda: features)
    cache = tmp_path / "portable_wesad_v2.parquet"
    sidecar = cache.with_suffix(".json")
    before = cache.read_bytes(), sidecar.read_bytes()
    changed = features.copy()
    changed[PORTABLE_FEATURE_COLUMNS[0]] += 1
    with pytest.raises(TypeError):
        cross_dataset._load_or_build_cache(
            "wesad", lambda: changed, source_provenance={"invalid": object()}, rebuild=True
        )
    assert (cache.read_bytes(), sidecar.read_bytes()) == before
