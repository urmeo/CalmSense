"""Enrollment split is leak-free and keeps both classes."""

import numpy as np
import pytest

from scripts.personalize import _sample_k, _stratified_split


def test_split_is_disjoint_and_covers_all():
    rng = np.random.RandomState(0)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ev, pool = _stratified_split(y, 0.5, rng)
    assert set(ev).isdisjoint(pool)
    assert sorted([*ev, *pool]) == list(range(len(y)))


def test_split_keeps_both_classes_each_side():
    rng = np.random.RandomState(1)
    y = np.array([0] * 6 + [1] * 6)
    ev, pool = _stratified_split(y, 0.5, rng)
    assert {0, 1} <= set(y[ev]) and {0, 1} <= set(y[pool])


def test_sample_k_balances_classes():
    rng = np.random.RandomState(2)
    y_pool = np.array([0] * 10 + [1] * 10)
    pick = _sample_k(y_pool, 6, rng)
    assert {0, 1} <= set(y_pool[pick])
    assert len(pick) <= 6
    counts = np.bincount(y_pool[pick], minlength=2)
    assert counts[0] == counts[1] == 3


@pytest.mark.parametrize("k", [0, 1, 5, 10, 30])
@pytest.mark.parametrize("class_counts", [(10, 10), (1, 20), (0, 0)])
def test_sample_k_uses_exact_available_budget(k, class_counts):
    y_pool = np.repeat([0, 1], class_counts)
    picked = _sample_k(y_pool, k, np.random.RandomState(2))
    assert len(picked) == min(k, len(y_pool))
    assert len(np.unique(picked)) == len(picked)
    assert np.issubdtype(picked.dtype, np.integer)
    if k >= 2 and all(class_counts):
        assert set(y_pool[picked]) == {0, 1}


def test_sample_k_rejects_negative_budget():
    with pytest.raises(ValueError, match="nonnegative"):
        _sample_k(np.array([0, 1]), -1, np.random.RandomState(2))


def test_empty_enrollment_pool_can_index_labels():
    y = np.array([0, 1])
    _, pool = _stratified_split(y, 0.5, np.random.RandomState(1))
    assert y[pool].size == 0


def test_synthetic_run_preserves_recorded_results(tmp_path, monkeypatch):
    from scripts import personalize
    from src import synthetic

    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    figures.mkdir()
    result = results / "personalization.json"
    figure = figures / "personalization.png"
    result.write_text("recorded results")
    figure.write_bytes(b"recorded figure")
    monkeypatch.setattr(personalize, "RESULTS_DIR", results)
    monkeypatch.setattr(personalize, "FIGURES_DIR", figures)
    monkeypatch.setattr(synthetic, "features", lambda **kwargs: (None, None, None))
    monkeypatch.setattr(personalize, "prepare_task", lambda *args: (None,) * 5)
    monkeypatch.setattr(
        personalize,
        "compute",
        lambda *args, **kwargs: {
            "uncalibrated": {"ece": 0.0, "brier": 0.0},
            "global": {"ece": 0.0, "brier": 0.0},
            "k_values": [],
        },
    )
    monkeypatch.setattr(personalize, "_plot", lambda out, path: path.write_bytes(b"demo figure"))

    personalize.run(synthetic=True)

    assert result.read_text() == "recorded results"
    assert figure.read_bytes() == b"recorded figure"
    assert (results / "demo" / "personalization.json").is_file()
    assert (figures / "demo" / "personalization.png").read_bytes() == b"demo figure"
