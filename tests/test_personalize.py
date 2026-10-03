"""Enrollment split is leak-free and keeps both classes."""

import json

import numpy as np
import pytest

from scripts import personalize
from scripts.personalize import _sample_k, _stratified_split
from src import synthetic as synthetic_data


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


@pytest.mark.parametrize("synthetic", [False, True])
def test_personalization_output_paths_protect_benchmark(tmp_path, monkeypatch, synthetic):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "outputs" / "figures"
    results_dir.mkdir()
    figures_dir.mkdir(parents=True)
    real_result = results_dir / "personalization.json"
    real_figure = figures_dir / "personalization.png"
    real_result.write_bytes(b"committed benchmark snapshot")
    real_figure.write_bytes(b"committed benchmark plot")
    monkeypatch.setattr(personalize, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(personalize, "FIGURES_DIR", figures_dir)

    def cached_features():
        assert not synthetic, "Synthetic demo read the real dataset cache"
        return None, None

    def generated_features(**kwargs):
        assert synthetic
        assert kwargs["cache"] is False
        return None, None, None

    monkeypatch.setattr(personalize, "load_cached", cached_features)
    monkeypatch.setattr(synthetic_data, "features", generated_features)
    monkeypatch.setattr(personalize, "prepare_task", lambda *args: (None,) * 5)
    metrics = {
        "k_values": [5],
        "uncalibrated": {"ece": 0.1, "brier": 0.1},
        "global": {"ece": 0.08, "brier": 0.08},
        "fewshot": {"5": {"ece": 0.05, "brier": 0.05}},
    }
    monkeypatch.setattr(personalize, "compute", lambda *args, **kwargs: metrics)
    monkeypatch.setattr(
        personalize, "provenance", lambda: {"git_sha": "test", "generated_at": "test"}
    )
    monkeypatch.setattr(personalize, "_plot", lambda out, path: path.write_bytes(b"new plot"))

    personalize.run(synthetic=synthetic)

    target_results = results_dir / "demo" if synthetic else results_dir
    target_figures = figures_dir / "demo" if synthetic else figures_dir
    assert json.loads((target_results / "personalization.json").read_text()) == metrics
    assert (target_figures / "personalization.png").read_bytes() == b"new plot"
    if synthetic:
        assert real_result.read_bytes() == b"committed benchmark snapshot"
        assert real_figure.read_bytes() == b"committed benchmark plot"
    else:
        assert not (results_dir / "demo").exists()
        assert not (figures_dir / "demo").exists()
