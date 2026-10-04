import json

import numpy as np
import pytest

from scripts import tuning
from src import synthetic


@pytest.mark.parametrize("demo_accuracy", [None, 0.42])
def test_synthetic_tuning_isolates_outputs_and_reads_only_demo_defaults(
    tmp_path, monkeypatch, demo_accuracy
):
    output_dir = tmp_path / "outputs"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "generated" / "figures"
    demo_dir = output_dir / "generated" / "demo"
    results_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    model = "Random Forest"
    real_metrics = results_dir / "metrics.json"
    real_metrics.write_text(
        json.dumps({"binary": {"models": [{"model": model, "accuracy_mean": 0.91}]}})
    )
    real_result = results_dir / "tuning.json"
    real_figure = figures_dir / "tuning.png"
    real_result.write_bytes(b"existing real tuning result")
    real_figure.write_bytes(b"existing real tuning plot")
    originals = {path: path.read_bytes() for path in (real_metrics, real_result, real_figure)}
    if demo_accuracy is not None:
        demo_results = demo_dir / "results"
        demo_results.mkdir(parents=True)
        (demo_results / "metrics.json").write_text(
            json.dumps({"binary": {"models": [{"model": model, "accuracy_mean": demo_accuracy}]}})
        )

    monkeypatch.setattr(tuning, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(tuning, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(tuning, "DEMO_DIR", demo_dir)

    def refuse_real_cache():
        pytest.fail("Synthetic tuning read the real dataset cache")

    def synthetic_features(**kwargs):
        assert kwargs["cache"] is False
        return None, None, None

    monkeypatch.setattr(tuning, "load_cached", refuse_real_cache)
    monkeypatch.setattr(synthetic, "features", synthetic_features)
    monkeypatch.setattr(tuning, "prepare_task", lambda *args: (None,) * 5)
    scores = {model: {"accuracy_mean": 0.72, "best_params": {}}}
    monkeypatch.setattr(tuning, "compute", lambda *args: scores)
    provenance = {"git_sha": "test", "generated_at": "test"}
    monkeypatch.setattr(tuning, "provenance", lambda: provenance)
    original_plot = tuning._plot
    plotted_defaults = []

    def plot(tuned, defaults, path):
        plotted_defaults.append(defaults)
        original_plot(tuned, defaults, path)

    monkeypatch.setattr(tuning, "_plot", plot)

    tuning.run(synthetic=True)

    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert json.loads((demo_dir / "results" / "tuning.json").read_text()) == {
        model: {"accuracy_mean": 0.72, "best_params": {}},
        "provenance": provenance,
    }
    demo_figure = demo_dir / "figures" / "tuning.png"
    if demo_accuracy is None:
        assert plotted_defaults == []
        assert not demo_figure.exists()
    else:
        assert plotted_defaults == [{model: demo_accuracy}]
        assert demo_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_nested_tuning_fails_on_invalid_candidate_instead_of_selecting_around_it(monkeypatch):
    monkeypatch.setitem(tuning.GRIDS, "lr", {"clf__C": [-1.0, 1.0]})
    groups = np.repeat(["S0", "S1", "S2", "S3"], 6)
    y = np.tile([0, 1], 12)
    X = np.random.RandomState(0).randn(24, 2)
    with pytest.raises(ValueError, match="C"):
        tuning.tune_model("lr", X, y, groups, inner_splits=2)


@pytest.mark.parametrize("inner_splits, groups", [(1, ["S0", "S1", "S2"]), (2, ["S0", "S1"])])
def test_nested_tuning_requires_valid_subject_counts(inner_splits, groups):
    with pytest.raises(ValueError, match="requires"):
        tuning.tune_model(
            "lr", np.zeros((len(groups), 1)), np.zeros(len(groups)), np.array(groups), inner_splits
        )


def test_tuning_plot_does_not_show_missing_default_as_zero(tmp_path, monkeypatch):
    bars = []
    original = tuning.plt.bar

    def bar(x, values, *args, **kwargs):
        bars.append(list(values))
        return original(x, values, *args, **kwargs)

    monkeypatch.setattr(tuning.plt, "bar", bar)
    tuning._plot(
        {"Random Forest": {"accuracy_mean": 0.8}, "LightGBM": {"accuracy_mean": 0.7}},
        {"Random Forest": 0.9},
        tmp_path / "plot.png",
    )
    assert bars == [[0.9], [0.8]]


@pytest.mark.parametrize("mismatch", [None, "schema", "protocol", "cohort", "duplicates"])
def test_tuning_default_comparison_requires_matching_protocol_schema_and_cohort(tmp_path, mismatch):
    from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION

    summary = {
        "benchmark_protocol_version": tuning.BENCHMARK_PROTOCOL_VERSION,
        "binary": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "models": [{"model": "Random Forest", "accuracy_mean": 0.91}],
            "per_subject": [{"subject": "S0"}, {"subject": "S1"}],
        },
    }
    if mismatch == "schema":
        summary["binary"]["feature_schema_version"] -= 1
    elif mismatch == "protocol":
        summary["benchmark_protocol_version"] -= 1
    elif mismatch == "cohort":
        summary["binary"]["per_subject"][1]["subject"] = "S2"
    elif mismatch == "duplicates":
        summary["binary"]["per_subject"][1]["subject"] = "S0"
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    defaults = tuning._defaults(tmp_path, groups=np.array(["S1", "S0", "S0"]))
    assert defaults == ({"Random Forest": 0.91} if mismatch is None else {})


def test_tuning_retires_only_generated_stale_plot_when_defaults_are_incompatible(
    tmp_path, monkeypatch
):
    results, figures = tmp_path / "results", tmp_path / "generated" / "figures"
    results.mkdir()
    figures.mkdir(parents=True)
    (results / "metrics.json").write_text(json.dumps({"benchmark_protocol_version": 1}))
    stale = figures / "tuning.png"
    stale.write_bytes(b"stale generated comparison")
    historical = tmp_path / "figures" / "tuning.png"
    historical.parent.mkdir()
    historical.write_bytes(b"preserved historical figure")
    monkeypatch.setattr(tuning, "RESULTS_DIR", results)
    monkeypatch.setattr(tuning, "FIGURES_DIR", figures)
    monkeypatch.setattr(tuning, "load_cached", lambda: (None, None))
    monkeypatch.setattr(
        tuning, "prepare_task", lambda *args: (None, None, np.array(["S0", "S1"]), None, None)
    )
    monkeypatch.setattr(tuning, "compute", lambda *args: {"Random Forest": {"accuracy_mean": 0.6}})
    tuning.run()
    assert (results / "tuning.json").exists()
    assert not stale.exists()
    assert historical.read_bytes() == b"preserved historical figure"
