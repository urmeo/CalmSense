"""Guard the docs against regressions to the original scaffold README.

The first commit shipped a templated README that named models the project never
trains (Transformer with cross-modal attention, BiLSTM, CatBoost, EfficientNet),
ranked one "Best overall" (the real finding is a statistical tie), and listed
eight notebooks (01 to 08) that do not exist; the repo ships a single notebook.
Those claims were removed in the 2026-05-28 rewrite. These tests fail if any of
them reappear, or if the docs ever point at a notebook file that is not on disk.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()

# Names that appear only in the scaffold: the project never trains these.
PHANTOM_MODELS = ["Transformer", "BiLSTM", "CatBoost", "EfficientNet", "cross-modal attention"]


def test_readme_names_no_untrained_models():
    lower = README.lower()
    found = [name for name in PHANTOM_MODELS if name.lower() in lower]
    assert not found, f"README names models the project never trains: {found}"


def test_docs_make_no_best_overall_ranking_claim():
    # The scaffold crowned a single model "Best overall"; the honest result is that
    # the four feature models are statistically tied (Friedman p = 0.81).
    assert "best overall" not in README.lower(), "README makes a 'Best overall' ranking claim"


def test_referenced_notebooks_exist():
    refs = sorted(set(re.findall(r"notebooks/[\w./-]+\.ipynb", README)))
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    assert not missing, f"Docs reference notebooks that do not exist: {missing}"


def test_no_scaffold_numbered_notebook_series():
    # The scaffold listed notebooks/01_.. through notebooks/08_..; none exist.
    assert not re.search(r"notebooks/0[1-8]_", README), (
        "README references the scaffold's numbered 01-08 notebook series"
    )


@pytest.mark.parametrize("location", ["empty", "root", "notebooks"])
def test_notebook_setup_reuses_checkout(tmp_path, monkeypatch, location):
    root = tmp_path / "CalmSense"

    def checkout():
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "calibration.py").touch()
        (root / "pyproject.toml").touch()
        (root / "notebooks").mkdir()

    if location != "empty":
        checkout()
    start = {"empty": tmp_path, "root": root, "notebooks": root / "notebooks"}[location]
    monkeypatch.chdir(start)
    commands = []

    def run(command, *, check):
        assert check
        commands.append(command)
        if command[:2] == ["git", "clone"]:
            assert Path(command[-1]) == root
            checkout()

    monkeypatch.setattr(subprocess, "run", run)
    notebook = json.loads((ROOT / "notebooks" / "CalmSense.ipynb").read_text())
    setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(setup["source"])
    for _ in range(2):
        exec(compile(source, "notebook setup", "exec"), {})
        assert Path.cwd() == root
    assert sum(command[:2] == ["git", "clone"] for command in commands) == (location == "empty")
    assert commands[-1] == [sys.executable, "-m", "pip", "install", "-q", "-e", "."]
    assert not (root / "CalmSense").exists()


def test_no_em_or_en_dashes_anywhere():
    """The author's style bans em/en dashes in code, comments, and docs; guard the
    whole tree so a stray dash cannot slip back in (an earlier pass missed several)."""
    targets = (
        sorted(ROOT.glob("src/**/*.py"))
        + sorted(ROOT.glob("scripts/*.py"))
        + sorted(ROOT.glob("tests/*.py"))
        + sorted(ROOT.glob("frontend/src/**/*.ts"))
        + sorted(ROOT.glob("frontend/src/**/*.tsx"))
        + sorted(ROOT.glob("docs/*.md"))
        + sorted(ROOT.glob("notebooks/*.ipynb"))
        + [
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "results" / "README.md",
            ROOT / "data" / "raw" / "README.md",
        ]
    )
    en_dash, em_dash = chr(0x2013), chr(0x2014)  # by codepoint, so this guard never flags itself
    offenders = [
        str(p.relative_to(ROOT))
        for p in targets
        if p.exists() and (en_dash in p.read_text() or em_dash in p.read_text())
    ]
    assert not offenders, f"em/en dash found in: {offenders}"
