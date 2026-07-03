"""Guard the docs against regressions to the original scaffold README.

The first commit shipped a templated README that named models the project never
trains (Transformer with cross-modal attention, BiLSTM, CatBoost, EfficientNet),
ranked one "Best overall" (the real finding is a statistical tie), and listed
eight notebooks (01 to 08) that do not exist; the repo ships a single notebook.
Those claims were removed in the 2026-05-28 rewrite. These tests fail if any of
them reappear, or if the docs ever point at a notebook file that is not on disk.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
PAPER = (ROOT / "PAPER.md").read_text()

# Names that appear only in the scaffold: the project never trains these.
PHANTOM_MODELS = ["Transformer", "BiLSTM", "CatBoost", "EfficientNet", "cross-modal attention"]


def test_readme_names_no_untrained_models():
    lower = README.lower()
    found = [name for name in PHANTOM_MODELS if name.lower() in lower]
    assert not found, f"README names models the project never trains: {found}"


def test_docs_make_no_best_overall_ranking_claim():
    # The scaffold crowned a single model "Best overall"; the honest result is that
    # the four feature models are statistically tied (Friedman p = 0.81).
    for name, text in (("README.md", README), ("PAPER.md", PAPER)):
        assert "best overall" not in text.lower(), f"{name} makes a 'Best overall' ranking claim"


def test_referenced_notebooks_exist():
    refs = sorted(set(re.findall(r"notebooks/[\w./-]+\.ipynb", README + PAPER)))
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    assert not missing, f"Docs reference notebooks that do not exist: {missing}"


def test_no_scaffold_numbered_notebook_series():
    # The scaffold listed notebooks/01_.. through notebooks/08_..; none exist.
    assert not re.search(r"notebooks/0[1-8]_", README), (
        "README references the scaffold's numbered 01-08 notebook series"
    )
