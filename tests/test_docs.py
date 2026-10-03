"""Documentation describes implemented models and references existing artifacts."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


# These models are not implemented by the project.
PHANTOM_MODELS = ["Transformer", "BiLSTM", "CatBoost", "EfficientNet", "cross-modal attention"]


def test_readme_names_no_untrained_models():
    lower = README.lower()
    found = [name for name in PHANTOM_MODELS if name.lower() in lower]
    assert not found, f"README names models the project never trains: {found}"


def test_docs_make_no_best_overall_ranking_claim():
    # No significant difference was detected among the four feature models.
    assert "best overall" not in README.lower(), "README makes a 'Best overall' ranking claim"


def test_readme_local_links_and_images_exist():
    """Check the overview's Markdown destinations and HTML image sources."""
    refs = re.findall(r"\]\(([^\s)]+)\)", README) + re.findall(r'(?:src|href)="([^"]+)"', README)
    for ref in refs:
        url = urlsplit(ref)
        if url.scheme or not url.path:
            continue
        assert (ROOT / unquote(url.path)).exists(), f"Missing README asset: {ref}"


def test_readme_table_update_is_idempotent(tmp_path, monkeypatch):
    """Updating the overview must preserve its prose and use the committed results."""
    from scripts import update_readme_tables as tables

    path = tmp_path / "README.md"
    path.write_text(README)
    monkeypatch.setattr(tables, "README", path)
    tables.main()
    assert path.read_text() == README
    tables.main()
    assert path.read_text() == README


def test_readme_table_update_rejects_missing_markers():
    from scripts.update_readme_tables import _replace

    with pytest.raises(SystemExit, match="not found in README.md"):
        _replace("overview without generated tables", "calibration", "replacement")


def test_referenced_notebooks_exist():
    refs = sorted(set(re.findall(r"notebooks/[\w./-]+\.ipynb", README)))
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    assert not missing, f"Docs reference notebooks that do not exist: {missing}"


def test_no_scaffold_numbered_notebook_series():
    # The numbered 01-08 notebook series does not exist.
    assert not re.search(r"notebooks/0[1-8]_", README), (
        "README references the scaffold's numbered 01-08 notebook series"
    )


def test_no_em_or_en_dashes_anywhere():
    """Guard documentation and source punctuation."""
    targets = (
        sorted(ROOT.glob("src/**/*.py"))
        + sorted(ROOT.glob("scripts/*.py"))
        + sorted(ROOT.glob("tests/*.py"))
        + sorted(ROOT.glob("frontend/src/**/*.ts"))
        + sorted(ROOT.glob("frontend/src/**/*.tsx"))
        + sorted(ROOT.glob("outputs/dashboard/**/*.ts"))
        + sorted(ROOT.glob("docs/*.md"))
        + sorted(ROOT.glob("notebooks/*.ipynb"))
        + [
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
        ]
    )
    en_dash, em_dash = chr(0x2013), chr(0x2014)  # by codepoint, so this guard never flags itself
    offenders = [
        str(p.relative_to(ROOT))
        for p in targets
        if p.exists() and (en_dash in p.read_text() or em_dash in p.read_text())
    ]
    assert not offenders, f"em/en dash found in: {offenders}"
