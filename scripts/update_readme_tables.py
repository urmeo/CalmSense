"""Update the README calibration and personalization tables from committed results.

Run after scripts/calibration.py and scripts/personalize.py so the project overview
reflects the committed JSON. This updates documentation only; it does not fit models.
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import BINARY_BRIER_DEFINITION, normalize_binary_calibration
from src.config import RESULTS_DIR
from src.utils import atomic_write_text

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = RESULTS_DIR


def _f(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError("README metrics must be finite")
    return f"{x:.3f}"


def _replace(text: str, name: str, body: str) -> str:
    start, end = f"<!-- AUTOGEN:{name} START -->", f"<!-- AUTOGEN:{name} END -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"AUTOGEN markers for '{name}' not found in README.md")
    return pat.sub(f"{start}\n{body}\n{end}", text)


def _calibration_table() -> str:
    with open(RESULTS / "calibration.json") as fh:
        d = normalize_binary_calibration(json.load(fh))
    head = "| Evaluation | ECE | MCE | Brier |\n| --- | :-: | :-: | :-: |"
    rows = [
        ("Subject-mixed 5-fold, non-overlapping", "within_subject"),
        ("LOSO, matched non-overlapping", "loso_matched"),
        ("LOSO, all windows", "loso"),
        ("LOSO, all windows + isotonic", "recalibrated_isotonic"),
    ]
    lines = [
        f"| {label} | {_f(d[k]['ece'])} | {_f(d[k]['mce'])} | {_f(d[k]['brier'])} |"
        for label, k in rows
    ]
    return "\n".join([head] + lines)


def _personalization_table() -> str:
    with open(RESULTS / "personalization.json") as fh:
        d = json.load(fh)
    if d.get("brier_definition", BINARY_BRIER_DEFINITION) != BINARY_BRIER_DEFINITION:
        raise ValueError("Personalization Brier values must use positive-class MSE")
    head = "| Recalibration / requested enrollment budget | ECE | Brier |\n| --- | :-: | :-: |"
    rows = [("None (LOSO)", d["uncalibrated"]), ("Global (training subjects)", d["global"])]
    # k values are whatever personalize.py used (K_VALUES), not hard-coded
    for k in sorted(d["fewshot"], key=int):
        rows.append((f"Per-subject, budget {k}", d["fewshot"][k]))
    lines = [f"| {label} | {_f(s['ece'])} | {_f(s['brier'])} |" for label, s in rows]
    return "\n".join([head] + lines)


def main() -> None:
    if (
        not (RESULTS / "calibration.json").exists()
        or not (RESULTS / "personalization.json").exists()
    ):
        sys.exit("Missing results: run scripts/calibration.py and scripts/personalize.py first.")
    text = README.read_text()
    text = _replace(text, "calibration", _calibration_table())
    text = _replace(text, "personalization", _personalization_table())
    atomic_write_text(README, text)
    print(f"README calibration and personalization tables updated from {RESULTS}.")


if __name__ == "__main__":
    main()
