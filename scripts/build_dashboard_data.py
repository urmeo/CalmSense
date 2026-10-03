"""Assemble experiment results into the TypeScript module the dashboard reads."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.calibration import BINARY_BRIER_DEFINITION, normalize_binary_calibration
from src.config import OUTPUT_DIR, RESULTS_DIR

DASHBOARD_RESULTS = OUTPUT_DIR / "dashboard" / "results.ts"

# Keys the dashboard consumes per task (per-subject lists stay out of the bundle)
TASK_KEYS = [
    "n_windows",
    "n_features",
    "classes",
    "models",
    "best_model",
    "loso_accuracy",
    "loso_pooled_accuracy",
    "loso_matched_accuracy",
    "within_subject_accuracy",
    "optimism_gap_pts",
]


def _load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path).to_dict("records") if path.exists() else None


def run():
    metrics = _load_json("metrics.json")
    if metrics is None:
        raise SystemExit(
            f"{RESULTS_DIR / 'metrics.json'} missing. Run scripts/run_experiment.py first."
        )

    out = {}
    for task in ("binary", "multiclass"):
        if task in metrics:
            out[task] = {k: metrics[task].get(k) for k in TASK_KEYS}

    shap = _load_csv("shap_top_features.csv")
    if shap:
        out["shap"] = shap[:12]
    for key, fname in [
        ("stats", "stats.json"),
        ("wrist", "wrist.json"),
        ("cross_dataset", "cross_dataset.json"),
        ("calibration", "calibration.json"),
        ("personalization", "personalization.json"),
        ("tuning", "tuning.json"),
    ]:
        data = _load_json(fname)
        if data is not None:
            if key == "calibration":
                data = normalize_binary_calibration(data)
            elif key == "personalization":
                data.setdefault("brier_definition", BINARY_BRIER_DEFINITION)
            out[key] = data
    ablation = _load_csv("ablation.csv")
    if ablation:
        out["ablation"] = ablation

    payload = json.dumps(out, indent=2, allow_nan=False)
    DASHBOARD_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_RESULTS.write_text(
        f"const data = {payload};\n\nexport default data;\n", encoding="utf-8"
    )
    print(f"Wrote {DASHBOARD_RESULTS} with sections: {sorted(out)}")


if __name__ == "__main__":
    run()
