"""Threshold-free discrimination and one operating point for the binary task.

The rest of the benchmark reports accuracy and F1 at the default 0.5 threshold, but
the project's thesis is that the probability matters. This computes AUROC and AUPRC
(threshold-free) from the pooled out-of-fold LOSO probabilities, and one operating
point for the shipped model (random forest) at the Youden-J threshold: sensitivity,
specificity, PPV, NPV. Same folds as scripts/run_experiment.py, so the numbers line up.

Run inside `make reproduce` (needs cached features from run_experiment.py first).
xgboost/lightgbm need OpenMP (brew install libomp on macOS); models that cannot import
are skipped with an "available": false marker rather than failing the whole run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

from scripts.run_experiment import CLF_NAMES, RESULTS_DIR, build_pipeline, load_cached, prepare_task
from src.utils import provenance

FEATURE_MODELS = ["lr", "rf", "xgb", "lgbm"]
POINT_MODEL = "rf"  # the shipped model


def loso_pos_proba(key, X, y, groups):
    """Pooled out-of-fold P(class == 1), aligned with pooled true labels."""
    logo = LeaveOneGroupOut()
    p1, true = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = build_pipeline(key)
        pipe.fit(X[train_idx], y[train_idx])
        clf = pipe.named_steps["clf"]
        pos = list(clf.classes_).index(1)  # column for the positive (stress) class
        p1.extend(pipe.predict_proba(X[test_idx])[:, pos])
        true.extend(y[test_idx])
    return np.asarray(true), np.asarray(p1)


def operating_point(y_true, p1):
    """Youden-J threshold and the confusion-derived rates at that threshold."""
    fpr, tpr, thr = roc_curve(y_true, p1)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    pred = (p1 >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    safe = lambda num, den: float(num / den) if den else float("nan")  # noqa: E731
    return {
        "rule": "Youden J (max sensitivity + specificity - 1)",
        "threshold": t,
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
    }


def run():
    cached = load_cached()
    if cached is None:
        raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
    features_df, x_raw = cached
    X, y, groups, _, _ = prepare_task(features_df, x_raw, [1, 2])  # binary: baseline vs stress

    out = {"task": "binary", "n_windows": int(len(y)), "models": []}
    for key in FEATURE_MODELS:
        name = CLF_NAMES[key]
        try:
            y_true, p1 = loso_pos_proba(key, X, y, groups)
        except Exception as e:  # missing OpenMP for xgb/lgbm, etc.
            out["models"].append({"model": name, "available": False, "reason": str(e)[:80]})
            print(f"  {name:20s} skipped ({str(e)[:40]})")
            continue
        row = {
            "model": name,
            "available": True,
            "auroc": float(roc_auc_score(y_true, p1)),
            "auprc": float(average_precision_score(y_true, p1)),
        }
        if key == POINT_MODEL:
            row["operating_point"] = operating_point(y_true, p1)
        out["models"].append(row)
        print(f"  {name:20s} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}")

    out["provenance"] = provenance()
    path = RESULTS_DIR / "threshold_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    run()
