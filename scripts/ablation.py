"""Ablate feature groups to test whether stress detection is physiology- or motion-driven."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.run_experiment import (
    FIGURES_DIR,
    RESULTS_DIR,
    build_pipeline,
    load_cached,
    loso_evaluate,
    prepare_task,
)

# Canonical feature-group prefixes; every feature column is named "<GROUP>_...".
# Subsets below are defined against these, so a prefix rename is caught by the
# coverage guard in run() instead of silently dropping features from a subset.
FEATURE_GROUPS = ["HRV", "EDA", "TEMP", "RESP", "ACC"]

# Feature-group subsets (by name prefix)
SUBSETS = {
    "All features": FEATURE_GROUPS,
    "No motion (HRV+EDA+TEMP+RESP)": ["HRV", "EDA", "TEMP", "RESP"],
    "Autonomic (HRV+EDA)": ["HRV", "EDA"],
    "HRV only": ["HRV"],
    "EDA only": ["EDA"],
    "Motion only (ACC)": ["ACC"],
}


def _group_of(col: str) -> str:
    return col.split("_")[0]


def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    cached = load_cached()
    if cached is None:
        raise SystemExit("No cached features. Run scripts/run_experiment.py first.")
    features_df, x_raw = cached

    X, y, groups, feature_cols, _ = prepare_task(features_df, x_raw, [1, 2])

    # Fail loudly if any feature is unaccounted for: the "All features" subset must
    # cover every column, or the ablation would silently compare the wrong sets.
    unknown = sorted({_group_of(c) for c in feature_cols} - set(FEATURE_GROUPS))
    if unknown:
        raise SystemExit(f"Feature groups not in FEATURE_GROUPS: {unknown}")

    rows = []
    for name, prefixes in SUBSETS.items():
        cols = [i for i, c in enumerate(feature_cols) if _group_of(c) in prefixes]
        res = loso_evaluate(lambda: build_pipeline("rf"), X[:, cols], y, groups)
        rows.append(
            {
                "subset": name,
                "n_features": len(cols),
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "f1_macro_mean": res["f1_macro_mean"],
            }
        )
        print(f"  {name:32s} ({len(cols):2d} feat)  acc={res['accuracy_mean']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    order = df.iloc[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh(order["subset"], order["accuracy_mean"], xerr=order["accuracy_std"], color="#3498db")
    plt.axvline(df.iloc[0]["accuracy_mean"], color="#e74c3c", ls="--", label="all features")
    plt.xlabel("LOSO accuracy (Random Forest, binary)")
    plt.title("Feature-group ablation")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation.png", dpi=150)
    plt.close()
    print(f"\nWrote {RESULTS_DIR / 'ablation.csv'} and ablation.png")


if __name__ == "__main__":
    run()
