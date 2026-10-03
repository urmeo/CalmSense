"""Reproduce the full CalmSense LOSO benchmark from raw WESAD data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DEMO_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from src.dataset import WindowedDataset, load_cached
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION
from src.models.ml.classifiers import get_classifier
from src.utils import provenance, save_verified_joblib, set_seed

TASKS = {
    "binary": {"keep": [1, 2], "names": ["baseline", "stress"]},
    "multiclass": {"keep": [1, 2, 3], "names": ["baseline", "stress", "amusement"]},
}

CLASSIFIERS = ["lr", "rf", "xgb", "lgbm"]
BENCHMARK_PROTOCOL_VERSION = 2
CLF_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}


def build_pipeline(clf_key: str) -> Pipeline:
    """Keep imputation and scaling inside the estimator fitted for each CV fold."""
    estimator = get_classifier(clf_key)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ]
    )


def _fit_params(pipe, y_train):
    """Balance XGBoost (others balance via class_weight)."""
    if pipe.named_steps["clf"].__class__.__name__ == "XGBClassifier":
        return {"clf__sample_weight": compute_sample_weight("balanced", y_train)}
    return {}


def loso_evaluate(pipeline_factory, X, y, groups):
    """Fit preprocessing on training subjects and predict the held-out subject."""
    logo = LeaveOneGroupOut()
    classes = np.unique(y)
    pooled_true, pooled_pred = [], []
    per_subject = []

    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx], **_fit_params(pipe, y[train_idx]))
        pred = pipe.predict(X[test_idx])

        pooled_true.extend(y[test_idx])
        pooled_pred.extend(pred)
        per_subject.append(
            {
                "subject": groups[test_idx][0],
                "n": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(
                    y[test_idx], pred, labels=classes, average="macro", zero_division=0
                ),
            }
        )

    pooled_true = np.array(pooled_true)
    pooled_pred = np.array(pooled_pred)
    subj_df = pd.DataFrame(per_subject)
    # Subject means weight people equally; pooled metrics weight their window counts.
    return {
        "accuracy_mean": float(subj_df["accuracy"].mean()),
        "accuracy_std": float(subj_df["accuracy"].std()),
        "f1_macro_mean": float(subj_df["f1_macro"].mean()),
        "f1_macro_std": float(subj_df["f1_macro"].std()),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "pooled_accuracy": float(accuracy_score(pooled_true, pooled_pred)),
        "per_subject": subj_df,
        "y_true": pooled_true,
        "y_pred": pooled_pred,
        "classes": classes,
    }


def cnn_loso(x_raw, y, groups):
    from src.models.dl.cnn_1d import CNN1DClassifier

    logo = LeaveOneGroupOut()
    pooled_true, pooled_pred, per_subject = [], [], []
    n_folds = len(np.unique(groups))

    for fold, (train_idx, test_idx) in enumerate(logo.split(x_raw, y, groups), 1):
        print(f"    1D-CNN fold {fold}/{n_folds}", flush=True)
        model = CNN1DClassifier(in_channels=x_raw.shape[1], random_state=SEED)
        model.fit(x_raw[train_idx], y[train_idx], groups=groups[train_idx])
        pred = model.predict(x_raw[test_idx])
        pooled_true.extend(y[test_idx])
        pooled_pred.extend(pred)
        per_subject.append(
            {
                "subject": groups[test_idx][0],
                "n": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(
                    y[test_idx], pred, labels=np.unique(y), average="macro", zero_division=0
                ),
            }
        )

    subj_df = pd.DataFrame(per_subject)
    return {
        "accuracy_mean": float(subj_df["accuracy"].mean()),
        "accuracy_std": float(subj_df["accuracy"].std()),
        "f1_macro_mean": float(subj_df["f1_macro"].mean()),
        "f1_macro_std": float(subj_df["f1_macro"].std()),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
        "pooled_accuracy": float(accuracy_score(pooled_true, pooled_pred)),
        "per_subject": subj_df,
        "y_true": np.array(pooled_true),
        "y_pred": np.array(pooled_pred),
        "classes": np.unique(y),
    }


def nonoverlap_mask(groups):
    """Keep alternating chronological windows per subject at the default 50% overlap."""
    keep = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        keep[idx[::2]] = True
    return keep


def kfold_accuracy(pipeline_factory, X, y, groups) -> float:
    """Subject-mixed 5-fold pooled accuracy for the optimism gap.

    Keep alternating windows to remove direct signal overlap; the same subjects
    can still appear in training and validation folds.
    """
    keep = nonoverlap_mask(groups)
    Xk, yk = X[keep], y[keep]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pooled_true, pooled_pred = [], []
    for train_idx, test_idx in skf.split(Xk, yk):
        pipe = pipeline_factory()
        pipe.fit(Xk[train_idx], yk[train_idx], **_fit_params(pipe, yk[train_idx]))
        pooled_true.extend(yk[test_idx])
        pooled_pred.extend(pipe.predict(Xk[test_idx]))
    return float(accuracy_score(pooled_true, pooled_pred))


def plot_confusion(result, names, title, path):
    cm = confusion_matrix(
        result["y_true"], result["y_pred"], labels=result["classes"], normalize="true"
    )
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(rows, path):
    df = pd.DataFrame(rows).sort_values("accuracy_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(df["model"], df["accuracy_mean"], xerr=df["accuracy_std"], color="#3498db")
    plt.xlabel("LOSO accuracy")
    plt.title("Subject-independent model comparison")
    plt.xlim(0, 1)
    for i, (acc, _) in enumerate(zip(df["accuracy_mean"], df["accuracy_std"])):
        plt.text(acc + 0.01, i, f"{acc:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_subject(subj_df, path):
    df = subj_df.sort_values("accuracy")
    plt.figure(figsize=(8, 4))
    plt.bar(df["subject"], df["accuracy"], color="#2ecc71")
    plt.axhline(df["accuracy"].mean(), color="#e74c3c", ls="--", label="mean")
    plt.ylabel("LOSO accuracy")
    plt.xlabel("Held-out subject")
    plt.title("Per-subject performance (best model)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gap(loso_acc, kfold_acc, path):
    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(
        ["LOSO\n(subject-independent)", "5-fold\n(within-subject)"],
        [loso_acc, kfold_acc],
        color=["#3498db", "#e67e22"],
    )
    for b, v in zip(bars, [loso_acc, kfold_acc]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Optimism gap: {(kfold_acc - loso_acc) * 100:.1f} pts")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_embedding(X, y, names, path):
    """Fit a descriptive PCA on all windows; it is not used to score classifiers."""
    from sklearn.decomposition import PCA

    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    plt.figure(figsize=(6, 5))
    for cls, name in enumerate(names):
        m = y == np.unique(y)[cls]
        plt.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.5, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Feature space (PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def shap_analysis(X, y, feature_names, fig_dir):
    """Explain an XGBoost fit on all available windows, separate from LOSO scoring."""
    import shap

    pipe = build_pipeline("xgb")
    pipe.fit(X, y, **_fit_params(pipe, y))
    Xt = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X))
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[-1]  # last (positive) class; for binary values[-1] == values[1]
    shap_vals = np.asarray(values)
    if shap_vals.ndim == 3:  # (samples, features, classes)
        shap_vals = shap_vals[:, :, -1]

    shap.summary_plot(shap_vals, Xt, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    arr = np.abs(shap_vals).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": arr})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    return importance


def prepare_task(features_df, x_raw, keep):
    if len(x_raw) != len(features_df) or len(set(keep)) != len(keep) or not keep:
        raise ValueError(
            "Task labels must be distinct and raw windows must align with feature rows"
        )
    mask = features_df["label"].isin(keep).to_numpy()
    sub = features_df[mask].reset_index(drop=True)
    if sub.empty:
        raise ValueError("No feature windows match the task labels")
    meta = ["subject_id", "window_id", "label", "label_name"]
    feature_cols = [c for c in sub.columns if c not in meta]
    if not feature_cols or sub["subject_id"].isna().any():
        raise ValueError("Task windows require features and nonmissing subject identifiers")
    X = sub[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    # Omit columns with no observed values; median imputation remains inside each fold.
    keep_cols = ~np.isnan(X).all(axis=0)
    X = X[:, keep_cols]
    feature_cols = [c for c, k in zip(feature_cols, keep_cols) if k]
    if not feature_cols:
        raise ValueError("Task contains no finite feature values")
    # 0-indexed labels in `keep` order
    remap = {label: i for i, label in enumerate(keep)}
    y = sub["label"].map(remap).to_numpy()
    groups = sub["subject_id"].to_numpy()
    return X, y, groups, feature_cols, x_raw[mask]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="run on generated data, no WESAD")
    args = parser.parse_args()
    set_seed(SEED)

    # Synthetic runs share a separate output root and never overwrite real-WESAD results.
    results_dir = DEMO_DIR / "results" if args.synthetic else RESULTS_DIR
    figures_dir = DEMO_DIR / "figures" if args.synthetic else FIGURES_DIR
    models_dir = DEMO_DIR / "models" if args.synthetic else MODELS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from src.synthetic import features as synth_features

        print("Building synthetic dataset (demo only)...")
        # Never cache: synthetic features must not overwrite a real WESAD cache.
        features_df, x_raw, _ = synth_features(n_subjects=8, block_sec=150, cache=False)
    else:
        cached = None if (args.rebuild or args.subjects) else load_cached()
        if cached is None:
            print("Building dataset from raw WESAD...")
            features_df, x_raw, _ = WindowedDataset().build(subjects=args.subjects)
        else:
            print("Loaded cached dataset.")
            features_df, x_raw = cached

    print(f"Windows: {len(features_df)} | subjects: {features_df['subject_id'].nunique()}")
    summary = {}

    for task, cfg in TASKS.items():
        print(f"\n=== Task: {task} ===")
        X, y, groups, feature_cols, x_raw_task = prepare_task(features_df, x_raw, cfg["keep"])
        rows, results = [], {}

        for key in CLASSIFIERS:
            res = loso_evaluate(lambda k=key: build_pipeline(k), X, y, groups)
            results[key] = res
            rows.append(
                {
                    "model": CLF_NAMES[key],
                    "accuracy_mean": res["accuracy_mean"],
                    "accuracy_std": res["accuracy_std"],
                    "f1_macro_mean": res["f1_macro_mean"],
                    "balanced_accuracy": res["balanced_accuracy"],
                }
            )
            print(
                f"  {CLF_NAMES[key]:20s} acc={res['accuracy_mean']:.3f} f1={res['f1_macro_mean']:.3f}"
            )

        if not args.no_cnn:
            cnn_res = cnn_loso(x_raw_task, y, groups)
            results["cnn"] = cnn_res
            rows.append(
                {
                    "model": "1D-CNN",
                    "accuracy_mean": cnn_res["accuracy_mean"],
                    "accuracy_std": cnn_res["accuracy_std"],
                    "f1_macro_mean": cnn_res["f1_macro_mean"],
                    "balanced_accuracy": cnn_res["balanced_accuracy"],
                }
            )
            print(
                f"  {'1D-CNN':20s} acc={cnn_res['accuracy_mean']:.3f} f1={cnn_res['f1_macro_mean']:.3f}"
            )

        best_key = max(rows, key=lambda r: r["accuracy_mean"])["model"]
        best = max(results.items(), key=lambda kv: kv[1]["accuracy_mean"])

        plot_model_comparison(rows, figures_dir / f"{task}_model_comparison.png")
        plot_confusion(
            best[1], cfg["names"], f"{task} ({best_key})", figures_dir / f"{task}_confusion.png"
        )
        plot_per_subject(best[1]["per_subject"], figures_dir / f"{task}_per_subject.png")
        plot_embedding(X, y, cfg["names"], figures_dir / f"{task}_pca.png")

        if best[0] in CLASSIFIERS:
            # Same non-overlapping windows on both bars: only the CV scheme differs.
            m = nonoverlap_mask(groups)
            gap_factory = lambda k=best[0]: build_pipeline(k)  # noqa: E731
            loso_matched = loso_evaluate(gap_factory, X[m], y[m], groups[m])["pooled_accuracy"]
            kf_acc = kfold_accuracy(gap_factory, X, y, groups)
            plot_gap(loso_matched, kf_acc, figures_dir / f"{task}_optimism_gap.png")
        else:
            loso_matched = None
            kf_acc = None

        pd.DataFrame(rows).to_csv(results_dir / f"{task}_model_comparison.csv", index=False)
        best[1]["per_subject"].to_csv(results_dir / f"{task}_per_subject.csv", index=False)

        summary[task] = {
            "n_windows": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
            "n_features": int(X.shape[1]),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "classes": cfg["names"],
            "models": rows,
            "best_model": best_key,
            "loso_accuracy": best[1]["accuracy_mean"],
            "loso_pooled_accuracy": best[1]["pooled_accuracy"],
            "loso_matched_accuracy": loso_matched,
            "within_subject_accuracy": kf_acc,
            "optimism_gap_pts": (
                round((kf_acc - loso_matched) * 100, 1)
                if kf_acc is not None and loso_matched is not None
                else None
            ),
            "per_subject": best[1]["per_subject"].to_dict("records"),
        }

        # Refit for inference after evaluation; this full-data fit supplies no LOSO scores.
        if task == "binary":
            top_clf = max(
                [(k, results[k]) for k in CLASSIFIERS],
                key=lambda kv: kv[1]["accuracy_mean"],
            )[0]
            importance = shap_analysis(X, y, feature_cols, figures_dir)
            importance.to_csv(results_dir / "shap_top_features.csv", index=False)
            final = build_pipeline(top_clf)
            final.fit(X, y, **_fit_params(final, y))
            save_verified_joblib(
                {
                    "pipeline": final,
                    "features": feature_cols,
                    "classes": cfg["names"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                },
                models_dir / "stress_classifier.joblib",
            )
            print(f"  Saved inference model ({CLF_NAMES[top_clf]}) + SHAP.")

    summary["provenance"] = provenance()
    summary["benchmark_protocol_version"] = BENCHMARK_PROTOCOL_VERSION
    summary["methodology"] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "cnn_validation": None if args.no_cnn else "training_subject_holdout",
        "cnn_normalization": None if args.no_cnn else "inner_training_windows_only",
        "f1_classes": "full_task_class_set",
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {results_dir}")
    if not args.synthetic:
        print("Run scripts/build_dashboard_data.py to refresh the dashboard.")


if __name__ == "__main__":
    run()
