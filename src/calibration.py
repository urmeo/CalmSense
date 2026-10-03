"""Calibration metrics for classifier probabilities (Guo et al., 2017)."""

from copy import deepcopy
from typing import Dict, List, Union

import numpy as np

Array = Union[np.ndarray, list]
BINARY_BRIER_DEFINITION = "positive_class_mse"


def _validated_predictions(y: Array, proba: Array) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    if y.ndim != 1 or not len(y):
        raise ValueError("Calibration requires a nonempty one-dimensional label array")
    if proba.ndim not in (1, 2) or len(proba) != len(y):
        raise ValueError("Probabilities must have one row per label")
    if proba.ndim == 2 and proba.shape[1] < 2:
        raise ValueError("Probability matrices require at least two class columns")
    if not np.isfinite(proba).all() or np.any((proba < 0) | (proba > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    n_classes = 2 if proba.ndim == 1 else proba.shape[1]
    if not np.isin(y, np.arange(n_classes)).all():
        raise ValueError("Labels must be zero-based class indices matching the probabilities")
    if proba.ndim == 2 and not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("Class probabilities must sum to one in every row")
    return y.astype(np.intp, copy=False), proba


def _confidence_correct(y: Array, proba: Array):
    """Use confidence in the predicted class, paired with prediction correctness."""
    y, proba = _validated_predictions(y, proba)
    if proba.ndim == 1:
        conf = np.maximum(proba, 1.0 - proba)
        # Match argmax on [P(0), P(1)]: a tie chooses the first class.
        pred = (proba > 0.5).astype(int)
    else:
        conf = proba.max(axis=1)
        pred = proba.argmax(axis=1)
    return conf, (pred == y).astype(float)


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)


def reliability_curve(y: Array, proba: Array, n_bins: int = 15) -> List[Dict[str, float]]:
    """Return confidence, accuracy, and count for nonempty equal-width bins."""
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def expected_calibration_error(y: Array, proba: Array, n_bins: int = 15) -> float:
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    n = len(conf)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def maximum_calibration_error(y: Array, proba: Array, n_bins: int = 15) -> float:
    conf, correct = _confidence_correct(y, proba)
    idx = _bin_index(conf, n_bins)
    gaps = [
        abs(correct[idx == b].mean() - conf[idx == b].mean())
        for b in range(n_bins)
        if (idx == b).any()
    ]
    return float(max(gaps)) if gaps else 0.0


def brier_score(y: Array, proba: Array) -> float:
    """Binary positive-class MSE, independent of vector/two-column representation.

    For three or more classes, return the summed multiclass Brier score.
    """
    y, proba = _validated_predictions(y, proba)
    if proba.ndim == 1:
        return float(np.mean((proba - y) ** 2))
    if proba.shape[1] == 2:
        return float(np.mean((proba[:, 1] - y) ** 2))
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def normalize_binary_calibration(result: dict) -> dict:
    """Copy binary Brier values onto the positive-class MSE scale.

    Missing ``brier_definition`` means ``two_class_sum``; halve scores and paired gaps.
    Preserve ECE, p-values, timestamps, and the source dictionary.
    """
    out = deepcopy(result)
    definition = out.get("brier_definition", "two_class_sum")
    if definition == BINARY_BRIER_DEFINITION:
        return out
    if definition != "two_class_sum":
        raise ValueError(f"Unknown binary Brier definition: {definition}")
    # Binary class errors are equal, so their sum is twice the positive-class MSE.
    for key in (
        "loso",
        "loso_matched",
        "within_subject",
        "recalibrated_isotonic",
        "recalibrated_sigmoid",
    ):
        if key in out:
            out[key]["brier"] /= 2.0
    gap = out.get("gap_significance")
    if gap:
        gap["mean_brier_gap"] /= 2.0
        gap["ci95"] = [v / 2.0 for v in gap["ci95"]]
        for row in gap.get("per_subject", {}).values():
            row["loso"] /= 2.0
            row["within"] /= 2.0
    out["brier_definition"] = BINARY_BRIER_DEFINITION
    out["brier_rescaled_from"] = "two_class_sum"
    return out


def summary(y: Array, proba: Array, n_bins: int = 15) -> Dict[str, object]:
    return {
        "ece": expected_calibration_error(y, proba, n_bins),
        "mce": maximum_calibration_error(y, proba, n_bins),
        "brier": brier_score(y, proba),
        "reliability": reliability_curve(y, proba, n_bins),
    }


def net_benefit(y: Array, p_pos: Array, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit at each probability threshold."""
    y, p_pos = _validated_predictions(y, p_pos)
    if p_pos.ndim != 1:
        raise ValueError("net_benefit requires a positive-class probability vector")
    thresholds = np.asarray(thresholds, dtype=float)
    if (
        thresholds.ndim != 1
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError("Thresholds must be a finite vector between zero and one")
    n = len(y)
    out = []
    for pt in thresholds:
        if pt >= 1.0:
            out.append(0.0)  # Boundary convention avoids infinite false-positive weight at 1.
            continue
        flagged = p_pos >= pt
        tp = np.sum(flagged & (y == 1))
        fp = np.sum(flagged & (y == 0))
        out.append(tp / n - (fp / n) * (pt / (1.0 - pt)))
    return np.array(out)
