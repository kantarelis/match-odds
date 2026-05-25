"""Proper scoring rules and a reliability helper for match-outcome probabilities.

The headline metrics are **log-loss** and **Brier score** (proper scoring rules); accuracy is
reported but is never the primary objective (CLAUDE.md -> ML / Modeling Discipline). Every model in
``matchodds.modeling`` emits probabilities in the fixed class order :data:`CLASSES` = ``("H", "D",
"A")``; these helpers assume that order and a matching integer encoding (``H -> 0``, ``D -> 1``,
``A -> 2``). They are pure and deterministic — no RNG.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

CLASSES: tuple[str, str, str] = ("H", "D", "A")
_CLASS_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(CLASSES)}

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def encode_labels(results: Iterable[str]) -> IntArray:
    """Map ``H`` / ``D`` / ``A`` outcome labels to their ``[H, D, A]`` indices (0 / 1 / 2)."""
    return np.asarray([_CLASS_TO_INDEX[str(result)] for result in results], dtype=np.intp)


def decode_labels(indices: Iterable[int]) -> list[str]:
    """Inverse of :func:`encode_labels`: indices back to ``H`` / ``D`` / ``A`` labels."""
    return [CLASSES[int(index)] for index in indices]


def _as_proba(proba: FloatArray) -> FloatArray:
    array = np.asarray(proba, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(CLASSES):
        raise ValueError(f"expected a (n, {len(CLASSES)}) probability matrix, got shape {array.shape}")
    return array


def _as_labels(y_true: IntArray) -> IntArray:
    return np.asarray(y_true, dtype=np.intp)


def log_loss(y_true: IntArray, proba: FloatArray, *, eps: float = 1e-15) -> float:
    """Multiclass log-loss (cross-entropy) of the true-class probabilities.

    Lower is better; 0 is perfect. Probabilities are clipped to ``[eps, 1]`` so a confident miss
    stays finite. Rows of ``proba`` are assumed to sum to 1 (every model guarantees this).
    """
    probs = _as_proba(proba)
    labels = _as_labels(y_true)
    true_class = np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.log(true_class).mean())


def brier_score(y_true: IntArray, proba: FloatArray) -> float:
    """Multiclass Brier score: mean squared error between the probability vector and the one-hot
    outcome. Ranges ``[0, 2]``; lower is better; 0 is perfect."""
    probs = _as_proba(proba)
    labels = _as_labels(y_true)
    onehot = np.eye(len(CLASSES), dtype=np.float64)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def accuracy(y_true: IntArray, proba: FloatArray) -> float:
    """Fraction of matches whose most-probable class is the true outcome (reported, not optimised)."""
    probs = _as_proba(proba)
    labels = _as_labels(y_true)
    return float((probs.argmax(axis=1) == labels).mean())


def score_summary(y_true: IntArray, proba: FloatArray) -> dict[str, float]:
    """Bundle the headline metrics for one set of predictions."""
    return {
        "log_loss": log_loss(y_true, proba),
        "brier": brier_score(y_true, proba),
        "accuracy": accuracy(y_true, proba),
    }


def reliability_curve(
    y_true: IntArray, proba: FloatArray, *, n_bins: int = 10
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Pooled one-vs-rest calibration curve for a reliability diagram.

    Every predicted class-probability (the full ``(n, 3)`` matrix flattened) is paired with whether
    that class was the realised outcome, then grouped into ``n_bins`` equal-width bins over
    ``[0, 1]``. Returns ``(mean_predicted, observed_frequency, count)`` per bin; empty bins are
    ``NaN`` with a count of 0. A perfectly calibrated model sits on the diagonal.
    """
    probs = _as_proba(proba)
    labels = _as_labels(y_true)
    predicted = probs.ravel()
    observed = np.eye(len(CLASSES), dtype=np.float64)[labels].ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, n_bins - 1)

    mean_predicted = np.full(n_bins, np.nan, dtype=np.float64)
    observed_frequency = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.intp)
    for current in range(n_bins):
        in_bin = bin_index == current
        size = int(in_bin.sum())
        counts[current] = size
        if size:
            mean_predicted[current] = predicted[in_bin].mean()
            observed_frequency[current] = observed[in_bin].mean()
    return mean_predicted, observed_frequency, counts
