"""Unit tests for the proper-scoring-rule metrics and the reliability helper."""

import math

import numpy as np
import pytest

from matchodds.modeling import metrics


def test_class_order_and_label_roundtrip():
    assert metrics.CLASSES == ("H", "D", "A")
    encoded = metrics.encode_labels(["H", "D", "A", "H"])
    assert list(encoded) == [0, 1, 2, 0]
    assert metrics.decode_labels([0, 1, 2]) == ["H", "D", "A"]
    # Round-trip a full cycle.
    assert metrics.decode_labels(metrics.encode_labels(["A", "H", "D"])) == ["A", "H", "D"]


def test_log_loss_matches_hand_computed():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true = np.array([0, 1])
    expected = -(math.log(0.7) + math.log(0.8)) / 2
    assert metrics.log_loss(y_true, proba) == pytest.approx(expected)


def test_brier_matches_hand_computed():
    proba = np.array([[0.7, 0.2, 0.1]])
    y_true = np.array([0])
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01
    assert metrics.brier_score(y_true, proba) == pytest.approx(0.14)


def test_perfect_predictions_score_zero():
    proba = np.eye(3)
    y_true = np.array([0, 1, 2])
    assert metrics.log_loss(y_true, proba) == pytest.approx(0.0)
    assert metrics.brier_score(y_true, proba) == pytest.approx(0.0)
    assert metrics.accuracy(y_true, proba) == pytest.approx(1.0)


def test_log_loss_clips_confident_miss_to_finite():
    proba = np.array([[0.0, 1.0, 0.0]])
    y_true = np.array([0])
    value = metrics.log_loss(y_true, proba)
    assert math.isfinite(value) and value > 0.0


def test_accuracy_counts_argmax():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    y_true = np.array([0, 0, 1])  # argmax is [0, 2, 1] -> 2 of 3 correct
    assert metrics.accuracy(y_true, proba) == pytest.approx(2 / 3)


def test_score_summary_bundles_the_three_metrics():
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    y_true = np.array([0, 1])
    summary = metrics.score_summary(y_true, proba)
    assert set(summary) == {"log_loss", "brier", "accuracy"}
    assert summary["log_loss"] == pytest.approx(metrics.log_loss(y_true, proba))
    assert summary["brier"] == pytest.approx(metrics.brier_score(y_true, proba))
    assert summary["accuracy"] == pytest.approx(metrics.accuracy(y_true, proba))


def test_reliability_curve_bins_deterministically():
    proba = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]])
    y_true = np.array([0, 2])
    # Flattened predicted = [0.9, 0.1, 0.0, 0.2, 0.3, 0.5]; observed = [1, 0, 0, 0, 0, 1].
    # With 2 bins split at 0.5: bin0 = {0.1, 0.0, 0.2, 0.3}, bin1 = {0.9, 0.5}.
    mean_predicted, observed_frequency, counts = metrics.reliability_curve(y_true, proba, n_bins=2)
    assert list(counts) == [4, 2]
    assert mean_predicted == pytest.approx([0.15, 0.7])
    assert observed_frequency == pytest.approx([0.0, 1.0])
    assert int(counts.sum()) == proba.size


def test_reliability_curve_marks_empty_bins_nan():
    proba = np.array([[0.1, 0.1, 0.8]])
    y_true = np.array([2])
    mean_predicted, observed_frequency, counts = metrics.reliability_curve(y_true, proba, n_bins=10)
    assert counts[0] == 0
    assert math.isnan(mean_predicted[0]) and math.isnan(observed_frequency[0])
    assert int(counts.sum()) == proba.size


def test_metrics_reject_wrong_shaped_probabilities():
    with pytest.raises(ValueError):
        metrics.log_loss(np.array([0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        metrics.brier_score(np.array([0]), np.array([[0.5, 0.5]]))
