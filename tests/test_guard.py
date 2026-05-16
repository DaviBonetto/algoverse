from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")

from pre_refusal_signatures.guard import decide, fit_guard_thresholds


def test_guard_decision_thresholds():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.2, 2.0]])
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.80, 0.90])
    thresholds, centroid = fit_guard_thresholds(X, y, probabilities, harmful_quantile=0.0)
    assert centroid.shape == (2,)
    assert decide(0.85, 0.0, thresholds) == "FLAG"
    assert decide(0.01, thresholds.benign_centroid_distance_threshold + 1.0, thresholds) == "REVIEW"

