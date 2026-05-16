from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardThresholds:
    flag_threshold: float
    review_threshold: float
    benign_centroid_distance_threshold: float


def centroid_distance(vector, centroid) -> float:
    import numpy as np

    return float(np.linalg.norm(vector - centroid))


def fit_guard_thresholds(X_layer, y, probabilities, harmful_quantile: float = 0.20, review_margin: float = 0.10):
    import numpy as np

    harmful_probs = probabilities[y == 1]
    if harmful_probs.size == 0:
        raise ValueError("Cannot fit guard without harmful examples")
    flag_threshold = float(np.quantile(harmful_probs, harmful_quantile))
    review_threshold = max(0.0, flag_threshold - review_margin)

    benign_vectors = X_layer[y == 0]
    if benign_vectors.size == 0:
        raise ValueError("Cannot fit centroid guard without benign examples")
    centroid = benign_vectors.mean(axis=0)
    benign_distances = np.linalg.norm(benign_vectors - centroid, axis=1)
    distance_threshold = float(np.quantile(benign_distances, 0.95))
    return GuardThresholds(flag_threshold, review_threshold, distance_threshold), centroid


def decide(harmful_probability: float, benign_distance: float, thresholds: GuardThresholds) -> str:
    if harmful_probability >= thresholds.flag_threshold:
        return "FLAG"
    if harmful_probability >= thresholds.review_threshold or benign_distance > thresholds.benign_centroid_distance_threshold:
        return "REVIEW"
    return "ALLOW"

