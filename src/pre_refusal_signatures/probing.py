from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeResult:
    layer: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_splits: int


def load_state_file(path: str | Path):
    import numpy as np

    return np.load(Path(path), allow_pickle=True)


def _safe_cv_splits(y, requested: int) -> int:
    import numpy as np

    _, counts = np.unique(y, return_counts=True)
    return int(max(2, min(requested, counts.min())))


def train_layer_probes(X, y, cv_folds: int = 5, random_seed: int = 42):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = _safe_cv_splits(y, cv_folds)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    results: list[ProbeResult] = []
    layer_probabilities = {}

    for layer_idx in range(X.shape[1]):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed),
        )
        probabilities = cross_val_predict(
            probe,
            X[:, layer_idx, :],
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        try:
            roc_auc = float(roc_auc_score(y, probabilities))
        except ValueError:
            roc_auc = float("nan")

        results.append(
            ProbeResult(
                layer=layer_idx,
                accuracy=float(accuracy_score(y, predictions)),
                precision=float(precision_score(y, predictions, zero_division=0)),
                recall=float(recall_score(y, predictions, zero_division=0)),
                f1=float(f1_score(y, predictions, zero_division=0)),
                roc_auc=roc_auc,
                n_splits=n_splits,
            )
        )
        layer_probabilities[layer_idx] = probabilities

    return results, layer_probabilities


def best_layer(results: list[ProbeResult]) -> int:
    if not results:
        raise ValueError("No probe results available")
    return max(results, key=lambda result: (result.f1, result.accuracy)).layer


def fit_probe(X_layer, y, random_seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed),
    )
    return probe.fit(X_layer, y)

