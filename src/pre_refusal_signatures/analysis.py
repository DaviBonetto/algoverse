from __future__ import annotations

from pathlib import Path


REFUSAL_TOKENS = ["Sorry", "sorry", "cannot", "can't", "unable", "refuse", "I"]


def state_metadata(state) -> dict[str, object]:
    import numpy as np

    y = state["y"]
    n = int(len(y))
    return {
        "n": n,
        "n_layers": int(state["X"].shape[1]),
        "hidden_dim": int(state["X"].shape[2]),
        "n_benign": int((y == 0).sum()),
        "n_harmful": int((y == 1).sum()),
        "model_name": str(state["model_name"]),
        "has_families": "families" in state.files,
        "has_pair_ids": "pair_ids" in state.files,
    }


def get_state_array(state, key: str, fallback_key: str | None = None, default: str = ""):
    import numpy as np

    if key in state.files:
        return state[key]
    if fallback_key and fallback_key in state.files:
        return state[fallback_key]
    return np.array([default] * len(state["y"]))


def metric_dict(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    predictions = (probabilities >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def normalize_rows(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def class_mean_direction(X_layer, y):
    import numpy as np

    harmful = X_layer[y == 1]
    benign = X_layer[y == 0]
    if len(harmful) == 0 or len(benign) == 0:
        raise ValueError("Both classes are required to compute a class-mean direction.")
    direction = harmful.mean(axis=0) - benign.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def projection_scores(X_layer, direction):
    return X_layer @ direction


def write_json(data: dict, path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

