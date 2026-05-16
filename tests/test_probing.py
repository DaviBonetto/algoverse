from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from pre_refusal_signatures.probing import best_layer, train_layer_probes


def test_train_layer_probes_finds_signal_layer():
    rng = np.random.default_rng(7)
    y = np.array([0, 1] * 12)
    X = rng.normal(size=(24, 4, 8)).astype("float32")
    X[:, 2, 0] += (2 * y - 1) * 3.0
    results, probabilities = train_layer_probes(X, y, cv_folds=4, random_seed=1)
    assert len(results) == 4
    assert best_layer(results) == 2
    assert probabilities[2].shape == (24,)

