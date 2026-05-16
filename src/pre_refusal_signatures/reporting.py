from __future__ import annotations

from dataclasses import asdict
from pathlib import Path


def write_metrics_csv(results, path: str | Path) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(result) for result in results]).to_csv(path, index=False)


def write_predictions_csv(ids, y, probabilities, layer: int, path: str | Path) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ids,
            "target": y,
            "harmful_probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
            "layer": layer,
        }
    ).to_csv(path, index=False)

