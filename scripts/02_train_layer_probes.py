from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.probing import best_layer, load_state_file, train_layer_probes
from pre_refusal_signatures.reporting import write_metrics_csv, write_predictions_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train layer-wise linear probes.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    ids = state["ids"]
    results, probabilities_by_layer = train_layer_probes(X, y, args.cv_folds, args.seed)
    chosen_layer = best_layer(results)

    metrics_path = Path(args.reports_dir) / "layer_probe_metrics.csv"
    predictions_path = Path(args.outputs_dir) / "probe_predictions.csv"
    write_metrics_csv(results, metrics_path)
    write_predictions_csv(ids, y, probabilities_by_layer[chosen_layer], chosen_layer, predictions_path)
    best = [result for result in results if result.layer == chosen_layer][0]
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    print(f"best_layer={chosen_layer} f1={best.f1:.3f} accuracy={best.accuracy:.3f}")


if __name__ == "__main__":
    main()

