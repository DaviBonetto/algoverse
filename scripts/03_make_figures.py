from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.plotting import make_confusion_matrix, make_layer_curve, make_pca_plot
from pre_refusal_signatures.probing import best_layer, load_state_file, train_layer_probes


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate probe figures and error analysis.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--metrics", default="reports/layer_probe_metrics.csv")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    ids = state["ids"]
    prompts = state["prompts"]
    metrics = pd.read_csv(args.metrics)
    majority_baseline = max(float((y == 0).mean()), float((y == 1).mean()))
    metrics["majority_baseline"] = majority_baseline

    results, probabilities_by_layer = train_layer_probes(X, y, cv_folds=5, random_seed=42)
    chosen_layer = best_layer(results)
    probabilities = probabilities_by_layer[chosen_layer]

    figures_dir = Path(args.figures_dir)
    reports_dir = Path(args.reports_dir)
    make_layer_curve(metrics, figures_dir / "layer_accuracy_curve.png")
    make_pca_plot(X[:, chosen_layer, :], y, figures_dir / "best_layer_pca.png")
    make_confusion_matrix(y, probabilities, figures_dir / "confusion_matrix.png")

    predictions = (probabilities >= 0.5).astype(int)
    false_positive = [i for i, (true, pred) in enumerate(zip(y, predictions)) if true == 0 and pred == 1]
    false_negative = [i for i, (true, pred) in enumerate(zip(y, predictions)) if true == 1 and pred == 0]
    lines = [
        "# Error Analysis",
        "",
        f"Best layer selected by F1: `{chosen_layer}`.",
        f"False positives: `{len(false_positive)}`.",
        f"False negatives: `{len(false_negative)}`.",
        "",
        "## False Positives",
    ]
    for idx in false_positive[:5]:
        lines.append(f"- `{ids[idx]}` p={probabilities[idx]:.3f}: {prompts[idx]}")
    lines.extend(["", "## False Negatives"])
    for idx in false_negative[:5]:
        lines.append(f"- `{ids[idx]}` p={probabilities[idx]:.3f}: {prompts[idx]}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "This analysis is diagnostic, not a deployment claim. Errors should be read as evidence about the limitations of small curated probing datasets and linear separability.",
        ]
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"best_layer={chosen_layer}")
    print(f"figures={figures_dir}")
    print(f"error_analysis={reports_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()

