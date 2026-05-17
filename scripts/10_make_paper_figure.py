from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Build a multi-panel paper-style summary figure.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    layer = pd.read_csv(reports / "layer_probe_metrics.csv")
    baselines = pd.read_csv(reports / "baseline_comparison.csv")
    heldout = pd.read_csv(reports / "family_heldout_results.csv")
    geometry = pd.read_csv(reports / "direction_geometry.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(layer["layer"], layer["f1"], marker="o", label="F1")
    ax.plot(layer["layer"], layer["roc_auc"], marker="o", label="ROC-AUC")
    ax.set_title("A. Layer-wise hidden-state probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[0, 1]
    labels = baselines["baseline"].str.replace("_", "\n")
    ax.bar(labels, baselines["f1"], color="#457b9d")
    ax.set_title("B. Hidden states vs controls")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[1, 0]
    ax.plot(heldout["layer"], heldout["f1"], marker="o", label="F1")
    ax.plot(heldout["layer"], heldout["roc_auc"], marker="o", label="ROC-AUC")
    ax.set_title("C. Family-heldout generalization")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(geometry["layer"], geometry["projection_margin"], marker="o", label="Projection margin")
    twin = ax.twinx()
    twin.plot(geometry["layer"], geometry["direction_stability_mean_cosine"], marker="o", color="#e76f51", label="Direction stability")
    ax.set_title("D. Direction geometry")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Class-mean projection margin")
    twin.set_ylabel("Mean fold-direction cosine")
    twin.set_ylim(-1, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle("Pre-Refusal Geometry Atlas", fontsize=16)
    plt.tight_layout()
    plt.savefig(figures / "paper_summary_figure.png", dpi=200)
    plt.close()
    print(f"saved={figures / 'paper_summary_figure.png'}")


if __name__ == "__main__":
    main()

