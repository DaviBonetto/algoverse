from __future__ import annotations

from pathlib import Path


def make_layer_curve(metrics, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = 0.5
    if "majority_baseline" in metrics:
        baseline = float(metrics["majority_baseline"].iloc[0])

    plt.figure(figsize=(9, 5))
    plt.plot(metrics["layer"], metrics["accuracy"], marker="o", label="Accuracy")
    plt.plot(metrics["layer"], metrics["f1"], marker="o", label="F1")
    plt.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Majority baseline")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("Layer-wise linear probe performance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_pca_plot(X_layer, y, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=42).fit_transform(X_layer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    for label, name, color in [(0, "benign", "#2a9d8f"), (1, "harmful-intent", "#d62828")]:
        mask = y == label
        plt.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.78, s=38, color=color)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Best-layer hidden states projected with PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_confusion_matrix(y, probabilities, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "harmful"],
        yticklabels=["benign", "harmful"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Best-layer probe confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

