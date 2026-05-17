from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.analysis import metric_dict
from pre_refusal_signatures.probing import best_layer, load_state_file, train_layer_probes


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description="Run text and control baselines.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    prompts = state["prompts"]
    cv = StratifiedKFold(n_splits=min(args.cv_folds, int(np.bincount(y).min())), shuffle=True, random_state=args.seed)

    rows = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_probs = cross_val_predict(dummy, np.zeros((len(y), 1)), y, cv=cv, method="predict_proba")
    rows.append({"baseline": "majority_class", "layer": -1, **metric_dict(y, dummy_probs[:, 1] if dummy_probs.shape[1] > 1 else np.zeros_like(y))})

    length_feature = np.array([[len(prompt), len(prompt.split())] for prompt in prompts], dtype="float32")
    length_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed))
    length_probs = cross_val_predict(length_model, length_feature, y, cv=cv, method="predict_proba")[:, 1]
    rows.append({"baseline": "prompt_length_only", "layer": -1, **metric_dict(y, length_probs)})

    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    tfidf_probs = cross_val_predict(tfidf_model, prompts, y, cv=cv, method="predict_proba")[:, 1]
    rows.append({"baseline": "tfidf_prompt_text", "layer": -1, **metric_dict(y, tfidf_probs)})

    results, layer_probs = train_layer_probes(X, y, args.cv_folds, args.seed)
    chosen_layer = best_layer(results)
    rows.append({"baseline": "best_hidden_state_probe", "layer": chosen_layer, **metric_dict(y, layer_probs[chosen_layer])})

    final_layer = X.shape[1] - 1
    rows.append({"baseline": "final_layer_probe", "layer": final_layer, **metric_dict(y, layer_probs[final_layer])})

    rng = np.random.default_rng(args.seed)
    shuffled_y = rng.permutation(y)
    _, shuffled_layer_probs = train_layer_probes(X, shuffled_y, args.cv_folds, args.seed)
    shuffled_probs = shuffled_layer_probs[chosen_layer]
    rows.append({"baseline": "shuffled_label_control", "layer": chosen_layer, **metric_dict(y, shuffled_probs)})

    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "baseline_comparison.csv", index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    labels = df["baseline"].str.replace("_", "\n")
    plt.bar(labels, df["f1"], color=["#999999", "#8ecae6", "#219ebc", "#d62828", "#f77f00", "#6c757d"])
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.title("Probe performance vs text/control baselines")
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_comparison.png", dpi=180)
    plt.close()

    print(df.to_string(index=False))
    print(f"saved={reports_dir / 'baseline_comparison.csv'}")
    print(f"figure={figures_dir / 'baseline_comparison.png'}")


if __name__ == "__main__":
    main()

