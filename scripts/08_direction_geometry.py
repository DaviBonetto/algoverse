from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.analysis import class_mean_direction, get_state_array, projection_scores
from pre_refusal_signatures.probing import best_layer, load_state_file, train_layer_probes


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    parser = argparse.ArgumentParser(description="Analyze harmful-intent direction geometry.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    ids = state["ids"]
    pair_ids = get_state_array(state, "pair_ids", default="")
    results, _ = train_layer_probes(X, y, args.cv_folds, args.seed)
    chosen_layer = best_layer(results)

    rows = []
    cv = StratifiedKFold(n_splits=min(args.cv_folds, int(np.bincount(y).min())), shuffle=True, random_state=args.seed)
    for layer in range(X.shape[1]):
        global_dir = class_mean_direction(X[:, layer, :], y)
        projections = projection_scores(X[:, layer, :], global_dir)
        margin = float(projections[y == 1].mean() - projections[y == 0].mean())

        fold_dirs = []
        for train_idx, _ in cv.split(X[:, layer, :], y):
            fold_dirs.append(class_mean_direction(X[train_idx, layer, :], y[train_idx]))
        fold_dirs = np.stack(fold_dirs)
        cosine_matrix = fold_dirs @ fold_dirs.T
        upper = cosine_matrix[np.triu_indices_from(cosine_matrix, k=1)]

        rows.append(
            {
                "layer": layer,
                "projection_margin": margin,
                "direction_stability_mean_cosine": float(upper.mean()),
                "direction_stability_min_cosine": float(upper.min()),
                "is_best_probe_layer": layer == chosen_layer,
            }
        )

    pair_rows = []
    for pair_id in sorted(set(pair_ids)):
        if not pair_id:
            continue
        idx = np.where(pair_ids == pair_id)[0]
        if len(idx) != 2 or len(set(y[idx])) != 2:
            continue
        layer = chosen_layer
        direction = class_mean_direction(X[:, layer, :], y)
        proj = projection_scores(X[idx, layer, :], direction)
        harmful_proj = float(proj[y[idx] == 1][0])
        benign_proj = float(proj[y[idx] == 0][0])
        pair_rows.append(
            {
                "pair_id": pair_id,
                "harmful_id": str(ids[idx][y[idx] == 1][0]),
                "benign_id": str(ids[idx][y[idx] == 0][0]),
                "layer": layer,
                "harmful_projection": harmful_proj,
                "benign_projection": benign_proj,
                "pair_margin": harmful_proj - benign_proj,
            }
        )

    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    df.to_csv(reports_dir / "direction_geometry.csv", index=False)
    pair_df.to_csv(reports_dir / "counterfactual_pair_margins.csv", index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["layer"], df["projection_margin"], marker="o")
    axes[0].axvline(chosen_layer, color="red", linestyle="--", label=f"best layer {chosen_layer}")
    axes[0].set_title("Harmful-benign direction margin")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Projection margin")
    axes[0].legend()
    axes[1].plot(df["layer"], df["direction_stability_mean_cosine"], marker="o", color="#2a9d8f")
    axes[1].axvline(chosen_layer, color="red", linestyle="--")
    axes[1].set_ylim(-1, 1.05)
    axes[1].set_title("Direction stability across CV folds")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean cosine")
    plt.tight_layout()
    plt.savefig(figures_dir / "direction_geometry.png", dpi=180)
    plt.close()

    if not pair_df.empty:
        plt.figure(figsize=(10, 4.5))
        sorted_pairs = pair_df.sort_values("pair_margin")
        plt.bar(sorted_pairs["pair_id"], sorted_pairs["pair_margin"], color="#6a4c93")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Harmful projection - benign projection")
        plt.title(f"Counterfactual pair margins at layer {chosen_layer}")
        plt.tight_layout()
        plt.savefig(figures_dir / "counterfactual_pair_margins.png", dpi=180)
        plt.close()

    print(f"best_probe_layer={chosen_layer}")
    print(f"saved={reports_dir / 'direction_geometry.csv'}")
    print(f"pairs={reports_dir / 'counterfactual_pair_margins.csv'}")


if __name__ == "__main__":
    main()

