from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.analysis import get_state_array, metric_dict
from pre_refusal_signatures.probing import fit_probe, load_state_file


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import LeaveOneGroupOut

    parser = argparse.ArgumentParser(description="Evaluate hidden-state probes with family-heldout splits.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    groups = get_state_array(state, "families", fallback_key="categories", default="unknown")

    rows = []
    logo = LeaveOneGroupOut()
    for layer in range(X.shape[1]):
        all_probs = np.full(len(y), np.nan)
        valid_groups = []
        for train_idx, test_idx in logo.split(X[:, layer, :], y, groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            probe = fit_probe(X[train_idx, layer, :], y[train_idx], args.seed)
            all_probs[test_idx] = probe.predict_proba(X[test_idx, layer, :])[:, 1]
            valid_groups.append(str(groups[test_idx][0]))
        mask = ~np.isnan(all_probs)
        if mask.sum() and len(np.unique(y[mask])) == 2:
            rows.append({"layer": layer, "heldout_groups": len(set(valid_groups)), "n_eval": int(mask.sum()), **metric_dict(y[mask], all_probs[mask])})

    df = pd.DataFrame(rows)
    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / "family_heldout_results.csv", index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["layer"], df["f1"], marker="o", label="F1")
    plt.plot(df["layer"], df["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Family-heldout generalization by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "family_heldout_curve.png", dpi=180)
    plt.close()

    best = df.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    print(f"best_layer={int(best.layer)} f1={best.f1:.3f} roc_auc={best.roc_auc:.3f}")
    print(f"saved={reports_dir / 'family_heldout_results.csv'}")


if __name__ == "__main__":
    main()

