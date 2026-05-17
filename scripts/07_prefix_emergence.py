from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.config import ensure_dirs, load_config
from pre_refusal_signatures.dataset import PromptRecord, load_prompts
from pre_refusal_signatures.extraction import create_synthetic_hidden_states, extract_hidden_states, select_balanced_records
from pre_refusal_signatures.probing import load_state_file, train_layer_probes


def truncate_prompt(prompt: str, fraction: float) -> str:
    words = prompt.split()
    if not words:
        return prompt
    keep = max(1, int(round(len(words) * fraction)))
    return " ".join(words[:keep])


def prefix_records(records: list[PromptRecord], fraction: float) -> list[PromptRecord]:
    return [replace(record, prompt=truncate_prompt(record.prompt, fraction)) for record in records]


def main() -> None:
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Measure when harmful-intent signal emerges over prompt prefixes and layers.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--fractions", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--outputs-dir", default="outputs/prefix")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = args.data or config.get("data_path", "data/prompts.jsonl")
    records = load_prompts(data_path, min_per_label=1)
    records = select_balanced_records(records, args.max_prompts)

    outputs_dir = Path(args.outputs_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dirs(outputs_dir, reports_dir, figures_dir)

    rows = []
    for fraction in args.fractions:
        frac_records = prefix_records(records, fraction)
        state_path = outputs_dir / f"hidden_states_prefix_{int(fraction * 100):03d}.npz"
        if args.synthetic:
            shape = create_synthetic_hidden_states(frac_records, state_path, seed=int(config.get("random_seed", 42)) + int(fraction * 100))
        else:
            shape = extract_hidden_states(
                frac_records,
                model_name=args.model or config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                output_path=state_path,
                max_length=int(config.get("max_length", 512)),
                device=args.device or config.get("device", "auto"),
                dtype=config.get("dtype", "auto"),
            )
        state = load_state_file(state_path)
        results, _ = train_layer_probes(state["X"], state["y"], int(config.get("cv_folds", 5)), int(config.get("random_seed", 42)))
        for result in results:
            rows.append(
                {
                    "prefix_fraction": fraction,
                    "prefix_percent": int(fraction * 100),
                    "layer": result.layer,
                    "accuracy": result.accuracy,
                    "f1": result.f1,
                    "roc_auc": result.roc_auc,
                    "shape": str(shape),
                }
            )
        print(f"fraction={fraction:.2f} shape={shape} saved={state_path}")

    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "prefix_emergence.csv", index=False)

    heatmap = df.pivot(index="prefix_percent", columns="layer", values="f1").sort_index()
    plt.figure(figsize=(12, 4.8))
    sns.heatmap(heatmap, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "F1"})
    plt.xlabel("Layer")
    plt.ylabel("Prompt prefix (%)")
    plt.title("Pre-refusal signal emergence over prompt time and model depth")
    plt.tight_layout()
    plt.savefig(figures_dir / "prefix_emergence_heatmap.png", dpi=180)
    plt.close()

    best_by_prefix = df.sort_values(["prefix_fraction", "f1", "accuracy"], ascending=[True, False, False]).groupby("prefix_fraction").head(1)
    best_by_prefix.to_csv(reports_dir / "prefix_best_layers.csv", index=False)
    print(f"saved={reports_dir / 'prefix_emergence.csv'}")
    print(f"figure={figures_dir / 'prefix_emergence_heatmap.png'}")


if __name__ == "__main__":
    main()

