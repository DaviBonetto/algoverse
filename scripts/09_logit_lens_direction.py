from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.analysis import REFUSAL_TOKENS, class_mean_direction
from pre_refusal_signatures.probing import load_state_file


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Project class-mean directions through the model unembedding.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--tokens", nargs="*", default=REFUSAL_TOKENS)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    model_name = args.model or str(state["model_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    lm_head = model.get_output_embeddings().weight.detach().cpu().numpy()

    token_rows = []
    token_ids = []
    for token in args.tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not encoded:
            continue
        token_id = encoded[0]
        token_ids.append((token, token_id))

    rows = []
    for layer in range(X.shape[1]):
        direction = class_mean_direction(X[:, layer, :], y)
        logits = lm_head @ direction
        refusal_values = []
        for token, token_id in token_ids:
            value = float(logits[token_id])
            refusal_values.append(value)
            token_rows.append({"layer": layer, "token": token, "token_id": token_id, "direction_logit": value})
        rows.append({"layer": layer, "mean_refusal_token_logit": float(np.mean(refusal_values)), "max_refusal_token_logit": float(np.max(refusal_values))})

    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    token_df = pd.DataFrame(token_rows)
    df.to_csv(reports_dir / "logit_lens_direction.csv", index=False)
    token_df.to_csv(reports_dir / "logit_lens_refusal_tokens.csv", index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["layer"], df["mean_refusal_token_logit"], marker="o", label="Mean refusal-token logit")
    plt.plot(df["layer"], df["max_refusal_token_logit"], marker="o", label="Max refusal-token logit")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Direction projected through lm_head")
    plt.title("Does the harmful-intent direction point toward refusal tokens?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "logit_lens_direction.png", dpi=180)
    plt.close()

    print(f"model={model_name}")
    print(f"tokens={token_ids}")
    print(f"saved={reports_dir / 'logit_lens_direction.csv'}")


if __name__ == "__main__":
    main()

