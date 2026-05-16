from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.extraction import extract_hidden_states
from pre_refusal_signatures.guard import centroid_distance, decide, fit_guard_thresholds
from pre_refusal_signatures.probing import best_layer, fit_probe, load_state_file, train_layer_probes


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description="Run a proof-of-concept early guard.")
    parser.add_argument("--states", default="outputs/hidden_states.npz")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    state = load_state_file(args.states)
    X = state["X"]
    y = state["y"]
    model_name = args.model or str(state["model_name"])
    results, probabilities_by_layer = train_layer_probes(X, y, cv_folds=5, random_seed=42)
    chosen_layer = best_layer(results)
    probe = fit_probe(X[:, chosen_layer, :], y)
    thresholds, benign_centroid = fit_guard_thresholds(
        X[:, chosen_layer, :],
        y,
        probabilities_by_layer[chosen_layer],
    )

    if model_name == "synthetic-control":
        seed = abs(hash(args.prompt)) % (2**32)
        rng = np.random.default_rng(seed)
        prompt_state = benign_centroid + rng.normal(0, 0.15, size=benign_centroid.shape)
        backend = "synthetic-control"
    else:
        from pre_refusal_signatures.dataset import PromptRecord

        temp_path = Path("outputs") / "_guard_prompt_hidden_states.npz"
        record = PromptRecord(id="guard_prompt", label="benign", category="demo", prompt=args.prompt)
        extract_hidden_states([record], model_name=model_name, output_path=temp_path, device=args.device)
        prompt_state = load_state_file(temp_path)["X"][0, chosen_layer, :]
        backend = "model-forward-pass"
    harmful_probability = float(probe.predict_proba(prompt_state.reshape(1, -1))[0, 1])
    distance = centroid_distance(prompt_state, benign_centroid)
    decision = decide(harmful_probability, distance, thresholds)

    print(f"model={model_name}")
    print(f"backend={backend}")
    print(f"layer={chosen_layer}")
    print(f"harmful_probability={harmful_probability:.3f}")
    print(f"benign_centroid_distance={distance:.3f}")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
