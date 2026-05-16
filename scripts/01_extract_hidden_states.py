from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.config import ensure_dirs, load_config
from pre_refusal_signatures.dataset import load_prompts
from pre_refusal_signatures.extraction import (
    create_synthetic_hidden_states,
    extract_hidden_states,
    select_balanced_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer-wise hidden-state vectors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic states for pipeline smoke tests.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))
    ensure_dirs(output_dir)
    records = load_prompts(config.get("data_path", "data/prompts.jsonl"), min_per_label=1)
    records = select_balanced_records(records, args.max_prompts)
    output_path = Path(args.output) if args.output else output_dir / "hidden_states.npz"

    if args.synthetic:
        shape = create_synthetic_hidden_states(
            records,
            output_path,
            seed=int(config.get("random_seed", 42)),
        )
        print(f"saved={output_path}")
        print(f"shape={shape}")
        print("backend=synthetic-control")
        return

    shape = extract_hidden_states(
        records,
        model_name=args.model or config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        output_path=output_path,
        max_length=int(config.get("max_length", 512)),
        device=args.device or config.get("device", "auto"),
        dtype=config.get("dtype", "auto"),
    )
    print(f"saved={output_path}")
    print(f"shape={shape}")


if __name__ == "__main__":
    main()
