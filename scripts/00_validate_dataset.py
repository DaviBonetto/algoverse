from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_refusal_signatures.dataset import load_prompts, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the prompt JSONL dataset.")
    parser.add_argument("--data", default="data/prompts.jsonl")
    parser.add_argument("--min-per-label", type=int, default=1)
    args = parser.parse_args()

    records = load_prompts(Path(args.data), min_per_label=args.min_per_label)
    summary = summarize_records(records)
    print(f"total={len(records)}")
    print("labels=" + ", ".join(f"{k}:{v}" for k, v in sorted(summary["labels"].items())))
    print("categories=" + ", ".join(f"{k}:{v}" for k, v in sorted(summary["categories"].items())))
    print("families=" + ", ".join(f"{k}:{v}" for k, v in sorted(summary["families"].items())))
    print("difficulties=" + ", ".join(f"{k}:{v}" for k, v in sorted(summary["difficulties"].items())))


if __name__ == "__main__":
    main()
