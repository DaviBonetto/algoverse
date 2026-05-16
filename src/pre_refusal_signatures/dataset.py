from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALID_LABELS = {"harmful", "benign"}
REQUIRED_KEYS = {"id", "label", "category", "prompt"}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    label: str
    category: str
    prompt: str

    @property
    def target(self) -> int:
        return 1 if self.label == "harmful" else 0


class DatasetValidationError(ValueError):
    """Raised when the prompt JSONL file violates the expected schema."""


def _validate_row(row: dict, line_no: int) -> PromptRecord:
    missing = REQUIRED_KEYS - set(row)
    if missing:
        raise DatasetValidationError(f"Line {line_no}: missing keys {sorted(missing)}")

    extra = set(row) - REQUIRED_KEYS
    if extra:
        raise DatasetValidationError(f"Line {line_no}: unexpected keys {sorted(extra)}")

    values = {key: row[key] for key in REQUIRED_KEYS}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"Line {line_no}: {key} must be a non-empty string")

    label = row["label"].strip().lower()
    if label not in VALID_LABELS:
        raise DatasetValidationError(f"Line {line_no}: invalid label {row['label']!r}")

    return PromptRecord(
        id=row["id"].strip(),
        label=label,
        category=row["category"].strip(),
        prompt=row["prompt"].strip(),
    )


def load_prompts(path: str | Path, min_per_label: int = 1) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetValidationError(f"Line {line_no}: row must be a JSON object")
            record = _validate_row(row, line_no)
            if record.id in seen_ids:
                raise DatasetValidationError(f"Line {line_no}: duplicate id {record.id!r}")
            seen_ids.add(record.id)
            records.append(record)

    if not records:
        raise DatasetValidationError("Dataset is empty")

    counts = Counter(record.label for record in records)
    for label in sorted(VALID_LABELS):
        if counts[label] < min_per_label:
            raise DatasetValidationError(
                f"Need at least {min_per_label} {label} examples, found {counts[label]}"
            )
    return records


def summarize_records(records: Iterable[PromptRecord]) -> dict[str, Counter]:
    records = list(records)
    return {
        "labels": Counter(record.label for record in records),
        "categories": Counter(record.category for record in records),
    }

