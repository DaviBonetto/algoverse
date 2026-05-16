from __future__ import annotations

import pytest

from pre_refusal_signatures.dataset import DatasetValidationError, load_prompts


def write_jsonl(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_valid_dataset(tmp_path):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(
        path,
        [
            '{"id":"h1","label":"harmful","category":"misuse","prompt":"unsafe intent"}',
            '{"id":"b1","label":"benign","category":"math","prompt":"what is 2+2?"}',
        ],
    )
    records = load_prompts(path)
    assert [record.target for record in records] == [1, 0]


def test_missing_key_raises(tmp_path):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(path, ['{"id":"h1","label":"harmful","prompt":"x"}'])
    with pytest.raises(DatasetValidationError, match="missing keys"):
        load_prompts(path)


def test_invalid_label_raises(tmp_path):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(path, ['{"id":"h1","label":"risky","category":"x","prompt":"x"}'])
    with pytest.raises(DatasetValidationError, match="invalid label"):
        load_prompts(path)


def test_duplicate_id_raises(tmp_path):
    path = tmp_path / "prompts.jsonl"
    write_jsonl(
        path,
        [
            '{"id":"x","label":"harmful","category":"a","prompt":"x"}',
            '{"id":"x","label":"benign","category":"b","prompt":"y"}',
        ],
    )
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        load_prompts(path)

