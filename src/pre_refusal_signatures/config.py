from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        yaml = None

    with Path(path).open("r", encoding="utf-8") as handle:
        if yaml is not None:
            data = yaml.safe_load(handle) or {}
        else:
            data = {}
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition(":")
                value = value.strip()
                if value.lower() in {"true", "false"}:
                    parsed: Any = value.lower() == "true"
                else:
                    try:
                        parsed = int(value)
                    except ValueError:
                        try:
                            parsed = float(value)
                        except ValueError:
                            parsed = value
                data[key.strip()] = parsed
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return data


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
