from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_data_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "DeepFundBench"
        / "data"
    )


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def filter_before_trading_date(
    items: list[dict[str, Any]],
    trading_date: str,
    key: str,
) -> list[dict[str, Any]]:
    cutoff = datetime.strptime(trading_date, "%Y-%m-%d")
    filtered = []
    for item in items:
        raw = item.get(key, "")
        if not raw:
            filtered.append(item)
            continue
        try:
            normalized = raw.replace("Z", "+00:00") if "Z" in raw else raw
            publish_time = datetime.fromisoformat(normalized).replace(tzinfo=None)
        except (TypeError, ValueError):
            try:
                publish_time = datetime.strptime(raw[:10], "%Y-%m-%d")
            except (TypeError, ValueError):
                filtered.append(item)
                continue
        if publish_time <= cutoff:
            filtered.append(item)
    return filtered
