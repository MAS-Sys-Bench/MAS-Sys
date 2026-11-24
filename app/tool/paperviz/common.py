from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_paperviz_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "experiments" / "PaperBananaBench" / "data"


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_task_dir(task_name: str) -> Path:
    if task_name not in {"diagram", "plot"}:
        raise ValueError(f"Unsupported PaperViz task_name '{task_name}'.")
    return resolve_paperviz_data_dir() / task_name
