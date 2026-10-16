"""PinchBench task loader.

Parses Markdown task files with YAML frontmatter into structured Task objects.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TaskSession:
    """A single session within a multi-session task."""

    id: str
    prompt: str
    new_session: bool = False


@dataclass
class Task:
    """A PinchBench benchmark task."""

    task_id: str
    name: str
    category: str
    grading_type: str  # "automated", "llm_judge", "hybrid"
    timeout_seconds: int
    workspace_files: List[Dict[str, str]]
    prompt: str
    expected_behavior: str
    grading_criteria: List[str]
    grading_weights: Optional[Dict[str, float]] = None
    llm_judge_rubric: Optional[str] = None
    file_path: Optional[Path] = None
    sessions: Optional[List[TaskSession]] = None

    @property
    def is_multi_session(self) -> bool:
        return self.sessions is not None and len(self.sessions) > 0


def _split_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split YAML frontmatter from Markdown body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    return frontmatter, body


def _extract_section(body: str, heading: str) -> str:
    """Extract content under a ## heading until the next ## heading."""
    pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_grading_criteria(section: str) -> List[str]:
    """Parse checklist items from grading criteria section."""
    return re.findall(r"- \[[ x]\]\s+(.*)", section)


def _extract_llm_judge_rubric(body: str) -> Optional[str]:
    """Extract LLM Judge Rubric section if present."""
    rubric = _extract_section(body, "LLM Judge Rubric")
    return rubric if rubric else None


def load_task(filepath: Path) -> Task:
    """Load a single task from a Markdown file.

    Args:
        filepath: Path to the task Markdown file.

    Returns:
        Parsed Task object.
    """
    text = filepath.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)

    # Parse sessions for multi-session tasks
    sessions = None
    if "sessions" in fm:
        sessions = [
            TaskSession(
                id=s["id"],
                prompt=s["prompt"].strip(),
                new_session=s.get("new_session", False),
            )
            for s in fm["sessions"]
        ]

    return Task(
        task_id=fm.get("id", filepath.stem),
        name=fm.get("name", filepath.stem),
        category=fm.get("category", "unknown"),
        grading_type=fm.get("grading_type", "automated"),
        timeout_seconds=fm.get("timeout_seconds", 300),
        workspace_files=fm.get("workspace_files", []),
        prompt=_extract_section(body, "Prompt"),
        expected_behavior=_extract_section(body, "Expected Behavior"),
        grading_criteria=_extract_grading_criteria(
            _extract_section(body, "Grading Criteria")
        ),
        grading_weights=fm.get("grading_weights"),
        llm_judge_rubric=_extract_llm_judge_rubric(body),
        file_path=filepath,
        sessions=sessions,
    )


def load_all_tasks(
    tasks_dir: Path,
    task_ids: Optional[List[str]] = None,
) -> List[Task]:
    """Load all tasks from data directory.

    Args:
        tasks_dir: Directory containing task_*.md files.
        task_ids: If given, only load tasks with matching IDs.

    Returns:
        Sorted list of Task objects.
    """
    tasks = []
    for md_file in sorted(tasks_dir.glob("task_*.md")):
        if md_file.name == "TASK_TEMPLATE.md":
            continue
        task = load_task(md_file)
        if task_ids and task.task_id not in task_ids:
            continue
        tasks.append(task)
    return tasks
