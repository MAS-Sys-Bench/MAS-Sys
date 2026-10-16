"""PinchBench workspace manager.

Prepares isolated workspace directories for each task, copying fixture files
and creating inline content as specified in the task definition.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.logger import logger

if TYPE_CHECKING:
    from experiments.PinchBench.task_loader import Task


def prepare_workspace(
    task: "Task",  # noqa: F821
    base_dir: Path,
    assets_dir: Path,
    run_id: Optional[str] = None,
) -> Path:
    """Prepare an isolated workspace directory for a task.

    Each task gets its own directory so that file operations are isolated.
    Workspace files declared in the task frontmatter are either:
    - Copied from assets_dir (when ``source`` key is present), or
    - Created inline (when ``path`` + ``content`` keys are present).

    Args:
        task: The Task object with workspace_files specification.
        base_dir: Parent directory under which workspace dirs are created.
        assets_dir: Directory containing shared fixture files.
        run_id: Optional run identifier for uniqueness.

    Returns:
        Path to the prepared workspace directory.
    """
    dir_name = task.task_id
    if run_id:
        dir_name = f"{run_id}_{dir_name}"
    workspace = base_dir / dir_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    for wf in task.workspace_files:
        if "source" in wf:
            # Copy file from assets directory
            src = assets_dir / wf["source"]
            dest_name = wf.get("dest", wf["source"])
            dest = workspace / dest_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dest)
                logger.debug(f"[PinchBench] Copied {src} -> {dest}")
            else:
                logger.warning(f"[PinchBench] Asset not found: {src}")
        elif "path" in wf and "content" in wf:
            # Create file with inline content
            dest = workspace / wf["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(wf["content"], encoding="utf-8")
            logger.debug(f"[PinchBench] Created inline file: {dest}")

    return workspace


def cleanup_workspace(workspace: Path) -> None:
    """Remove a task workspace directory.

    Args:
        workspace: Path to the workspace directory to remove.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
        logger.debug(f"[PinchBench] Cleaned up workspace: {workspace}")
