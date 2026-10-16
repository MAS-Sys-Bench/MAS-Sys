"""PinchBench generation pipeline.

Executes PinchBench tasks through the MAS-Sys MAS system using PlanAgent
to dynamically plan the multi-agent topology for each task.

Flow:
  task → prepare workspace → PlanAgent.plan() → MASConfig
  → FlowFactory.from_config() → flow.execute(prompt) → collect transcript
"""

import asyncio
import base64
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import config
from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.mas.plan_agent import PlanAgent
from app.sandbox.client import SANDBOX_CLIENT
from app.tracer import tracer

from experiments.PinchBench.task_loader import Task
from experiments.PinchBench.transcript_adapter import (
    collect_all_messages,
    convert_messages_to_transcript,
)
from experiments.PinchBench.workspace_manager import (
    cleanup_workspace,
    prepare_workspace,
)

_THIS_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _THIS_DIR / "assets"


def build_pinchbench_task_prompt(task: Task) -> str:
    """Return the original PinchBench task prompt."""
    return task.prompt


def _persist_raw_output(workspace: Path, raw_output: str) -> None:
    (workspace / "raw_output.json").write_text(raw_output, encoding="utf-8")


def _materialize_image_output(task: Task, workspace: Path, raw_output: str) -> None:
    try:
        payload = json.loads(raw_output)
    except Exception:
        return

    if payload.get("mode") != "image":
        return

    image_b64 = payload.get("image_base64")
    if not image_b64:
        return

    match = re.search(r'Save it as "([^"]+)"', task.prompt)
    filename = match.group(1) if match else "generated_image.png"
    target = workspace / filename
    try:
        target.write_bytes(base64.b64decode(image_b64))
    except Exception as exc:
        logger.warning(
            f"[PinchBench] {task.task_id}: Failed to materialize image output at {target}: {exc}"
        )


def _sync_sandbox_artefacts(task_id: str, workspace: Path) -> None:
    """Copy files created by the agent from the sandbox temp dir to *workspace*.

    After ``Agent.run()`` finishes it calls ``SANDBOX_CLIENT.cleanup()`` which
    removes the Docker container but leaves the host temp directory (the bind
    mount backing ``/workspace``) intact.  The agent writes files under
    ``/workspace/<task_id>/`` inside the container which maps to
    ``<temp_dir>/<task_id>/`` on the host.

    Because tools may recreate the sandbox (losing volume-bindings), we scan
    all ``sandbox_workspace_*`` temp directories for the most recently modified
    one that contains a ``<task_id>`` subdirectory.
    """
    import tempfile

    temp_root = Path(tempfile.gettempdir())
    candidates = sorted(
        (
            d / task_id
            for d in temp_root.glob("sandbox_workspace_*")
            if (d / task_id).is_dir() and any((d / task_id).iterdir())
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        logger.warning(
            f"[PinchBench] {task_id}: No sandbox temp dir with artefacts found"
        )
        return

    src = candidates[0]  # most recently modified
    copied = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied += 1

    logger.info(
        f"[PinchBench] {task_id}: Copied {copied} file(s) from {src} "
        f"to {workspace}"
    )


async def generate_one(
    task: Task,
    workdir: Path,
    timeout: Optional[int] = None,
    cleanup: bool = True,
    run_id: Optional[str] = None,
    runtime: str = "legacy",
) -> Dict[str, Any]:
    """Execute one PinchBench task via PlanAgent.

    Args:
        task: PinchBench Task object.
        workdir: Parent directory for workspace directories.
        timeout: Per-task timeout in seconds (overrides task default).
        cleanup: Whether to remove workspace after execution.
        run_id: Optional run identifier for workspace uniqueness.

    Returns:
        Result dict with task_id, transcript, workspace_path, and metadata.
    """
    task_timeout = timeout or task.timeout_seconds
    start_time = time.time()

    result: Dict[str, Any] = {
        "task_id": task.task_id,
        "task_name": task.name,
        "transcript": [],
        "workspace_path": "",
        "raw_output": "",
        "error": None,
        "elapsed_seconds": 0.0,
    }

    tracer.reset()
    tracer.start_e2e()

    workspace = None
    original_workspace_root = config.workspace_root
    try:
        # Step 1: Prepare workspace
        workspace = prepare_workspace(
            task=task,
            base_dir=workdir,
            assets_dir=_ASSETS_DIR,
            run_id=run_id,
        )
        result["workspace_path"] = str(workspace)
        logger.info(
            f"[PinchBench] {task.task_id}: Workspace ready at {workspace}"
        )

        # Step 2: PlanAgent decides topology
        task_prompt = build_pinchbench_task_prompt(task)
        plan_agent = PlanAgent(runtime=runtime)
        mas_config = await plan_agent.plan(task_prompt, runtime=runtime)
        logger.info(
            f"[PinchBench] {task.task_id}: PlanAgent chose "
            f"topology={mas_config.topology}, "
            f"agents={[a.instance_id for a in mas_config.agents]}"
        )

        agent_workspace = str(workspace)  # path the agent sees in prompt
        use_sandbox = bool(config.sandbox and config.sandbox.use_sandbox)
        if use_sandbox:
            agent_workspace = f"/workspace/{task.task_id}"

        # Point agent prompts/tools at the active task workspace instead of the
        # process-global default workspace directory.
        config.set_workspace_root(Path(agent_workspace), create=not use_sandbox)

        # Step 3: Build and execute flow
        flow = FlowFactory.from_config(
            mas_config,
            plan_agent=plan_agent,
            runtime=runtime,
        )

        sandbox_sync_mode = "none"
        # PinchBench workspaces are seeded via copy_to because bind mounts are
        # unreliable in our current Docker setup. The sandbox client now
        # remembers copied paths and replays them automatically if a tool
        # lazily recreates the container mid-task.
        if use_sandbox:
            SANDBOX_CLIENT._volume_bindings = None
            SANDBOX_CLIENT._copied_paths = []
            await SANDBOX_CLIENT.create(config=config.sandbox)
            await SANDBOX_CLIENT.copy_to(str(workspace), agent_workspace)
            sandbox_sync_mode = "copy"
            logger.info(
                f"[PinchBench] {task.task_id}: Sandbox enabled — "
                f"copied {workspace} -> {agent_workspace}"
            )

        # Use the same task prompt for planning and execution; only prepend
        # the concrete workspace path that the execution agents must use.
        execution_prompt = (
            f"Your workspace directory is: {agent_workspace}\n"
            f"All task files are in this directory. Work within this directory.\n\n"
            f"{task_prompt}"
        )

        raw_output = await asyncio.wait_for(
            flow.execute(execution_prompt),
            timeout=task_timeout,
        )
        result["raw_output"] = raw_output
        _persist_raw_output(workspace, raw_output)
        _materialize_image_output(task, workspace, raw_output)

        # Copy artefacts from sandbox temp dir to host workspace.
        # Agent.run() already called SANDBOX_CLIENT.cleanup() which
        # destroyed the container but left the temp directory intact.
        if use_sandbox and sandbox_sync_mode == "copy":
            _sync_sandbox_artefacts(task.task_id, workspace)

        # Step 4: Collect transcript
        messages = collect_all_messages(flow)
        transcript = convert_messages_to_transcript(messages)
        result["transcript"] = transcript

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 2)
        logger.info(
            f"[PinchBench] {task.task_id}: Done in {elapsed:.1f}s, "
            f"transcript={len(transcript)} events"
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 2)
        result["error"] = f"Timed out after {elapsed:.1f}s"
        logger.warning(
            f"[PinchBench] {task.task_id}: Timed out after {elapsed:.1f}s"
        )
        # Still try to recover artefacts on timeout
        if use_sandbox and sandbox_sync_mode == "copy":
            _sync_sandbox_artefacts(task.task_id, workspace)

    except Exception as e:
        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 2)
        result["error"] = str(e)
        logger.error(
            f"[PinchBench] {task.task_id}: Failed after {elapsed:.1f}s — {e}"
        )

    finally:
        config.set_workspace_root(original_workspace_root, create=True)
        tracer.end_e2e()
        _trace_dir = workspace if workspace and workspace.exists() else workdir
        tracer.save(str(_trace_dir), prefix="")

        if use_sandbox:
            try:
                await SANDBOX_CLIENT.cleanup()
            except Exception as exc:
                logger.warning(
                    f"[PinchBench] {task.task_id}: sandbox cleanup failed: {exc}"
                )

        if cleanup and workspace and workspace.exists():
            # Don't clean up yet if we need the workspace for evaluation
            pass

    return result


async def generate_batch(
    tasks: List[Task],
    workdir: Path,
    timeout: Optional[int] = None,
    cleanup: bool = True,
    run_id: Optional[str] = None,
    runtime: str = "legacy",
) -> List[Dict[str, Any]]:
    """Execute multiple PinchBench tasks sequentially.

    Args:
        tasks: List of Task objects to execute.
        workdir: Parent directory for workspace directories.
        timeout: Per-task timeout override in seconds.
        cleanup: Whether to remove workspaces after execution.
        run_id: Optional run identifier.

    Returns:
        List of result dicts from generate_one().
    """
    results = []
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        logger.info(
            f"[PinchBench] [{idx}/{total}] Processing {task.task_id}: {task.name}"
        )
        result = await generate_one(
            task=task,
            workdir=workdir,
            timeout=timeout,
            cleanup=cleanup,
            run_id=run_id,
            runtime=runtime,
        )
        results.append(result)

    return results


def write_results(path: str, results: List[Dict[str, Any]]) -> None:
    """Write generation results to a pretty-printed JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def read_results(path: str) -> List[Dict[str, Any]]:
    """Read generation results from a JSON file.

    Also supports legacy JSONL format (one JSON object per line) for
    backward compatibility.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # Try JSON array first, fall back to JSONL
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]
