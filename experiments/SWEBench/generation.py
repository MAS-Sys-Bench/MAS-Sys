"""Patch generation via MAS-Sys MAS pipeline for SWE-bench.

Bridges SWE-bench instances to the MAS execution API.
Uses PlanAgent to dynamically plan the MAS architecture for each instance,
then lets agents interactively explore the repo, edit code, and produce a fix.

Flow:
  instance → clone repo → PlanAgent.plan() → MASConfig
  → FlowFactory.from_config() → flow.execute(prompt) → agents edit files
  → git diff → model_patch
"""

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.config import config
from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.mas.plan_agent import PlanAgent
from app.tracer import tracer
from experiments.SWEBench.sandbox_env import resolve_sandbox_config_for_instance

from experiments.SWEBench.utils import (
    build_task_prompt,
    cleanup_instance_repo,
    get_diff,
    setup_instance_repo,
)


# ── Generation pipeline ─────────────────────────────────────────────────────


def _safe_instance_id(instance_id: str) -> str:
    return instance_id.replace("/", "__")


def _extract_patch_from_host_repo(repo_dir: Optional[str]) -> str:
    if not repo_dir or not os.path.exists(repo_dir):
        return ""
    return get_diff(repo_dir)


async def _extract_patch_from_sandbox(container_repo: Optional[str]) -> str:
    if not container_repo:
        return ""
    try:
        from app.sandbox.client import SANDBOX_CLIENT

        exit_code, output = await SANDBOX_CLIENT.exec_simple(
            (
                f"cd {container_repo} && "
                f"git -c safe.directory={container_repo} diff"
            ),
            timeout=30,
        )
        if exit_code == 0:
            return output
    except Exception:
        return ""
    return ""


def _copy_instance_artifacts(
    *,
    workdir: str,
    result_dir: Optional[str],
    instance_id: str,
) -> None:
    if not result_dir:
        return

    safe_id = _safe_instance_id(instance_id)
    src_trace = Path(workdir) / f"{safe_id}_trace.json"
    src_meta = Path(workdir) / f"{safe_id}_metadata.json"
    instance_dir = Path(result_dir) / safe_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    if src_trace.exists():
        shutil.copy2(src_trace, instance_dir / "trace.json")
    if src_meta.exists():
        shutil.copy2(src_meta, instance_dir / "metadata.json")

async def generate_one(
    instance: Dict,
    workdir: str,
    result_dir: Optional[str] = None,
    model_name: str = "mas-sys",
    timeout: int = 600,
    cleanup: bool = True,
    runtime: str = "legacy",
) -> Dict:
    """Generate a patch for one SWE-bench instance via PlanAgent.

    Args:
        instance: SWE-bench instance dict.
        workdir: Parent directory for repo working copies.
        model_name: Model identifier for the prediction output.
        timeout: Per-instance timeout in seconds.
        cleanup: Whether to remove the repo working copy after generation.

    Returns:
        Prediction dict with instance_id, model_name_or_path, model_patch.
    """
    instance_id = instance["instance_id"]
    safe_id = _safe_instance_id(instance_id)
    start_time = time.time()

    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": "",
        "generation_status": "started",
        "generation_error": None,
        "generation_error_type": None,
        "elapsed_seconds": None,
    }

    tracer.reset()
    tracer.start_e2e()

    repo_dir = None
    container_repo = None
    try:
        # Step 1: Setup repo
        logger.info(f"[SWE-bench] {instance_id}: Setting up repo...")
        repo_dir = setup_instance_repo(instance, workdir)
        logger.info(f"[SWE-bench] {instance_id}: Repo ready at {repo_dir}")

        runtime_sandbox_config = config.sandbox
        if config.sandbox.use_sandbox:
            runtime_sandbox_config = resolve_sandbox_config_for_instance(
                instance=instance,
                repo_dir=repo_dir,
                base_config=config.sandbox,
            )

        agent_repo_dir = repo_dir  # path the agent sees in the prompt
        container_repo = f"/workspace/{safe_id}"
        if config.sandbox.use_sandbox:
            # SWE-bench instance images prepare the checked-out repo at /testbed.
            # Do not mount the host checkout over it, or the official repo setup
            # performed during image build would be lost.
            container_repo = "/testbed"
            agent_repo_dir = container_repo

        task_prompt = build_task_prompt(instance, agent_repo_dir)

        # Step 2: PlanAgent decides topology
        plan_agent = PlanAgent(runtime=runtime)
        mas_config = await plan_agent.plan(task_prompt, runtime=runtime)
        logger.info(
            f"[SWE-bench] {instance_id}: PlanAgent chose "
            f"topology={mas_config.topology}, "
            f"agents={[a.instance_id for a in mas_config.agents]}"
        )

        # Step 3: Build and execute flow
        flow = FlowFactory.from_config(
            mas_config,
            plan_agent=plan_agent,
            runtime=runtime,
        )

        # When sandbox is enabled, mount the repo into the container so
        # tools (bash, str_replace_editor) can access it inside Docker.
        if config.sandbox.use_sandbox:
            from app.sandbox.client import SANDBOX_CLIENT

            await SANDBOX_CLIENT.create(
                config=runtime_sandbox_config,
                volume_bindings={},
            )
            logger.info(
                f"[SWE-bench] {instance_id}: Sandbox enabled — "
                f"repo prepared at {container_repo}; image={runtime_sandbox_config.image}"
            )

        await asyncio.wait_for(
            flow.execute(task_prompt),
            timeout=timeout,
        )

        # Step 4: Extract patch via git diff (primary) or from output (fallback)
        model_patch = _extract_patch_from_host_repo(repo_dir)

        if not model_patch.strip() and config.sandbox.use_sandbox:
            model_patch = await _extract_patch_from_sandbox(container_repo)
            if model_patch:
                logger.info(
                    f"[SWE-bench] {instance_id}: Host git diff empty, "
                    f"recovered patch from sandbox git diff ({len(model_patch)} chars)"
                )

        prediction["model_patch"] = model_patch
        prediction["generation_status"] = "success" if model_patch.strip() else "empty_patch"
        prediction["elapsed_seconds"] = round(time.time() - start_time, 2)

        elapsed = time.time() - start_time
        logger.info(
            f"[SWE-bench] {instance_id}: Done in {elapsed:.1f}s, "
            f"patch={len(model_patch)} chars"
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(
            f"[SWE-bench] {instance_id}: Timed out after {elapsed:.1f}s"
        )
        # Still try to get partial diff if agent made some edits
        prediction["model_patch"] = _extract_patch_from_host_repo(repo_dir)
        if not prediction["model_patch"].strip() and config.sandbox.use_sandbox:
            prediction["model_patch"] = await _extract_patch_from_sandbox(container_repo)
        prediction["generation_status"] = (
            "timeout_with_patch" if prediction["model_patch"].strip() else "timeout"
        )
        prediction["generation_error"] = f"Timed out after {elapsed:.1f}s"
        prediction["generation_error_type"] = "TimeoutError"
        prediction["elapsed_seconds"] = round(elapsed, 2)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[SWE-bench] {instance_id}: Failed after {elapsed:.1f}s — {e}"
        )
        prediction["model_patch"] = _extract_patch_from_host_repo(repo_dir)
        if not prediction["model_patch"].strip() and config.sandbox.use_sandbox:
            prediction["model_patch"] = await _extract_patch_from_sandbox(container_repo)
        prediction["generation_status"] = (
            "error_with_patch" if prediction["model_patch"].strip() else "error"
        )
        prediction["generation_error"] = str(e)
        prediction["generation_error_type"] = type(e).__name__
        prediction["elapsed_seconds"] = round(elapsed, 2)

    finally:
        tracer.end_e2e()
        tracer.save(workdir, prefix=f"{safe_id}_")
        _copy_instance_artifacts(
            workdir=workdir,
            result_dir=result_dir,
            instance_id=instance_id,
        )

        # Clean up sandbox container (recreated per-instance with volume bindings)
        if config.sandbox.use_sandbox:
            try:
                from app.sandbox.client import SANDBOX_CLIENT
                await SANDBOX_CLIENT.cleanup()
            except Exception:
                pass
        if cleanup and repo_dir and os.path.exists(repo_dir):
            cleanup_instance_repo(repo_dir)

    return prediction


async def generate_batch(
    instances: List[Dict],
    workdir: str,
    result_dir: Optional[str] = None,
    model_name: str = "mas-sys",
    timeout: int = 600,
    cleanup: bool = True,
    runtime: str = "legacy",
) -> List[Dict]:
    """Generate patches for multiple SWE-bench instances sequentially.

    Args:
        instances: List of SWE-bench instance dicts.
        workdir: Parent directory for repo working copies.
        model_name: Model identifier for predictions.
        timeout: Per-instance timeout in seconds.
        cleanup: Whether to remove repo copies after each instance.

    Returns:
        List of prediction dicts.
    """
    predictions = []
    total = len(instances)

    for idx, instance in enumerate(instances, 1):
        logger.info(
            f"[SWE-bench] [{idx}/{total}] Processing {instance['instance_id']}"
        )
        pred = await generate_one(
            instance=instance,
            workdir=workdir,
            result_dir=result_dir,
            model_name=model_name,
            timeout=timeout,
            cleanup=cleanup,
            runtime=runtime,
        )
        predictions.append(pred)

    return predictions
