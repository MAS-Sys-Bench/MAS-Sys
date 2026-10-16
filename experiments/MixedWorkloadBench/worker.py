from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_EXPERIMENTS_ROOT = _PROJECT_ROOT / "experiments"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))


PINCH_TASKS_DIR = _PROJECT_ROOT / "experiments" / "PinchBench" / "tasks"
DEEPFUND_DATA_DIR = _PROJECT_ROOT / "experiments" / "DeepFundBench" / "data"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_swebench_replay_mas_config(
    job_spec: Dict[str, Any],
    mas_config_data: Dict[str, Any],
) -> bool:
    """Align old replay plans with the SWE verifier's fixed output contract."""
    if job_spec.get("benchmark") != "swebench":
        return False

    agents = mas_config_data.get("agents")
    fsm = mas_config_data.get("fsm")
    if not isinstance(agents, list) or not isinstance(fsm, dict):
        return False

    verifier_ids = {
        agent.get("instance_id")
        for agent in agents
        if isinstance(agent, dict) and agent.get("agent_key") == "swe_verifier"
    }
    verifier_ids.discard(None)
    if not verifier_ids:
        return False

    changed = False
    decision_nodes = fsm.get("decision_nodes")
    if not isinstance(decision_nodes, list):
        return False
    for node in decision_nodes:
        if not isinstance(node, dict) or node.get("node") not in verifier_ids:
            continue
        if node.get("decision_field") != "decision":
            node["decision_field"] = "decision"
            changed = True
    return changed


def _save_deadline_debug_snapshot(
    job_spec: Dict[str, Any],
    start_time: float,
    signum: int | None = None,
) -> None:
    job_dir = Path(job_spec["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    signal_payload = {
        "job_id": job_spec.get("job_id"),
        "benchmark": job_spec.get("benchmark"),
        "logical_id": job_spec.get("logical_id"),
        "signal": signum,
        "pid": os.getpid(),
        "received_time": time.time(),
        "elapsed_seconds": round(time.time() - start_time, 4),
        "reason": "job deadline termination requested",
    }

    try:
        from app.tracer import tracer

        try:
            tracer.end_e2e()
        except Exception as exc:  # pragma: no cover - best effort debug capture
            errors.append(f"tracer.end_e2e failed: {exc}")
        try:
            tracer.save(job_dir, prefix="deadline_debug_")
        except Exception as exc:  # pragma: no cover - best effort debug capture
            errors.append(f"tracer.save failed: {exc}")
    except Exception as exc:  # pragma: no cover - best effort debug capture
        errors.append(f"tracer import failed: {exc}")

    if errors:
        signal_payload["errors"] = errors
    _write_json(job_dir / "deadline_debug_signal.json", signal_payload)

    result_path = job_dir / "result.json"
    if not result_path.exists():
        _write_json(
            result_path,
            {
                "job_id": job_spec.get("job_id"),
                "benchmark": job_spec.get("benchmark"),
                "logical_id": job_spec.get("logical_id"),
                "status": "failed",
                "error": "job deadline termination requested",
                "elapsed_seconds": signal_payload["elapsed_seconds"],
                "payload": {},
            },
        )


def _cleanup_current_sandbox_sync() -> None:
    """Best-effort cleanup for the process-global Docker sandbox.

    Deadline termination may interrupt an async benchmark before its normal
    ``finally`` block completes. Remove the active container by ID first so a
    leaked sandbox cannot keep a veth attached to docker0.
    """
    try:
        from app.sandbox.client import SANDBOX_CLIENT

        sandbox = getattr(SANDBOX_CLIENT, "sandbox", None)
        container = getattr(sandbox, "container", None)
        container_id = getattr(container, "id", None)
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", str(container_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        SANDBOX_CLIENT.sandbox = None
    except Exception:
        pass


def _cleanup_current_sandbox_async() -> None:
    try:
        from app.sandbox.client import SANDBOX_CLIENT

        asyncio.run(SANDBOX_CLIENT.cleanup())
    except Exception:
        _cleanup_current_sandbox_sync()


def _install_deadline_debug_handler(
    job_spec: Dict[str, Any],
    start_time: float,
) -> None:
    state = {"handled": False}

    def _handle_deadline(signum: int, _frame: Any) -> None:
        if not state["handled"]:
            state["handled"] = True
            _save_deadline_debug_snapshot(job_spec, start_time, signum=signum)
            _cleanup_current_sandbox_sync()
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, _handle_deadline)


@contextmanager
def _replay_plan_agent(job_spec: Dict[str, Any]) -> Iterator[None]:
    replay_plans = list(job_spec.get("replay_plans") or [])
    if not replay_plans:
        yield
        return

    from app.mas.plan_agent import PlanAgent
    from app.mas.schema import MASConfig
    from app.tracer import tracer

    original_plan = PlanAgent.plan
    state = {"index": 0}

    async def replay_plan(self: PlanAgent, task: str, runtime: str | None = None) -> MASConfig:
        del runtime
        index = state["index"]
        if index >= len(replay_plans):
            raise RuntimeError(
                f"Replay PlanAgent was called more times than prepared for {job_spec['job_id']}"
            )

        record = replay_plans[index]
        state["index"] = index + 1
        expected_hash = record.get("task_prompt_sha256")
        if expected_hash and expected_hash != _sha256_text(task):
            raise RuntimeError(
                "Replay PlanAgent task prompt mismatch for "
                f"{job_spec['job_id']} plan #{index + 1}: expected {expected_hash}, "
                f"got {_sha256_text(task)}"
            )

        sleep_s = float(record.get("planning_duration_s") or 0.0)
        start_time = time.time()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        end_time = time.time()

        mas_config_data = record.get("mas_config")
        if not isinstance(mas_config_data, dict):
            raise RuntimeError(
                f"Replay PlanAgent record for {job_spec['job_id']} does not contain mas_config"
            )
        replay_normalized = _normalize_swebench_replay_mas_config(
            job_spec,
            mas_config_data,
        )
        output_raw = record.get("raw_output")
        if not isinstance(output_raw, str) or not output_raw.strip():
            output_raw = json.dumps(mas_config_data, ensure_ascii=False)
        elif replay_normalized:
            output_raw = json.dumps(mas_config_data, ensure_ascii=False)
        self.last_raw_output = output_raw
        self.last_plan_start_time = start_time
        self.last_plan_end_time = end_time
        self.last_plan_duration_s = round(end_time - start_time, 4)
        self.last_input_tokens = 0
        self.last_completion_tokens = 0

        tracer.record_planning(
            input_text=task,
            output_raw=output_raw,
            start_time=start_time,
            end_time=end_time,
            input_tokens=0,
            completion_tokens=0,
        )
        return MASConfig(**mas_config_data)

    PlanAgent.plan = replay_plan
    try:
        yield
        if state["index"] != len(replay_plans):
            raise RuntimeError(
                f"Replay PlanAgent consumed {state['index']} of {len(replay_plans)} "
                f"prepared plans for {job_spec['job_id']}"
            )
    finally:
        PlanAgent.plan = original_plan


@contextmanager
def _llm_transport_context(runtime: str) -> Iterator[None]:
    previous = os.environ.get("MAS_LLM_TRANSPORT")
    if runtime == "orla":
        os.environ["MAS_LLM_TRANSPORT"] = "orla"
    else:
        os.environ.pop("MAS_LLM_TRANSPORT", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MAS_LLM_TRANSPORT", None)
        else:
            os.environ["MAS_LLM_TRANSPORT"] = previous


def _run_context(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    return job_spec["run_context"]


def _remaining_job_seconds(job_spec: Dict[str, Any], floor_seconds: int = 1) -> int:
    run_context = _run_context(job_spec)
    deadline = float(
        run_context.get("job_deadline_time")
        or run_context.get("experiment_deadline_time")
    )
    remaining = int(deadline - time.time())
    return max(floor_seconds, remaining)


def _run_research_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    from app.tracer import tracer
    from experiments.ResearchBench.generation.proposal_writing import write_proposal

    payload = job_spec["payload"]
    entry = payload["paper_entry"]
    paper_id = payload["paper_id"]
    job_dir = Path(job_spec["job_dir"])

    session_id = f"mixed:research:{job_spec['job_id']}"
    tracer.activate_session(session_id, reset=True)
    tracer.start_e2e()

    try:
        paper_data = entry["paper_data"]
        author_data = entry["author_data"]
        ref_proposal = paper_data["evaluation_format"]["content"]
        profiles = list(author_data.values())
        ref_abstracts = [ref["abstract"] for ref in paper_data.get("references", [])]
        gen_proposal = write_proposal(
            profiles,
            ref_abstracts,
            runtime=job_spec.get("runtime", "legacy"),
        )
        result = {
            "status": "success",
            "benchmark": "research_bench",
            "logical_id": paper_id,
            "paper_id": paper_id,
            "result_payload": {
                "paper_id": paper_id,
                "ref_proposal": ref_proposal,
                "gen_proposal": gen_proposal,
            },
        }
    finally:
        tracer.end_e2e()
        tracer.save(job_dir)
        tracer.clear_session(session_id)

    return result


def _run_pinch_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    from experiments.PinchBench.generation import generate_one as generate_pinch_one
    from experiments.PinchBench.task_loader import load_all_tasks

    payload = job_spec["payload"]
    job_dir = Path(job_spec["job_dir"])
    timeout = _remaining_job_seconds(job_spec)
    task_id = payload["task_id"]

    task_map = {task.task_id: task for task in load_all_tasks(PINCH_TASKS_DIR)}
    task = task_map[task_id]
    workdir = job_dir / "workdir"
    result = __import__("asyncio").run(
        generate_pinch_one(
            task=task,
            workdir=workdir,
            timeout=timeout,
            cleanup=False,
            run_id=job_spec["job_id"],
            runtime=job_spec.get("runtime", "legacy"),
        )
    )
    workspace_path = Path(result["workspace_path"]) if result.get("workspace_path") else workdir
    trace_path = workspace_path / "trace.json"
    meta_path = workspace_path / "metadata.json"
    if not trace_path.exists() or not meta_path.exists():
        legacy_trace_path = workspace_path / f"{task_id}_trace.json"
        legacy_meta_path = workspace_path / f"{task_id}_metadata.json"
        if legacy_trace_path.exists() and legacy_meta_path.exists():
            trace_path = legacy_trace_path
            meta_path = legacy_meta_path
    if trace_path.exists() and meta_path.exists():
        shutil.copy2(trace_path, job_dir / "trace.json")
        shutil.copy2(meta_path, job_dir / "metadata.json")
        llm_calls_path = workspace_path / "llm_calls.json"
        if llm_calls_path.exists():
            shutil.copy2(llm_calls_path, job_dir / "llm_calls.json")
    return {
        "status": "success" if not result.get("error") else "failed",
        "benchmark": "pinchbench",
        "logical_id": task_id,
        "task_id": task_id,
        "result_payload": result,
    }


def _run_swe_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    from experiments.SWEBench.generation import generate_one as generate_swe_one

    payload = job_spec["payload"]
    job_dir = Path(job_spec["job_dir"])
    timeout = _remaining_job_seconds(job_spec)
    workdir = job_dir / "workdir"
    if not payload.get("instance"):
        raise RuntimeError(
            "SWE-bench instance payload is unavailable. Rebuild the mixed workload plan "
            "in an environment with the `datasets` package installed."
        )
    instance = payload["instance"]
    result = __import__("asyncio").run(
        generate_swe_one(
            instance=instance,
            workdir=str(workdir),
            model_name="mas-sys",
            timeout=timeout,
            cleanup=False,
            runtime=job_spec.get("runtime", "legacy"),
        )
    )
    safe_id = payload["instance_id"].replace("/", "__")
    trace_path = workdir / f"{safe_id}_trace.json"
    meta_path = workdir / f"{safe_id}_metadata.json"
    if trace_path.exists() and meta_path.exists():
        shutil.copy2(trace_path, job_dir / "trace.json")
        shutil.copy2(meta_path, job_dir / "metadata.json")
    generation_status = str(result.get("generation_status") or "")
    return {
        "status": "success"
        if generation_status == "success" and result.get("model_patch", "").strip()
        else "failed",
        "benchmark": "swebench",
        "logical_id": payload["instance_id"],
        "instance_id": payload["instance_id"],
        "result_payload": result,
    }


def _run_paperbanana_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    from app.tracer import tracer
    from experiments.PaperBananaBench.run_benchmark import _generate_single

    payload = job_spec["payload"]
    sample = payload["sample"]
    job_dir = Path(job_spec["job_dir"])
    timeout = _remaining_job_seconds(job_spec)

    session_id = f"mixed:paperbanana:{job_spec['job_id']}"
    tracer.activate_session(session_id, reset=True)
    tracer.start_e2e()

    try:
        result = __import__("asyncio").run(
            _generate_single(
                sample,
                "diagram",
                retrieval_setting="auto",
                max_critic_rounds=3,
                timeout=timeout,
                runtime=job_spec.get("runtime", "legacy"),
            )
        )
    finally:
        tracer.end_e2e()
        tracer.save(job_dir)
        tracer.clear_session(session_id)

    return {
        "status": "success" if not result.get("error") else "failed",
        "benchmark": "paperbananabench",
        "logical_id": payload["sample_id"],
        "sample_id": payload["sample_id"],
        "result_payload": result,
    }


def _run_deepfund_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    from experiments.DeepFundBench.run_deepfund import run_experiment

    payload = job_spec["payload"]
    job_dir = Path(job_spec["job_dir"])
    timeout = _remaining_job_seconds(job_spec)
    results_dir = job_dir / "deepfund_results"

    result = __import__("asyncio").run(
        run_experiment(
            cfg=payload["config"],
            data_dir=str(DEEPFUND_DATA_DIR),
            results_dir=str(results_dir),
            timeout=timeout,
            runtime=job_spec.get("runtime", "legacy"),
        )
    )

    out_dir = Path(result["out_dir"]) if result.get("out_dir") else None
    trading_date = payload["trading_date"]
    if out_dir and out_dir.exists():
        trace_path = out_dir / f"{trading_date}_trace.json"
        meta_path = out_dir / f"{trading_date}_metadata.json"
        if trace_path.exists() and meta_path.exists():
            shutil.copy2(trace_path, job_dir / "trace.json")
            shutil.copy2(meta_path, job_dir / "metadata.json")

    return {
        "status": "success" if not result.get("error") else "failed",
        "benchmark": "deepfundbench",
        "logical_id": payload["logical_id"],
        "trading_date": trading_date,
        "ticker": payload.get("ticker"),
        "result_payload": result,
    }


def _run_job(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = job_spec["benchmark"]
    if benchmark == "research_bench":
        return _run_research_job(job_spec)
    if benchmark == "pinchbench":
        return _run_pinch_job(job_spec)
    if benchmark == "swebench":
        return _run_swe_job(job_spec)
    if benchmark == "paperbananabench":
        return _run_paperbanana_job(job_spec)
    if benchmark == "deepfundbench":
        return _run_deepfund_job(job_spec)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _run_preflight() -> None:
    imports = [
        ("research_bench", "experiments.ResearchBench.generation.proposal_writing"),
        ("pinchbench", "experiments.PinchBench.generation"),
        ("pinchbench", "experiments.PinchBench.task_loader"),
        ("swebench", "experiments.SWEBench.generation"),
        ("paperbananabench", "experiments.PaperBananaBench.run_benchmark"),
        ("deepfundbench", "experiments.DeepFundBench.run_deepfund"),
    ]
    imported = []
    for benchmark, module_name in imports:
        __import__(module_name)
        imported.append({"benchmark": benchmark, "module": module_name})
    print(json.dumps({"status": "ok", "imports": imported}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker for the mixed workload benchmark.")
    parser.add_argument("--job-spec", required=False, help="Path to job_spec.json")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate worker imports and runtime dependencies, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        _run_preflight()
        return
    if not args.job_spec:
        raise SystemExit("--job-spec is required unless --preflight is used")

    job_spec = _read_json(Path(args.job_spec))
    job_dir = Path(job_spec["job_dir"])

    start_time = time.time()
    _install_deadline_debug_handler(job_spec, start_time)
    status = "success"
    error = None
    payload: Dict[str, Any] = {}
    try:
        with _llm_transport_context(job_spec.get("runtime", "legacy")):
            with _replay_plan_agent(job_spec):
                payload = _run_job(job_spec)
        status = payload.get("status", "success")
    except Exception as exc:  # pragma: no cover - best effort failure capture
        status = "failed"
        error = str(exc)
    finally:
        _cleanup_current_sandbox_async()
    result = {
        "job_id": job_spec["job_id"],
        "benchmark": job_spec["benchmark"],
        "logical_id": job_spec["logical_id"],
        "status": status,
        "error": error,
        "elapsed_seconds": round(time.time() - start_time, 4),
        "payload": payload,
    }
    _write_json(job_dir / "result.json", result)
    print(json.dumps({"job_id": job_spec["job_id"], "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
