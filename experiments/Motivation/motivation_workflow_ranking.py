#!/usr/bin/env python3
"""Workflow-level serving-condition ranking benchmark.

Five distinct agent workflows are compared under five distinct serving
conditions.  Each workflow is intentionally shaped to be sensitive to a
different serving bottleneck:

* star_longctx: high fan-out, long-context KV/admission pressure.
* dag_2wave: moderate fan-out in two dependency waves.
* chain_compress: serial, low peak KV, later steps see compressed context.
* tool_react: many small request/tool-style turns, CPU/tokenizer/API overhead.
* decode_debate: fewer requests but long decode, GPU decode contention.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from workflow_helpers import (
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_REVISION,
    PROJECT_ROOT,
    build_context,
    mean,
    pctl,
    stop_process,
    tail,
    timestamp,
    wait_ready,
    write_rank_bump_svg,
    write_rank_heatmap_svg,
)


@dataclass(frozen=True)
class ServingCondition:
    name: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    kv_cache_memory_bytes: int | None
    cpu_cores: str | None = None
    pressure_interval_s: float = 0.0
    pressure_max_tokens: int = 0
    pressure_context_tokens: int = 256
    pressure_concurrency_cap: int = 2
    cpu_pressure_workers: int = 0


CONDITIONS: dict[str, ServingCondition] = {
    "C1_abundant": ServingCondition(
        name="C1_abundant",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=16,
        kv_cache_memory_bytes=32 * 1024**3,
    ),
    "C2_kv_tight": ServingCondition(
        name="C2_kv_tight",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        kv_cache_memory_bytes=2 * 1024**3,
    ),
    "C3_cpu_tight": ServingCondition(
        name="C3_cpu_tight",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=16,
        kv_cache_memory_bytes=32 * 1024**3,
        cpu_cores="0",
        cpu_pressure_workers=3,
    ),
    "C4_gpu_pressure": ServingCondition(
        name="C4_gpu_pressure",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=16,
        kv_cache_memory_bytes=32 * 1024**3,
        pressure_interval_s=0.02,
        pressure_max_tokens=1536,
        pressure_context_tokens=256,
        pressure_concurrency_cap=8,
    ),
    "C5_mixed_tight": ServingCondition(
        name="C5_mixed_tight",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        kv_cache_memory_bytes=2 * 1024**3,
        cpu_cores="0",
        pressure_interval_s=0.50,
        pressure_max_tokens=256,
        pressure_context_tokens=192,
        pressure_concurrency_cap=1,
        cpu_pressure_workers=1,
    ),
}

WORKFLOWS = [
    "W_star_longctx",
    "W_dag_2wave",
    "W_chain_compress",
    "W_tool_react",
    "W_decode_debate",
]


def launch_vllm(
    *,
    condition: ServingCondition,
    args: argparse.Namespace,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    cmd = [
        args.vllm_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--max-model-len",
        str(condition.max_model_len),
        "--gpu-memory-utilization",
        str(condition.gpu_memory_utilization),
        "--max-num-seqs",
        str(condition.max_num_seqs),
        "--no-enable-log-requests",
        "--no-enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if condition.kv_cache_memory_bytes is not None:
        cmd.extend(["--kv-cache-memory-bytes", str(condition.kv_cache_memory_bytes)])
    if condition.cpu_cores:
        cmd = ["taskset", "-c", condition.cpu_cores, *cmd]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    env.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

    log_fh = log_path.open("wb")
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._motivation_log_fh = log_fh  # type: ignore[attr-defined]
    return process


async def llm_call(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    tag: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        stream=False,
    )
    ended = time.perf_counter()
    usage = response.usage
    text = response.choices[0].message.content or ""
    return {
        "tag": tag,
        "latency_s": ended - started,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "preview": text[:180],
    }


async def tool_llm_call(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    tag: str,
) -> dict[str, Any]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_metric",
                "description": "Look up a benchmark metric from synthetic telemetry.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "window": {"type": "string"},
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "return_json": {"type": "boolean"},
                    },
                    "required": ["metric", "window", "filters", "return_json"],
                },
            },
        }
    ]
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "lookup_metric"}},
        temperature=0.0,
        max_tokens=max_tokens,
        stream=False,
    )
    ended = time.perf_counter()
    usage = response.usage
    message = response.choices[0].message
    tool_calls = list(message.tool_calls or [])
    preview = ""
    if tool_calls:
        preview = tool_calls[0].function.arguments[:180]
    else:
        preview = (message.content or "")[:180]
    return {
        "tag": tag,
        "latency_s": ended - started,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "tool_call_count": len(tool_calls),
        "preview": preview,
    }


def messages_for(
    *,
    role: str,
    marker: str,
    task: str,
    context: str,
    extra: str = "",
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a deterministic MAS benchmark agent. "
                "Follow the requested output shape exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Role: {role}\nMarker: {marker}\n"
                f"Task: {task}\n"
                f"{extra}\n\nContext:\n{context}"
            ),
        },
    ]


def context_slice(context: str, fraction: float) -> str:
    lines = context.splitlines()
    return "\n".join(lines[: max(1, int(len(lines) * fraction))])


def parse_cpu_cores(cpu_cores: str | None) -> set[int] | None:
    if not cpu_cores:
        return None
    result: set[int] = set()
    for part in cpu_cores.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))
    return result


async def workflow_star_longctx(
    client: AsyncOpenAI,
    model: str,
    context: str,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    workers = max(2, args.star_workers)
    task = "Return compact JSON with marker, risk, mitigation. Keep under 55 words."
    worker_calls = await asyncio.gather(
        *[
            llm_call(
                client=client,
                model=model,
                messages=messages_for(
                    role=f"star_worker_{idx}",
                    marker=f"{run_id}-star-worker-{idx}",
                    task=task,
                    context=context,
                ),
                max_tokens=args.short_max_tokens,
                tag=f"star.worker_{idx}",
            )
            for idx in range(workers)
        ]
    )
    aggregate = "\n".join(call["preview"] for call in worker_calls)
    center = await llm_call(
        client=client,
        model=model,
        messages=messages_for(
            role="star_center",
            marker=f"{run_id}-star-center",
            task="Merge worker JSON into one compact final JSON. Keep under 70 words.",
            context=context,
            extra=f"Worker outputs:\n{aggregate}",
        ),
        max_tokens=args.short_max_tokens,
        tag="star.center",
    )
    return [*worker_calls, center]


async def workflow_dag_2wave(
    client: AsyncOpenAI,
    model: str,
    context: str,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    width = max(2, args.dag_width)
    task = "Return compact JSON with marker, risk, mitigation. Keep under 55 words."
    wave0 = await asyncio.gather(
        *[
            llm_call(
                client=client,
                model=model,
                messages=messages_for(
                    role=f"dag_wave0_{idx}",
                    marker=f"{run_id}-dag-w0-{idx}",
                    task=task,
                    context=context,
                ),
                max_tokens=args.short_max_tokens,
                tag=f"dag.wave0_{idx}",
            )
            for idx in range(width)
        ]
    )
    wave0_summary = "\n".join(call["preview"] for call in wave0)
    wave1 = await asyncio.gather(
        *[
            llm_call(
                client=client,
                model=model,
                messages=messages_for(
                    role=f"dag_wave1_{idx}",
                    marker=f"{run_id}-dag-w1-{idx}",
                    task=task,
                    context=context,
                    extra=f"Upstream wave summaries:\n{wave0_summary}",
                ),
                max_tokens=args.short_max_tokens,
                tag=f"dag.wave1_{idx}",
            )
            for idx in range(width)
        ]
    )
    all_summary = "\n".join(call["preview"] for call in [*wave0, *wave1])
    reducer = await llm_call(
        client=client,
        model=model,
        messages=messages_for(
            role="dag_reducer",
            marker=f"{run_id}-dag-reducer",
            task="Return a compact final JSON merged from all upstream outputs.",
            context=context,
            extra=f"Upstream outputs:\n{all_summary}",
        ),
        max_tokens=args.short_max_tokens,
        tag="dag.reducer",
    )
    return [*wave0, *wave1, reducer]


async def workflow_chain_compress(
    client: AsyncOpenAI,
    model: str,
    context: str,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    calls = []
    previous = ""
    for idx in range(args.chain_steps):
        step_context = context if idx == 0 else context_slice(context, args.chain_context_fraction)
        task = (
            "Return compact JSON with marker, selected evidence, and compressed state. "
            "Keep under 60 words."
        )
        result = await llm_call(
            client=client,
            model=model,
            messages=messages_for(
                role=f"chain_compress_{idx}",
                marker=f"{run_id}-chain-{idx}",
                task=task,
                context=step_context,
                extra=f"Previous compressed state:\n{previous}" if previous else "",
            ),
            max_tokens=args.short_max_tokens,
            tag=f"chain.step_{idx}",
        )
        calls.append(result)
        previous = result["preview"]
    return calls


async def workflow_tool_react(
    client: AsyncOpenAI,
    model: str,
    context: str,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    small_context = context_slice(context, args.tool_context_fraction)

    async def one_agent(agent_idx: int) -> list[dict[str, Any]]:
        agent_calls = []
        scratchpad = ""
        for step in range(args.tool_steps):
            tool_schema = {
                "tool": "lookup_metric",
                "arguments": {
                    "metric": f"m{agent_idx}_{step}",
                    "filters": ["alpha", "beta", "queue", "cache_marker"],
                    "return_json": True,
                },
            }
            result = await tool_llm_call(
                client=client,
                model=model,
                messages=messages_for(
                    role=f"tool_react_agent_{agent_idx}",
                    marker=f"{run_id}-tool-a{agent_idx}-s{step}",
                    task="Call lookup_metric with the requested arguments. Do not answer directly.",
                    context=small_context,
                    extra=(
                        f"Required tool schema:\n{json.dumps(tool_schema)}\n"
                        f"Scratchpad:\n{scratchpad}"
                    ),
                ),
                max_tokens=args.tool_max_tokens,
                tag=f"tool.agent_{agent_idx}.step_{step}",
            )
            # Local parse/serialize simulates app-side tool plumbing and JSON handling.
            payload = {
                "tool_result": {
                    "agent": agent_idx,
                    "step": step,
                    "ok": True,
                    "payload_preview": result["preview"][:80],
                    "records": [
                        {
                            "metric": f"m{k}",
                            "value": (agent_idx + 1) * (step + 1) * (k + 1),
                            "tags": ["alpha", "beta", "queue", "cache"],
                        }
                        for k in range(12)
                    ],
                }
            }
            encoded = json.dumps(payload)
            for _ in range(args.tool_cpu_work):
                encoded = json.dumps(json.loads(encoded), sort_keys=True)
            _ = json.dumps(
                {
                    "tool_result": {
                        "agent": agent_idx,
                        "step": step,
                        "ok": True,
                        "payload_preview": result["preview"][:80],
                    }
                }
            )
            scratchpad = f"Tool result for step {step}: ok; preview={result['preview'][:90]}"
            agent_calls.append(result)
            if args.tool_delay_s > 0:
                await asyncio.sleep(args.tool_delay_s)
        return agent_calls

    nested = await asyncio.gather(
        *[one_agent(agent_idx) for agent_idx in range(args.tool_agents)]
    )
    return [call for calls in nested for call in calls]


async def workflow_decode_debate(
    client: AsyncOpenAI,
    model: str,
    context: str,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    debate_context = context_slice(context, args.decode_context_fraction)
    calls = []
    transcript = ""
    for round_idx in range(args.decode_rounds):
        round_calls = await asyncio.gather(
            *[
                llm_call(
                    client=client,
                    model=model,
                    messages=messages_for(
                        role=f"decode_debater_{side}",
                        marker=f"{run_id}-decode-r{round_idx}-{side}",
                        task=(
                            "Write a detailed argument with evidence, counterargument, "
                            "and conclusion. Target a long answer."
                        ),
                        context=debate_context,
                        extra=f"Transcript so far:\n{transcript}",
                    ),
                    max_tokens=args.decode_max_tokens,
                    tag=f"decode.round_{round_idx}.{side}",
                )
                for side in ("pro", "con")
            ]
        )
        calls.extend(round_calls)
        transcript += "\n".join(call["preview"] for call in round_calls) + "\n"
    judge = await llm_call(
        client=client,
        model=model,
        messages=messages_for(
            role="decode_judge",
            marker=f"{run_id}-decode-judge",
            task="Write a detailed final judgment and synthesis. Target a long answer.",
            context=debate_context,
            extra=f"Debate transcript:\n{transcript}",
        ),
        max_tokens=args.decode_max_tokens,
        tag="decode.judge",
    )
    return [*calls, judge]


WORKFLOW_RUNNERS = {
    "W_star_longctx": workflow_star_longctx,
    "W_dag_2wave": workflow_dag_2wave,
    "W_chain_compress": workflow_chain_compress,
    "W_tool_react": workflow_tool_react,
    "W_decode_debate": workflow_decode_debate,
}


async def run_workflow_once(
    *,
    client: AsyncOpenAI,
    model: str,
    context: str,
    workflow: str,
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    calls = await WORKFLOW_RUNNERS[workflow](client, model, context, run_id, args)
    ended = time.perf_counter()
    return {
        "workflow": workflow,
        "run_id": run_id,
        "e2e_duration_s": ended - started,
        "llm_calls": calls,
        "llm_call_count": len(calls),
        "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in calls),
        "completion_tokens": sum(call.get("completion_tokens") or 0 for call in calls),
    }


async def pressure_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    condition: ServingCondition,
    stop_event: asyncio.Event,
    args: argparse.Namespace,
    pressure_log: Path,
) -> None:
    if condition.pressure_interval_s <= 0 or condition.pressure_max_tokens <= 0:
        return
    context = build_context(condition.pressure_context_tokens)
    active: set[asyncio.Task[Any]] = set()
    request_idx = 0

    async def one_pressure_request(idx: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = await llm_call(
                client=client,
                model=model,
                messages=messages_for(
                    role="background_gpu_pressure",
                    marker=f"{condition.name}-pressure-{idx}",
                    task=(
                        "Write a long background answer. This request exists only "
                        "to create decode-side serving pressure."
                    ),
                    context=context,
                ),
                max_tokens=condition.pressure_max_tokens,
                tag=f"pressure.{idx}",
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            result = {"error": repr(exc)}
            status = "error"
        return {
            "idx": idx,
            "status": status,
            "duration_s": round(time.perf_counter() - started, 4),
            **result,
        }

    with pressure_log.open("w", encoding="utf-8") as f:
        while not stop_event.is_set():
            done = {task for task in active if task.done()}
            for task in done:
                active.remove(task)
                try:
                    f.write(json.dumps(task.result(), ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as exc:  # noqa: BLE001
                    f.write(json.dumps({"status": "error", "error": repr(exc)}) + "\n")
                    f.flush()
            if len(active) < condition.pressure_concurrency_cap:
                task = asyncio.create_task(one_pressure_request(request_idx))
                active.add(task)
                request_idx += 1
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=condition.pressure_interval_s)
            except asyncio.TimeoutError:
                pass

        if active:
            await asyncio.wait(active, timeout=args.request_timeout_s)
            for task in list(active):
                if task.done():
                    try:
                        f.write(json.dumps(task.result(), ensure_ascii=False) + "\n")
                    except Exception as exc:  # noqa: BLE001
                        f.write(json.dumps({"status": "error", "error": repr(exc)}) + "\n")
                else:
                    task.cancel()


def launch_cpu_pressure(condition: ServingCondition) -> list[subprocess.Popen[bytes]]:
    if not condition.cpu_cores or condition.cpu_pressure_workers <= 0:
        return []
    workers = []
    code = (
        "import math\n"
        "x = 0.0\n"
        "while True:\n"
        "    for i in range(200000):\n"
        "        x += math.sqrt((i % 97) + 1.0)\n"
    )
    for _ in range(condition.cpu_pressure_workers):
        workers.append(
            subprocess.Popen(
                ["taskset", "-c", condition.cpu_cores, sys.executable, "-c", code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    return workers


def stop_cpu_pressure(workers: list[subprocess.Popen[bytes]]) -> None:
    for worker in workers:
        if worker.poll() is not None:
            continue
        try:
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


async def run_condition(
    *,
    condition: ServingCondition,
    args: argparse.Namespace,
    artifact_root: Path,
    context: str,
) -> list[dict[str, Any]]:
    condition_dir = artifact_root / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    log_path = condition_dir / "vllm.log"
    base_url = f"http://127.0.0.1:{args.port}/v1"
    original_affinity: set[int] | None = None
    condition_affinity = parse_cpu_cores(condition.cpu_cores)
    if condition_affinity is not None and args.pin_orchestrator_cpu:
        try:
            original_affinity = set(os.sched_getaffinity(0))
            os.sched_setaffinity(0, condition_affinity)
        except Exception:  # noqa: BLE001
            original_affinity = None
    process: subprocess.Popen[bytes] | None = None
    client: AsyncOpenAI | None = None
    pressure_task: asyncio.Task[Any] | None = None
    stop_pressure = asyncio.Event()
    cpu_pressure_workers: list[subprocess.Popen[bytes]] = []
    try:
        process = launch_vllm(condition=condition, args=args, log_path=log_path)
        cpu_pressure_workers = launch_cpu_pressure(condition)
        await wait_ready(base_url, args.startup_timeout_s)
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {process.returncode}")
        http_client = httpx.AsyncClient(
            timeout=args.request_timeout_s,
            limits=httpx.Limits(max_connections=96, max_keepalive_connections=32),
        )
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=args.request_timeout_s,
            max_retries=0,
            http_client=http_client,
        )

        if condition.pressure_interval_s > 0:
            pressure_task = asyncio.create_task(
                pressure_loop(
                    client=client,
                    model=args.served_model_name,
                    condition=condition,
                    stop_event=stop_pressure,
                    args=args,
                    pressure_log=condition_dir / "pressure.jsonl",
                )
            )
            await asyncio.sleep(args.pressure_warmup_s)

        workflows = [name for name in args.workflows.split(",") if name]
        raw_path = condition_dir / "runs.jsonl"
        results = []

        for warmup in range(args.warmups):
            for workflow in workflows:
                await run_workflow_once(
                    client=client,
                    model=args.served_model_name,
                    context=context,
                    workflow=workflow,
                    run_id=f"{condition.name}-warmup{warmup}-{workflow}",
                    args=args,
                )

        with raw_path.open("w", encoding="utf-8") as f:
            for rep in range(args.repetitions):
                order = workflows[:]
                random.Random(args.seed + rep).shuffle(order)
                for workflow in order:
                    run = await run_workflow_once(
                        client=client,
                        model=args.served_model_name,
                        context=context,
                        workflow=workflow,
                        run_id=f"{condition.name}-rep{rep}-{workflow}",
                        args=args,
                    )
                    run["condition"] = condition.name
                    run["rep"] = rep
                    results.append(run)
                    f.write(json.dumps(run, ensure_ascii=False) + "\n")
                    f.flush()
        return results
    except Exception:
        print(f"\n[{condition.name}] vLLM log tail:\n{tail(log_path)}")
        raise
    finally:
        stop_pressure.set()
        if pressure_task is not None:
            try:
                await pressure_task
            except asyncio.CancelledError:
                pass
        if client is not None:
            await client.close()
        if process is not None:
            stop_process(process)
        stop_cpu_pressure(cpu_pressure_workers)
        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
            except Exception:  # noqa: BLE001
                pass


def write_summary(
    artifact_root: Path,
    results: list[dict[str, Any]],
    *,
    condition_order: list[str],
    workflow_order: list[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["condition"], row["workflow"]), []).append(row)

    summary_rows = []
    for condition in condition_order:
        for workflow in workflow_order:
            rows = grouped.get((condition, workflow), [])
            if not rows:
                continue
            durations = [row["e2e_duration_s"] for row in rows]
            prompt_tokens = [row.get("prompt_tokens", 0) for row in rows]
            completion_tokens = [row.get("completion_tokens", 0) for row in rows]
            call_counts = [row.get("llm_call_count", 0) for row in rows]
            summary_rows.append(
                {
                    "condition": condition,
                    "topology": workflow,
                    "workflow": workflow,
                    "n": len(rows),
                    "mean_s": round(mean(durations), 4),
                    "median_s": round(statistics.median(durations), 4),
                    "p90_s": round(pctl(durations, 0.90), 4),
                    "min_s": round(min(durations), 4),
                    "max_s": round(max(durations), 4),
                    "mean_llm_calls": round(mean(call_counts), 1),
                    "mean_prompt_tokens": round(mean(prompt_tokens), 1),
                    "mean_completion_tokens": round(mean(completion_tokens), 1),
                }
            )

    rank_rows = []
    rank_table = {}
    for condition in condition_order:
        rows = [row for row in summary_rows if row["condition"] == condition]
        rows.sort(key=lambda row: row["median_s"])
        rank_table[condition] = [row["workflow"] for row in rows]
        for rank, row in enumerate(rows, start=1):
            rank_rows.append(
                {
                    "condition": condition,
                    "topology": row["workflow"],
                    "workflow": row["workflow"],
                    "rank": rank,
                    "median_s": row["median_s"],
                    "p90_s": row["p90_s"],
                    "mean_prompt_tokens": row["mean_prompt_tokens"],
                    "mean_completion_tokens": row["mean_completion_tokens"],
                }
            )

    if not summary_rows:
        return {"summary_rows": [], "rank_rows": [], "rank_table_by_median": {}}

    with (artifact_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (artifact_root / "ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rank_rows)

    write_rank_bump_svg(
        artifact_root / "workflow_ranking_bump.svg",
        rank_rows,
        condition_order=condition_order,
        topology_order=workflow_order,
    )
    write_rank_heatmap_svg(
        artifact_root / "workflow_ranking_heatmap.svg",
        rank_rows,
        condition_order=condition_order,
        topology_order=workflow_order,
    )

    summary = {
        "summary_rows": summary_rows,
        "rank_rows": rank_rows,
        "rank_table_by_median": rank_table,
        "artifact_root": str(artifact_root),
        "workflow_ranking_bump_svg": str(artifact_root / "workflow_ranking_bump.svg"),
        "workflow_ranking_heatmap_svg": str(artifact_root / "workflow_ranking_heatmap.svg"),
    }
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


async def main_async(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    artifact_root = Path(
        args.output_dir
        or PROJECT_ROOT / "experiments" / "Motivation" / "results" / timestamp()
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    context = build_context(args.context_tokens)
    condition_order = [name for name in args.conditions.split(",") if name]
    workflow_order = [name for name in args.workflows.split(",") if name]
    conditions = [CONDITIONS[name] for name in condition_order]
    conditions = [replace(c, cpu_cores=str(args.cpu_core)) if c.cpu_cores else c for c in conditions]

    manifest = {
        "model": args.model,
        "model_revision": args.revision,
        "pin_orchestrator_cpu": args.pin_orchestrator_cpu,
        "pressure_warmup_s": args.pressure_warmup_s,
        "served_model_name": args.served_model_name,
        "gpu": args.gpu,
        "conditions": [condition.__dict__ for condition in conditions],
        "workflows": workflow_order,
        "context_tokens_target": args.context_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "workflow_params": {
            "short_max_tokens": args.short_max_tokens,
            "star_workers": args.star_workers,
            "dag_width": args.dag_width,
            "chain_steps": args.chain_steps,
            "chain_context_fraction": args.chain_context_fraction,
            "tool_agents": args.tool_agents,
            "tool_steps": args.tool_steps,
            "tool_context_fraction": args.tool_context_fraction,
            "tool_max_tokens": args.tool_max_tokens,
            "tool_cpu_work": args.tool_cpu_work,
            "tool_delay_s": args.tool_delay_s,
            "decode_rounds": args.decode_rounds,
            "decode_context_fraction": args.decode_context_fraction,
            "decode_max_tokens": args.decode_max_tokens,
        },
        "seed": args.seed,
    }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_results: list[dict[str, Any]] = []
    for condition in conditions:
        print(f"[condition] {condition.name}", flush=True)
        condition_results = await run_condition(
            condition=condition,
            args=args,
            artifact_root=artifact_root,
            context=context,
        )
        all_results.extend(condition_results)
        partial = write_summary(
            artifact_root,
            all_results,
            condition_order=condition_order,
            workflow_order=workflow_order,
        )
        print(json.dumps(partial["rank_table_by_median"], indent=2), flush=True)

    summary = write_summary(
        artifact_root,
        all_results,
        condition_order=condition_order,
        workflow_order=workflow_order,
    )
    print(f"[done] artifacts: {artifact_root}", flush=True)
    print(json.dumps(summary["rank_table_by_median"], indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="Model revision (defaults to the paper snapshot for the default model)")
    parser.add_argument("--cpu-core", type=int, default=0, help="CPU core for the two CPU-constrained conditions")
    parser.add_argument("--served-model-name", default="motivation-qwen")
    parser.add_argument("--vllm-python", default=DEFAULT_PYTHON)
    parser.add_argument("--hf-home", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--conditions",
        default="C1_abundant,C2_kv_tight,C3_cpu_tight,C4_gpu_pressure,C5_mixed_tight",
        help=f"Comma-separated subset of: {','.join(CONDITIONS)}",
    )
    parser.add_argument("--workflows", default=",".join(WORKFLOWS))
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--pressure-warmup-s", type=float, default=2.0)
    parser.add_argument("--pin-orchestrator-cpu", action="store_true", default=True)
    parser.add_argument("--no-pin-orchestrator-cpu", dest="pin_orchestrator_cpu", action="store_false")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--short-max-tokens", type=int, default=64)
    parser.add_argument("--star-workers", type=int, default=11)
    parser.add_argument("--dag-width", type=int, default=5)
    parser.add_argument("--chain-steps", type=int, default=8)
    parser.add_argument("--chain-context-fraction", type=float, default=0.25)
    parser.add_argument("--tool-agents", type=int, default=8)
    parser.add_argument("--tool-steps", type=int, default=8)
    parser.add_argument("--tool-context-fraction", type=float, default=0.08)
    parser.add_argument("--tool-max-tokens", type=int, default=16)
    parser.add_argument("--tool-cpu-work", type=int, default=300)
    parser.add_argument("--tool-delay-s", type=float, default=0.01)
    parser.add_argument("--decode-rounds", type=int, default=2)
    parser.add_argument("--decode-context-fraction", type=float, default=0.25)
    parser.add_argument("--decode-max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.revision is None and args.model == DEFAULT_MODEL:
        args.revision = DEFAULT_REVISION

    unknown_conditions = [name for name in args.conditions.split(",") if name not in CONDITIONS]
    if unknown_conditions:
        raise SystemExit(f"Unknown condition(s): {', '.join(unknown_conditions)}")
    unknown_workflows = [name for name in args.workflows.split(",") if name not in WORKFLOW_RUNNERS]
    if unknown_workflows:
        raise SystemExit(f"Unknown workflow(s): {', '.join(unknown_workflows)}")
    cpu_conditions = any(CONDITIONS[name].cpu_cores for name in args.conditions.split(","))
    if cpu_conditions and args.cpu_core not in os.sched_getaffinity(0):
        parser.error("--cpu-core must be in the process CPU affinity mask")
    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
