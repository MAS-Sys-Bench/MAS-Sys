from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.SWEBench.utils import load_swebench_dataset


DEFAULT_RESULTS_DIR = _THIS_DIR / "results"
DEFAULT_PREPARE_DIR = _THIS_DIR / "prepare"
DEFAULT_CONFIG = _THIS_DIR / "configs" / "example.yaml"
DEFAULT_LLM_CONFIG = _PROJECT_ROOT / "config" / "config.toml"
DEFAULT_SUMMARY_CSV = _THIS_DIR / "summary.csv"
RESEARCH_BENCH_PATH = _PROJECT_ROOT / "experiments" / "ResearchBench" / "data" / "high_impact_paper_bench.json"
PAPERBANANA_PATH = _PROJECT_ROOT / "experiments" / "PaperBananaBench" / "data" / "diagram" / "test.json"
PINCH_TASKS_DIR = _PROJECT_ROOT / "experiments" / "PinchBench" / "tasks"
DEEPFUND_DATA_DIR = _PROJECT_ROOT / "experiments" / "DeepFundBench" / "data"
SWE_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
SWE_DATASET_SPLIT = "test"
DEEPFUND_CONFIGS = [
    _PROJECT_ROOT / "experiments" / "DeepFundBench" / "configs" / "main.yaml",
]
DEADLINE_TERMINATION_GRACE_S = float(
    os.environ.get("MIXED_WORKLOAD_DEADLINE_GRACE_S", "10")
)
JOB_DEADLINE_S = float(os.environ.get("MIXED_WORKLOAD_JOB_DEADLINE_S", "1200"))


@dataclass
class BenchmarkPool:
    name: str
    sampled_items: List[Dict[str, Any]]


SUMMARY_TABLES: List[tuple[str, List[str]]] = [
    (
        "Experiment Setup",
        [
            "Random seed",
            "Test window (min)",
            "Target total JPS",
            "Submitted JPS",
            "Backend",
            "Runtime",
            "Hardware",
            "Models",
        ],
    ),
    (
        "Completion Results",
        [
            "Submitted jobs",
            "Jobs completed within test window",
            "Succeeded jobs",
            "Failed jobs",
            "Killed jobs",
        ],
    ),
    (
        "E2E Job",
        [
            "Mean job runtime (s)",
            "P95 job runtime (s)",
            "Max job runtime (s)",
            "Jobs with job queueing delay",
            "Mean job queueing delay (s)",
            "Max job queueing delay (s)",
        ],
    ),
    (
        "Token Metrics",
        [
            "Jobs with token metrics",
            "Mean total tokens/job",
            "P95 total tokens/job",
            "Max total tokens/job",
            "Mean input tokens/job",
            "Mean output tokens/job",
        ],
    ),
    (
        "Tool Metrics",
        [
            "Jobs with tool latency metric",
            "Mean tool-call latency/job (s)",
            "Max tool-call latency/job (s)",
        ],
    ),
    (
        "Inference Metrics",
        [
            "Jobs with metrics",
            "LLM calls",
            "Mean client latency total/job (s)",
            "P95 client latency total/job (s)",
            "Max client latency total/job (s)",
            "Mean backend latency total/job (s)",
            "P95 backend latency total/job (s)",
            "Max backend latency total/job (s)",
            "Mean backend queue wait total/job (s)",
            "P95 backend queue wait total/job (s)",
            "Max backend queue wait total/job (s)",
        ],
    ),
    (
        "Concurrency Metrics",
        [
            "Mean concurrent jobs",
            "Max concurrent jobs",
            "Mean concurrent LLM calls",
            "Max concurrent LLM calls",
        ],
    ),
    (
        "Agent Graph Metrics",
        [
            "Jobs with agent graph metrics",
            "Mean agents/job",
            "Max agents/job",
            "Mean agent graph depth",
            "Max agent graph depth",
            "Mean critical path latency (s)",
            "Max critical path latency (s)",
            "Mean parallelism",
            "Max parallelism",
        ],
    ),
    (
        "Resource Metrics",
        [
            "Resource samples",
            "Mean CPU usage (%)",
            "Max CPU usage (%)",
            "Mean SM util (%)",
            "Max SM util (%)",
            "Mean KV cache usage",
            "Max KV cache usage",
            "Mean KV reuse rate",
            "Max KV reuse rate",
        ],
    ),
    (
        "Backend Health",
        [
            "Backend health status",
            "Backend health checks",
            "Backend health failed checks",
            "First backend health failure elapsed (s)",
        ],
    ),
    (
        "ORLA Metrics",
        [
            "Jobs with ORLA metrics",
            "Mean ORLA queue wait/job (s)",
            "P95 ORLA queue wait/job (s)",
            "Mean ORLA scheduler decision/job (s)",
            "Mean ORLA dispatch/job (s)",
            "Mean ORLA backend request latency/job (s)",
            "P95 ORLA backend request latency/job (s)",
            "Total ORLA estimated cost (USD)",
        ],
    ),
    (
        "Run Timing",
        [
            "Run completion elapsed (s)",
        ],
    ),
    (
        "Tool Breakdown",
        [
            "Jobs with tool breakdown metrics",
            "Mean tool calls/job",
            "Max tool calls/job",
        ],
    ),
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _load_simple_yaml(path)


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx].rstrip()
    return line


def _load_simple_yaml(path: str) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Any]] = [(-1, root)]

    for idx, raw_line in enumerate(lines):
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML structure near: {raw_line}")
            parent.append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "":
            next_container: Any
            next_line_index = idx + 1
            next_container = {}
            for future in lines[next_line_index:]:
                future_clean = _strip_inline_comment(future).rstrip()
                if not future_clean.strip():
                    continue
                future_stripped = future_clean.strip()
                if future_stripped.startswith("- "):
                    next_container = []
                break
            parent[key] = next_container
            stack.append((indent, next_container))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _generate_trading_days(start_date: str, end_date: str) -> List[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days: List[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _normalize_ratios(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        raise ValueError("ratio must sum to a positive value")
    return {k: float(v) / total for k, v in raw.items()}


def _load_research_pool() -> BenchmarkPool:
    raw = _read_json(RESEARCH_BENCH_PATH)
    population = [
        {
            "logical_id": paper_id,
            "paper_id": paper_id,
            "paper_entry": entry,
        }
        for paper_id, entry in raw.items()
    ]
    return BenchmarkPool(
        name="research_bench",
        sampled_items=population,
    )


def _load_pinch_pool() -> BenchmarkPool:
    from experiments.PinchBench.task_loader import load_all_tasks

    population = [
        {
            "logical_id": task.task_id,
            "task_id": task.task_id,
            "task_name": task.name,
        }
        for task in load_all_tasks(PINCH_TASKS_DIR)
    ]
    return BenchmarkPool(
        name="pinchbench",
        sampled_items=population,
    )


def _load_swe_pool() -> BenchmarkPool:
    instances = load_swebench_dataset(dataset_name=SWE_DATASET_NAME, split=SWE_DATASET_SPLIT)
    population = [
        {
            "logical_id": inst["instance_id"],
            "instance_id": inst["instance_id"],
            "instance": inst,
        }
        for inst in instances
    ]
    return BenchmarkPool(
        name="swebench",
        sampled_items=population,
    )


def _load_paperbanana_pool() -> BenchmarkPool:
    samples = _read_json(PAPERBANANA_PATH)
    population = [
        {
            "logical_id": sample["id"],
            "sample_id": sample["id"],
            "sample": sample,
        }
        for sample in samples
    ]
    return BenchmarkPool(
        name="paperbananabench",
        sampled_items=population,
    )


def _load_deepfund_pool() -> BenchmarkPool:
    population: List[Dict[str, Any]] = []
    for cfg_path in DEEPFUND_CONFIGS:
        cfg = _load_yaml(str(cfg_path))
        for trading_date in _generate_trading_days(cfg["start_date"], cfg["end_date"]):
            for ticker in cfg.get("tickers", []):
                ticker_day_cfg = dict(cfg)
                ticker_day_cfg["start_date"] = trading_date
                ticker_day_cfg["end_date"] = trading_date
                ticker_day_cfg["tickers"] = [ticker]
                logical_id = f"{cfg['exp_name']}:{trading_date}:{ticker}"
                population.append(
                    {
                        "logical_id": logical_id,
                        "exp_name": cfg["exp_name"],
                        "trading_date": trading_date,
                        "ticker": ticker,
                        "config": ticker_day_cfg,
                    }
                )
    return BenchmarkPool(
        name="deepfundbench",
        sampled_items=population,
    )


def build_sample_manifest(config: Dict[str, Any]) -> Dict[str, BenchmarkPool]:
    return {
        "research_bench": _load_research_pool(),
        "pinchbench": _load_pinch_pool(),
        "swebench": _load_swe_pool(),
        "paperbananabench": _load_paperbanana_pool(),
        "deepfundbench": _load_deepfund_pool(),
    }


def build_arrival_plan(
    config: Dict[str, Any],
    pools: Dict[str, BenchmarkPool],
) -> List[Dict[str, Any]]:
    rng = random.Random(int(config["seed"]))
    ratios = _normalize_ratios(config["ratio"])
    total_jps = float(config["total_jps"])
    duration_s = int(config["duration_minutes"]) * 60
    if total_jps <= 0:
        raise ValueError("total_jps must be > 0")

    bench_names = list(ratios.keys())
    cumulative = []
    running = 0.0
    for bench_name in bench_names:
        running += ratios[bench_name]
        cumulative.append((bench_name, running))

    arrival_plan: List[Dict[str, Any]] = []
    offset_s = 0.0
    while True:
        offset_s += rng.expovariate(total_jps)
        if offset_s > duration_s:
            break
        draw = rng.random()
        benchmark_name = bench_names[-1]
        for candidate, threshold in cumulative:
            if draw <= threshold:
                benchmark_name = candidate
                break
        item = rng.choice(pools[benchmark_name].sampled_items)
        dispatch_id = len(arrival_plan) + 1
        arrival_plan.append(
            {
                "dispatch_id": dispatch_id,
                "job_id": f"job_{dispatch_id:05d}",
                "arrival_offset_s": round(offset_s, 6),
                "benchmark": benchmark_name,
                "logical_id": item["logical_id"],
                "payload": item,
            }
        )
    return arrival_plan


def _sanitize_arrival_plan_for_export(arrival_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "dispatch_id": item["dispatch_id"],
            "job_id": item["job_id"],
            "arrival_offset_s": item["arrival_offset_s"],
            "benchmark": item["benchmark"],
            "logical_id": item["logical_id"],
        }
        for item in arrival_plan
    ]


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            result: Dict[str, Any] = {}
            current: Dict[str, Any] | None = None
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = _strip_inline_comment(raw_line).strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current = result
                    for part in line[1:-1].split("."):
                        current = current.setdefault(part, {})
                    continue
                if current is None or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                current[key.strip()] = _parse_scalar(raw_value)
            return result

    with path.open("rb") as f:
        return tomllib.load(f)


def _safe_llm_prepare_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    visible_keys = (
        "model",
        "size",
        "capabilities",
        "cost",
    )
    return {key: entry.get(key) for key in visible_keys if key in entry}


def _load_plan_agent_visible_config(path: Path = DEFAULT_LLM_CONFIG) -> Dict[str, Any]:
    if not path.exists():
        return {"error": "config.toml not found"}

    raw_config = _load_toml(path)
    llm_config = raw_config.get("llm", {}) if isinstance(raw_config, dict) else {}
    snapshot: Dict[str, Any] = {"llm": {}}

    for family in ("lang", "vision", "image"):
        family_config = llm_config.get(family, {})
        if not isinstance(family_config, dict):
            continue
        entries: Dict[str, Dict[str, Any]] = {}
        for name, entry in family_config.items():
            if not isinstance(entry, dict):
                continue
            entries[name] = _safe_llm_prepare_entry(entry)
        if entries:
            snapshot["llm"][family] = entries
    return snapshot


def _load_llm_runtime_snapshot(path: Path = DEFAULT_LLM_CONFIG) -> Dict[str, Any]:
    if not path.exists():
        return {"error": "config.toml not found"}

    raw_config = _load_toml(path)
    llm_config = raw_config.get("llm", {}) if isinstance(raw_config, dict) else {}
    snapshot: Dict[str, Any] = {}
    for family in ("lang", "vision", "image"):
        family_config = llm_config.get(family, {})
        if not isinstance(family_config, dict):
            continue
        entries: Dict[str, Dict[str, Any]] = {}
        for name, entry in family_config.items():
            if not isinstance(entry, dict):
                continue
            entries[name] = {
                "model": entry.get("model"),
                "backend": entry.get("backend"),
            }
        if entries:
            snapshot[family] = entries
    return snapshot


def _build_config_snapshot(
    config: Dict[str, Any],
    arrival_plan: List[Dict[str, Any]],
    prepare_fingerprint: str | None = None,
) -> Dict[str, Any]:
    snapshot = dict(config)
    snapshot["llm_runtime"] = _load_llm_runtime_snapshot()
    snapshot["arrival_plan"] = _sanitize_arrival_plan_for_export(arrival_plan)
    if prepare_fingerprint is not None:
        snapshot["prepare_fingerprint"] = prepare_fingerprint
    return snapshot


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return model
    raise TypeError(f"Cannot serialize model object of type {type(model).__name__}")


def _build_replay_task_prompt(plan_entry: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    benchmark = plan_entry["benchmark"]
    payload = plan_entry["payload"]

    if benchmark == "research_bench":
        from experiments.ResearchBench.generation.proposal_writing import (
            _build_author_profile,
            build_proposal_task_prompt,
        )

        entry = payload["paper_entry"]
        paper_data = entry["paper_data"]
        author_data = entry["author_data"]
        profiles = [_build_author_profile(profile) for profile in author_data.values()]
        ref_abstracts = [ref["abstract"] for ref in paper_data.get("references", [])]
        return build_proposal_task_prompt(profiles, ref_abstracts), {}

    if benchmark == "pinchbench":
        from experiments.PinchBench.generation import build_pinchbench_task_prompt
        from experiments.PinchBench.task_loader import load_all_tasks

        task_map = {task.task_id: task for task in load_all_tasks(PINCH_TASKS_DIR)}
        task = task_map[payload["task_id"]]
        return build_pinchbench_task_prompt(task), {}

    if benchmark == "swebench":
        from app.config import config as app_config
        from experiments.SWEBench.utils import build_task_prompt

        if not app_config.sandbox.use_sandbox:
            raise RuntimeError(
                "Prepared replay for SWE-bench requires sandbox.use_sandbox=true. "
                "Without sandboxing, the PlanAgent prompt embeds the per-run host "
                "checkout path and cannot be replayed exactly."
            )
        instance = payload["instance"]
        return build_task_prompt(instance, "/testbed"), {}

    if benchmark == "paperbananabench":
        from experiments.PaperBananaBench.run_benchmark import (
            build_paperbanana_task_prompt,
        )

        return (
            build_paperbanana_task_prompt(
                payload["sample"],
                "diagram",
                retrieval_setting="auto",
                max_critic_rounds=3,
            ),
            {},
        )

    if benchmark == "deepfundbench":
        from experiments.DeepFundBench.run_deepfund import build_task

        cfg = payload["config"]
        initial_cashflow = cfg.get("cashflow", 100_000.0)
        portfolio = {"cashflow": initial_cashflow, "positions": {}}
        return (
            build_task(
                cfg=cfg,
                ticker=payload["ticker"],
                trading_date=payload["trading_date"],
                portfolio=portfolio,
                decision_memory=[],
            ),
            {},
        )

    raise ValueError(f"Unsupported benchmark for prepared replay: {benchmark}")


async def _prepare_plan_record(
    plan_entry: Dict[str, Any],
    job_path: Path,
) -> Dict[str, Any]:
    from app.mas.plan_agent import PlanAgent

    task_prompt, _assets = _build_replay_task_prompt(plan_entry)

    plan_agent = PlanAgent()
    mas_config = await plan_agent.plan(task_prompt)
    mas_config_data = _dump_model(mas_config)

    plan_record: Dict[str, Any] = {
        "job_id": plan_entry["job_id"],
        "dispatch_id": plan_entry["dispatch_id"],
        "benchmark": plan_entry["benchmark"],
        "logical_id": plan_entry["logical_id"],
        "arrival_offset_s": plan_entry["arrival_offset_s"],
        "task_prompt_sha256": _sha256_text(task_prompt),
        "planning_start_time": plan_agent.last_plan_start_time,
        "planning_end_time": plan_agent.last_plan_end_time,
        "planning_duration_s": plan_agent.last_plan_duration_s,
        "planner_input_tokens": plan_agent.last_input_tokens,
        "planner_completion_tokens": plan_agent.last_completion_tokens,
        "mas_config": mas_config_data,
    }
    _write_json(job_path, plan_record)
    return plan_record


CONFIG_PREPARE_CONTENT_KEYS = ("plan_agent_visible_config", "sampling_config")
CONFIG_PREPARE_KEYS = set(CONFIG_PREPARE_CONTENT_KEYS) | {"prepare_fingerprint"}


def _config_prepare_extra_keys(config_prepare: Dict[str, Any]) -> set[str]:
    return set(config_prepare) - CONFIG_PREPARE_KEYS


def _config_prepare_fingerprint(config_prepare: Dict[str, Any]) -> str:
    comparable = {key: config_prepare.get(key) for key in CONFIG_PREPARE_CONTENT_KEYS}
    payload = json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)
    return _sha256_text(payload)


def _build_config_prepare(
    config: Dict[str, Any],
    pools: Dict[str, BenchmarkPool],
    arrival_plan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "plan_agent_visible_config": _load_plan_agent_visible_config(),
        "sampling_config": {
            "seed": config.get("seed"),
            "duration_minutes": config.get("duration_minutes"),
            "total_jps": config.get("total_jps"),
            "ratio": config.get("ratio"),
            "pool_sizes": {name: len(pool.sampled_items) for name, pool in pools.items()},
            "arrival_plan": _sanitize_arrival_plan_for_export(arrival_plan),
        },
    }
    return {
        "plan_agent_visible_config": base["plan_agent_visible_config"],
        "prepare_fingerprint": _config_prepare_fingerprint(base),
        "sampling_config": base["sampling_config"],
    }


def _prepare_number(path: Path) -> int | None:
    if not path.name.startswith("prepare_"):
        return None
    try:
        return int(path.name.removeprefix("prepare_"))
    except ValueError:
        return None


def _next_prepare_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    numbers = [
        number
        for path in base_dir.glob("prepare_*")
        if path.is_dir() and (number := _prepare_number(path)) is not None
    ]
    return base_dir / f"prepare_{(max(numbers) + 1) if numbers else 0}"


async def _prepare_workload(
    config: Dict[str, Any],
    prepare_dir: Path,
    pools: Dict[str, BenchmarkPool],
    arrival_plan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prepare_dir.mkdir(parents=True, exist_ok=False)
    config_prepare = _build_config_prepare(
        config=config,
        pools=pools,
        arrival_plan=arrival_plan,
    )
    _write_json(prepare_dir / "config_prepare.json", config_prepare)

    for index, entry in enumerate(arrival_plan, 1):
        print(
            json.dumps(
                {
                    "prepare": "planning",
                    "index": index,
                    "total": len(arrival_plan),
                    "job_id": entry["job_id"],
                    "benchmark": entry["benchmark"],
                    "logical_id": entry["logical_id"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        await _prepare_plan_record(entry, prepare_dir / f"{entry['job_id']}.json")

    return {
        "prepare_dir": str(prepare_dir),
        "config_prepare_path": str(prepare_dir / "config_prepare.json"),
        "prepare_fingerprint": config_prepare.get("prepare_fingerprint"),
        "planned_jobs": len(arrival_plan),
    }


def _resolve_prepare_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "config_prepare.json":
        return candidate.parent
    return candidate


def _load_prepare_dir(path: str | Path) -> Dict[str, Any]:
    prepare_dir = _resolve_prepare_dir(str(path))
    config_path = prepare_dir / "config_prepare.json"
    config_prepare = _read_json(config_path)
    if not isinstance(config_prepare, dict):
        raise ValueError(f"config_prepare.json is not a JSON object: {config_path}")
    extra_keys = _config_prepare_extra_keys(config_prepare)
    if extra_keys:
        raise ValueError(
            f"config_prepare.json has fields that are not part of prepare fingerprint: "
            f"{sorted(extra_keys)}"
        )
    sampling_config = config_prepare.get("sampling_config")
    if not isinstance(sampling_config, dict):
        raise ValueError(f"config_prepare.json has no sampling_config: {config_path}")
    arrival_plan = sampling_config.get("arrival_plan")
    if not isinstance(arrival_plan, list):
        raise ValueError(f"config_prepare.json has no sampling_config.arrival_plan: {config_path}")

    job_records: Dict[str, Dict[str, Any]] = {}
    for entry in arrival_plan:
        if not isinstance(entry, dict) or not entry.get("job_id"):
            raise ValueError(f"Invalid arrival plan entry in {config_path}")
        job_id = str(entry["job_id"])
        job_path = prepare_dir / f"{job_id}.json"
        record = _read_json(job_path)
        if not isinstance(record, dict):
            raise ValueError(f"Replay plan is not a JSON object: {job_path}")
        job_records[job_id] = record

    return {
        "prepare_dir": prepare_dir,
        "config_prepare": config_prepare,
        "job_records": job_records,
    }


def _find_matching_prepare(
    base_dir: Path,
    expected_config_prepare: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not base_dir.exists():
        return None
    expected_fingerprint = expected_config_prepare.get("prepare_fingerprint")
    candidates = [
        path
        for path in base_dir.glob("prepare_*")
        if path.is_dir() and _prepare_number(path) is not None
    ]
    candidates.sort(key=lambda path: _prepare_number(path) or -1, reverse=True)
    for candidate in candidates:
        config_path = candidate / "config_prepare.json"
        if not config_path.exists():
            continue
        config_prepare = _read_json_if_exists(config_path, {})
        if not isinstance(config_prepare, dict):
            continue
        if _config_prepare_extra_keys(config_prepare):
            continue
        fingerprint = _config_prepare_fingerprint(config_prepare)
        if fingerprint == expected_fingerprint:
            return _load_prepare_dir(candidate)
    return None


def _attach_prepare_replay(
    arrival_plan: List[Dict[str, Any]],
    prepare: Dict[str, Any],
) -> List[Dict[str, Any]]:
    config_prepare = prepare["config_prepare"]
    prepared_arrival = (
        config_prepare.get("sampling_config", {}).get("arrival_plan")
        if isinstance(config_prepare.get("sampling_config"), dict)
        else None
    )
    current_sanitized = _sanitize_arrival_plan_for_export(arrival_plan)
    if prepared_arrival != current_sanitized:
        raise ValueError(
            f"Prepare directory {prepare['prepare_dir']} does not match the current sampling plan"
        )

    job_records = prepare["job_records"]
    replay_plan: List[Dict[str, Any]] = []
    for entry in arrival_plan:
        job_id = entry["job_id"]
        record = job_records.get(job_id)
        if record is None:
            raise ValueError(f"Prepare directory {prepare['prepare_dir']} is missing {job_id}.json")
        if (
            record.get("benchmark") != entry["benchmark"]
            or record.get("logical_id") != entry["logical_id"]
        ):
            raise ValueError(f"Replay plan {job_id}.json does not match current sampling")
        replay_entry = dict(entry)
        replay_entry["replay_plans"] = [record]
        replay_plan.append(replay_entry)
    return replay_plan


def _ensure_runtime_assets(arrival_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    swe_entries = [
        entry
        for entry in arrival_plan
        if entry.get("benchmark") == "swebench"
    ]
    if not swe_entries:
        return {"swebench_images": {}}

    from app.config import config as app_config

    if not app_config.sandbox.use_sandbox:
        return {
            "swebench_images": {},
            "warning": "sandbox.use_sandbox=false; SWE-bench instance images were not prepared",
        }

    from experiments.SWEBench.sandbox_env import ensure_instance_image

    images: Dict[str, str] = {}
    for index, entry in enumerate(swe_entries, 1):
        logical_id = entry["logical_id"]
        if logical_id in images:
            continue
        print(
            json.dumps(
                {
                    "runtime_prepare": "swebench_image",
                    "index": index,
                    "total": len(swe_entries),
                    "job_id": entry["job_id"],
                    "logical_id": logical_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        images[logical_id] = ensure_instance_image(entry["payload"]["instance"])
    return {"swebench_images": images}


def _stats(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        frac = idx - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "p50": round(percentile(0.50), 4),
        "p95": round(percentile(0.95), 4),
        "p99": round(percentile(0.99), 4),
        "max": round(ordered[-1], 4),
    }


def _compact_stats(values: List[float]) -> Dict[str, float | None]:
    full = _stats(values)
    return {
        "count": full["count"],
        "mean": full["mean"],
        "p95": full["p95"],
        "max": full["max"],
    }


def _as_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    result: List[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_max_stats(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def _normalize_tool_name(name: Any) -> str:
    text = str(name or "unknown").strip()
    return text or "unknown"


def _sum_or_none(values: List[float]) -> float | None:
    return round(sum(values), 4) if values else None


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return default


def _backend_prefill_start_time(metrics: Dict[str, Any]) -> float | None:
    value = _as_float(metrics.get("backend_prefill_start_ts"))
    if value is not None and value > 1_000_000_000:
        return value
    return None


def _load_llm_call_records(job_dir: Path) -> List[Dict[str, Any]]:
    for trace_name in ("trace.json", "deadline_debug_trace.json"):
        trace = _read_json_if_exists(job_dir / trace_name, {})
        if isinstance(trace, dict):
            calls = trace.get("llm_calls")
            if isinstance(calls, list):
                return [call for call in calls if isinstance(call, dict)]

    for calls_name in ("llm_calls.json", "deadline_debug_llm_calls.json"):
        calls = _read_json_if_exists(job_dir / calls_name, [])
        if isinstance(calls, list):
            return [call for call in calls if isinstance(call, dict)]
    return []


def _compute_job_queueing_delay(llm_call_records: List[Dict[str, Any]]) -> float | None:
    agent_calls = [
        call
        for call in llm_call_records
        if isinstance(call, dict) and call.get("scope") == "agent"
    ]
    if not agent_calls:
        return None

    first_call = min(
        agent_calls,
        key=lambda call: _as_float(call.get("start_time")) or float("inf"),
    )
    prompt_ready_time = _as_float(first_call.get("prompt_ready_time"))
    metrics = first_call.get("response_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    prefill_start_time = _backend_prefill_start_time(metrics)
    if prompt_ready_time is None or prefill_start_time is None:
        return None

    delay = prefill_start_time - prompt_ready_time
    if delay < -0.01:
        return None
    return round(max(0.0, delay), 4)


def _phase_duration(phase: Dict[str, Any]) -> float | None:
    duration = _as_float(phase.get("duration_s"))
    if duration is not None:
        return duration
    start = _as_float(phase.get("start_time"))
    end = _as_float(phase.get("end_time"))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def _compute_agent_graph_metrics(meta: Dict[str, Any]) -> Dict[str, Any]:
    flow = meta.get("flow", {}) if isinstance(meta, dict) else {}
    phases = flow.get("phases", []) if isinstance(flow, dict) else []
    if not isinstance(phases, list):
        phases = []

    agents = meta.get("agents", {}) if isinstance(meta, dict) else {}
    agent_ids = set(agents.keys()) if isinstance(agents, dict) else set()
    agent_latency_total_s = 0.0
    critical_path_latency_s = 0.0
    agent_graph_depth = 0

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_type = phase.get("type")
        if not phase.get("count_toward_breakdown", True):
            continue

        sub_phases = phase.get("sub_phases") or []
        if phase_type == "parallel_batch" and isinstance(sub_phases, list) and sub_phases:
            sub_durations: List[float] = []
            for sub_phase in sub_phases:
                if not isinstance(sub_phase, dict):
                    continue
                agent_id = sub_phase.get("agent_id")
                if agent_id:
                    agent_ids.add(str(agent_id))
                duration = _phase_duration(sub_phase)
                if duration is not None:
                    sub_durations.append(duration)
                    agent_latency_total_s += duration
            if sub_durations:
                critical_path_latency_s += max(sub_durations)
                agent_graph_depth += 1
            else:
                duration = _phase_duration(phase)
                if duration is not None:
                    critical_path_latency_s += duration
            continue

        for agent_id in phase.get("agent_ids") or []:
            if agent_id:
                agent_ids.add(str(agent_id))
        duration = _phase_duration(phase)
        if duration is None:
            continue
        if phase_type in {"agent_run", "orla_stage"}:
            critical_path_latency_s += duration
            agent_latency_total_s += duration
            agent_graph_depth += 1

    parallelism = None
    if critical_path_latency_s > 0:
        parallelism = round(agent_latency_total_s / critical_path_latency_s, 4)

    return {
        "num_agents": len(agent_ids),
        "agent_graph_depth": agent_graph_depth if phases else None,
        "critical_path_latency_s": (
            round(critical_path_latency_s, 4) if critical_path_latency_s > 0 else None
        ),
        "agent_e2e_latency_total_s": (
            round(agent_latency_total_s, 4) if agent_latency_total_s > 0 else None
        ),
        "parallelism": parallelism,
    }


def _clip_interval(
    start: Any,
    end: Any,
    window_start: float,
    window_end: float,
) -> tuple[float, float] | None:
    start_value = _as_float(start)
    end_value = _as_float(end)
    if start_value is None or end_value is None:
        return None
    clipped_start = max(start_value, window_start)
    clipped_end = min(end_value, window_end)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _concurrency_stats(
    intervals: List[tuple[float, float]],
    window_start: float | None,
    window_end: float | None,
) -> Dict[str, float | None]:
    if window_start is None or window_end is None or window_end <= window_start:
        return {"mean": None, "max": None}
    events: List[tuple[float, int]] = []
    for start, end in intervals:
        clipped = _clip_interval(start, end, window_start, window_end)
        if clipped is None:
            continue
        events.append((clipped[0], 1))
        events.append((clipped[1], -1))
    if not events:
        return {"mean": 0.0, "max": 0}

    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    max_current = 0
    last_time = window_start
    area = 0.0
    for timestamp, delta in events:
        if timestamp > last_time:
            area += current * (timestamp - last_time)
            last_time = timestamp
        current += delta
        max_current = max(max_current, current)
    if window_end > last_time:
        area += current * (window_end - last_time)
    return {
        "mean": round(area / (window_end - window_start), 4),
        "max": max_current,
    }


def _collect_llm_call_intervals(
    run_dir: Path,
    job_ids: set[str] | None = None,
) -> List[tuple[float, float]]:
    intervals: List[tuple[float, float]] = []
    jobs_dir = run_dir / "jobs"
    if not jobs_dir.exists():
        return intervals
    for job_dir in jobs_dir.glob("job_*"):
        if not job_dir.is_dir():
            continue
        if job_ids is not None and job_dir.name not in job_ids:
            continue
        for call in _load_llm_call_records(job_dir):
            if call.get("scope") != "agent":
                continue
            interval = _clip_interval(
                call.get("start_time"),
                call.get("end_time"),
                float("-inf"),
                float("inf"),
            )
            if interval is not None:
                intervals.append(interval)
    return intervals


def _metric_endpoint_from_base_url(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "/metrics", "", "", ""))


def _health_endpoint_from_base_url(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def _parse_gpu_ids(raw: Any) -> List[int]:
    result: List[int] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            result.append(int(text))
        except ValueError:
            continue
    return result


def _load_metrics_endpoints(path: Path = DEFAULT_LLM_CONFIG) -> List[str]:
    if not path.exists():
        return []
    raw_config = _load_toml(path)
    llm_config = raw_config.get("llm", {}) if isinstance(raw_config, dict) else {}
    endpoints: List[str] = []
    for family in ("lang", "vision", "image"):
        family_config = llm_config.get(family, {})
        if not isinstance(family_config, dict):
            continue
        for entry in family_config.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("backend") or "").lower() not in {"vllm", "sglang", "vllm-omni", "continuum"}:
                continue
            endpoint = _metric_endpoint_from_base_url(str(entry.get("base_url") or ""))
            if endpoint and endpoint not in endpoints:
                endpoints.append(endpoint)
    return endpoints


def _load_health_endpoints(path: Path = DEFAULT_LLM_CONFIG) -> List[str]:
    if not path.exists():
        return []
    raw_config = _load_toml(path)
    llm_config = raw_config.get("llm", {}) if isinstance(raw_config, dict) else {}
    endpoints: List[str] = []
    for family in ("lang", "vision", "image"):
        family_config = llm_config.get(family, {})
        if not isinstance(family_config, dict):
            continue
        for entry in family_config.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("backend") or "").lower() not in {"vllm", "sglang", "vllm-omni", "continuum"}:
                continue
            endpoint = _health_endpoint_from_base_url(str(entry.get("base_url") or ""))
            if endpoint and endpoint not in endpoints:
                endpoints.append(endpoint)
    return endpoints


def _load_resource_gpu_ids(path: Path = DEFAULT_LLM_CONFIG) -> List[int]:
    if not path.exists():
        return []
    raw_config = _load_toml(path)
    llm_config = raw_config.get("llm", {}) if isinstance(raw_config, dict) else {}
    gpu_ids: List[int] = []
    for family in ("lang", "vision", "image"):
        family_config = llm_config.get(family, {})
        if not isinstance(family_config, dict):
            continue
        for entry in family_config.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("backend") or "").lower() not in {"vllm", "sglang", "vllm-omni", "continuum"}:
                continue
            for gpu_id in _parse_gpu_ids(entry.get("gpus")):
                if gpu_id not in gpu_ids:
                    gpu_ids.append(gpu_id)
    return sorted(gpu_ids)


def _parse_prometheus_metric_values(text: str) -> Dict[str, List[float]]:
    parsed: Dict[str, List[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        value = _as_float(parts[1])
        if value is None:
            continue
        parsed.setdefault(name, []).append(value)
    return parsed


def _parse_prometheus_metrics(text: str) -> Dict[str, float]:
    return {
        name: values[-1]
        for name, values in _parse_prometheus_metric_values(text).items()
        if values
    }


def _merge_prometheus_metric_values(metrics_texts: List[str]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for text in metrics_texts:
        for name, values in _parse_prometheus_metric_values(text).items():
            merged.setdefault(name, []).extend(values)
    return merged


def _first_metric(metrics: Dict[str, List[float]], names: List[str], mode: str = "last") -> float | None:
    for name in names:
        values = metrics.get(name)
        if not values:
            continue
        if mode == "sum":
            return sum(values)
        if mode == "mean":
            return statistics.fmean(values)
        if mode == "max":
            return max(values)
        return values[-1]
    return None


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _sum_metrics(metrics: Dict[str, List[float]], names: List[str]) -> float | None:
    values = [
        value
        for name in names
        for value in metrics.get(name, [])
    ]
    if not values:
        return None
    return sum(values)


def _extract_kv_metrics(metrics_texts: List[str]) -> Dict[str, float | None]:
    merged = _merge_prometheus_metric_values(metrics_texts)
    vllm_prefix_queries = _sum_metrics(
        merged,
        [
            "vllm:prefix_cache_queries_total",
            "vllm_prefix_cache_queries_total",
            "vllm:external_prefix_cache_queries_total",
            "vllm_external_prefix_cache_queries_total",
        ],
    )
    vllm_prefix_hits = _sum_metrics(
        merged,
        [
            "vllm:prefix_cache_hits_total",
            "vllm_prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits_total",
            "vllm_external_prefix_cache_hits_total",
        ],
    )
    sglang_cache_hit_rate = _first_metric(
        merged,
        [
            "sglang:cache_hit_rate",
            "sglang_cache_hit_rate",
        ],
        mode="mean",
    )
    return {
        "kv_cache_usage": _first_metric(
            merged,
            [
                "vllm:kv_cache_usage_perc",
                "vllm_kv_cache_usage_perc",
                "vllm:gpu_cache_usage_perc",
                "vllm_gpu_cache_usage_perc",
                "sglang:kv_cache_usage",
                "sglang_kv_cache_usage",
                "sglang:token_usage",
                "sglang_token_usage",
            ],
            mode="max",
        ),
        "kv_cache_reuse_rate": _first_metric(
            merged,
            [
                "kv_cache_reuse_rate",
                "sglang:kv_cache_reuse_rate",
                "sglang_kv_cache_reuse_rate",
                "vllm:kv_cache_reuse_rate",
                "vllm_kv_cache_reuse_rate",
            ],
            mode="mean",
        ),
        "kv_cache_hit_rate": sglang_cache_hit_rate,
        "kv_cache_query_tokens_total": vllm_prefix_queries,
        "kv_cache_hit_tokens_total": vllm_prefix_hits,
    }


def _sample_gpu_sm_util(gpu_indices: List[int] | None = None) -> float | None:
    selected = set(gpu_indices or [])
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        values = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            if selected and index not in selected:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            values.append(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
        return statistics.fmean(values) if values else None
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    numeric: List[float] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if selected and index not in selected:
            continue
        value = _as_float(parts[1])
        if value is not None:
            numeric.append(value)
    return statistics.fmean(numeric) if numeric else None


class _ResourceMonitor:
    def __init__(
        self,
        metrics_endpoints: List[str],
        gpu_indices: List[int] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        self.metrics_endpoints = metrics_endpoints
        self.gpu_indices = sorted(set(gpu_indices or []))
        self.interval_s = interval_s
        self.samples: List[Dict[str, Any]] = []
        self._stop = asyncio.Event()
        self._last_cpu_times: Dict[int, float] = {}
        self._last_cpu_sample_time: float | None = None
        self._last_docker_cpu_times: Dict[str, float] = {}
        self._last_docker_cpu_sample_time: float | None = None
        self._last_docker_cpu_usage: float | None = None
        self._next_docker_cpu_sample_time: float = 0.0
        self._last_kv_query_tokens_total: float | None = None
        self._last_kv_hit_tokens_total: float | None = None

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(await asyncio.to_thread(self._sample_once))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                continue

    def _sample_once(self) -> Dict[str, Any]:
        metrics_texts: List[str] = []
        for endpoint in self.metrics_endpoints:
            try:
                with urlopen(endpoint, timeout=1.0) as response:
                    metrics_texts.append(response.read().decode("utf-8", errors="replace"))
            except Exception:
                continue
        process_cpu = self._sample_process_tree_cpu_usage()
        docker_cpu = self._sample_docker_container_cpu_usage()
        combined_cpu_values = [
            value
            for value in (process_cpu, docker_cpu)
            if value is not None
        ]
        sample: Dict[str, Any] = {
            "timestamp": time.time(),
            "cpu_usage": min(100.0, sum(combined_cpu_values)) if combined_cpu_values else None,
            "cpu_usage_process_tree": process_cpu,
            "cpu_usage_docker_containers": docker_cpu,
            "gpu_sm_util": _sample_gpu_sm_util(self.gpu_indices),
        }
        kv_metrics = _extract_kv_metrics(metrics_texts)
        needs_counter_rate = (
            kv_metrics.get("kv_cache_reuse_rate") is None
            or kv_metrics.get("kv_cache_hit_rate") is None
        )
        if needs_counter_rate:
            counter_rate = self._sample_kv_cache_counter_rate(
                kv_metrics.get("kv_cache_hit_tokens_total"),
                kv_metrics.get("kv_cache_query_tokens_total"),
            )
            if kv_metrics.get("kv_cache_reuse_rate") is None:
                kv_metrics["kv_cache_reuse_rate"] = counter_rate
            if kv_metrics.get("kv_cache_hit_rate") is None:
                kv_metrics["kv_cache_hit_rate"] = counter_rate
        else:
            self._update_kv_counters(
                kv_metrics.get("kv_cache_hit_tokens_total"),
                kv_metrics.get("kv_cache_query_tokens_total"),
            )
        sample.update(kv_metrics)
        return sample

    def _update_kv_counters(
        self,
        hit_tokens_total: float | None,
        query_tokens_total: float | None,
    ) -> None:
        if hit_tokens_total is not None:
            self._last_kv_hit_tokens_total = hit_tokens_total
        if query_tokens_total is not None:
            self._last_kv_query_tokens_total = query_tokens_total

    def _sample_kv_cache_counter_rate(
        self,
        hit_tokens_total: float | None,
        query_tokens_total: float | None,
    ) -> float | None:
        previous_hits = self._last_kv_hit_tokens_total
        previous_queries = self._last_kv_query_tokens_total
        self._update_kv_counters(hit_tokens_total, query_tokens_total)
        if (
            hit_tokens_total is None
            or query_tokens_total is None
            or previous_hits is None
            or previous_queries is None
        ):
            return None
        hit_delta = hit_tokens_total - previous_hits
        query_delta = query_tokens_total - previous_queries
        if hit_delta < 0 or query_delta <= 0:
            return None
        return _ratio_or_none(hit_delta, query_delta)

    def _sample_process_tree_cpu_usage(self) -> float | None:
        try:
            import psutil  # type: ignore

            root = psutil.Process(os.getpid())
            processes = [root] + root.children(recursive=True)
            now = time.time()
            current: Dict[int, float] = {}
            for proc in processes:
                try:
                    cpu_times = proc.cpu_times()
                    current[proc.pid] = float(cpu_times.user + cpu_times.system)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            previous = self._last_cpu_times
            previous_time = self._last_cpu_sample_time
            self._last_cpu_times = current
            self._last_cpu_sample_time = now
            if previous_time is None:
                return None
            wall_delta = now - previous_time
            if wall_delta <= 0:
                return None
            cpu_delta = 0.0
            for pid, cpu_time in current.items():
                prior = previous.get(pid)
                if prior is not None and cpu_time >= prior:
                    cpu_delta += cpu_time - prior
            cpu_count = psutil.cpu_count() or 1
            return min(100.0, (cpu_delta / wall_delta) * 100.0 / cpu_count)
        except Exception:
            return None

    @staticmethod
    def _is_experiment_docker_container(container: Any) -> bool:
        name = str(getattr(container, "name", "") or "")
        if name.startswith("sandbox_"):
            return True
        return False

    def _sample_docker_container_cpu_usage(self) -> float | None:
        try:
            import psutil  # type: ignore

            now = time.time()
            if now < self._next_docker_cpu_sample_time:
                return self._last_docker_cpu_usage
            self._next_docker_cpu_sample_time = now + 5.0

            ps_result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=^/sandbox_",
                    "--format",
                    "{{.ID}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.2,
            )
            if ps_result.returncode != 0:
                self._next_docker_cpu_sample_time = now + 30.0
                return None
            container_ids = [
                line.strip()
                for line in ps_result.stdout.splitlines()
                if line.strip()
            ]
            if not container_ids:
                self._last_docker_cpu_usage = 0.0
                return 0.0

            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{json .}}",
                    *container_ids,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if result.returncode != 0:
                self._next_docker_cpu_sample_time = now + 30.0
                return None

            cpu_delta = 0.0
            seen_container = False
            for line in result.stdout.splitlines():
                try:
                    stats = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(stats.get("Name") or stats.get("Container") or "")
                if not name.startswith("sandbox_"):
                    continue
                raw_cpu = str(stats.get("CPUPerc") or "").strip().rstrip("%")
                try:
                    cpu_delta += float(raw_cpu)
                    seen_container = True
                except ValueError:
                    continue
            if not seen_container:
                self._last_docker_cpu_usage = 0.0
                return 0.0
            cpu_count = psutil.cpu_count() or 1
            self._last_docker_cpu_usage = min(100.0, cpu_delta / cpu_count)
            return self._last_docker_cpu_usage
        except Exception:
            self._next_docker_cpu_sample_time = time.time() + 30.0
            return None


class _BackendHealthMonitor:
    def __init__(
        self,
        health_endpoints: List[str],
        interval_s: float = 5.0,
    ) -> None:
        self.health_endpoints = health_endpoints
        self.interval_s = interval_s
        self.samples: List[Dict[str, Any]] = []
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        if not self.health_endpoints:
            return
        while not self._stop.is_set():
            self.samples.append(await asyncio.to_thread(self._sample_once))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                continue

    def _sample_once(self) -> Dict[str, Any]:
        checked_at = time.time()
        endpoint_results: List[Dict[str, Any]] = []
        for endpoint in self.health_endpoints:
            start = time.time()
            result: Dict[str, Any] = {
                "endpoint": endpoint,
                "ok": False,
            }
            try:
                with urlopen(endpoint, timeout=1.0) as response:
                    status_code = int(getattr(response, "status", 0) or 0)
                    result.update(
                        {
                            "ok": 200 <= status_code < 300,
                            "status_code": status_code,
                            "latency_s": round(time.time() - start, 4),
                        }
                    )
            except Exception as exc:
                result.update(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": round(time.time() - start, 4),
                    }
                )
            endpoint_results.append(result)
        return {
            "timestamp": checked_at,
            "ok": all(item.get("ok") for item in endpoint_results),
            "endpoints": endpoint_results,
        }


def _aggregate_backend_health_samples(
    samples: List[Dict[str, Any]],
    *,
    wall_start_time: float | None = None,
) -> Dict[str, Any]:
    endpoint_checks = [
        endpoint
        for sample in samples
        if isinstance(sample, dict)
        for endpoint in sample.get("endpoints", []) or []
        if isinstance(endpoint, dict)
    ]
    failed_checks = [
        endpoint for endpoint in endpoint_checks if not bool(endpoint.get("ok"))
    ]
    first_failure_time = None
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("ok", True):
            continue
        ts = _as_float(sample.get("timestamp"))
        if ts is not None:
            first_failure_time = ts
            break
    first_failure_elapsed = (
        round(first_failure_time - wall_start_time, 4)
        if first_failure_time is not None and wall_start_time is not None
        else None
    )
    if not samples:
        status = "not_checked"
    elif failed_checks:
        status = "degraded"
    else:
        status = "ok"
    return {
        "status": status,
        "samples": len(samples),
        "checks": len(endpoint_checks),
        "failed_checks": len(failed_checks),
        "first_failure_time": first_failure_time,
        "first_failure_elapsed_s": first_failure_elapsed,
    }


def _aggregate_resource_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    def values_for(key: str) -> List[float]:
        values: List[float] = []
        for sample in samples:
            value = _as_float(sample.get(key))
            if value is not None:
                values.append(value)
        return values

    cpu_usage = _mean_max_stats(values_for("cpu_usage"))
    gpu_sm_util = _mean_max_stats(values_for("gpu_sm_util"))
    kv_cache_usage = _mean_max_stats(values_for("kv_cache_usage"))
    kv_cache_reuse_rate = _mean_max_stats(values_for("kv_cache_reuse_rate"))
    kv_cache_hit_rate = _mean_max_stats(values_for("kv_cache_hit_rate"))
    return {
        "samples": len(samples),
        "mean_cpu_usage": cpu_usage["mean"],
        "max_cpu_usage": cpu_usage["max"],
        "mean_sm_util": gpu_sm_util["mean"],
        "max_sm_util": gpu_sm_util["max"],
        "mean_kv_cache_usage": kv_cache_usage["mean"],
        "max_kv_cache_usage": kv_cache_usage["max"],
        "mean_kv_cache_reuse_rate": kv_cache_reuse_rate["mean"],
        "max_kv_cache_reuse_rate": kv_cache_reuse_rate["max"],
        "mean_kv_cache_hit_rate": kv_cache_hit_rate["mean"],
        "max_kv_cache_hit_rate": kv_cache_hit_rate["max"],
    }


def _load_inference_summary(job_dir: Path) -> Dict[str, Any]:
    meta_path = job_dir / "metadata.json"
    debug_deadline_snapshot = False
    if not meta_path.exists():
        meta_path = job_dir / "deadline_debug_metadata.json"
        debug_deadline_snapshot = True
        if not meta_path.exists():
            return {
                "llm_call_count": 0,
                "metrics_available": False,
                "reason": "metadata.json not found",
            }

    meta = _read_json(meta_path)
    llm_call_records = _load_llm_call_records(job_dir)
    llm_calls = meta.get("llm_calls", {}) if isinstance(meta, dict) else {}
    token_summary = meta.get("token_summary", {}) if isinstance(meta, dict) else {}
    agents = meta.get("agents", {}) if isinstance(meta, dict) else {}
    agent_graph_metrics = _compute_agent_graph_metrics(meta if isinstance(meta, dict) else {})
    client_latencies = _as_float_list(llm_calls.get("client_latency_s"))
    ttfts = _as_float_list(llm_calls.get("ttft_s"))
    backend_ttfts = _as_float_list(llm_calls.get("backend_ttft_s"))
    backend_queue_waits = _as_float_list(llm_calls.get("backend_queue_wait_s"))
    backend_latencies = _as_float_list(llm_calls.get("backend_latency_s"))
    kv_cache_reuse_rates = _as_float_list(llm_calls.get("kv_cache_reuse_rate"))
    kv_cache_reuse_stats = _stats(kv_cache_reuse_rates)
    orla_metric_fields = (
        "orla_ttft_s",
        "orla_tpot_s",
        "orla_queue_wait_s",
        "orla_scheduler_decision_s",
        "orla_dispatch_s",
        "orla_backend_request_latency_s",
        "orla_estimated_cost_usd",
    )
    orla_metrics = {
        field: _as_float_list(llm_calls.get(field))
        for field in orla_metric_fields
    }
    tool_call_latency_s = 0.0
    tool_call_count = 0
    tool_time_by_name: Dict[str, float] = {}
    if isinstance(agents, dict):
        for agent in agents.values():
            if not isinstance(agent, dict):
                continue
            for tool_call in agent.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_count += 1
                tool_name = _normalize_tool_name(tool_call.get("tool_name"))
                try:
                    duration = float(tool_call.get("duration_s") or 0.0)
                except (TypeError, ValueError):
                    continue
                tool_call_latency_s += duration
                tool_time_by_name[tool_name] = tool_time_by_name.get(tool_name, 0.0) + duration

    return {
        "llm_call_count": int(llm_calls.get("count") or len(client_latencies) or 0),
        "debug_deadline_snapshot": debug_deadline_snapshot,
        "metrics_available": bool(
            client_latencies
            or ttfts
            or backend_ttfts
            or backend_queue_waits
            or backend_latencies
            or any(orla_metrics.values())
        ),
        "prompt_capture_enabled": bool(llm_calls.get("prompt_capture_enabled")),
        "client_latency_total_s": _sum_or_none(client_latencies),
        "client_latency_s": _stats(client_latencies),
        "client_ttft_s": _stats(ttfts),
        "backend_ttft_total_s": _sum_or_none(backend_ttfts),
        "backend_ttft_s": _stats(backend_ttfts),
        "backend_queue_wait_total_s": _sum_or_none(backend_queue_waits),
        "backend_queue_wait_s": _stats(backend_queue_waits),
        "backend_latency_total_s": _sum_or_none(backend_latencies),
        "backend_latency_s": _stats(backend_latencies),
        "kv_cache_reuse_rate": kv_cache_reuse_stats,
        "orla_ttft_s": _stats(orla_metrics["orla_ttft_s"]),
        "orla_tpot_s": _stats(orla_metrics["orla_tpot_s"]),
        "orla_queue_wait_s": _stats(orla_metrics["orla_queue_wait_s"]),
        "orla_scheduler_decision_s": _stats(orla_metrics["orla_scheduler_decision_s"]),
        "orla_dispatch_s": _stats(orla_metrics["orla_dispatch_s"]),
        "orla_backend_request_latency_s": _stats(
            orla_metrics["orla_backend_request_latency_s"]
        ),
        "orla_queue_wait_total_s": _sum_or_none(orla_metrics["orla_queue_wait_s"]),
        "orla_scheduler_decision_total_s": _sum_or_none(
            orla_metrics["orla_scheduler_decision_s"]
        ),
        "orla_dispatch_total_s": _sum_or_none(orla_metrics["orla_dispatch_s"]),
        "orla_backend_request_latency_total_s": _sum_or_none(
            orla_metrics["orla_backend_request_latency_s"]
        ),
        "orla_estimated_cost_usd": _stats(orla_metrics["orla_estimated_cost_usd"]),
        "orla_estimated_cost_usd_sum": _sum_or_none(
            orla_metrics["orla_estimated_cost_usd"]
        ),
        "job_kv_cache_reuse_rate_mean": kv_cache_reuse_stats["mean"],
        "job_kv_cache_reuse_rate_max": kv_cache_reuse_stats["max"],
        "job_queueing_delay_s": _compute_job_queueing_delay(llm_call_records),
        "job_input_tokens": token_summary.get("total_input_tokens"),
        "job_output_tokens": token_summary.get("total_completion_tokens"),
        "job_total_tokens": token_summary.get("total_tokens"),
        "job_tool_call_latency_s": round(tool_call_latency_s, 4),
        "job_tool_call_count": tool_call_count,
        "job_tool_time_by_name_s": {
            name: round(duration, 4)
            for name, duration in sorted(tool_time_by_name.items())
        },
        **agent_graph_metrics,
    }


async def _run_worker_subprocess(
    job_spec_path: Path,
    job_deadline_monotonic: float,
) -> tuple[int, str, str, bool]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "experiments.MixedWorkloadBench.worker",
        "--job-spec",
        str(job_spec_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(_PROJECT_ROOT),
        start_new_session=True,
    )
    killed_for_deadline = False
    remaining_s = max(0.0, job_deadline_monotonic - time.monotonic())
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=remaining_s)
    except asyncio.TimeoutError:
        killed_for_deadline = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=max(0.0, DEADLINE_TERMINATION_GRACE_S),
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await proc.communicate()
    return (
        proc.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        killed_for_deadline,
    )


def _run_worker_preflight() -> None:
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.MixedWorkloadBench.worker",
            "--preflight",
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "MixedWorkloadBench worker preflight failed before scheduling any jobs.\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def _cleanup_existing_sandbox_containers() -> None:
    """Remove old leftover MAS sandbox containers before a new benchmark run.

    Each job owns its sandbox for at most that worker process lifetime. Any
    old ``sandbox_*`` container visible before scheduling starts is stale for
    this benchmark runner and can exhaust docker0/veth capacity.
    """
    if os.environ.get("MAS_SKIP_SANDBOX_PREFLIGHT_CLEANUP") == "1":
        return
    max_age_s = float(os.environ.get("MAS_SANDBOX_PREFLIGHT_CLEANUP_AGE_S", "1800"))
    try:
        import docker  # type: ignore

        client = docker.from_env()
        containers = client.containers.list(
            all=True,
            filters={"name": "sandbox_"},
        )
    except Exception as exc:
        print(
            f"Warning: unable to inspect existing sandbox containers: {exc}",
            file=sys.stderr,
        )
        return

    now = datetime.now(timezone.utc)
    container_ids: List[str] = []
    for container in containers:
        name = str(getattr(container, "name", "") or "")
        if not name.startswith("sandbox_"):
            continue
        created_raw = str(container.attrs.get("Created") or "")
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            age_s = (now - created_at).total_seconds()
        except Exception:
            age_s = max_age_s
        if age_s >= max_age_s:
            container_ids.append(container.id)

    if not container_ids:
        return
    rm = subprocess.run(
        ["docker", "rm", "-f", *container_ids],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if rm.returncode != 0:
        print(
            "Warning: failed to remove some stale sandbox containers before run:\n"
            f"{rm.stderr[-4000:]}",
            file=sys.stderr,
        )
    else:
        print(
            json.dumps(
                {
                    "sandbox_preflight_cleanup": "removed_stale_containers",
                    "count": len(container_ids),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


async def _execute_plan(
    config: Dict[str, Any],
    run_dir: Path,
    arrival_plan: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    monotonic_start = time.monotonic()
    wall_start = time.time()
    duration_s = int(config["duration_minutes"]) * 60
    arrival_window_end_wall = wall_start + duration_s
    run_timing = {
        "wall_start_time": wall_start,
        "arrival_window_end_time": arrival_window_end_wall,
        "experiment_deadline_time": arrival_window_end_wall,
        "global_deadline_enabled": False,
        "job_deadline_s": JOB_DEADLINE_S,
    }
    job_results: List[Dict[str, Any]] = []
    resource_monitor = _ResourceMonitor(
        _load_metrics_endpoints(),
        gpu_indices=_load_resource_gpu_ids(),
    )
    resource_monitor_task = asyncio.create_task(resource_monitor.run())
    backend_health_monitor = _BackendHealthMonitor(_load_health_endpoints())
    backend_health_monitor_task = asyncio.create_task(backend_health_monitor.run())

    async def run_one(plan_entry: Dict[str, Any]) -> None:
        await asyncio.sleep(max(0.0, plan_entry["arrival_offset_s"] - (time.monotonic() - monotonic_start)))
        job_start_monotonic = time.monotonic()
        start_wall = time.time()
        job_deadline_monotonic = job_start_monotonic + JOB_DEADLINE_S
        job_deadline_wall = start_wall + JOB_DEADLINE_S

        job_dir = jobs_dir / plan_entry["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        job_spec = {
            "job_id": plan_entry["job_id"],
            "dispatch_id": plan_entry["dispatch_id"],
            "benchmark": plan_entry["benchmark"],
            "logical_id": plan_entry["logical_id"],
            "runtime": plan_entry.get("runtime", config.get("runtime", "legacy")),
            "payload": plan_entry["payload"],
            "replay_plans": plan_entry.get("replay_plans", []),
            "job_dir": str(job_dir),
            "run_context": {
                "planned_arrival_offset_s": plan_entry["arrival_offset_s"],
                "wall_start_time": wall_start,
                "arrival_window_end_time": arrival_window_end_wall,
                "experiment_deadline_time": job_deadline_wall,
                "job_start_time": start_wall,
                "job_deadline_time": job_deadline_wall,
                "job_deadline_s": JOB_DEADLINE_S,
            },
        }
        job_spec_path = job_dir / "job_spec.json"
        with job_spec_path.open("w", encoding="utf-8") as f:
            json.dump(job_spec, f, indent=2, ensure_ascii=False)

        try:
            return_code, stdout_text, stderr_text, killed_for_deadline = await _run_worker_subprocess(
                job_spec_path,
                job_deadline_monotonic=job_deadline_monotonic,
            )
        finally:
            try:
                job_spec_path.unlink()
            except FileNotFoundError:
                pass
        end_wall = time.time()
        job_result_path = job_dir / "result.json"
        worker_result: Dict[str, Any] = {}
        if job_result_path.exists():
            worker_result = _read_json(job_result_path)
        else:
            worker_result = {
                "status": "failed",
                "error": "worker did not produce result.json",
            }
        inference_summary = _load_inference_summary(job_dir)

        job_results.append(
            {
                "job_id": plan_entry["job_id"],
                "dispatch_id": plan_entry["dispatch_id"],
                "benchmark": plan_entry["benchmark"],
                "logical_id": plan_entry["logical_id"],
                "planned_arrival_offset_s": plan_entry["arrival_offset_s"],
                "start_time": start_wall,
                "end_time": end_wall,
                "job_deadline_time": job_deadline_wall,
                "job_deadline_s": JOB_DEADLINE_S,
                "run_time_s": round(end_wall - start_wall, 4),
                "worker_returncode": return_code,
                "worker_stdout_tail": stdout_text[-4000:],
                "worker_stderr_tail": stderr_text[-4000:],
                "killed_for_deadline": killed_for_deadline,
                "worker_result_path": str(job_result_path),
                "inference": inference_summary,
                "worker_result": worker_result,
            }
        )

    try:
        await asyncio.gather(*(run_one(entry) for entry in arrival_plan))
    finally:
        run_timing["run_completion_time"] = time.time()
        resource_monitor.stop()
        backend_health_monitor.stop()
        await asyncio.gather(resource_monitor_task, backend_health_monitor_task)
        resource_path = run_dir / "resource_metrics.json"
        _write_json(
            resource_path,
            {
                "sample_interval_s": resource_monitor.interval_s,
                "metrics_endpoints": resource_monitor.metrics_endpoints,
                "gpu_indices": resource_monitor.gpu_indices,
                "samples": resource_monitor.samples,
            },
        )
        run_timing["resource_metrics_path"] = str(resource_path)
        run_timing["resources"] = _aggregate_resource_samples(resource_monitor.samples)
        backend_health_path = run_dir / "backend_health.json"
        backend_health = _aggregate_backend_health_samples(
            backend_health_monitor.samples,
            wall_start_time=wall_start,
        )
        _write_json(
            backend_health_path,
            {
                "sample_interval_s": backend_health_monitor.interval_s,
                "health_endpoints": backend_health_monitor.health_endpoints,
                "summary": backend_health,
                "samples": backend_health_monitor.samples,
            },
        )
        run_timing["backend_health_path"] = str(backend_health_path)
        run_timing["backend_health"] = backend_health
    return sorted(job_results, key=lambda item: item["dispatch_id"]), run_timing


def _summarize_posthoc_evaluations(job_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, int]] = {}
    for item in job_results:
        benchmark = item["benchmark"]
        evaluation = item["worker_result"].get("evaluation", {})
        status = evaluation.get("status", "not_run")
        bucket = summary.setdefault(benchmark, {})
        bucket[status] = bucket.get(status, 0) + 1
    return summary


def _job_succeeded(item: Dict[str, Any]) -> bool:
    return item.get("worker_result", {}).get("status") == "success"


def _job_killed(item: Dict[str, Any]) -> bool:
    return bool(item.get("killed_for_deadline"))


def _job_metric_eligible(item: Dict[str, Any]) -> bool:
    return _job_succeeded(item) and not _job_killed(item)


def _job_failed(item: Dict[str, Any]) -> bool:
    return not _job_succeeded(item) and not _job_killed(item)


def _posthoc_failed_generation_status(item: Dict[str, Any]) -> str:
    return "killed_for_deadline" if _job_killed(item) else "skipped_failed_generation"


def _export_error(item: Dict[str, Any], worker_result: Dict[str, Any]) -> str | None:
    if _job_killed(item):
        return "killed for deadline"
    if worker_result.get("status") == "success":
        return None

    payload = worker_result.get("payload", {})
    result_payload = payload.get("result_payload", {}) if isinstance(payload, dict) else {}
    return (
        worker_result.get("error")
        or (payload.get("error") if isinstance(payload, dict) else None)
        or (result_payload.get("error") if isinstance(result_payload, dict) else None)
        or (result_payload.get("generation_error") if isinstance(result_payload, dict) else None)
        or "generation did not succeed"
    )


def _job_dir_for_result(run_dir: Path, item: Dict[str, Any]) -> Path:
    worker_result_path = item.get("worker_result_path")
    if worker_result_path:
        return Path(worker_result_path).parent
    return run_dir / "jobs" / item["job_id"]


def _write_posthoc_evaluation(
    run_dir: Path,
    item: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> None:
    job_dir = _job_dir_for_result(run_dir, item)
    eval_path = job_dir / "posthoc_evaluation.json"
    worker_result = item.setdefault("worker_result", {})
    worker_result["evaluation"] = evaluation
    worker_result["evaluation_path"] = str(eval_path)
    item["evaluation_path"] = str(eval_path)
    _write_json(eval_path, evaluation)


def _parse_csv_date(raw: Any) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text.split()[0])
    except ValueError:
        return None


def _load_next_close_price(
    ticker: str,
    trading_date: str,
    data_dir: Path = DEEPFUND_DATA_DIR,
) -> tuple[date, float] | None:
    csv_path = data_dir / "daily" / f"{ticker.upper()}.csv"
    target_date = _parse_csv_date(trading_date)
    if target_date is None or not csv_path.exists():
        return None

    candidates: List[tuple[date, float]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_date = _parse_csv_date(row.get("date"))
                close_price = _as_float(row.get("close"))
                if row_date is None or close_price is None:
                    continue
                if row_date > target_date:
                    candidates.append((row_date, close_price))
    except OSError:
        return None

    if not candidates:
        return None

    next_date = min(candidate[0] for candidate in candidates)
    same_day_prices = [price for day, price in candidates if day == next_date]
    return next_date, same_day_prices[-1]


def _deepfund_profit_loss_label(pnl: float) -> str:
    if pnl > 1e-9:
        return "profit"
    if pnl < -1e-9:
        return "loss"
    return "flat"


def _find_deepfund_record_path(job_dir: Path, payload: Dict[str, Any]) -> Path | None:
    result_payload = payload.get("result_payload", {}) if isinstance(payload, dict) else {}
    trading_date = payload.get("trading_date") or result_payload.get("trading_date")

    candidate_dirs: List[Path] = []
    out_dir = result_payload.get("out_dir")
    if out_dir:
        candidate_dirs.append(Path(out_dir))
    candidate_dirs.extend(sorted((job_dir / "deepfund_results").glob("*")))

    candidate_files: List[Path] = []
    for candidate_dir in candidate_dirs:
        try:
            is_dir = candidate_dir.is_dir()
        except OSError:
            is_dir = False
        if not is_dir:
            continue
        try:
            files = [
                path
                for path in sorted(candidate_dir.glob("*.json"))
                if not (
                    path.name.endswith("_trace.json")
                    or path.name.endswith("_metadata.json")
                    or path.name.endswith("_llm_calls.json")
                    or path.name == "evaluation_report.json"
                )
            ]
        except OSError:
            continue
        if trading_date:
            exact = candidate_dir / f"{trading_date}.json"
            if exact in files:
                return exact
        candidate_files.extend(files)

    return candidate_files[0] if candidate_files else None


def _build_deepfund_posthoc_evaluation(
    job_dir: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    record_path = _find_deepfund_record_path(job_dir, payload)
    if record_path is None:
        return {
            "status": "missing_result",
            "reason": "DeepFundBench did not produce a trading-day result file",
        }

    record = _read_json_if_exists(record_path, None)
    if not isinstance(record, dict):
        return {
            "status": "missing_result",
            "reason": f"DeepFundBench result file is not readable: {record_path}",
        }

    config = record.get("config", {}) if isinstance(record.get("config"), dict) else {}
    portfolio = record.get("portfolio", {}) if isinstance(record.get("portfolio"), dict) else {}
    positions = portfolio.get("positions", {}) if isinstance(portfolio.get("positions"), dict) else {}

    initial_value = _as_float(config.get("initial_cashflow"))
    final_value = _as_float(portfolio.get("total_value"))
    cashflow = _as_float(portfolio.get("cashflow")) or 0.0
    if final_value is None:
        final_value = cashflow + sum(
            _as_float((position or {}).get("value")) or 0.0
            for position in positions.values()
            if isinstance(position, dict)
        )
    if initial_value is None:
        initial_value = final_value

    pnl = round(final_value - initial_value, 4)
    return_pct = round((pnl / initial_value) * 100.0, 6) if initial_value else None
    decisions = record.get("decisions", []) if isinstance(record.get("decisions"), list) else []

    metrics: Dict[str, Any] = {
        "initial_value": round(initial_value, 4),
        "final_value": round(final_value, 4),
        "pnl": pnl,
        "return_pct": return_pct,
        "profit_loss": _deepfund_profit_loss_label(pnl),
        "decisions_total": len(decisions),
        "decisions_buy": sum(
            1
            for decision in decisions
            if isinstance(decision, dict) and decision.get("action") == "Buy"
        ),
        "decisions_sell": sum(
            1
            for decision in decisions
            if isinstance(decision, dict) and decision.get("action") == "Sell"
        ),
        "decisions_hold": sum(
            1
            for decision in decisions
            if isinstance(decision, dict) and decision.get("action") == "Hold"
        ),
    }

    trading_date = str(record.get("trading_date") or "")
    next_value = cashflow
    next_prices: Dict[str, Dict[str, Any]] = {}
    next_dates: List[date] = []
    missing_next_prices: List[str] = []
    for ticker, position in positions.items():
        if not isinstance(position, dict):
            continue
        shares = int(_as_float(position.get("shares")) or 0)
        if shares <= 0:
            continue
        next_close = _load_next_close_price(str(ticker), trading_date)
        if next_close is None:
            missing_next_prices.append(str(ticker))
            continue
        next_date, close_price = next_close
        next_dates.append(next_date)
        next_prices[str(ticker)] = {
            "date": next_date.isoformat(),
            "close": round(close_price, 4),
            "shares": shares,
        }
        next_value += shares * close_price

    if not missing_next_prices:
        next_pnl = round(next_value - initial_value, 4)
        metrics.update(
            {
                "next_trading_day_value": round(next_value, 4),
                "next_trading_day_pnl": next_pnl,
                "next_trading_day_return_pct": (
                    round((next_pnl / initial_value) * 100.0, 6)
                    if initial_value
                    else None
                ),
                "next_trading_day_profit_loss": _deepfund_profit_loss_label(next_pnl),
            }
        )
        if next_dates:
            metrics["next_trading_date"] = max(next_dates).isoformat()
        if next_prices:
            metrics["next_trading_day_prices"] = next_prices

    evaluation: Dict[str, Any] = {
        "status": "completed",
        "metrics": metrics,
        "result_path": str(record_path),
    }
    if missing_next_prices:
        evaluation["warning"] = (
            "next-trading-day PnL skipped because close prices were missing for "
            + ", ".join(missing_next_prices)
        )
    return evaluation


def _build_swebench_posthoc_evaluation(
    report: Dict[str, Any] | None,
    instance_id: str,
    predictions_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "benchmark": "swebench",
        "instance_id": instance_id,
        "predictions_path": str(predictions_path),
        "report_path": str(report_path),
    }
    if report is None:
        return {
            **base,
            "status": "missing_report",
            "reason": "SWE-bench Docker evaluation did not produce a report",
        }

    resolved_ids = set(report.get("resolved_ids", []))
    unresolved_ids = set(report.get("unresolved_ids", []))
    error_ids = set(report.get("error_ids", []))
    completed_ids = set(report.get("completed_ids", []))
    empty_patch_ids = set(report.get("empty_patch_ids", []))

    if instance_id in resolved_ids:
        swe_status = "resolved"
        posthoc_status = "completed"
        resolved = True
    elif instance_id in unresolved_ids:
        swe_status = "unresolved"
        posthoc_status = "completed"
        resolved = False
    elif instance_id in error_ids:
        swe_status = "error"
        posthoc_status = "failed"
        resolved = False
    elif instance_id in empty_patch_ids:
        swe_status = "empty_patch"
        posthoc_status = "skipped_no_patch"
        resolved = False
    elif instance_id in completed_ids:
        swe_status = "completed"
        posthoc_status = "completed"
        resolved = None
    else:
        swe_status = "not_reported"
        posthoc_status = "missing_instance_report"
        resolved = None

    return {
        **base,
        "status": posthoc_status,
        "metrics": {
            "resolved": resolved,
            "swebench_status": swe_status,
        },
    }


def _run_swebench_posthoc_batch(
    run_dir: Path,
    job_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    from experiments.SWEBench.evaluation import (
        compute_resolved_rate,
        run_swebench_evaluation,
    )
    from experiments.SWEBench.utils import write_predictions

    candidates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for item in job_results:
        if item["benchmark"] != "swebench":
            continue
        worker_result = item.get("worker_result", {})
        payload = worker_result.get("payload", {})
        result_payload = payload.get("result_payload", {})
        instance_id = result_payload.get("instance_id") or payload.get("instance_id")
        model_patch = result_payload.get("model_patch", "")
        if worker_result.get("status") != "success" or not str(model_patch).strip():
            continue
        if not instance_id:
            continue
        candidates.append(
            (
                item,
                {
                    "instance_id": instance_id,
                    "model_name_or_path": result_payload.get(
                        "model_name_or_path", "mas-sys"
                    ),
                    "model_patch": model_patch,
                },
            )
        )

    if not candidates:
        return {}

    by_instance_id: Dict[str, List[tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for item, prediction in candidates:
        by_instance_id.setdefault(prediction["instance_id"], []).append(
            (item, prediction)
        )

    evaluations: Dict[str, Dict[str, Any]] = {}

    def run_batch(
        predictions: List[Dict[str, Any]],
        item_by_instance_id: Dict[str, Dict[str, Any]],
        artifact_dir: Path,
        run_id: str,
    ) -> None:
        predictions_path = artifact_dir / "predictions.jsonl"
        report_path = artifact_dir / "report.json"
        write_predictions(str(predictions_path), predictions)

        instance_ids = [prediction["instance_id"] for prediction in predictions]
        report = run_swebench_evaluation(
            predictions_path=str(predictions_path),
            dataset_name=SWE_DATASET_NAME,
            split=SWE_DATASET_SPLIT,
            run_id=run_id,
            max_workers=4,
            timeout=1800,
            instance_ids=instance_ids,
            output_dir=str(artifact_dir),
            report_path=str(report_path),
        )
        batch_metrics = compute_resolved_rate(report or {})

        for instance_id, item in item_by_instance_id.items():
            evaluation = _build_swebench_posthoc_evaluation(
                report,
                instance_id,
                predictions_path,
                report_path,
            )
            evaluation["batch_metrics"] = batch_metrics
            evaluations[item["job_id"]] = evaluation

    unique_predictions: List[Dict[str, Any]] = []
    unique_item_by_instance_id: Dict[str, Dict[str, Any]] = {}
    duplicate_entries: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for instance_id, entries in by_instance_id.items():
        if len(entries) == 1:
            item, prediction = entries[0]
            unique_predictions.append(prediction)
            unique_item_by_instance_id[instance_id] = item
        else:
            duplicate_entries.extend(entries)

    if unique_predictions:
        run_batch(
            unique_predictions,
            unique_item_by_instance_id,
            run_dir / "posthoc_artifacts" / "swebench",
            f"mixed_posthoc_{run_dir.name}",
        )

    for item, prediction in duplicate_entries:
        run_batch(
            [prediction],
            {
                prediction["instance_id"]: item,
            },
            run_dir / "posthoc_artifacts" / "swebench" / item["job_id"],
            f"mixed_posthoc_{run_dir.name}_{item['job_id']}",
        )
    return evaluations


async def _run_posthoc_evaluations(run_dir: Path, job_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from experiments.PaperBananaBench.eval_toolkits import evaluate_sample
    from experiments.PaperBananaBench.run_benchmark import (
        DATA_DIR as PAPERBANANA_DATA_DIR,
        _build_vision_eval_config,
    )
    from experiments.PinchBench.evaluation import evaluate_task
    from experiments.PinchBench.task_loader import load_all_tasks
    from experiments.ResearchBench.eval import compute_proposal_metrics

    vision_cfg = _build_vision_eval_config()
    pinch_tasks = {task.task_id: task for task in load_all_tasks(PINCH_TASKS_DIR)}
    swebench_evaluations = _run_swebench_posthoc_batch(run_dir, job_results)

    for item in job_results:
        worker_result = item.get("worker_result", {})
        payload = worker_result.get("payload", {})
        benchmark = item["benchmark"]
        evaluation: Dict[str, Any]

        if worker_result.get("status") != "success":
            evaluation = {
                "status": _posthoc_failed_generation_status(item),
                "reason": _export_error(item, worker_result),
            }
        elif benchmark == "research_bench":
            result_payload = payload.get("result_payload", {})
            metrics = compute_proposal_metrics(
                result_payload.get("ref_proposal", ""),
                result_payload.get("gen_proposal", ""),
            )
            evaluation = {
                "status": "completed",
                "metrics": metrics,
            }
        elif benchmark == "pinchbench":
            result_payload = payload.get("result_payload", {})
            task_id = result_payload.get("task_id") or payload.get("task_id") or item["logical_id"]
            task = pinch_tasks.get(task_id)
            if task is None:
                evaluation = {
                    "status": "missing_task_definition",
                    "reason": f"PinchBench task definition not found for {task_id}",
                }
            else:
                evaluated = await evaluate_task(
                    task,
                    result_payload.get("transcript", []),
                    result_payload.get("workspace_path", ""),
                )
                evaluation = {
                    "status": "completed",
                    "metrics": {
                        "combined_score": evaluated.get("combined_score"),
                        "automated_average": evaluated.get("automated_average"),
                        "llm_judge_average": evaluated.get("llm_judge_average"),
                    },
                    "result": evaluated,
                }
        elif benchmark == "swebench":
            evaluation = swebench_evaluations.get(item["job_id"])
            if evaluation is None:
                result_payload = payload.get("result_payload", {})
                reason = (
                    worker_result.get("error")
                    or payload.get("error")
                    or result_payload.get("generation_error")
                    or "generation did not produce a non-empty patch"
                )
                evaluation = {
                    "status": _posthoc_failed_generation_status(item),
                    "reason": reason,
                }
        elif benchmark == "paperbananabench":
            result_payload = dict(payload.get("result_payload", {}))
            if result_payload.get("error"):
                evaluation = {
                    "status": _posthoc_failed_generation_status(item),
                    "reason": result_payload.get("error"),
                }
            elif not result_payload.get("eval_image_field"):
                evaluation = {
                    "status": "skipped_no_generated_image",
                    "reason": "generation did not expose an evaluation image field",
                }
            else:
                evaluated = await evaluate_sample(
                    sample=result_payload,
                    task_name="diagram",
                    data_dir=PAPERBANANA_DATA_DIR,
                    **vision_cfg,
                )
                payload["result_payload"] = evaluated
                score_metrics = {
                    key: evaluated.get(key)
                    for key in (
                        "instruction_faithfulness_score",
                        "conciseness_score",
                        "readability_score",
                        "aesthetics_score",
                        "overall_score",
                    )
                    if evaluated.get(key) is not None
                }
                evaluation = {
                    "status": "completed",
                    "metrics": score_metrics,
                }
        elif benchmark == "deepfundbench":
            evaluation = _build_deepfund_posthoc_evaluation(
                _job_dir_for_result(run_dir, item),
                payload,
            )
        else:
            evaluation = {
                "status": "not_run",
                "reason": "benchmark does not define a posthoc evaluation step in MixedWorkloadBench",
            }

        _write_posthoc_evaluation(run_dir, item, evaluation)

    return job_results


def _sanitize_job_results_for_export(job_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for item in job_results:
        worker_result = item.get("worker_result", {})
        evaluation = _compact_evaluation(worker_result.get("evaluation"))
        sanitized.append(
            {
                "job_id": item["job_id"],
                "benchmark": item["benchmark"],
                "logical_id": item["logical_id"],
                "planned_arrival_offset_s": item["planned_arrival_offset_s"],
                "status": worker_result.get("status"),
                "error": _export_error(item, worker_result),
                "e2e_time_s": item["run_time_s"],
                "evaluation": evaluation,
                "inference": _compact_inference_summary(item.get("inference")),
            }
        )
    return sanitized


def _compact_evaluation(evaluation: Any) -> Dict[str, Any] | None:
    if not isinstance(evaluation, dict):
        return None

    compact: Dict[str, Any] = {"status": evaluation.get("status")}
    if evaluation.get("reason"):
        compact["reason"] = evaluation.get("reason")

    metrics = evaluation.get("metrics")
    if isinstance(metrics, dict):
        primary_keys = (
            "overall_score",
            "combined_score",
            "resolved",
            "swebench_status",
            "resolved_rate",
            "gpt_metric_score",
            "rouge_l",
            "embedding_sim",
            "pnl",
            "return_pct",
            "profit_loss",
            "next_trading_day_pnl",
            "next_trading_day_return_pct",
            "next_trading_day_profit_loss",
        )
        primary = {key: metrics[key] for key in primary_keys if key in metrics}
        if primary:
            compact["metrics"] = primary

    batch_metrics = evaluation.get("batch_metrics")
    if isinstance(batch_metrics, dict):
        compact["batch_metrics"] = {
            key: batch_metrics[key]
            for key in ("resolved_rate", "resolved", "total")
            if key in batch_metrics
        }
    return compact


def _compact_inference_summary(inference: Any) -> Dict[str, Any]:
    if not isinstance(inference, dict):
        return {"metrics_available": False}
    if not inference.get("metrics_available"):
        compact = {
            "metrics_available": False,
            "llm_calls": int(inference.get("llm_call_count") or 0),
        }
        if inference.get("debug_deadline_snapshot"):
            compact["debug_deadline_snapshot"] = True
        if inference.get("reason"):
            compact["reason"] = inference.get("reason")
        if inference.get("job_tool_call_count") is not None:
            compact["tool_calls"] = inference.get("job_tool_call_count")
        return compact

    compact: Dict[str, Any] = {
        "metrics_available": True,
        "llm_calls": int(inference.get("llm_call_count") or 0),
    }
    if inference.get("debug_deadline_snapshot"):
        compact["debug_deadline_snapshot"] = True
    scalar_keys = (
        "job_queueing_delay_s",
        "client_latency_total_s",
        "backend_ttft_total_s",
        "backend_queue_wait_total_s",
        "backend_latency_total_s",
        "critical_path_latency_s",
        "agent_e2e_latency_total_s",
        "parallelism",
        "orla_estimated_cost_usd_sum",
        "orla_queue_wait_total_s",
        "orla_scheduler_decision_total_s",
        "orla_dispatch_total_s",
        "orla_backend_request_latency_total_s",
        "job_tool_call_count",
    )
    for key in scalar_keys:
        if inference.get(key) is not None:
            compact[key] = inference.get(key)
    for key in ("num_agents", "agent_graph_depth"):
        if inference.get(key) is not None:
            compact[key] = inference.get(key)

    stat_fields = {
        "client_latency_s": ("p95", "client_latency_p95_s"),
        "client_ttft_s": ("p50", "client_ttft_p50_s"),
        "backend_ttft_s": ("p50", "backend_ttft_p50_s"),
        "backend_queue_wait_s": ("p95", "backend_queue_wait_p95_s"),
        "backend_latency_s": ("p95", "backend_latency_p95_s"),
        "kv_cache_reuse_rate": ("mean", "kv_cache_reuse_rate_mean"),
        "orla_ttft_s": ("p50", "orla_ttft_p50_s"),
        "orla_tpot_s": ("p50", "orla_tpot_p50_s"),
        "orla_queue_wait_s": ("p95", "orla_queue_wait_p95_s"),
        "orla_scheduler_decision_s": ("p95", "orla_scheduler_decision_p95_s"),
        "orla_dispatch_s": ("p95", "orla_dispatch_p95_s"),
        "orla_backend_request_latency_s": (
            "p95",
            "orla_backend_request_latency_p95_s",
        ),
    }
    for source_key, (stat_key, output_key) in stat_fields.items():
        stats = inference.get(source_key)
        if isinstance(stats, dict) and stats.get(stat_key) is not None:
            compact[output_key] = stats.get(stat_key)
    tool_times = inference.get("job_tool_time_by_name_s")
    if isinstance(tool_times, dict) and tool_times:
        compact["tool_time_by_name_s"] = tool_times
    return compact


def _inference_values(job_results: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for item in job_results:
        value = (item.get("inference") or {}).get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _aggregate_inference(job_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    orla_metric_keys = (
        "orla_queue_wait_total_s",
        "orla_scheduler_decision_total_s",
        "orla_dispatch_total_s",
        "orla_backend_request_latency_total_s",
        "orla_estimated_cost_usd_sum",
    )
    jobs_with_orla_metrics = sum(
        1
        for item in job_results
        if any(
            (item.get("inference") or {}).get(key) is not None
            for key in orla_metric_keys
        )
    )

    return {
        "jobs_with_metrics": sum(
            1 for item in job_results if (item.get("inference") or {}).get("metrics_available")
        ),
        "llm_calls": sum(
            int((item.get("inference") or {}).get("llm_call_count") or 0)
            for item in job_results
        ),
        "client_latency_total_s": _compact_stats(
            _inference_values(job_results, "client_latency_total_s")
        ),
        "backend_ttft_total_s": _compact_stats(
            _inference_values(job_results, "backend_ttft_total_s")
        ),
        "backend_queue_wait_total_s": _compact_stats(
            _inference_values(job_results, "backend_queue_wait_total_s")
        ),
        "backend_latency_total_s": _compact_stats(
            _inference_values(job_results, "backend_latency_total_s")
        ),
        "kv_cache_reuse_rate": _aggregate_job_kv_cache_reuse_rate(job_results),
        "orla": {
            "jobs_with_metrics": jobs_with_orla_metrics,
            "ttft_s": _compact_stats(
                _inference_values(job_results, "orla_ttft_p50_s")
            ),
            "tpot_s": _compact_stats(
                _inference_values(job_results, "orla_tpot_p50_s")
            ),
            "queue_wait_s": _compact_stats(
                _inference_values(job_results, "orla_queue_wait_p95_s")
            ),
            "scheduler_decision_s": _compact_stats(
                _inference_values(job_results, "orla_scheduler_decision_p95_s")
            ),
            "dispatch_s": _compact_stats(
                _inference_values(job_results, "orla_dispatch_p95_s")
            ),
            "backend_request_latency_s": _compact_stats(
                _inference_values(job_results, "orla_backend_request_latency_p95_s")
            ),
            "estimated_cost_usd": _compact_stats(
                _inference_values(job_results, "orla_estimated_cost_usd_sum")
            ),
            "queue_wait_total_s": _compact_stats(
                _inference_values(job_results, "orla_queue_wait_total_s")
            ),
            "scheduler_decision_total_s": _compact_stats(
                _inference_values(job_results, "orla_scheduler_decision_total_s")
            ),
            "dispatch_total_s": _compact_stats(
                _inference_values(job_results, "orla_dispatch_total_s")
            ),
            "backend_request_latency_total_s": _compact_stats(
                _inference_values(job_results, "orla_backend_request_latency_total_s")
            ),
            "estimated_cost_usd_total": _sum_or_none(
                _inference_values(job_results, "orla_estimated_cost_usd_sum")
            ),
        },
    }


def _aggregate_job_tokens(job_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | None]]:
    return {
        "job_total_tokens": _compact_stats(_inference_values(job_results, "job_total_tokens")),
        "job_input_tokens": _compact_stats(_inference_values(job_results, "job_input_tokens")),
        "job_output_tokens": _compact_stats(_inference_values(job_results, "job_output_tokens")),
    }


def _aggregate_job_tool_call_latency(job_results: List[Dict[str, Any]]) -> Dict[str, float | None]:
    values: List[float] = []
    for item in job_results:
        if not _job_succeeded(item):
            continue
        value = (item.get("inference") or {}).get("job_tool_call_latency_s")
        try:
            values.append(float(value or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return _compact_stats(values)


def _aggregate_job_tool_breakdown(job_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    call_counts: List[float] = []
    tool_names: set[str] = set()
    per_job_tool_times: List[Dict[str, float]] = []

    for item in job_results:
        if not _job_succeeded(item):
            continue
        inference = item.get("inference") or {}
        try:
            call_counts.append(float(inference.get("job_tool_call_count") or 0))
        except (TypeError, ValueError):
            call_counts.append(0.0)

        raw_tool_times = inference.get("job_tool_time_by_name_s")
        job_tool_times: Dict[str, float] = {}
        if isinstance(raw_tool_times, dict):
            for raw_name, raw_duration in raw_tool_times.items():
                name = _normalize_tool_name(raw_name)
                try:
                    duration = float(raw_duration or 0.0)
                except (TypeError, ValueError):
                    duration = 0.0
                job_tool_times[name] = job_tool_times.get(name, 0.0) + duration
                tool_names.add(name)
        per_job_tool_times.append(job_tool_times)

    by_tool = {
        name: _mean_max_stats(
            [job_tool_times.get(name, 0.0) for job_tool_times in per_job_tool_times]
        )
        for name in sorted(tool_names)
    }
    return {
        "jobs": len(call_counts),
        "tool_calls_per_job": _mean_max_stats(call_counts),
        "tool_time_per_job_s": by_tool,
    }


def _aggregate_inference_mean_max(job_results: List[Dict[str, Any]], key: str) -> Dict[str, float | None]:
    return _mean_max_stats(_inference_values(job_results, key))


def _aggregate_job_kv_cache_reuse_rate(job_results: List[Dict[str, Any]]) -> Dict[str, float | None]:
    means = _inference_values(job_results, "job_kv_cache_reuse_rate_mean")
    maxes = _inference_values(job_results, "job_kv_cache_reuse_rate_max")
    return {
        "count": len(means),
        "mean": round(statistics.fmean(means), 4) if means else None,
        "max": round(max(maxes), 4) if maxes else None,
    }


def _aggregate_concurrency(
    run_dir: Path,
    job_results: List[Dict[str, Any]],
    run_timing: Dict[str, Any],
) -> Dict[str, Any]:
    window_start = _as_float(run_timing.get("wall_start_time"))
    window_end = _as_float(
        run_timing.get("run_completion_time")
        or run_timing.get("experiment_deadline_time")
    )
    job_intervals = []
    for item in job_results:
        interval = _clip_interval(
            item.get("start_time"),
            item.get("end_time"),
            float("-inf"),
            float("inf"),
        )
        if interval is not None:
            job_intervals.append(interval)
    jobs = _concurrency_stats(job_intervals, window_start, window_end)
    llm_calls = _concurrency_stats(
        _collect_llm_call_intervals(
            run_dir,
            {str(item["job_id"]) for item in job_results},
        ),
        window_start,
        window_end,
    )
    return {
        "job_scope": "all_jobs",
        "jobs": len(job_results),
        "mean_concurrent_jobs": jobs["mean"],
        "max_concurrent_jobs": jobs["max"],
        "mean_concurrent_llm_calls": llm_calls["mean"],
        "max_concurrent_llm_calls": llm_calls["max"],
    }


def _with_resource_scope(resources: Dict[str, Any], job_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "job_scope": "all_jobs",
        "jobs": len(job_results),
        **resources,
    }


def _load_resource_samples(run_timing: Dict[str, Any]) -> List[Dict[str, Any]]:
    resource_path = run_timing.get("resource_metrics_path")
    if not resource_path:
        return []
    resource_metrics = _read_json_if_exists(Path(str(resource_path)), {})
    samples = resource_metrics.get("samples") if isinstance(resource_metrics, dict) else None
    if not isinstance(samples, list):
        return []
    return [sample for sample in samples if isinstance(sample, dict)]


def _load_backend_health_summary(run_timing: Dict[str, Any]) -> Dict[str, Any]:
    embedded = run_timing.get("backend_health")
    if isinstance(embedded, dict):
        return embedded
    health_path = run_timing.get("backend_health_path")
    if not health_path:
        return {}
    backend_health = _read_json_if_exists(Path(str(health_path)), {})
    summary = backend_health.get("summary") if isinstance(backend_health, dict) else None
    return summary if isinstance(summary, dict) else {}


def _aggregate_results(
    config: Dict[str, Any],
    run_dir: Path,
    pools: Dict[str, BenchmarkPool],
    arrival_plan: List[Dict[str, Any]],
    job_results: List[Dict[str, Any]],
    run_timing: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    per_benchmark: Dict[str, Dict[str, Any]] = {}
    posthoc_by_benchmark = _summarize_posthoc_evaluations(job_results)
    metric_jobs = [item for item in job_results if _job_metric_eligible(item)]
    run_values = [item["run_time_s"] for item in metric_jobs]
    for name, pool in pools.items():
        bench_jobs = [item for item in job_results if item["benchmark"] == name]
        bench_metric_jobs = [item for item in bench_jobs if _job_metric_eligible(item)]
        succeeded = sum(1 for item in bench_jobs if _job_succeeded(item))
        killed = sum(1 for item in bench_jobs if _job_killed(item))
        failed = sum(1 for item in bench_jobs if _job_failed(item))
        per_benchmark[name] = {
            "pool_size": len(pool.sampled_items),
            "submitted_jobs": len(bench_jobs),
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "killed_jobs": killed,
            "metric_jobs": len(bench_metric_jobs),
            "run_time_s": _compact_stats(
                [item["run_time_s"] for item in bench_metric_jobs]
            ),
            "job_queueing_delay": _aggregate_inference_mean_max(
                bench_metric_jobs, "job_queueing_delay_s"
            ),
            "tokens": _aggregate_job_tokens(bench_metric_jobs),
            "job_tool_call_latency": _aggregate_job_tool_call_latency(bench_metric_jobs),
            "job_tool_breakdown": _aggregate_job_tool_breakdown(bench_metric_jobs),
            "agents_per_job": _aggregate_inference_mean_max(
                bench_metric_jobs, "num_agents"
            ),
            "agent_graph_depth": _aggregate_inference_mean_max(
                bench_metric_jobs, "agent_graph_depth"
            ),
            "critical_path_latency": _aggregate_inference_mean_max(
                bench_metric_jobs, "critical_path_latency_s"
            ),
            "parallelism": _aggregate_inference_mean_max(
                bench_metric_jobs, "parallelism"
            ),
            "inference": _aggregate_inference(bench_metric_jobs),
            "posthoc_evaluation": posthoc_by_benchmark.get(name, {}),
        }

    succeeded_total = sum(1 for item in job_results if _job_succeeded(item))
    killed_total = sum(1 for item in job_results if _job_killed(item))
    failed_total = sum(1 for item in job_results if _job_failed(item))
    duration_s = int(config["duration_minutes"]) * 60
    run_timing = run_timing or {}
    arrival_window_end_time = _as_float(run_timing.get("arrival_window_end_time"))
    if arrival_window_end_time is None:
        arrival_window_end_time = _as_float(run_timing.get("experiment_deadline_time"))
    completed_within_window = (
        sum(
            1
            for item in job_results
            if (
                _as_float(item.get("end_time")) is not None
                and arrival_window_end_time is not None
                and (_as_float(item.get("end_time")) or 0.0) <= arrival_window_end_time
            )
        )
        if arrival_window_end_time is not None
        else None
    )
    resource_samples = _load_resource_samples(run_timing)
    backend_health = _load_backend_health_summary(run_timing)

    return {
        "run_dir": str(run_dir),
        "wall_start_time": run_timing.get("wall_start_time"),
        "arrival_window_end_time": run_timing.get("arrival_window_end_time"),
        "experiment_deadline_time": run_timing.get("experiment_deadline_time"),
        "run_completion_time": run_timing.get("run_completion_time"),
        "global_deadline_enabled": run_timing.get("global_deadline_enabled"),
        "job_deadline_s": run_timing.get("job_deadline_s"),
        "submitted_jobs": len(arrival_plan),
        "completed_jobs": len(job_results),
        "completed_within_window_jobs": completed_within_window,
        "succeeded_jobs": succeeded_total,
        "failed_jobs": failed_total,
        "killed_jobs": killed_total,
        "metric_jobs": len(metric_jobs),
        "target_jps": float(config["total_jps"]),
        "submitted_jps": round(len(arrival_plan) / duration_s, 6) if duration_s else None,
        "run_time_s": _compact_stats(run_values),
        "job_queueing_delay": _aggregate_inference_mean_max(
            metric_jobs, "job_queueing_delay_s"
        ),
        "concurrency": _aggregate_concurrency(run_dir, job_results, run_timing),
        "resources": _with_resource_scope(
            _aggregate_resource_samples(resource_samples),
            job_results,
        ),
        "backend_health": backend_health,
        "tokens": _aggregate_job_tokens(metric_jobs),
        "job_tool_call_latency": _aggregate_job_tool_call_latency(metric_jobs),
        "job_tool_breakdown": _aggregate_job_tool_breakdown(metric_jobs),
        "agents_per_job": _aggregate_inference_mean_max(metric_jobs, "num_agents"),
        "agent_graph_depth": _aggregate_inference_mean_max(
            metric_jobs, "agent_graph_depth"
        ),
        "critical_path_latency": _aggregate_inference_mean_max(
            metric_jobs, "critical_path_latency_s"
        ),
        "parallelism": _aggregate_inference_mean_max(metric_jobs, "parallelism"),
        "kv_cache_reuse_rate": _aggregate_job_kv_cache_reuse_rate(metric_jobs),
        "inference": _aggregate_inference(metric_jobs),
        "per_benchmark": per_benchmark,
    }


def _iter_summary_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for table_index, (section, metrics) in enumerate(SUMMARY_TABLES):
        if table_index > 0:
            rows.append({"Section": "", "Metric": ""})
        for metric in metrics:
            rows.append({"Section": section, "Metric": metric})
    return rows


def _read_summary_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return ["Section", "Metric"], _iter_summary_rows()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or ["Section", "Metric"])
        rows = [dict(row) for row in reader]
    if "Section" not in fieldnames:
        fieldnames.insert(0, "Section")
    if "Metric" not in fieldnames:
        fieldnames.insert(1, "Metric")
    if not rows:
        rows = _iter_summary_rows()
    return fieldnames, rows


def _write_summary_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _ensure_summary_csv(path: Path = DEFAULT_SUMMARY_CSV) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    _write_summary_csv(path, ["Section", "Metric"], _iter_summary_rows())


def _unique_summary_column(fieldnames: List[str], run_dir: Path) -> str:
    base = run_dir.name.removeprefix("mixed_workload_")
    column = base
    suffix = 2
    while column in fieldnames:
        column = f"{base}_{suffix}"
        suffix += 1
    return column


def _stat(data: Dict[str, Any], key: str) -> Any:
    if not isinstance(data, dict):
        return None
    return data.get(key)


def _elapsed_seconds(start: Any, end: Any) -> float | None:
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if end_f < start_f:
        return None
    return round(end_f - start_f, 4)


def _summary_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _summarize_llm_runtime() -> tuple[str, str]:
    snapshot = _load_llm_runtime_snapshot()
    backends: List[str] = []
    models: List[str] = []
    for family in ("lang", "vision", "image"):
        entries = snapshot.get(family, {})
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            backend = entry.get("backend")
            model = entry.get("model")
            if backend and str(backend) not in backends:
                backends.append(str(backend))
            if model and str(model) not in models:
                models.append(str(model))
    return ", ".join(backends) or "N/A", ", ".join(models) or "N/A"


def _summary_values(
    config: Dict[str, Any],
    overall: Dict[str, Any],
) -> Dict[tuple[str, str], Any]:
    backend, models = _summarize_llm_runtime()
    hardware = config.get("hardware") or os.getenv("MIXED_WORKLOAD_HARDWARE") or "N/A"
    run_time = overall.get("run_time_s", {})
    queueing = overall.get("job_queueing_delay", {})
    tokens = overall.get("tokens", {})
    token_total = tokens.get("job_total_tokens", {}) if isinstance(tokens, dict) else {}
    token_input = tokens.get("job_input_tokens", {}) if isinstance(tokens, dict) else {}
    token_output = tokens.get("job_output_tokens", {}) if isinstance(tokens, dict) else {}
    tool_latency = overall.get("job_tool_call_latency", {})
    tool_breakdown = overall.get("job_tool_breakdown", {})
    tool_calls_per_job = (
        tool_breakdown.get("tool_calls_per_job", {})
        if isinstance(tool_breakdown, dict)
        else {}
    )
    tool_time_per_job = (
        tool_breakdown.get("tool_time_per_job_s", {})
        if isinstance(tool_breakdown, dict)
        else {}
    )
    inference = overall.get("inference", {})
    client_latency = inference.get("client_latency_total_s", {}) if isinstance(inference, dict) else {}
    backend_latency = inference.get("backend_latency_total_s", {}) if isinstance(inference, dict) else {}
    backend_queue = inference.get("backend_queue_wait_total_s", {}) if isinstance(inference, dict) else {}
    orla = inference.get("orla", {}) if isinstance(inference, dict) else {}
    orla_queue = orla.get("queue_wait_total_s", {}) if isinstance(orla, dict) else {}
    orla_scheduler = orla.get("scheduler_decision_total_s", {}) if isinstance(orla, dict) else {}
    orla_dispatch = orla.get("dispatch_total_s", {}) if isinstance(orla, dict) else {}
    orla_backend = orla.get("backend_request_latency_total_s", {}) if isinstance(orla, dict) else {}
    orla_jobs = orla.get("jobs_with_metrics") if isinstance(orla, dict) else None
    has_orla_metrics = bool(orla_jobs)
    runtime = "orla" if has_orla_metrics else str(config.get("runtime") or "legacy")
    concurrency = overall.get("concurrency", {})
    resources = overall.get("resources", {})
    backend_health = overall.get("backend_health", {})
    agents = overall.get("agents_per_job", {})
    graph_depth = overall.get("agent_graph_depth", {})
    critical_path = overall.get("critical_path_latency", {})
    parallelism = overall.get("parallelism", {})
    job_kv_reuse = overall.get("kv_cache_reuse_rate", {})
    mean_kv_reuse = _stat(job_kv_reuse, "mean")
    max_kv_reuse = _stat(job_kv_reuse, "max")
    if mean_kv_reuse is None and isinstance(resources, dict):
        mean_kv_reuse = resources.get("mean_kv_cache_reuse_rate")
    if max_kv_reuse is None and isinstance(resources, dict):
        max_kv_reuse = resources.get("max_kv_cache_reuse_rate")

    values = {
        ("Experiment Setup", "Random seed"): config.get("seed"),
        ("Experiment Setup", "Test window (min)"): config.get("duration_minutes"),
        ("Experiment Setup", "Target total JPS"): overall.get("target_jps"),
        ("Experiment Setup", "Submitted JPS"): overall.get("submitted_jps"),
        ("Experiment Setup", "Backend"): backend,
        ("Experiment Setup", "Runtime"): runtime,
        ("Experiment Setup", "Hardware"): hardware,
        ("Experiment Setup", "Models"): models,
        ("Completion Results", "Submitted jobs"): overall.get("submitted_jobs"),
        ("Completion Results", "Jobs completed within test window"): overall.get(
            "completed_within_window_jobs"
        ),
        ("Completion Results", "Succeeded jobs"): overall.get("succeeded_jobs"),
        ("Completion Results", "Failed jobs"): overall.get("failed_jobs"),
        ("Completion Results", "Killed jobs"): overall.get("killed_jobs"),
        ("E2E Job", "Mean job runtime (s)"): _stat(run_time, "mean"),
        ("E2E Job", "P95 job runtime (s)"): _stat(run_time, "p95"),
        ("E2E Job", "Max job runtime (s)"): _stat(run_time, "max"),
        ("E2E Job", "Jobs with job queueing delay"): _stat(queueing, "count"),
        ("E2E Job", "Mean job queueing delay (s)"): _stat(queueing, "mean"),
        ("E2E Job", "Max job queueing delay (s)"): _stat(queueing, "max"),
        ("Token Metrics", "Jobs with token metrics"): _stat(token_total, "count"),
        ("Token Metrics", "Mean total tokens/job"): _stat(token_total, "mean"),
        ("Token Metrics", "P95 total tokens/job"): _stat(token_total, "p95"),
        ("Token Metrics", "Max total tokens/job"): _stat(token_total, "max"),
        ("Token Metrics", "Mean input tokens/job"): _stat(token_input, "mean"),
        ("Token Metrics", "Mean output tokens/job"): _stat(token_output, "mean"),
        ("Tool Metrics", "Jobs with tool latency metric"): _stat(tool_latency, "count"),
        ("Tool Metrics", "Mean tool-call latency/job (s)"): _stat(tool_latency, "mean"),
        ("Tool Metrics", "Max tool-call latency/job (s)"): _stat(tool_latency, "max"),
        ("Inference Metrics", "Jobs with metrics"): inference.get("jobs_with_metrics") if isinstance(inference, dict) else None,
        ("Inference Metrics", "LLM calls"): inference.get("llm_calls") if isinstance(inference, dict) else None,
        ("Inference Metrics", "Mean client latency total/job (s)"): _stat(client_latency, "mean"),
        ("Inference Metrics", "P95 client latency total/job (s)"): _stat(client_latency, "p95"),
        ("Inference Metrics", "Max client latency total/job (s)"): _stat(client_latency, "max"),
        ("Inference Metrics", "Mean backend latency total/job (s)"): _stat(backend_latency, "mean"),
        ("Inference Metrics", "P95 backend latency total/job (s)"): _stat(backend_latency, "p95"),
        ("Inference Metrics", "Max backend latency total/job (s)"): _stat(backend_latency, "max"),
        ("Inference Metrics", "Mean backend queue wait total/job (s)"): _stat(backend_queue, "mean"),
        ("Inference Metrics", "P95 backend queue wait total/job (s)"): _stat(backend_queue, "p95"),
        ("Inference Metrics", "Max backend queue wait total/job (s)"): _stat(backend_queue, "max"),
        ("Concurrency Metrics", "Mean concurrent jobs"): concurrency.get("mean_concurrent_jobs") if isinstance(concurrency, dict) else None,
        ("Concurrency Metrics", "Max concurrent jobs"): concurrency.get("max_concurrent_jobs") if isinstance(concurrency, dict) else None,
        ("Concurrency Metrics", "Mean concurrent LLM calls"): concurrency.get("mean_concurrent_llm_calls") if isinstance(concurrency, dict) else None,
        ("Concurrency Metrics", "Max concurrent LLM calls"): concurrency.get("max_concurrent_llm_calls") if isinstance(concurrency, dict) else None,
        ("Agent Graph Metrics", "Jobs with agent graph metrics"): _stat(agents, "count"),
        ("Agent Graph Metrics", "Mean agents/job"): _stat(agents, "mean"),
        ("Agent Graph Metrics", "Max agents/job"): _stat(agents, "max"),
        ("Agent Graph Metrics", "Mean agent graph depth"): _stat(graph_depth, "mean"),
        ("Agent Graph Metrics", "Max agent graph depth"): _stat(graph_depth, "max"),
        ("Agent Graph Metrics", "Mean critical path latency (s)"): _stat(critical_path, "mean"),
        ("Agent Graph Metrics", "Max critical path latency (s)"): _stat(critical_path, "max"),
        ("Agent Graph Metrics", "Mean parallelism"): _stat(parallelism, "mean"),
        ("Agent Graph Metrics", "Max parallelism"): _stat(parallelism, "max"),
        ("Resource Metrics", "Resource samples"): resources.get("samples") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Mean CPU usage (%)"): resources.get("mean_cpu_usage") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Max CPU usage (%)"): resources.get("max_cpu_usage") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Mean SM util (%)"): resources.get("mean_sm_util") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Max SM util (%)"): resources.get("max_sm_util") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Mean KV cache usage"): resources.get("mean_kv_cache_usage") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Max KV cache usage"): resources.get("max_kv_cache_usage") if isinstance(resources, dict) else None,
        ("Resource Metrics", "Mean KV reuse rate"): mean_kv_reuse,
        ("Resource Metrics", "Max KV reuse rate"): max_kv_reuse,
        ("Backend Health", "Backend health status"): backend_health.get("status") if isinstance(backend_health, dict) else None,
        ("Backend Health", "Backend health checks"): backend_health.get("checks") if isinstance(backend_health, dict) else None,
        ("Backend Health", "Backend health failed checks"): backend_health.get("failed_checks") if isinstance(backend_health, dict) else None,
        ("Backend Health", "First backend health failure elapsed (s)"): backend_health.get("first_failure_elapsed_s") if isinstance(backend_health, dict) else None,
        ("ORLA Metrics", "Jobs with ORLA metrics"): orla_jobs if has_orla_metrics else None,
        ("ORLA Metrics", "Mean ORLA queue wait/job (s)"): _stat(orla_queue, "mean") if has_orla_metrics else None,
        ("ORLA Metrics", "P95 ORLA queue wait/job (s)"): _stat(orla_queue, "p95") if has_orla_metrics else None,
        ("ORLA Metrics", "Mean ORLA scheduler decision/job (s)"): _stat(orla_scheduler, "mean") if has_orla_metrics else None,
        ("ORLA Metrics", "Mean ORLA dispatch/job (s)"): _stat(orla_dispatch, "mean") if has_orla_metrics else None,
        ("ORLA Metrics", "Mean ORLA backend request latency/job (s)"): _stat(orla_backend, "mean") if has_orla_metrics else None,
        ("ORLA Metrics", "P95 ORLA backend request latency/job (s)"): _stat(orla_backend, "p95") if has_orla_metrics else None,
        ("ORLA Metrics", "Total ORLA estimated cost (USD)"): orla.get("estimated_cost_usd_total") if has_orla_metrics and isinstance(orla, dict) else None,
        ("Run Timing", "Run completion elapsed (s)"): _elapsed_seconds(
            overall.get("wall_start_time"),
            overall.get("run_completion_time"),
        ),
        ("Tool Breakdown", "Jobs with tool breakdown metrics"): (
            tool_breakdown.get("jobs") if isinstance(tool_breakdown, dict) else None
        ),
        ("Tool Breakdown", "Mean tool calls/job"): _stat(tool_calls_per_job, "mean"),
        ("Tool Breakdown", "Max tool calls/job"): _stat(tool_calls_per_job, "max"),
    }
    if isinstance(tool_time_per_job, dict):
        for tool_name, stats in sorted(tool_time_per_job.items()):
            tool_label = _normalize_tool_name(tool_name)
            values[("Tool Breakdown", f"Mean {tool_label} time/job (s)")] = _stat(
                stats, "mean"
            )
            values[("Tool Breakdown", f"Max {tool_label} time/job (s)")] = _stat(
                stats, "max"
            )
    return values


def _append_summary_csv(
    config: Dict[str, Any],
    overall: Dict[str, Any],
    run_dir: Path,
    path: Path = DEFAULT_SUMMARY_CSV,
) -> None:
    fieldnames, rows = _read_summary_csv(path)
    expected_rows = _iter_summary_rows()
    if not rows:
        rows = expected_rows
    existing_keys = {(row.get("Section", ""), row.get("Metric", "")) for row in rows}
    for row in expected_rows:
        key = (row["Section"], row["Metric"])
        if row["Metric"] and key not in existing_keys:
            rows.append(dict(row))
            existing_keys.add(key)

    column = _unique_summary_column(fieldnames, run_dir)
    fieldnames.append(column)
    values = _summary_values(config, overall)
    for section, metric in values:
        key = (section, metric)
        if section == "Tool Breakdown" and metric and key not in existing_keys:
            rows.append({"Section": section, "Metric": metric})
            existing_keys.add(key)
    for row in rows:
        section = row.get("Section", "")
        metric = row.get("Metric", "")
        value = values.get((section, metric))
        if not metric or (section == "ORLA Metrics" and value is None):
            row[column] = ""
        else:
            row[column] = _summary_value(value)
    _write_summary_csv(path, fieldnames, rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAS-Sys mixed workload benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for run outputs (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize config and arrival plan without running jobs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Sample jobs, run PlanAgent once per job, and save replay plans under "
            "MixedWorkloadBench/prepare without executing MAS."
        ),
    )
    parser.add_argument(
        "--prepared-workload",
        type=str,
        default="",
        help=(
            "Optional path to a prepare_N directory or config_prepare.json. "
            "If omitted, the runner finds a matching prepare automatically."
        ),
    )
    parser.add_argument(
        "--prepare-dir",
        type=str,
        default=str(DEFAULT_PREPARE_DIR),
        help=f"Directory containing prepare_N workload plans (default: {DEFAULT_PREPARE_DIR})",
    )
    parser.add_argument(
        "--runtime",
        choices=("legacy", "orla"),
        default="legacy",
        help="Execution runtime for generated MAS jobs (default: legacy)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only and args.prepared_workload:
        raise SystemExit("--prepare-only cannot be combined with --prepared-workload")
    _run_worker_preflight()
    _cleanup_existing_sandbox_containers()

    config = _load_yaml(args.config)
    config["runtime"] = args.runtime
    pools = build_sample_manifest(config)
    arrival_plan = build_arrival_plan(config, pools)
    expected_config_prepare = _build_config_prepare(
        config=config,
        pools=pools,
        arrival_plan=arrival_plan,
    )

    prepare_base_dir = Path(args.prepare_dir)
    if args.prepare_only:
        existing_prepare = _find_matching_prepare(prepare_base_dir, expected_config_prepare)
        if existing_prepare is not None:
            result = {
                "prepare_dir": str(existing_prepare["prepare_dir"]),
                "config_prepare_path": str(Path(existing_prepare["prepare_dir"]) / "config_prepare.json"),
                "prepare_fingerprint": existing_prepare["config_prepare"].get("prepare_fingerprint"),
                "planned_jobs": len(arrival_plan),
                "reused": True,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        prepare_dir = _next_prepare_dir(prepare_base_dir)
        result = asyncio.run(
            _prepare_workload(
                config=config,
                prepare_dir=prepare_dir,
                pools=pools,
                arrival_plan=arrival_plan,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"mixed_workload_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "config_snapshot.json", _build_config_snapshot(config, arrival_plan))
        summary = {
            "run_dir": str(run_dir),
            "planned_jobs": len(arrival_plan),
            "pool_sizes": {name: len(pool.sampled_items) for name, pool in pools.items()},
            "dry_run": True,
        }
        _write_json(run_dir / "overall_result.json", summary)
        _ensure_summary_csv()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.prepared_workload:
        prepare = _load_prepare_dir(args.prepared_workload)
    else:
        prepare = _find_matching_prepare(prepare_base_dir, expected_config_prepare)
        if prepare is None:
            prepare_dir = _next_prepare_dir(prepare_base_dir)
            asyncio.run(
                _prepare_workload(
                    config=config,
                    prepare_dir=prepare_dir,
                    pools=pools,
                    arrival_plan=arrival_plan,
                )
            )
            prepare = _load_prepare_dir(prepare_dir)

    arrival_plan = _attach_prepare_replay(arrival_plan, prepare)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"mixed_workload_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "config_snapshot.json",
        _build_config_snapshot(
            config,
            arrival_plan,
            prepare["config_prepare"].get("prepare_fingerprint"),
        ),
    )

    runtime_assets = _ensure_runtime_assets(arrival_plan)
    _write_json(run_dir / "runtime_assets.json", runtime_assets)

    job_results, run_timing = asyncio.run(_execute_plan(config, run_dir, arrival_plan))
    job_results = asyncio.run(_run_posthoc_evaluations(run_dir, job_results))
    _write_json(run_dir / "job_results.json", _sanitize_job_results_for_export(job_results))

    overall = _aggregate_results(config, run_dir, pools, arrival_plan, job_results, run_timing)
    _write_json(run_dir / "overall_result.json", overall)
    _append_summary_csv(config, overall, run_dir)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
