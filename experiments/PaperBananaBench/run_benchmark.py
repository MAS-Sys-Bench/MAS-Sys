"""PaperBananaBench benchmark runner for MAS-Sys.

Iterates over test.json, routes each sample through PlanAgent →
FlowFactory → generic runtime flow, evaluates the generated image against the
GT, and incrementally saves results.

Usage:
    conda activate open_manus
    cd MAS-Sys
    python experiments/PaperBananaBench/run_benchmark.py \
        --task_name diagram \
        --split_name test \
        --retrieval_setting auto \
        --max_critic_rounds 3 \
        --max_concurrent 10 \
        --timeout 1200
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the MAS-Sys project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import config
from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.mas.plan_agent import PlanAgent
from app.tracer import tracer

from experiments.PaperBananaBench.eval_toolkits import evaluate_sample
from experiments.PaperBananaBench.task_prompt import TASK_PROMPT_TEMPLATE

# ── Paths ─────────────────────────────────────────────────────────────────────

BENCH_ROOT = _PROJECT_ROOT / "experiments" / "PaperBananaBench"
DATA_DIR = BENCH_ROOT / "data"
RESULT_DIR = BENCH_ROOT / "results"

# ── Helpers ───────────────────────────────────────────────────────────────────


def build_paperbanana_task_prompt(
    item: Dict[str, Any],
    task_name: str,
    retrieval_setting: str = "auto",
    max_critic_rounds: int = 3,
) -> str:
    """Convert a PaperBanana sample into the shared task prompt."""
    content = str(item.get("content", ""))
    visual_intent = str(item.get("visual_intent", ""))
    kind = "diagram" if task_name == "diagram" else "statistical plot"
    return TASK_PROMPT_TEMPLATE.format(
        kind=kind,
        task_name=task_name,
        retrieval_setting=retrieval_setting,
        max_critic_rounds=max_critic_rounds,
        content_preview=content,
        visual_intent=visual_intent,
    )


def _sanitize_trace_prefix(raw_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._")
    return safe_id or "unknown"


def _sample_index(raw_id: str, fallback_index: int) -> str:
    match = re.search(r"(\d+)$", str(raw_id))
    if match:
        return match.group(1)
    return str(fallback_index)


def _parse_json_dict(text: Any) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("[PaperBananaBench] Expected string output for artifact parsing.")
    raw = text.strip()
    if not raw:
        raise ValueError("[PaperBananaBench] Empty artifact output cannot be parsed.")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("[PaperBananaBench] Artifact JSON must be an object.")
    except Exception:
        raise ValueError("[PaperBananaBench] Artifact output is not valid JSON.") from None


def _extract_generated_image_base64(result: Dict[str, Any]) -> str:
    top_level = result.get("generated_image_base64", "")
    if isinstance(top_level, str) and top_level and not top_level.startswith("<base64:"):
        return top_level

    visualizer_output = result.get("visualizer_output", {})
    if isinstance(visualizer_output, dict):
        nested = visualizer_output.get("image_base64", "")
        if isinstance(nested, str) and nested and not nested.startswith("<base64:"):
            return nested
    return ""


def _guess_image_extension(image_b64: str) -> str:
    stripped = (image_b64 or "").lstrip()
    if stripped.startswith("iVBOR"):
        return ".png"
    if stripped.startswith("/9j/"):
        return ".jpg"
    if stripped.startswith("R0lGOD"):
        return ".gif"
    if stripped.startswith("UklGR"):
        return ".webp"
    return ".png"


def _save_generated_image(result: Dict[str, Any], path_without_ext: Path) -> str | None:
    image_b64 = _extract_generated_image_base64(result)
    if not image_b64:
        return None

    image_path = path_without_ext.with_suffix(_guess_image_extension(image_b64))
    try:
        image_path.write_bytes(base64.b64decode(image_b64))
        return str(image_path)
    except Exception as exc:
        logger.warning(f"[Benchmark] Failed to save generated image to {image_path}: {exc}")
        return None


def _extract_paperviz_artifacts(flow: Any) -> Dict[str, Any]:
    outputs = getattr(flow, "last_run_outputs", {}) or {}
    if not isinstance(outputs, dict):
        raise ValueError("[PaperBananaBench] Flow did not expose dict last_run_outputs.")

    visualizer_output: Dict[str, Any] = {}
    critic_output: Dict[str, Any] = {}
    for node in getattr(flow.mas_config, "agents", []):
        if node.agent_key in {"paperviz_diagram_visualizer", "paperviz_plot_visualizer"}:
            node_output = _parse_json_dict(outputs.get(node.instance_id, ""))
            visualizer_output = node_output
        elif node.agent_key == "paperviz_critic":
            node_output = _parse_json_dict(outputs.get(node.instance_id, ""))
            critic_output = node_output

    if not visualizer_output and not critic_output:
        raise ValueError("[PaperBananaBench] No parseable PaperViz artifacts were produced.")

    result: Dict[str, Any] = {
        "fsm_terminal_node": getattr(flow, "last_terminal_node", None),
        "fsm_rounds": getattr(flow, "last_run_rounds", 0),
    }
    if visualizer_output:
        result["visualizer_output"] = visualizer_output
        if visualizer_output.get("image_base64"):
            result["eval_image_field"] = "generated_image_base64"
            result["generated_image_base64"] = visualizer_output.get("image_base64")
        if visualizer_output.get("render_prompt"):
            result["render_prompt"] = visualizer_output.get("render_prompt")
        if visualizer_output.get("code"):
            result["generated_code"] = visualizer_output.get("code")
    if critic_output:
        result["critic_output"] = critic_output
        result["critic_decision"] = critic_output.get("decision")
        result["critic_suggestions"] = critic_output.get("critic_suggestions")
        result["revised_description"] = critic_output.get("revised_description")
    return result


def _build_vision_eval_config() -> Dict[str, Any]:
    """Return the configured default vision-model entry for benchmark evaluation."""
    vision_llm = config.require_llm_config("vision")
    return {
        "vision_base_url": vision_llm.base_url,
        "vision_api_key": vision_llm.api_key,
        "vision_model": vision_llm.model,
        "vision_max_tokens": vision_llm.max_tokens,
    }


async def _generate_single(
    item: Dict[str, Any],
    task_name: str,
    retrieval_setting: str,
    max_critic_rounds: int,
    timeout: int,
    runtime: str = "legacy",
) -> Dict[str, Any]:
    """Run generation only (PlanAgent → FlowFactory → artifact harvest) for one sample."""
    sample_id = item.get("id", "unknown")
    result: Dict[str, Any] = {**item}  # shallow copy; will be augmented

    try:
        # ── 1) PlanAgent ──────────────────────────────────────────
        task_prompt = build_paperbanana_task_prompt(
            item,
            task_name,
            retrieval_setting=retrieval_setting,
            max_critic_rounds=max_critic_rounds,
        )
        plan_agent = PlanAgent(runtime=runtime)
        mas_config = await plan_agent.plan(task_prompt, runtime=runtime)

        logger.info(
            f"[Benchmark] {sample_id}: PlanAgent chose topology={mas_config.topology}"
        )

        # ── 2) Build and execute flow ─────────────────────────────
        flow = FlowFactory.from_config(
            mas_config,
            plan_agent=plan_agent,
            runtime=runtime,
        )
        await asyncio.wait_for(flow.execute(task_prompt), timeout=timeout)

        # ── 3) Harvest artifacts ──────────────────────────────────
        if hasattr(flow, "last_run_artifacts") and flow.last_run_artifacts:
            result.update(flow.last_run_artifacts)
        else:
            result.update(_extract_paperviz_artifacts(flow))

    except asyncio.TimeoutError:
        logger.error(f"[Benchmark] {sample_id}: Timed out after {timeout}s")
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        logger.error(f"[Benchmark] {sample_id}: Error — {e}")
        result["error"] = str(e)

    return result


async def _process_single(
    item: Dict[str, Any],
    task_name: str,
    retrieval_setting: str,
    max_critic_rounds: int,
    timeout: int,
    vision_cfg: dict,
    runtime: str = "legacy",
) -> Dict[str, Any]:
    """Run the full pipeline (PlanAgent → FlowFactory → eval) for one sample."""
    result = await _generate_single(
        item,
        task_name,
        retrieval_setting=retrieval_setting,
        max_critic_rounds=max_critic_rounds,
        timeout=timeout,
        runtime=runtime,
    )

    if result.get("eval_image_field") and not result.get("error"):
        result = await evaluate_sample(
            sample=result,
            task_name=task_name,
            data_dir=DATA_DIR,
            **vision_cfg,
        )

    return result


async def run_benchmark(
    task_name: str,
    split_name: str,
    retrieval_setting: str,
    max_critic_rounds: int,
    max_concurrent: int,
    timeout: int,
    max_samples: int = 0,
    sample_offset: int = 0,
    save_generated_image: bool = False,
    runtime: str = "legacy",
) -> None:
    """Load test set and process samples with concurrency control."""
    # ── Load data ─────────────────────────────────────────────────
    input_path = DATA_DIR / task_name / f"{split_name}.json"
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data_list: List[Dict[str, Any]] = json.load(f)

    if sample_offset and sample_offset > 0:
        data_list = data_list[sample_offset:]
    if max_samples and max_samples > 0:
        data_list = data_list[:max_samples]
    print(f"Loaded {len(data_list)} samples from {input_path}")

    # ── Prepare output ────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULT_DIR / f"{task_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Vision LLM settings for evaluation (from config.toml) ────
    vision_cfg = _build_vision_eval_config()

    # ── Concurrency ───────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrent)
    all_results: List[Dict[str, Any]] = []
    eval_dims = ["instruction_faithfulness", "conciseness", "readability", "aesthetics", "overall"]
    async def _guarded(item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        sample_id = str(item.get("id", "unknown"))
        sample_no = _sample_index(sample_id, item_index)
        session_id = f"paperbanana:{sample_id}:{id(asyncio.current_task())}"

        tracer.activate_session(session_id, reset=True)
        tracer.start_e2e()

        queue_start = time.time()
        try:
            async with semaphore:
                queue_end = time.time()
                tracer.record_wait_phase(
                    wait_type="semaphore",
                    start_time=queue_start,
                    end_time=queue_end,
                    details={
                        "resource": "paperbanana_sample_slot",
                        "sample_id": sample_id,
                        "max_concurrent": max_concurrent,
                    },
                )

                result = await _process_single(
                    item,
                    task_name,
                    retrieval_setting,
                    max_critic_rounds,
                    timeout,
                    vision_cfg,
                    runtime,
                )
                result["sample_no"] = sample_no
                result["result_path"] = str(run_dir / f"result_{sample_no}.json")
                result["trace_path"] = str(run_dir / f"trace_{sample_no}.json")
                result["metadata_path"] = str(run_dir / f"metadata_{sample_no}.json")
                result["semaphore_wait_s"] = round(queue_end - queue_start, 4)
                return result
        finally:
            tracer.end_e2e()
            tracer.save(run_dir, prefix=f"{sample_no}_")
            raw_trace = run_dir / f"{sample_no}_trace.json"
            raw_meta = run_dir / f"{sample_no}_metadata.json"
            final_trace = run_dir / f"trace_{sample_no}.json"
            final_meta = run_dir / f"metadata_{sample_no}.json"
            if raw_trace.exists():
                raw_trace.replace(final_trace)
            if raw_meta.exists():
                raw_meta.replace(final_meta)
            tracer.clear_session(session_id)

    tasks = [
        asyncio.create_task(_guarded(item, idx))
        for idx, item in enumerate(data_list)
    ]

    completed = 0
    for future in asyncio.as_completed(tasks):
        result = await future
        all_results.append(result)
        completed += 1

        # Stats
        stats = {}
        for dim in eval_dims:
            key = f"{dim}_score"
            scores = [
                float(r.get(key))
                for r in all_results
                if r.get(key) is not None
            ]
            if scores:
                stats[dim[:5]] = f"{sum(scores) / len(scores):.2f}"

        stats_str = " | ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[{completed}/{len(data_list)}] {result.get('id', '?')} — {stats_str}")

        # Incremental save every 10 samples
        sample_no = result.get("sample_no", str(completed - 1))
        if save_generated_image:
            saved_image_path = _save_generated_image(
                result,
                run_dir / f"generated_{sample_no}",
            )
            if saved_image_path:
                result["generated_image_path"] = saved_image_path
        _save_single_result(result, run_dir / f"result_{sample_no}.json")

    _save_summary(all_results, run_dir / "results_summary.json")
    print(f"\n✅ Done — {len(all_results)} results saved to {run_dir}")

    # ── Print final summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for dim in eval_dims:
        key = f"{dim}_score"
        scores = [
            float(r.get(key))
            for r in all_results
            if r.get(key) is not None
        ]
        if not scores:
            continue
        print(
            f"  {dim:25s}: mean={sum(scores) / len(scores):.4f} "
            f"min={min(scores):.2f} max={max(scores):.2f} (n={len(scores)})"
        )


def _slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    entry = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > 50_000:
            entry[k] = f"<base64:{len(v)} chars>"
        else:
            entry[k] = v
    return entry


def _save_single_result(result: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_slim_result(result), f, ensure_ascii=False, indent=2)


def _save_summary(results: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_slim_result(r) for r in results], f, ensure_ascii=False, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PaperBananaBench on MAS-Sys (PlanAgent → generic flow runtime → eval)"
    )
    parser.add_argument(
        "--task_name", type=str, default="diagram", choices=["diagram", "plot"],
        help="Task type (default: diagram)",
    )
    parser.add_argument(
        "--split_name", type=str, default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--retrieval_setting", type=str, default="auto",
        choices=["auto", "manual", "random", "none"],
        help="Reference retrieval mode (default: auto)",
    )
    parser.add_argument(
        "--max_critic_rounds", type=int, default=3,
        help="Max critic revision rounds (default: 3)",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=10,
        help="Max concurrent sample processing (default: 10)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1200,
        help="Per-sample timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Max number of samples to run (0 = all, default: 0)",
    )
    parser.add_argument(
        "--sample_offset", type=int, default=0,
        help="Skip the first N samples before applying --max_samples (default: 0)",
    )
    parser.add_argument(
        "--save_generated_image",
        action="store_true",
        help="Save each final generated image to the run directory (default: off)",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["legacy", "orla", "orla_dry_run"],
        default="legacy",
        help="Execution runtime for benchmark generation; orla_dry_run only validates ORLA lowering",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            task_name=args.task_name,
            split_name=args.split_name,
            retrieval_setting=args.retrieval_setting,
            max_critic_rounds=args.max_critic_rounds,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            max_samples=args.max_samples,
            sample_offset=args.sample_offset,
            save_generated_image=args.save_generated_image,
            runtime=args.runtime,
        )
    )


if __name__ == "__main__":
    main()
