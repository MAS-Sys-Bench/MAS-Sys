#!/usr/bin/env python
"""Run PinchBench benchmark on MAS-Sys MAS system.

Usage:
    # Run all 23 tasks
    python -m experiments.PinchBench.run_pinchbench

    # Run specific tasks
    python -m experiments.PinchBench.run_pinchbench --task_ids task_00_sanity task_01_calendar

    # Generate only (skip evaluation)
    python -m experiments.PinchBench.run_pinchbench --skip_eval

    # Evaluate existing results
    python -m experiments.PinchBench.run_pinchbench --eval_only --results_file results/results.json

    # Custom timeout
    python -m experiments.PinchBench.run_pinchbench --timeout 600
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MAS_SYS_ROOT = _THIS_DIR.parent.parent
if str(_MAS_SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAS_SYS_ROOT))

from app.logger import logger

from experiments.PinchBench.evaluation import (
    evaluate_all,
    evaluate_task,
    print_results,
)
from experiments.PinchBench.generation import (
    generate_batch,
    read_results,
    write_results,
)
from experiments.PinchBench.task_loader import Task, load_all_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PinchBench benchmark on MAS-Sys MAS system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task selection
    parser.add_argument(
        "--task_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific task IDs to run (e.g. task_00_sanity task_01_calendar)",
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        default=str(_THIS_DIR / "tasks"),
        help="Directory containing task_*.md files",
    )

    # Execution settings
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-task timeout in seconds (overrides task default)",
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Keep workspace directories after execution (for debugging)",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["legacy", "orla", "orla_dry_run"],
        default="legacy",
        help="Execution runtime for benchmark generation; orla_dry_run only validates ORLA lowering",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_THIS_DIR / "results"),
        help="Directory to save results",
    )

    # Evaluation settings
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after generation",
    )

    # Eval-only mode
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate existing results, skip generation",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to existing results JSONL (for --eval_only)",
    )

    return parser.parse_args()


async def _run_evaluation(
    tasks: list[Task],
    gen_results: list[dict],
) -> dict:
    """Run evaluation for all generated results."""
    task_map = {t.task_id: t for t in tasks}
    eval_results = []

    for gen_result in gen_results:
        task_id = gen_result["task_id"]
        task = task_map.get(task_id)
        if task is None:
            logger.warning(f"[PinchBench] No task definition for {task_id}, skipping eval")
            continue

        transcript = gen_result.get("transcript", [])
        workspace_path = gen_result.get("workspace_path", "")

        eval_result = await evaluate_task(task, transcript, workspace_path)
        eval_results.append(eval_result)

    summary = await evaluate_all(eval_results)
    return summary


async def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load task definitions
    tasks_dir = Path(args.tasks_dir)
    tasks = load_all_tasks(tasks_dir, task_ids=args.task_ids)

    if not tasks:
        logger.error("[PinchBench] No tasks found")
        sys.exit(1)

    logger.info(f"[PinchBench] Loaded {len(tasks)} tasks")

    if args.eval_only:
        # Eval-only mode 
        if not args.results_file:
            print("Error: --results_file required when using --eval_only")
            sys.exit(1)

        gen_results = read_results(args.results_file)
        logger.info(
            f"[PinchBench] Loaded {len(gen_results)} results "
            f"from {args.results_file}"
        )

        summary = await _run_evaluation(tasks, gen_results)
        print_results(summary)

        # Save evaluation report
        report_file = output_dir / "eval_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"[PinchBench] Report saved to {report_file}")
        return

    # Generation mode 

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"pinchbench_{timestamp}"
    run_dir = output_dir / f"result_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir

    logger.info(
        f"[PinchBench] Starting generation: {len(tasks)} tasks, "
        f"timeout={args.timeout or 'per-task default'}s"
    )
    start_time = time.time()

    gen_results = await generate_batch(
        tasks=tasks,
        workdir=workdir,
        timeout=args.timeout,
        cleanup=not args.no_cleanup,
        run_id=None,
        runtime=args.runtime,
    )

    elapsed = time.time() - start_time
    logger.info(
        f"[PinchBench] Generation completed in {elapsed:.1f}s "
        f"({len(gen_results)} results)"
    )

    # Save generation results
    results_file = run_dir / f"result_{timestamp}.json"
    write_results(str(results_file), gen_results)
    logger.info(f"[PinchBench] Results saved to {results_file}")

    # Save run metadata
    meta_file = run_dir / f"metadata_{timestamp}.json"
    meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "result_dir": str(run_dir),
        "n_tasks": len(tasks),
        "n_results": len(gen_results),
        "timeout_override": args.timeout,
        "total_time_seconds": round(elapsed, 2),
        "task_ids": [t.task_id for t in tasks],
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Summary statistics
    errors = sum(1 for r in gen_results if r.get("error"))
    logger.info(
        f"[PinchBench] Completed: {len(gen_results) - errors}, "
        f"Errors: {errors}"
    )

    # Evaluation 
    if args.skip_eval:
        logger.info("[PinchBench] Skipping evaluation (--skip_eval)")
        return

    logger.info("[PinchBench] Starting evaluation...")
    summary = await _run_evaluation(tasks, gen_results)
    print_results(summary)

    # Save evaluation report
    report_file = run_dir / f"report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[PinchBench] Report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
