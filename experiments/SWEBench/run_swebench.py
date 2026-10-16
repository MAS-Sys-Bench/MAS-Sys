#!/usr/bin/env python
"""Run SWE-bench benchmark on MAS-Sys MAS system.

Usage:
    # Default: Run SWE-bench Lite (first 5 instances)
    python -m experiments.SWEBench.run_swebench --end 5

    # Specific instances
    python -m experiments.SWEBench.run_swebench --instance_ids django__django-11583 sympy__sympy-20590

    # Full Lite benchmark
    python -m experiments.SWEBench.run_swebench

    # Generate only (skip Docker evaluation)
    python -m experiments.SWEBench.run_swebench --end 5 --skip_eval

    # Evaluate existing predictions
    python -m experiments.SWEBench.run_swebench --eval_only --predictions_file results/preds.jsonl

This script follows the same PlanAgent-driven workflow as run_mas.py:
  1. Load SWE-bench instances from HuggingFace
  2. For each instance, clone repo → PlanAgent.plan() → agents edit code → git diff
  3. Save predictions as JSONL (instance_id + model_patch)
  4. Optionally run Docker-based evaluation via swebench harness
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

# Ensure project root is on path
_THIS_DIR = Path(__file__).resolve().parent
_MAS_SYS_ROOT = _THIS_DIR.parent.parent
if str(_MAS_SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAS_SYS_ROOT))

from app.logger import logger

from experiments.SWEBench.evaluation import (
    print_results,
    run_swebench_evaluation,
)
from experiments.SWEBench.generation import generate_batch
from experiments.SWEBench.utils import (
    load_swebench_dataset,
    read_predictions,
    write_predictions,
)


def _make_result_paths(base_output_dir: Path, timestamp: str) -> Tuple[Path, Path, Path, Path]:
    """Create the per-run result directory and standard file paths."""
    result_dir = base_output_dir / f"result_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    meta_file = result_dir / f"meta_{timestamp}.json"
    prediction_file = result_dir / f"prediction_{timestamp}.jsonl"
    report_file = result_dir / f"report_{timestamp}.json"
    return result_dir, meta_file, prediction_file, report_file


def _write_run_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench benchmark on MAS-Sys MAS system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name (default: princeton-nlp/SWE-bench_Lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )

    # Instance selection
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index of instances (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index of instances (exclusive)",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific instance IDs to evaluate",
    )

    # Generation settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="mas-sys",
        help="Model identifier for predictions (default: mas-sys)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-instance timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory for repo checkouts (default: temp dir)",
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Keep repo working copies after generation (for debugging)",
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
        help="Directory to save results (default: experiments/SWEBench/results/)",
    )

    # Evaluation settings
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip Docker evaluation after generation",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=4,
        help="Number of parallel Docker containers for evaluation (default: 4)",
    )
    parser.add_argument(
        "--eval_timeout",
        type=int,
        default=1800,
        help="Per-instance evaluation timeout in seconds (default: 1800)",
    )

    # Eval-only mode
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate existing predictions file, skip generation",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Path to existing predictions JSONL file (for --eval_only)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir, meta_file, preds_file, report_file = _make_result_paths(
        output_dir, timestamp
    )

    if args.eval_only:
        # ── Eval-only mode ───────────────────────────────────────────────
        if not args.predictions_file:
            print("Error: --predictions_file required when using --eval_only")
            sys.exit(1)

        predictions = read_predictions(args.predictions_file)
        logger.info(
            f"[SWE-bench] Loaded {len(predictions)} predictions "
            f"from {args.predictions_file}"
        )

        report = run_swebench_evaluation(
            predictions_path=args.predictions_file,
            dataset_name=args.dataset,
            split=args.split,
            run_id=f"eval_{timestamp}",
            max_workers=args.eval_workers,
            timeout=args.eval_timeout,
            output_dir=str(result_dir),
            report_path=str(report_file),
        )
        print_results(report, predictions)
        return

    # ── Generation mode ──────────────────────────────────────────────────

    # Load dataset
    logger.info(f"[SWE-bench] Loading dataset: {args.dataset} ({args.split})")
    all_instances = load_swebench_dataset(args.dataset, args.split)
    logger.info(f"[SWE-bench] Loaded {len(all_instances)} instances")

    # Select instances
    if args.instance_ids:
        id_set = set(args.instance_ids)
        instances = [inst for inst in all_instances if inst["instance_id"] in id_set]
        if len(instances) != len(id_set):
            found = {inst["instance_id"] for inst in instances}
            missing = id_set - found
            logger.warning(f"[SWE-bench] Instance IDs not found: {missing}")
    else:
        start = args.start or 0
        end = args.end or len(all_instances)
        instances = all_instances[start:end]

    if not instances:
        logger.error("[SWE-bench] No instances selected")
        sys.exit(1)

    logger.info(f"[SWE-bench] Selected {len(instances)} instances for generation")

    # Prepare working directory
    if args.workdir:
        workdir = args.workdir
        os.makedirs(workdir, exist_ok=True)
    else:
        workdir = str(result_dir / "workdir")
        os.makedirs(workdir, exist_ok=True)

    logger.info(f"[SWE-bench] Working directory: {workdir}")

    # Run generation
    run_name = f"planagent_{timestamp}"

    logger.info(
        f"[SWE-bench] Starting generation: PlanAgent-driven, "
        f"timeout={args.timeout}s/instance"
    )
    start_time = time.time()

    predictions = await generate_batch(
        instances=instances,
        workdir=workdir,
        result_dir=str(result_dir),
        model_name=args.model_name,
        timeout=args.timeout,
        cleanup=not args.no_cleanup,
        runtime=args.runtime,
    )

    elapsed = time.time() - start_time
    logger.info(
        f"[SWE-bench] Generation completed in {elapsed:.1f}s "
        f"({len(predictions)} predictions)"
    )

    # Save predictions
    write_predictions(str(preds_file), predictions)
    logger.info(f"[SWE-bench] Predictions saved to {preds_file}")

    # Summary statistics
    non_empty = sum(1 for p in predictions if p.get("model_patch", "").strip())
    logger.info(
        f"[SWE-bench] Non-empty patches: {non_empty}/{len(predictions)}"
    )

    # Save run metadata
    status_counts = {}
    for pred in predictions:
        status = pred.get("generation_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    meta = {
        "run_name": run_name,
        "timestamp": timestamp,
        "result_dir": str(result_dir),
        "dataset": args.dataset,
        "split": args.split,
        "n_instances": len(instances),
        "n_predictions": len(predictions),
        "n_non_empty": non_empty,
        "model_name": args.model_name,
        "timeout_per_instance": args.timeout,
        "total_time_seconds": elapsed,
        "instance_ids": [inst["instance_id"] for inst in instances],
        "generation": {
            "status_counts": status_counts,
            "instances": [
                {
                    "instance_id": pred.get("instance_id"),
                    "generation_status": pred.get("generation_status"),
                    "generation_error_type": pred.get("generation_error_type"),
                    "generation_error": pred.get("generation_error"),
                    "patch_chars": len(pred.get("model_patch", "")),
                    "elapsed_seconds": pred.get("elapsed_seconds"),
                }
                for pred in predictions
            ],
        },
        "evaluation": None,
    }
    _write_run_meta(meta_file, meta)

    # ── Docker evaluation ────────────────────────────────────────────────
    if args.skip_eval:
        logger.info("[SWE-bench] Skipping Docker evaluation (--skip_eval)")
        meta["evaluation"] = {
            "status": "skipped_by_flag",
            "report_file": None,
            "evaluated_instance_ids": [],
        }
        _write_run_meta(meta_file, meta)
        print_results(None, predictions)
    else:
        instance_ids = [p["instance_id"] for p in predictions if p.get("model_patch", "").strip()]
        if not instance_ids:
            logger.warning(
                "[SWE-bench] No non-empty patches were generated; skipping Docker evaluation "
                "to avoid evaluating the full dataset split."
            )
            meta["evaluation"] = {
                "status": "skipped_no_patch",
                "report_file": None,
                "evaluated_instance_ids": [],
            }
            _write_run_meta(meta_file, meta)
            print_results(None, predictions)
            return

        logger.info("[SWE-bench] Starting Docker evaluation...")
        eval_run_id = f"mas-sys_{run_name}"

        # Only evaluate instances we generated

        report = run_swebench_evaluation(
            predictions_path=str(preds_file),
            dataset_name=args.dataset,
            split=args.split,
            run_id=eval_run_id,
            max_workers=args.eval_workers,
            timeout=args.eval_timeout,
            instance_ids=instance_ids if instance_ids else None,
            output_dir=str(result_dir),
            report_path=str(report_file),
        )

        meta["evaluation"] = {
            "status": "completed" if report is not None else "missing_report",
            "report_file": str(report_file) if report_file.exists() else None,
            "evaluated_instance_ids": instance_ids,
        }
        _write_run_meta(meta_file, meta)

        print_results(report, predictions)

        # Save report
        if report:
            report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"[SWE-bench] Report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
