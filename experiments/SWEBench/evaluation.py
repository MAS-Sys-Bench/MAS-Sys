"""Evaluation wrapper for the embedded SWE-bench harness."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from app.config import PROJECT_ROOT
from app.logger import logger


def run_swebench_evaluation(
    predictions_path: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    run_id: str = "mas-sys_eval",
    max_workers: int = 4,
    timeout: int = 1800,
    instance_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Optional[Dict]:
    """Run SWE-bench evaluation using the embedded harness.

    Args:
        predictions_path: Path to the predictions JSONL file.
        dataset_name: HuggingFace dataset name or local JSON path.
        split: Dataset split.
        run_id: Unique identifier for this evaluation run.
        max_workers: Number of parallel Docker containers.
        timeout: Per-instance timeout in seconds.
        instance_ids: Optional subset of instance IDs to evaluate.
        output_dir: Directory where harness-level output files should be written.
        report_path: Optional final path for the aggregated report JSON.

    Returns:
        Report dict if available, or None if evaluation failed.
    """
    predictions_path = str(Path(predictions_path).resolve())
    if report_path is not None:
        report_path = str(Path(report_path).resolve())

    cmd = [
        sys.executable, "-m", "experiments.SWEBench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--split", split,
        "--predictions_path", predictions_path,
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
    ]

    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)

    logger.info(f"[SWE-bench] Starting Docker evaluation: {' '.join(cmd)}")
    project_root = Path(PROJECT_ROOT).resolve()
    runtime_output_dir = (
        Path(output_dir).resolve() if output_dir else project_root
    )
    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath_entries = [str(project_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=runtime_output_dir,
            env=env,
            timeout=timeout * max(len(instance_ids) if instance_ids else 10, 1) + 600,
        )

        if result.returncode != 0:
            logger.error(f"[SWE-bench] Evaluation failed:\n{result.stderr}")
            return None

        logger.info(f"[SWE-bench] Evaluation completed successfully")
        if result.stdout:
            logger.info(f"[SWE-bench] Output:\n{result.stdout[-2000:]}")

    except subprocess.TimeoutExpired:
        logger.error("[SWE-bench] Evaluation subprocess timed out")
        return None
    except FileNotFoundError:
        logger.error(
            "[SWE-bench] Python executable not found for embedded harness run"
        )
        return None

    # Try to read the report file
    report = _find_report(
        run_id,
        predictions_path,
        output_dir=runtime_output_dir,
        report_path=report_path,
    )
    return report


def _find_report(
    run_id: str,
    predictions_path: str,
    output_dir: Optional[Path] = None,
    report_path: Optional[str] = None,
) -> Optional[Dict]:
    """Locate and parse the SWE-bench evaluation report JSON.

    SWE-bench writes reports to a logs/ directory structure.
    """
    report_file = _find_report_file(run_id, predictions_path, output_dir=output_dir)
    if report_file is None:
        logger.warning("[SWE-bench] Could not find evaluation report file")
        return None

    if report_path:
        destination = Path(report_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if report_file.resolve() != destination:
            shutil.move(str(report_file), str(destination))
            report_file = destination

    try:
        with open(report_file) as f:
            report = json.load(f)
        logger.info(f"[SWE-bench] Found report: {report_file}")
        return report
    except (json.JSONDecodeError, IOError):
        logger.warning(
            f"[SWE-bench] Failed to read evaluation report file: {report_file}"
        )
        return None


def _find_report_file(
    run_id: str,
    predictions_path: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Locate the aggregated SWE-bench evaluation report file."""
    project_root = Path(PROJECT_ROOT).resolve()
    predictions_dir = Path(predictions_path).resolve().parent

    # SWE-bench report locations vary by runner. Search likely absolute roots
    # instead of relying on the current working directory.
    candidate_dirs = [
        output_dir.resolve() if output_dir else None,
        project_root,
        project_root / "logs",
        project_root / "logs" / "run_evaluation",
        project_root / "evaluation_results",
        predictions_dir,
        predictions_dir / "logs",
    ]
    search_dirs = []
    seen_dirs = set()
    for candidate in candidate_dirs:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        search_dirs.append(resolved)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for report_file in search_dir.rglob(f"*{run_id}*.json"):
            return report_file

    return None


def compute_resolved_rate(report: Dict) -> Dict:
    """Compute resolved rate from a SWE-bench evaluation report.

    Args:
        report: Report dict from swebench evaluation.

    Returns:
        Summary dict with resolved_rate and counts.
    """
    if not report:
        return {"resolved_rate": 0.0, "resolved": 0, "total": 0}

    # Handle different report formats
    if "resolved_instances" in report and "total_instances" in report:
        resolved = report["resolved_instances"]
        total = report["total_instances"]
    elif "resolved_ids" in report:
        resolved = len(report.get("resolved_ids", []))
        total = report.get("total_instances", resolved)
    else:
        # Count from per-instance results
        resolved = sum(1 for v in report.values()
                       if isinstance(v, dict) and v.get("resolved", False))
        total = sum(1 for v in report.values() if isinstance(v, dict))

    rate = resolved / total if total > 0 else 0.0

    return {
        "resolved_rate": rate,
        "resolved": resolved,
        "total": total,
    }


def print_results(report: Optional[Dict], predictions: Optional[List[Dict]] = None) -> None:
    """Print evaluation results summary."""
    print("\n" + "=" * 60)
    print("SWE-bench Evaluation Results")
    print("=" * 60)

    if report is None:
        print("\n  No evaluation report available.")
        print("  Run Docker evaluation separately if needed.")
    else:
        summary = compute_resolved_rate(report)
        print(f"\n--- Resolution Rate ---")
        print(f"  Resolved: {summary['resolved']}/{summary['total']} "
              f"({summary['resolved_rate']:.1%})")

        if "resolved_ids" in report:
            print(f"\n--- Resolved Instances ---")
            for rid in report["resolved_ids"]:
                print(f"  ✓ {rid}")

    if predictions:
        empty = sum(1 for p in predictions if not p.get("model_patch", "").strip())
        print(f"\n--- Generation Summary ---")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Non-empty patches: {len(predictions) - empty}")
        print(f"  Empty patches:     {empty}")

    print("=" * 60)
