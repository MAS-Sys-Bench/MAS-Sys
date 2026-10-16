"""PinchBench evaluation engine.

Supports three grading modes:
- automated: Runs per-task Python grade() functions.
- llm_judge: Sends transcript + rubric to an LLM for scoring.
- hybrid: Combines automated and LLM judge scores by weight.
"""

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.llm import LLM
from app.logger import logger

if TYPE_CHECKING:
    from experiments.PinchBench.task_loader import Task

_THIS_DIR = Path(__file__).resolve().parent
_GRADERS_DIR = _THIS_DIR / "graders"

# Map task_id to grader module name.
# Tasks with only llm_judge grading have no entry here.
_GRADER_MODULE_MAP = {
    "task_00_sanity": "task_00_sanity",
    "task_01_calendar": "task_01_calendar",
    "task_02_stock": "task_02_stock",
    "task_04_weather": "task_04_weather",
    "task_08_memory": "task_08_memory",
    "task_09_files": "task_09_files",
    "task_10_workflow": "task_10_workflow",
    "task_11_clawdhub": "task_11_clawdhub",
    "task_12_skill_search": "task_12_skill_search",
    "task_13_image_gen": "task_13_image_gen",
    "task_16_email_triage": "task_16_email_triage",
    "task_18_market_research": "task_18_market_research",
    "task_17_email_search": "task_17_email_search",
    "task_19_spreadsheet_summary": "task_19_spreadsheet_summary",
    "task_21_openclaw_comprehension": "task_21_openclaw_comprehension",
    "task_22_second_brain": "task_22_second_brain",
}


def _load_grader(task_id: str):
    """Dynamically import a grader module and return its grade() function."""
    module_name = _GRADER_MODULE_MAP.get(task_id)
    if module_name is None:
        return None
    mod = importlib.import_module(
        f"experiments.PinchBench.graders.{module_name}"
    )
    return mod.grade


def run_automated_grading(
    task_id: str,
    transcript: List[Dict],
    workspace_path: str,
) -> Optional[Dict[str, float]]:
    """Run automated grading for a task.

    Returns:
        Dict of criterion → score, or None if no grader exists.
    """
    grade_fn = _load_grader(task_id)
    if grade_fn is None:
        return None
    try:
        scores = grade_fn(transcript, workspace_path)
        logger.info(f"[PinchBench] Automated scores for {task_id}: {scores}")
        return scores
    except Exception as e:
        logger.error(f"[PinchBench] Automated grading failed for {task_id}: {e}")
        return None


async def run_llm_judge_grading(
    task_id: str,
    transcript: List[Dict],
    rubric: str,
    prompt: str,
) -> Optional[Dict[str, float]]:
    """Run LLM judge grading for a task.

    Sends the task prompt, transcript, and rubric to the LLM and asks it
    to evaluate each criterion on a 0.0-1.0 scale.

    Returns:
        Dict mapping criterion names to float scores, or None on failure.
    """
    transcript_text = json.dumps(transcript, indent=2, ensure_ascii=False)
    # Truncate very long transcripts to fit context window
    if len(transcript_text) > 80000:
        transcript_text = transcript_text[:80000] + "\n... [truncated]"

    system_prompt = (
        "You are an expert evaluator for an AI agent benchmark called PinchBench. "
        "You will assess an agent's performance on a task by analyzing its transcript "
        "(the sequence of messages and tool calls exchanged during execution).\n\n"
        "Score EACH criterion in the rubric on a scale of 0.0 to 1.0.\n"
        "Return your evaluation as a JSON object mapping criterion names to scores.\n"
        "Example: {\"criterion_1\": 0.75, \"criterion_2\": 1.0}\n\n"
        "Only output the JSON object, no other text."
    )

    user_prompt = (
        f"## Task Prompt\n{prompt}\n\n"
        f"## Agent Transcript\n```json\n{transcript_text}\n```\n\n"
        f"## Evaluation Rubric\n{rubric}\n\n"
        "Please evaluate the agent's performance on each criterion in the rubric. "
        "Return a JSON object mapping criterion names to scores (0.0-1.0)."
    )

    try:
        llm = LLM(config_name="default")
        response = await llm.ask(
            messages=[{"role": "user", "content": user_prompt}],
            system_msgs=[{"role": "system", "content": system_prompt}],
            stream=False,
        )

        # Parse JSON from response (handle markdown code fences)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        scores = json.loads(text)

        # Normalize criterion keys and clamp scores to [0, 1]
        normalized: Dict[str, float] = {}
        for key, val in scores.items():
            name = key.strip().lower().replace(" ", "_")
            normalized[name] = max(0.0, min(1.0, float(val)))

        logger.info(f"[PinchBench] LLM judge scores for {task_id}: {normalized}")
        return normalized

    except Exception as e:
        logger.error(f"[PinchBench] LLM judge grading failed for {task_id}: {e}")
        return None


def _merge_scores(
    automated_scores: Optional[Dict[str, float]],
    llm_judge_scores: Optional[Dict[str, float]],
    grading_weights: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """Merge automated and LLM judge scores using the configured weights.

    Returns a dict with individual and combined scores.
    """
    result: Dict[str, Any] = {
        "automated_scores": automated_scores,
        "llm_judge_scores": llm_judge_scores,
    }

    auto_weight = 0.5
    judge_weight = 0.5
    if grading_weights:
        auto_weight = grading_weights.get("automated", 0.5)
        judge_weight = grading_weights.get("llm_judge", 0.5)

    auto_avg = 0.0
    if automated_scores:
        vals = [v for v in automated_scores.values() if isinstance(v, (int, float))]
        auto_avg = sum(vals) / len(vals) if vals else 0.0

    judge_avg = 0.0
    if llm_judge_scores:
        vals = [v for v in llm_judge_scores.values() if isinstance(v, (int, float))]
        judge_avg = sum(vals) / len(vals) if vals else 0.0

    if automated_scores and llm_judge_scores:
        combined = auto_avg * auto_weight + judge_avg * judge_weight
    elif automated_scores:
        combined = auto_avg
    elif llm_judge_scores:
        combined = judge_avg
    else:
        combined = 0.0

    result["automated_average"] = round(auto_avg, 4)
    result["llm_judge_average"] = round(judge_avg, 4)
    result["combined_score"] = round(combined, 4)

    return result


async def evaluate_task(
    task: "Task",  # noqa: F821
    transcript: List[Dict],
    workspace_path: str,
) -> Dict[str, Any]:
    """Evaluate one task using the appropriate grading strategy.

    Args:
        task: The Task object with grading metadata.
        transcript: PinchBench-formatted transcript (list of event dicts).
        workspace_path: Path to the task's workspace directory.

    Returns:
        Evaluation result dict.
    """
    result: Dict[str, Any] = {
        "task_id": task.task_id,
        "task_name": task.name,
        "grading_type": task.grading_type,
    }

    automated_scores: Optional[Dict[str, float]] = None
    llm_judge_scores: Optional[Dict[str, float]] = None

    if task.grading_type in ("automated", "hybrid"):
        automated_scores = run_automated_grading(
            task.task_id, transcript, workspace_path
        )

    if task.grading_type in ("llm_judge", "hybrid"):
        llm_judge_scores = await run_llm_judge_grading(
            task.task_id, transcript, task.llm_judge_rubric or "", task.prompt
        )

    merged = _merge_scores(
        automated_scores, llm_judge_scores, task.grading_weights
    )
    result.update(merged)

    return result


async def evaluate_all(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate evaluation results across all tasks.

    Args:
        results: List of per-task evaluation result dicts.

    Returns:
        Summary dict with per-task and aggregate scores.
    """
    total_score = 0.0
    scored = 0

    for r in results:
        score = r.get("combined_score", 0.0)
        total_score += score
        scored += 1

    avg_score = total_score / scored if scored else 0.0

    summary = {
        "total_tasks": len(results),
        "scored_tasks": scored,
        "average_score": round(avg_score, 4),
        "per_task": results,
    }

    return summary


def print_results(summary: Dict[str, Any]) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 60)
    print("PinchBench Evaluation Results")
    print("=" * 60)
    print(f"Total tasks:    {summary['total_tasks']}")
    print(f"Scored tasks:   {summary['scored_tasks']}")
    print(f"Average score:  {summary['average_score']:.2%}")
    print("-" * 60)

    for r in summary.get("per_task", []):
        task_id = r["task_id"]
        score = r.get("combined_score", 0.0)
        grading = r.get("grading_type", "?")
        print(f"  {task_id:<35} {score:.2%}  ({grading})")

    print("=" * 60)
