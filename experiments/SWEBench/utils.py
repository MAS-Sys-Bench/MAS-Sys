"""Utility functions for SWE-bench benchmark.

Handles dataset loading from HuggingFace, local repo management
(clone / checkout / diff), and prediction I/O.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from experiments.SWEBench.task_prompt import TASK_PROMPT_TEMPLATE

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _THIS_DIR / "results"
REPO_CACHE_DIR = _THIS_DIR / "repo_cache"


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_swebench_dataset(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
) -> List[Dict]:
    """Load SWE-bench dataset from HuggingFace.

    Returns a list of instance dicts with fields:
      instance_id, repo, base_commit, problem_statement, hints_text,
      patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, version, ...
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def instances_by_id(instances: List[Dict]) -> Dict[str, Dict]:
    """Index instances by instance_id."""
    return {inst["instance_id"]: inst for inst in instances}


# ── Git operations ───────────────────────────────────────────────────────────

def _run_git(args: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the given directory."""
    return subprocess.run(
        ["git", "-c", f"safe.directory={cwd}"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
        timeout=300,
    )


def clone_repo(repo: str, target_dir: str) -> None:
    """Clone a GitHub repo into target_dir.

    Uses REPO_CACHE_DIR as a bare-clone cache to avoid repeated downloads.
    """
    repo_slug = repo.replace("/", "__")
    cache_dir = REPO_CACHE_DIR / repo_slug

    if not cache_dir.exists():
        REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(cache_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )

    # Create working copy from bare cache
    subprocess.run(
        ["git", "clone", str(cache_dir), target_dir],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Add original remote for fetching missing objects
    _run_git(["remote", "set-url", "origin", f"https://github.com/{repo}.git"], cwd=target_dir)


def checkout_commit(repo_dir: str, commit: str) -> None:
    """Checkout a specific commit in the repo."""
    _run_git(["fetch", "origin"], cwd=repo_dir, check=False)
    _run_git(["checkout", commit], cwd=repo_dir)
    _run_git(["clean", "-fdx"], cwd=repo_dir)


def get_diff(repo_dir: str) -> str:
    """Get the unified diff of all changes in the working tree."""
    result = _run_git(["diff"], cwd=repo_dir, check=False)
    return result.stdout


def setup_instance_repo(instance: Dict, workdir: str) -> str:
    """Clone repo and checkout base_commit for one SWE-bench instance.

    Args:
        instance: SWE-bench instance dict.
        workdir: Parent directory for working copies.

    Returns:
        Absolute path to the repo working directory.
    """
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]

    # Create a per-instance work directory
    repo_dir = os.path.join(workdir, instance_id.replace("/", "__"))
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    clone_repo(repo, repo_dir)
    checkout_commit(repo_dir, base_commit)
    return repo_dir


def cleanup_instance_repo(repo_dir: str) -> None:
    """Remove the working copy of a repo."""
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)


# ── Patch extraction ─────────────────────────────────────────────────────────

def extract_diff_from_output(text: str) -> str:
    """Extract a unified diff / patch from LLM or agent output.

    Tries multiple strategies in order:
    1. <diff>...</diff> or <patch>...</patch> XML tags
    2. ```diff ... ``` or ```patch ... ``` fenced code blocks
    3. Any fenced code block
    4. Raw text (fallback)

    Compatible with SWE-bench's extract_diff() logic.
    """
    if not text:
        return ""

    diff_matches = []
    other_matches = []

    # Strategy 1: XML tags
    pattern = re.compile(r"<(diff|patch)>(.*?)</\1>", re.DOTALL)
    for tag, match in pattern.findall(text):
        diff_matches.append(match.strip())

    # Strategy 2: fenced code blocks
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for lang, match in pattern.findall(text):
        if lang in ("diff", "patch"):
            diff_matches.append(match.strip())
        else:
            other_matches.append(match.strip())

    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return text.strip()


# ── Prediction I/O ───────────────────────────────────────────────────────────

def write_predictions(filepath: str, predictions: List[Dict]) -> None:
    """Write predictions to a JSONL file (SWE-bench compatible)."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def read_predictions(filepath: str) -> List[Dict]:
    """Read predictions from a JSONL file."""
    predictions = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def build_task_prompt(instance: Dict, repo_dir: str) -> str:
    """Build the shared task prompt for one SWE-bench instance."""
    hints = str(instance.get("hints_text", "")).strip()
    hints_block = f"\n## Hints\n{hints}\n\n" if hints else "\n"
    return TASK_PROMPT_TEMPLATE.format(
        repo=instance["repo"],
        instance_id=instance["instance_id"],
        problem_statement=instance["problem_statement"],
        hints_block=hints_block,
        repo_dir=repo_dir,
    )
