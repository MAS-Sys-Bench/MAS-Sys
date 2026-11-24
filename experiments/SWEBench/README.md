# SWE-bench Experiment

Evaluates MAS-Sys on [SWE-bench](https://www.swebench.com/), a benchmark of real-world GitHub issues from popular Python repositories.

## Overview

Unlike HumanEval (single-function completion), SWE-bench requires agents to:
1. Explore a real codebase (clone + checkout at the bug-introducing commit)
2. Understand the issue from the problem statement
3. Locate and edit the relevant source files
4. Produce a unified diff patch that fixes the issue

This makes it a natural fit for MAS-Sys's tool-using agents (`str_replace_editor`, `bash`, etc.) and multi-agent topologies.

## Pipeline

```
SWE-bench Lite (534 instances, HuggingFace)
        │
        ▼
  Per instance:
    1. git clone {repo} → local temp dir
    2. git checkout {base_commit}
    3. PlanAgent.plan(task_desc) → MASConfig
    4. FlowFactory → flow.execute(prompt)
       └─ agents use str_replace_editor to read/edit files
    5. git diff → model_patch
    6. Output: {instance_id, model_name_or_path, model_patch}
        │
        ▼
  predictions.jsonl
        │
        ▼
    experiments.SWEBench.harness.run_evaluation (Docker)
        │
        ▼
  Resolved Rate
```

## Prerequisites

1. **Git** available on PATH
2. **Docker Desktop** running (required for evaluation; also required if sandbox mode is enabled)
3. **swebench** package installed:
   ```bash
   pip install -e path/to/SWE-bench
   ```
4. **datasets** package:
   ```bash
   pip install datasets
   ```
5. Sufficient disk space (~50GB+ for Docker images)

## Sandbox Mode

When `use_sandbox = true` in `config/config.toml` `[sandbox]` section, agent tools (`bash`, `str_replace_editor`, `python_execute`) run inside an isolated Docker container instead of on the host machine. The repository is mounted into the container at `/workspace/{instance_id}`.

This is recommended for safe and reproducible evaluation. To configure:

```toml
[sandbox]
use_sandbox = true
image = "python:3.12-slim"
network_enabled = true    # Required for agents to pip install dependencies
```

For SWE-bench generation, MAS-Sys now uses the embedded harness under experiments/SWEBench/harness to derive and build the repo/version-specific environment image automatically. No extra sandbox config keys are required. If sandbox mode is enabled, generation reuses that harness-built environment image as the agent runtime image for the current SWE-bench instance.

**Platform notes**:
- **Windows**: Requires Docker Desktop with WSL2 backend. The sandbox image is pulled automatically on first use.
- **Linux/macOS**: Docker Engine or Docker Desktop. Works out of the box.
- Evaluation runs through the embedded module path `experiments.SWEBench.harness.run_evaluation`.
- Harness image build artifacts and logs are stored under `logs/build_images/` and `logs/run_evaluation/`.

## Quick Start

```bash
# Run on first 3 instances (generation + evaluation)
python -m experiments.SWEBench.run_swebench --end 3

# Run on specific instances, skip Docker eval
python -m experiments.SWEBench.run_swebench \
    --instance_ids django__django-11583 \
    --skip_eval

# Generation only, keep repo checkouts for debugging
python -m experiments.SWEBench.run_swebench \
    --end 5 --skip_eval --no_cleanup \
    --workdir ./swebench_workdir

# Evaluate existing predictions
python -m experiments.SWEBench.run_swebench \
    --eval_only \
    --predictions_file results/predictions_planagent_20260317_150000.jsonl
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `princeton-nlp/SWE-bench_Lite` | HuggingFace dataset name |
| `--split` | `test` | Dataset split |
| `--start` | `None` | Start index (inclusive) |
| `--end` | `None` | End index (exclusive) |
| `--instance_ids` | `None` | Specific instance IDs |
| `--model_name` | `mas-sys` | Model identifier in predictions |
| `--timeout` | `600` | Per-instance generation timeout (seconds) |
| `--workdir` | temp dir | Directory for repo checkouts |
| `--no_cleanup` | `False` | Keep repo copies after generation |
| `--output_dir` | `results/` | Output directory |
| `--skip_eval` | `False` | Skip Docker evaluation |
| `--eval_workers` | `4` | Parallel Docker containers |
| `--eval_timeout` | `1800` | Per-instance eval timeout (seconds) |
| `--eval_only` | `False` | Only evaluate, skip generation |
| `--predictions_file` | `None` | Existing predictions for `--eval_only` |

## Output Files

After a run, the `results/` directory contains:

| File | Description |
|------|-------------|
| `predictions_{run_name}.jsonl` | SWE-bench compatible predictions |
| `meta_{run_name}.json` | Run metadata (timing, config, instance IDs) |
| `report_{run_name}.json` | Docker evaluation report (if evaluation ran) |

## Predictions Format

Each line in the JSONL file:
```json
{
    "instance_id": "django__django-11583",
    "model_name_or_path": "mas-sys",
    "model_patch": "diff --git a/django/utils/autoreload.py b/django/utils/autoreload.py\n..."
}
```

## Repo Caching

To avoid re-downloading repos, bare clones are cached in `experiments/SWEBench/repo_cache/`. Each instance creates a working copy from the cache, so multiple instances from the same repo share the download.

## Notes

- **Windows**: The `bash` tool is Unix-only. Agents will primarily use `str_replace_editor` (view/create/edit) and `python_execute` on Windows. PlanAgent may select `manus` or `swe` agent types accordingly.
- **Evaluation requires Docker**: Generation works without Docker, but evaluation must use Docker containers to reliably reproduce the test environment for each repository version.
- **Disk space**: Docker images for all SWE-bench Lite repos can total ~50GB+. Start with a small subset to gauge space requirements.
