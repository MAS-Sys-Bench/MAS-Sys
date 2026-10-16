import argparse
import json
import os
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure both MAS-Sys root and experiments/ are on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])  # MAS-Sys/
_EXPERIMENTS = str(Path(__file__).resolve().parents[1])   # MAS-Sys/experiments/
for _p in (_PROJECT_ROOT, _EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tqdm import tqdm

from app.logger import logger
from app.tracer import tracer
from ResearchBench.eval import compute_proposal_metrics
from ResearchBench.generation.proposal_writing import write_proposal
from ResearchBench.utils import load_benchmark

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parent / 'data' / 'paper_bench_medium.json'
)
RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
DEFAULT_NUM_PROCESSES = 4
INPUT_ALIASES = {
    'easy': 'paper_bench_easy.json',
    'medium': 'paper_bench_medium.json',
    'hard': 'paper_bench_hard.json',
    'high_impact': 'high_impact_paper_bench.json',
}


def resolve_input_path(input_name: str) -> str:
    alias = INPUT_ALIASES.get(input_name)
    if alias is not None:
        return str(Path(__file__).resolve().parent / 'data' / alias)
    return input_name


def resolve_input_label(input_name: str, input_path: str) -> str:
    if input_name in INPUT_ALIASES:
        return input_name
    return Path(input_path).stem


def make_result_dir(input_label: str, timestamp: str) -> Path:
    result_dir = RESULTS_ROOT / f'{input_label}_{timestamp}'
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def inference(
    paper_id: str,
    paper_data: Dict[str, Any],
    author_data: Dict[str, Any],
    ref_proposal: str,
    runtime: str = "legacy",
) -> Tuple[Dict[str, str], Dict[str, float]]:
    profiles: List[Any] = list(author_data.values())
    ref_abstracts = [ref['abstract'] for ref in paper_data.get('references', [])]

    gen_proposal = write_proposal(profiles, ref_abstracts, runtime=runtime)

    metrics = compute_proposal_metrics(ref_proposal, gen_proposal)
    results = {
        'paper_id': paper_id,
        'ref_proposal': ref_proposal,
        'gen_proposal': gen_proposal,
    }
    return results, metrics


def save_results(
    results: Dict[str, Any],
    metrics: Dict[str, float],
    output_path: Path,
    lock: Any,
    metrics_summary: Dict[str, List[float]],
) -> None:
    with open(output_path, 'w') as f:
        json.dump({**results, **metrics}, f)
        f.write('\n')

    with lock:
        # Update metrics summary
        for metric, score in metrics.items():
            if metric in metrics_summary:
                metrics_summary[metric].append(score)


def process_task(
    task: Tuple[
        str,
        Dict[str, Any],
        Dict[str, Any],
        str,
        int,
        str,
        str,
        str,
        Any,
        Dict[str, List[float]],
    ],
) -> None:
    (
        paper_id,
        paper_data,
        author_data,
        ref_proposal,
        instance_no,
        result_label,
        result_dir,
        runtime,
        lock,
        metrics_summary,
    ) = task

    session_id = f'researchbench_proposal:{paper_id}:{instance_no}:{os.getpid()}'
    tracer.activate_session(session_id, reset=True)
    tracer.start_e2e()

    try:
        results, metrics = inference(
            paper_id, paper_data, author_data, ref_proposal, runtime,
        )
        output_path = Path(result_dir) / f'{result_label}_{instance_no}.jsonl'
        save_results(results, metrics, output_path, lock, metrics_summary)
    finally:
        tracer.end_e2e()
        tracer.save(result_dir, prefix=f'{result_label}_{instance_no}_')
        tracer.clear_session(session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description='Research Proposal Evaluation (MAS-Sys)')
    parser.add_argument(
        '-i',
        '--input',
        '--input_path',
        dest='input_path',
        type=str,
        default='high_impact',
        help='Input benchmark alias or JSON path: easy, medium, hard, high_impact',
    )
    parser.add_argument(
        '-o',
        '--output',
        '--output_path',
        dest='output_path',
        type=str,
        default='',
        help='Deprecated: output is auto-generated under results/<label>_YYYYMMDD_HHMMSS/',
    )
    parser.add_argument(
        '-j',
        '--jobs',
        '--num_processes',
        dest='num_processes',
        type=int,
        default=DEFAULT_NUM_PROCESSES,
        help='Number of parallel worker processes to use',
    )
    parser.add_argument(
        '-n',
        '--instances',
        '--max-cases',
        '--max_cases',
        dest='max_cases',
        type=int,
        default=0,
        help='Number of experiment instances to run (0 = all remaining)',
    )
    parser.add_argument(
        '--runtime',
        type=str,
        choices=['legacy', 'orla', 'orla_dry_run'],
        default='legacy',
        help='Execution runtime to use for benchmark generation; orla_dry_run only validates ORLA lowering',
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_path)
    input_label = resolve_input_label(args.input_path, input_path)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    result_dir = make_result_dir(input_label, timestamp)
    output_template = result_dir / f'{input_label}_<instance_no>.jsonl'

    if args.output_path:
        logger.warning(
            f'Ignoring --output_path={args.output_path}. '
            f'Using auto-generated output template: {output_template}'
        )

    logger.info(f'Result directory: {result_dir}')
    logger.info(f'Result jsonl template: {input_label}_<instance_no>.jsonl')

    try:
        dataset = load_benchmark(input_path)
        if args.max_cases > 0:
            dataset = dict(list(dataset.items())[:args.max_cases])
        logger.info(f'Processing {len(dataset)} papers')

        # Create a manager for shared objects
        manager = Manager()
        lock = manager.Lock()
        metrics_summary = manager.dict(
            {
                metric: manager.list()
                for metric in [
                    'bleu',
                    'rouge_l',
                    'gpt_metric_score',
                    'bert_score',
                    'embedding_sim',
                    'fg_topic_consistency',
                    'fg_method_consistency',
                    'fg_factual_consistency',
                    'fg_claim_consistency',
                    'fg_app_context_consistency',
                    'fg_overall_similarity',
                    'quality_novelty_gen',
                    'quality_feasibility_gen',
                    'quality_novelty_ref',
                    'quality_feasibility_ref',
                ]
            }
        )

        # Prepare tasks
        tasks = [
            (
                paper_id,
                data['paper_data'],
                data['author_data'],
                data['paper_data']['evaluation_format']['content'],
                i + 1,
                input_label,
                str(result_dir),
                args.runtime,
                lock,
                metrics_summary,
            )
            for i, (paper_id, data) in enumerate(dataset.items())
        ]

        # Process tasks in parallel
        with Pool(processes=args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_task, tasks),
                    total=len(tasks),
                    desc='Processing papers',
                )
            )

        # Convert managed lists to regular lists for reporting
        local_metrics_summary = {
            metric: list(scores) for metric, scores in metrics_summary.items()
        }

        # Report average metrics
        for metric, scores in local_metrics_summary.items():
            if scores:
                average = sum(scores) / len(scores)
                logger.info(f"Average {metric.replace('_', ' ').upper()}: {average:.4f}")
    finally:
        pass


if __name__ == '__main__':
    main()
