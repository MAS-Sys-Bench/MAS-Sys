import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
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
from ResearchBench.eval import compute_review_metrics
from ResearchBench.generation.review_writing import write_review
from ResearchBench.utils import load_benchmark

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parent / 'data' / 'review_bench.json'
)
RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
DEFAULT_NUM_PROCESSES = 4
INPUT_ALIASES = {
    'review': 'review_bench.json',
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
    reviewer_data: Dict[str, Any],
    full_content: Dict[str, Any],
    strengths_bp_flatte: List[str],
    weaknesses_bp_flatte: List[str],
    human_scores: List[int],
    runtime: str = "legacy",
):
    ref_abstracts = [ref['abstract'] for ref in paper_data.get('references', [])]
    profiles_reviewers: List[Any] = list(reviewer_data.values())

    paper_content = ''
    for idx, section in enumerate(full_content):
        paper_content += f'{idx + 1}:\n\n'
        section_text = full_content[section]
        paper_content += section_text + '\n\n'

    generated_strength, generated_weakness, score, review_per_reviewer = write_review(
        paper_content, profiles_reviewers, ref_abstracts, runtime=runtime,
    )

    metrics = compute_review_metrics(
        strengths_bp_flatte,
        weaknesses_bp_flatte,
        generated_strength,
        generated_weakness,
    )
    avg_score = sum(human_scores) / len(human_scores)
    avg_generated_score = sum(score) / len(score)
    dist = abs(avg_score - avg_generated_score)
    results = {
        'paper_id': paper_id,
        'strengths_bp': strengths_bp_flatte,
        'weaknesses_bp': weaknesses_bp_flatte,
        'generated_strength': generated_strength,
        'generated_weakness': generated_weakness,
        'score': human_scores,
        'generated_scores': score,
        'avg_score': avg_score,
        'avg_generated_score': avg_generated_score,
        'score_diff': dist,
    }
    return results, metrics


def save_results(results: Dict[str, Any], metrics: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, 'w') as f:
        json.dump({**results, **metrics}, f)
        f.write('\n')


def process_task(
    args_tuple: Tuple[
        str,  # paper_id
        Dict[str, Any],  # paper_data
        Dict[str, Any],  # author_data
        Dict[str, Any],  # reviewer_data
        Dict[str, Any],  # full_content
        List[str],  # strengths_bp_flatten
        List[str],  # weaknesses_bp_flatten
        List[int],  # human_scores
        int,  # instance_no
        str,  # result_label
        str,  # result_dir
        str,  # runtime
    ],
) -> None:
    (
        paper_id,
        paper_data,
        author_data,
        reviewer_data,
        full_content,
        strengths_bp_flatten,
        weaknesses_bp_flatten,
        human_scores,
        instance_no,
        result_label,
        result_dir,
        runtime,
    ) = args_tuple

    session_id = f'researchbench_review:{paper_id}:{instance_no}:{os.getpid()}'
    tracer.activate_session(session_id, reset=True)
    tracer.start_e2e()

    try:
        results, metrics = inference(
            paper_id,
            paper_data,
            author_data,
            reviewer_data,
            full_content,
            strengths_bp_flatten,
            weaknesses_bp_flatten,
            human_scores,
            runtime,
        )
        output_path = Path(result_dir) / f'{result_label}_{instance_no}.jsonl'
        save_results(results, metrics, output_path)
    finally:
        tracer.end_e2e()
        tracer.save(result_dir, prefix=f'{result_label}_{instance_no}_')
        tracer.clear_session(session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description='Research Review Evaluation (MAS-Sys)')
    parser.add_argument(
        '-i',
        '--input',
        '--input_path',
        dest='input_path',
        type=str,
        default='review',
        help='Input benchmark alias or JSON path: review',
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

        # Prepare tasks
        tasks = []
        for i, (paper_id, data) in enumerate(dataset.items()):
            paper_data = data['paper_data']
            full_content = paper_data['full_content']
            author_data = data['author_data']
            reviewer_data = data['reviewer_data']
            reference_review = data['review_data']
            human_scores = [
                int(review.get('rating').split(':')[0]) for review in reference_review
            ]
            strengths_bp = [
                review.get('strength_evaluation_format', []) for review in reference_review
            ]
            strengths_bp_flatten = [item for sublist in strengths_bp for item in sublist]
            weaknesses_bp = [
                review.get('weakness_evaluation_format', []) for review in reference_review
            ]
            weaknesses_bp_flatten = [item for sublist in weaknesses_bp for item in sublist]

            tasks.append(
                (
                    paper_id,
                    paper_data,
                    author_data,
                    reviewer_data,
                    full_content,
                    strengths_bp_flatten,
                    weaknesses_bp_flatten,
                    human_scores,
                    i + 1,
                    input_label,
                    str(result_dir),
                    args.runtime,
                )
            )

        # Process tasks in parallel
        with Pool(processes=args.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_task, tasks),
                    total=len(tasks),
                    desc='Processing papers',
                )
            )
    finally:
        pass


if __name__ == '__main__':
    main()
