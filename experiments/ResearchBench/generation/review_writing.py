"""Review generation via MAS-Sys MAS pipeline.

Benchmark entry point that delegates everything to the app/ layer:
  PlanAgent.plan(task) → MASConfig → FlowFactory → DAGFlow.execute()

All agent logic, prompts, and tools live in app/. This module bridges
the benchmark data format to the MAS execution API, then reads reviewer
and chair outputs back from the instantiated DAG agents.
"""

import asyncio
import re
from typing import Any, Dict, List, Tuple

from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.mas.plan_agent import PlanAgent
from app.mas.schema import MASConfig
from experiments.ResearchBench.generation.review_prompt import (
    TASK_PROMPT_TEMPLATE,
)


# Helpers


def _build_reviewer_identity(p: Any) -> Dict[str, str]:
    if isinstance(p, dict):
        return {
            'name': str(p.get('name', '')).strip(),
            'affiliation': str(p.get('affiliation', '')).strip(),
        }
    return {
        'name': str(getattr(p, 'name', '')).strip(),
        'affiliation': str(getattr(p, 'affiliation', '')).strip(),
    }


def _normalize_publications(value: Any) -> List[Dict[str, str]]:
    publications = value.get("publications", []) if isinstance(value, dict) else getattr(value, "publications", [])
    normalized: List[Dict[str, str]] = []
    for pub in list(publications or [])[:10]:
        if isinstance(pub, dict):
            normalized.append(
                {
                    "title": str(pub.get("title", "")).strip(),
                    "abstract": str(pub.get("abstract", "")).strip(),
                }
            )
        else:
            normalized.append({"title": str(pub).strip(), "abstract": ""})
    return normalized


def _build_reviewer_profile(p: Any) -> Dict[str, Any]:
    identity = _build_reviewer_identity(p)
    return {
        **identity,
        "publications": _normalize_publications(p),
    }


def _format_reviewer_profiles(reviewers: List[Dict[str, Any]]) -> str:
    if not reviewers:
        return "No reviewer identities were provided."
    blocks = []
    for i, reviewer in enumerate(reviewers, start=1):
        pubs = reviewer.get("publications", [])
        pub_lines = []
        for j, pub in enumerate(pubs[:3], start=1):
            title = str(pub.get("title", "")).strip() or "Untitled"
            abstract = str(pub.get("abstract", "")).strip()
            pub_lines.append(f"[{j}] {title}" + (f"\n{abstract[:300]}" if abstract else ""))
        publications_text = "\n\n".join(pub_lines) if pub_lines else "No publications provided."
        blocks.append(
            f"Reviewer {i}:\n"
            f"Name: {reviewer.get('name', '').strip() or 'Unknown'}\n"
            f"Affiliation: {reviewer.get('affiliation', '').strip() or 'Unknown'}\n"
            f"Publications for agent_params.publications:\n{publications_text}"
        )
    return "\n\n".join(blocks)


def _format_citations(ref_contents: List[str]) -> str:
    refs = [ref for ref in ref_contents if ref]
    if not refs:
        return "No cited-paper abstracts were provided."
    return "\n\n".join(
        f"Cited Paper {i + 1} Abstract:\n{ref}"
        for i, ref in enumerate(refs)
    )


def build_review_task_prompt(
    paper_content: str,
    reviewer_profiles: List[Dict[str, Any]],
    ref_contents: List[str],
) -> str:
    return TASK_PROMPT_TEMPLATE.format(
        paper_content=paper_content,
        reviewer_profiles_text=_format_reviewer_profiles(reviewer_profiles),
        citations_text=_format_citations(ref_contents),
    )


def _parse_review_output(
    flow_output: str,
) -> Tuple[str, str, int]:
    """Parse one reviewer agent output string."""
    strength = _extract_tagged_section(flow_output, 'Review Strength')
    weakness = _extract_tagged_section(flow_output, 'Review Weakness')
    score = _extract_score(flow_output)
    return strength, weakness, score


def _collect_review_outputs(
    mas_config: MASConfig,
    flow: Any,
) -> Tuple[str, str, List[int], Dict[str, Dict[str, Any]]]:
    """Collect reviewer outputs from instantiated agents after DAG execution."""
    review_per_reviewer: Dict[str, Dict[str, Any]] = {}
    all_scores: List[int] = []
    last_strength = ''
    last_weakness = ''

    reviewer_nodes = [
        node for node in mas_config.agents if node.agent_key == 'research_review_writer'
    ]
    for node in reviewer_nodes:
        agent = flow.agents.get(node.instance_id)
        if not agent or not getattr(agent.memory, 'messages', None):
            continue
        assistant_messages = [m for m in agent.memory.messages if m.role == 'assistant']
        if not assistant_messages:
            continue
        review_text = assistant_messages[-1].content or ''
        strength, weakness, score = _parse_review_output(review_text)
        review_per_reviewer[node.instance_id] = {
            'strength': strength,
            'weakness': weakness,
            'score': score,
        }
        all_scores.append(score)
        last_strength = strength
        last_weakness = weakness

    if not review_per_reviewer:
        logger.warning(
            '[ReviewBench] Could not collect structured reviewer outputs.'
        )
        review_per_reviewer['reviewer_0'] = {
            'strength': '',
            'weakness': '',
            'score': 0,
        }
        all_scores = [0]

    return last_strength, last_weakness, all_scores, review_per_reviewer


def _extract_tagged_section(text: str, tag: str) -> str:
    """Extract text after a [Tag] heading until the next heading or end."""
    pattern = rf'\[{re.escape(tag)}\]\s*\n(.*?)(?=\n\[[^\n]+\]\n|$)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ''


def _extract_score(text: str) -> int:
    """Extract integer score from review output."""
    m = re.search(r"\[Review Score\]\s*\n\s*(10|[1-9])\b", text)
    if m:
        return int(m.group(1))
    # Try "Parsed score: N/10" first (from WriteReview tool)
    m = re.search(r'Parsed score:\s*(\d+)/10', text)
    if m:
        return int(m.group(1))
    # Try "score of N out of 10"
    m = re.search(r'score of (\d+) out of 10', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: any N/10
    m = re.search(r'\b(10|[1-9])\s*/\s*10\b', text)
    if m:
        return int(m.group(1))
    return 0


# Main entry point

async def _run_review_pipeline(
    paper_content: str,
    reviewer_profiles: List[Dict[str, Any]],
    ref_contents: List[str],
    runtime: str = "legacy",
) -> Tuple[str, str, List[int], Dict[str, Dict[str, Any]]]:
    """PlanAgent → FlowFactory → DAGFlow.execute() → collect."""
    task_prompt = build_review_task_prompt(
        paper_content=paper_content,
        reviewer_profiles=reviewer_profiles,
        ref_contents=ref_contents,
    )

    plan_agent = PlanAgent(runtime=runtime)
    mas_config = await plan_agent.plan(task_prompt, runtime=runtime)
    logger.info(
        f'[ReviewBench] PlanAgent topology={mas_config.topology}, '
        f'agents={[n.instance_id for n in mas_config.agents]}'
    )

    flow = FlowFactory.from_config(
        mas_config,
        plan_agent=plan_agent,
        runtime=runtime,
    )
    result = await flow.execute(task_prompt)
    logger.info(f'[ReviewBench] Result: {len(result)} chars.')

    return _collect_review_outputs(mas_config, flow)


def write_review(
    paper_content: str,
    profiles_reviewers: List[Any],
    ref_contents: List[str],
    runtime: str = "legacy",
) -> Tuple[str, str, List[int], Dict[str, Dict[str, Any]]]:
    """Generate peer reviews via the MAS-Sys MAS pipeline.

    Args:
        paper_content: Full text of the paper to review.
        profiles_reviewers: Reviewer profile dicts (top-5 used).
        ref_contents: Reference paper abstracts (used as citations context).

    Returns:
        Tuple of (strength, weakness, scores, review_per_reviewer).
    """
    profiles_reviewers = profiles_reviewers[:5]
    reviewer_profiles = [_build_reviewer_profile(p) for p in profiles_reviewers]
    return asyncio.run(
        _run_review_pipeline(
            paper_content,
            reviewer_profiles,
            ref_contents,
            runtime=runtime,
        )
    )
