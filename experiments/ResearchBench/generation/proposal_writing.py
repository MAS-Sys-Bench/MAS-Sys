"""Proposal generation via MAS-Sys MAS pipeline.

Benchmark entry point that delegates everything to the app/ layer:
  PlanAgent.plan(task) → MASConfig → FlowFactory → flow.execute()

All agent logic, prompts, and tools live in app/.  This module only
bridges the benchmark data format to the MAS execution API.
"""

import asyncio
from typing import Any, Dict, List

from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.mas.plan_agent import PlanAgent
from experiments.ResearchBench.generation.proposal_prompt import TASK_PROMPT_TEMPLATE


# Helpers


def _build_author_identity(p: Any) -> Dict[str, str]:
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


def _build_author_profile(p: Any) -> Dict[str, Any]:
    identity = _build_author_identity(p)
    return {
        **identity,
        "publications": _normalize_publications(p),
    }


def _format_author_profiles(author_profiles: List[Dict[str, Any]]) -> str:
    if not author_profiles:
        return "No author identities were provided."
    blocks = []
    for i, author in enumerate(author_profiles, start=1):
        pubs = author.get("publications", [])
        pub_lines = []
        for j, pub in enumerate(pubs[:3], start=1):
            title = str(pub.get("title", "")).strip() or "Untitled"
            abstract = str(pub.get("abstract", "")).strip()
            pub_lines.append(f"[{j}] {title}" + (f"\n{abstract[:300]}" if abstract else ""))
        publications_text = "\n\n".join(pub_lines) if pub_lines else "No publications provided."
        blocks.append(
            f"Author {i}:\n"
            f"Name: {author.get('name', '').strip() or 'Unknown'}\n"
            f"Affiliation: {author.get('affiliation', '').strip() or 'Unknown'}\n"
            f"Publications for agent_params.publications:\n{publications_text}"
        )
    return "\n\n".join(blocks)


def _format_references(ref_contents: List[str]) -> str:
    refs = [ref for ref in ref_contents if ref]
    if not refs:
        return "No reference papers were provided."
    return "\n\n".join(
        f"Reference {i + 1}:\n{ref}" for i, ref in enumerate(refs)
    )


def build_proposal_task_prompt(
    author_profiles: List[Dict[str, Any]],
    ref_contents: List[str],
) -> str:
    return TASK_PROMPT_TEMPLATE.format(
        author_profiles_text=_format_author_profiles(author_profiles),
        references_text=_format_references(ref_contents),
    )


# Main entry point

async def _run_proposal_pipeline(
    author_profiles: List[Dict[str, Any]],
    ref_contents: List[str],
    runtime: str = "legacy",
) -> str:
    """PlanAgent → FlowFactory → flow.execute()."""
    task_prompt = build_proposal_task_prompt(author_profiles, ref_contents)

    plan_agent = PlanAgent(runtime=runtime)
    mas_config = await plan_agent.plan(task_prompt, runtime=runtime)
    logger.info(
        f'[ProposalBench] PlanAgent topology={mas_config.topology}, '
        f'agents={[n.instance_id for n in mas_config.agents]}'
    )

    flow = FlowFactory.from_config(
        mas_config,
        plan_agent=plan_agent,
        runtime=runtime,
    )

    result = await flow.execute(task_prompt)
    logger.info(f'[ProposalBench] Result: {len(result)} chars.')
    return result


def write_proposal(
    profiles: List[Any],
    ref_contents: List[str],
    runtime: str = "legacy",
) -> str:
    """Generate a research proposal via the MAS-Sys MAS pipeline.

    Args:
        profiles: List of researcher profile dicts or objects with
            ``publications`` fields.
        ref_contents: List of reference paper abstracts.

    Returns:
        The generated research proposal as a string.
    """
    author_profiles = [_build_author_profile(p) for p in profiles]
    return asyncio.run(
        _run_proposal_pipeline(
            author_profiles,
            ref_contents,
            runtime=runtime,
        )
    )
