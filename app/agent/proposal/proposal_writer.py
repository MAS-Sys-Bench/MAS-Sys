from __future__ import annotations

import re
from typing import Any, Dict

from pydantic import Field, PrivateAttr

from app.agent.base import BaseAgent
from app.logger import logger
from app.prompt.proposal.proposal_writer import (
    CANDIDATE_PROMPT_TEMPLATE,
    GENERATE_PROFILE_SYSTEM_PROMPT,
    PAPER_WRITER_SYSTEM_TEMPLATE,
    PROFILE_PROMPT_TEMPLATE,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


class ProposalWriterAgent(BaseAgent):
    """Aligned proposal writer that first builds a persona, then writes one proposal candidate."""

    name: str = "proposal_writer"
    description: str = "Generates a researcher persona from publications, then writes one proposal candidate."
    system_prompt: str = PAPER_WRITER_SYSTEM_TEMPLATE
    max_steps: int = 2
    continuum_expected_llm_calls: int = 2
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    _runtime_author_name: str = PrivateAttr(default="")
    _runtime_affiliation: str = PrivateAttr(default="")
    _runtime_publications: str = PrivateAttr(default="No publications provided.")

    @staticmethod
    def _build_publications(profile: Any) -> str:
        if isinstance(profile, str):
            text = profile.strip()
            return text or "No publications provided."

        if isinstance(profile, list):
            if not profile:
                return "No publications provided."
            lines = []
            for idx, pub in enumerate(profile[:10], start=1):
                if isinstance(pub, dict):
                    title = str(pub.get("title", "")).strip()
                    abstract = str(pub.get("abstract", "")).strip()
                    if title and abstract:
                        lines.append(f"[{idx}] {title}\n{abstract}")
                    elif title:
                        lines.append(f"[{idx}] {title}")
                    elif abstract:
                        lines.append(f"[{idx}] {abstract}")
                else:
                    text = str(pub).strip()
                    if text:
                        lines.append(f"[{idx}] {text}")
            return "\n\n".join(lines) if lines else "No publications provided."

        if isinstance(profile, dict):
            pubs = profile.get("publications", [])
            if not pubs:
                return "No publications provided."
            lines = []
            for idx, pub in enumerate(pubs[:10], start=1):
                if isinstance(pub, dict):
                    title = str(pub.get("title", "")).strip()
                    abstract = str(pub.get("abstract", "")).strip()
                    if title and abstract:
                        lines.append(f"[{idx}] {title}\n{abstract}")
                    elif title:
                        lines.append(f"[{idx}] {title}")
                    elif abstract:
                        lines.append(f"[{idx}] {abstract}")
                else:
                    lines.append(f"[{idx}] {str(pub)}")
            return "\n\n".join(lines) if lines else "No publications provided."

        pubs = getattr(profile, "publications", None)
        if not pubs:
            return "No publications provided."
        return "\n\n".join(str(pub) for pub in pubs[:10])

    def initialize_from_params(self) -> None:
        params = self.agent_params
        self._runtime_author_name = str(params.get("author_name", ""))
        self._runtime_affiliation = str(params.get("affiliation", ""))
        publications = params.get("publications", [])
        self._runtime_publications = self._build_publications(publications)

    @staticmethod
    def build_profile_request(*, author_name: str, affiliation: str, publications: str) -> str:
        return PROFILE_PROMPT_TEMPLATE.format(
            author_name=author_name or "Unknown",
            affiliation=affiliation or "Unknown",
            publications=publications,
        )

    @staticmethod
    def build_candidate_request(*, related_papers: str) -> str:
        return CANDIDATE_PROMPT_TEMPLATE.format(related_papers=related_papers)

    @staticmethod
    def _extract_related_papers(request: str) -> str:
        text = (request or "").strip()
        if not text:
            return ""
        match = re.search(
            r"\[Reference Papers\]\s*(.*?)\s*(?:\[Requirements\]|\Z)",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        async with self.state_context(AgentState.RUNNING):
            author_name = self._runtime_author_name
            affiliation = self._runtime_affiliation
            publications = self._runtime_publications
            related_papers = self._extract_related_papers(request or "")

            self.current_step = 1
            self.start_single_step_trace()
            persona = await self._generate_persona(
                author_name=author_name,
                affiliation=affiliation,
                publications=publications,
            )
            self.end_single_step_trace()

            self.current_step = 2
            self.start_single_step_trace()
            result = await self._write_candidate(persona=persona, related_papers=related_papers)
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        raise NotImplementedError("ProposalWriterAgent uses a fixed two-stage aligned flow.")

    async def _generate_persona(self, *, author_name: str, affiliation: str, publications: str) -> str:
        messages = []
        for user_text, assistant_text in FEWSHOT_EXAMPLES:
            messages.append(Message.user_message(user_text))
            messages.append(Message.assistant_message(assistant_text))
        messages.append(
            Message.user_message(
                self.build_profile_request(
                    author_name=author_name,
                    affiliation=affiliation,
                    publications=publications,
                )
            )
        )

        logger.info(f"[ProposalWriterAgent] Generating persona for node '{self.name}'")
        response = await self.ask_with_trace(
            messages=messages,
            system_msgs=[Message.system_message(GENERATE_PROFILE_SYSTEM_PROMPT)],
            stream=False,
        )
        return response

    async def _write_candidate(self, *, persona: str, related_papers: str) -> str:
        messages = []
        for user_text, assistant_text in FEWSHOT_EXAMPLES:
            messages.append(Message.user_message(user_text))
            messages.append(Message.assistant_message(assistant_text))
        messages.append(Message.user_message(self.build_candidate_request(related_papers=related_papers)))

        logger.info(f"[ProposalWriterAgent] Writing proposal candidate for node '{self.name}'")
        response = await self.ask_with_trace(
            messages=messages,
            system_msgs=[Message.system_message(PAPER_WRITER_SYSTEM_TEMPLATE.format(profile=persona))],
            stream=False,
        )
        self.update_memory("assistant", response)
        return response
