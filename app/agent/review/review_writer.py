from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import Field, PrivateAttr

from app.agent.base import BaseAgent
from app.prompt.review.review_writer import (
    GENERATE_PROFILE_SYSTEM_PROMPT,
    PROFILE_PROMPT_TEMPLATE,
    REVIEW_OUTPUT_TEMPLATE,
    REVIEW_SCORE_SYSTEM_PROMPT,
    REVIEW_STRENGTH_SYSTEM_PROMPT,
    REVIEW_WEAKNESS_SYSTEM_PROMPT,
    SCORE_PROMPT_TEMPLATE,
    STRENGTH_PROMPT_TEMPLATE,
    WEAKNESS_PROMPT_TEMPLATE,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


class ReviewWriterAgent(BaseAgent):
    """Aligned review worker: persona -> strength -> weakness -> score."""

    name: str = "review_writer"
    description: str = "Generates reviewer persona, then writes strength, weakness, and score for one submission."
    max_steps: int = 4
    continuum_expected_llm_calls: int = 4
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    _runtime_reviewer_name: str = PrivateAttr(default="")
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
        self._runtime_reviewer_name = str(params.get("reviewer_name", ""))
        self._runtime_affiliation = str(params.get("affiliation", ""))
        publications = params.get("publications", [])
        self._runtime_publications = self._build_publications(publications)

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        proposal, shared_citations = self._split_shared_request(request or "")

        async with self.state_context(AgentState.RUNNING):
            reviewer_name = self._runtime_reviewer_name
            affiliation = self._runtime_affiliation
            publications = self._runtime_publications
            citations = shared_citations or str(self.agent_params.get("citations", ""))

            self.current_step = 1
            self.start_single_step_trace()
            persona = await self._generate_persona(
                reviewer_name=reviewer_name,
                affiliation=affiliation,
                publications=publications,
            )
            self.end_single_step_trace()

            self.current_step = 2
            self.start_single_step_trace()
            strength = await self._ask_with_prompt(
                prompt=STRENGTH_PROMPT_TEMPLATE.format(
                    profile=persona,
                    proposal=proposal,
                    citations=citations,
                ),
                system_prompt=REVIEW_STRENGTH_SYSTEM_PROMPT,
            )
            self.end_single_step_trace()

            self.current_step = 3
            self.start_single_step_trace()
            weakness = await self._ask_with_prompt(
                prompt=WEAKNESS_PROMPT_TEMPLATE.format(
                    profile=persona,
                    proposal=proposal,
                    citations=citations,
                ),
                system_prompt=REVIEW_WEAKNESS_SYSTEM_PROMPT,
            )
            self.end_single_step_trace()

            self.current_step = 4
            self.start_single_step_trace()
            score_text = await self._ask_with_prompt(
                prompt=SCORE_PROMPT_TEMPLATE.format(
                    profile=persona,
                    strength=strength,
                    weakness=weakness,
                ),
                system_prompt=REVIEW_SCORE_SYSTEM_PROMPT,
            )
            self.end_single_step_trace()

            score = self._extract_score(score_text)
            result = REVIEW_OUTPUT_TEMPLATE.format(
                persona=persona,
                strength=strength,
                weakness=weakness,
                score=score,
            )
            self.update_memory("assistant", result)
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result

    async def step(self) -> str:
        raise NotImplementedError("ReviewWriterAgent uses a fixed multi-stage aligned flow.")

    @staticmethod
    def _split_shared_request(request: str) -> tuple[str, str]:
        marker = "[Cited Paper Abstracts]"
        if marker not in request:
            return request, ""
        submission = request.split(marker, 1)[0].replace("[Submission]", "", 1).strip()
        citations = request.split(marker, 1)[1].strip()
        return submission, citations

    async def _generate_persona(
        self,
        *,
        reviewer_name: str,
        affiliation: str,
        publications: str,
    ) -> str:
        return await self._ask_with_prompt(
            prompt=PROFILE_PROMPT_TEMPLATE.format(
                reviewer_name=reviewer_name or "Unknown",
                affiliation=affiliation or "Unknown",
                publications=publications,
            ),
            system_prompt=GENERATE_PROFILE_SYSTEM_PROMPT,
        )

    async def _ask_with_prompt(self, *, prompt: str, system_prompt: str) -> str:
        messages = []
        for user_text, assistant_text in FEWSHOT_EXAMPLES:
            messages.append(Message.user_message(user_text))
            messages.append(Message.assistant_message(assistant_text))
        messages.append(Message.user_message(prompt))

        logger.info(f"[ReviewWriterAgent] Running node '{self.name}'")
        return await self.ask_with_trace(
            messages=messages,
            system_msgs=[Message.system_message(system_prompt)],
            stream=False,
        )

    @staticmethod
    def _extract_score(score_response: str) -> int:
        match = re.findall(r"\d+", score_response or "")
        if not match:
            return 0
        score = int(match[0])
        return max(1, min(score, 10))
