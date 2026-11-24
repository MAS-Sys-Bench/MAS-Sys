from __future__ import annotations

from typing import Optional

from app.agent.base import BaseAgent
from app.logger import logger
from app.prompt.review.metareview_chair import (
    FINAL_META_REVIEW_SYSTEM_PROMPT,
    METAREVIEW_PROMPT_TEMPLATE,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


class MetaReviewChairAgent(BaseAgent):
    """ResearchTown-style metareview chair agent."""

    name: str = "metareview_chair"
    description: str = "Synthesizes multiple reviews into a concise metareview and accept/reject decision."
    max_steps: int = 1
    continuum_expected_llm_calls: int = 1

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        reviews = request or ""
        self.update_memory("user", METAREVIEW_PROMPT_TEMPLATE.format(reviews=reviews))

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result = await self.step_with_reviews()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result

    async def step(self) -> str:
        raise NotImplementedError("MetaReviewChairAgent uses step_with_reviews().")

    async def step_with_reviews(self) -> str:
        messages = []
        for user_text, assistant_text in FEWSHOT_EXAMPLES:
            messages.append(Message.user_message(user_text))
            messages.append(Message.assistant_message(assistant_text))
        messages.extend(self.messages)

        logger.info(f"[MetaReviewChairAgent] Running node '{self.name}'")
        metareview = await self.ask_with_trace(
            messages=messages,
            system_msgs=[Message.system_message(FINAL_META_REVIEW_SYSTEM_PROMPT)],
            stream=False,
        )
        self.update_memory("assistant", metareview)
        self.state = AgentState.FINISHED
        return metareview
