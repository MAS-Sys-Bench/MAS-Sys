from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.logger import logger
from app.prompt.proposal.proposal_aggregator import (
    AGGREGATION_PROMPT_TEMPLATE,
    FINAL_PROPOSAL_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message

PROPOSAL_WRITER_FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


class ProposalAggregatorAgent(BaseAgent):
    """Aligned proposal aggregator that merges proposal candidates into one final proposal."""

    name: str = "proposal_aggregator"
    description: str = "Summarizes multiple proposal candidates into one final five-question proposal."
    system_prompt: str = FINAL_PROPOSAL_SYSTEM_PROMPT
    max_steps: int = 1
    continuum_expected_llm_calls: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def build_request(*, proposal_candidates: str) -> str:
        return AGGREGATION_PROMPT_TEMPLATE.format(
            proposal_candidates=proposal_candidates,
        )

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", self.build_request(proposal_candidates=request))

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result = await self.step()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        messages = []
        for user_text, assistant_text in PROPOSAL_WRITER_FEWSHOT_EXAMPLES:
            messages.append(Message.user_message(user_text))
            messages.append(Message.assistant_message(assistant_text))
        messages.extend(self.messages)

        logger.info(f"[ProposalAggregatorAgent] Running node '{self.name}'")
        response = await self.ask_with_trace(
            messages=messages,
            system_msgs=[Message.system_message(self.system_prompt)],
            stream=False,
        )
        self.update_memory("assistant", response)
        self.state = AgentState.FINISHED
        return response
