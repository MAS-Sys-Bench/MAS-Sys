from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.paperviz.common import extract_paperbanana_context, split_root_and_upstream, task_name_from_params
from app.logger import logger
from app.prompt.paperviz.stylist import (
    DIAGRAM_STYLIST_SYSTEM_PROMPT,
    PLOT_STYLIST_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message


class PaperVizStylistAgent(BaseAgent):
    """PaperViz stylist agent that refines a planner description with style guidance."""

    name: str = "paperviz_stylist"
    description: str = "Refines a figure plan into a polished publication-quality visual description."
    max_steps: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        prompt = self._build_prompt(request or "")
        self.update_memory("user", prompt)

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result = await self.step()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        system_prompt = self._system_prompt()
        logger.info(f"[PaperVizStylistAgent] Running node '{self.name}'")
        response = await self.ask_with_trace(
            messages=self.messages,
            system_msgs=[Message.system_message(system_prompt)],
            stream=False,
        )
        self.update_memory("assistant", response)
        self.state = AgentState.FINISHED
        return response

    def _system_prompt(self) -> str:
        return (
            PLOT_STYLIST_SYSTEM_PROMPT
            if task_name_from_params(self.agent_params) == "plot"
            else DIAGRAM_STYLIST_SYSTEM_PROMPT
        )

    def _build_prompt(self, detailed_description: str) -> str:
        params = self.agent_params
        root_text, upstream_text = split_root_and_upstream(detailed_description)
        context = extract_paperbanana_context(root_text)
        task_name = str(params.get("task_name") or context.get("task_name") or task_name_from_params(params))
        task_name = "plot" if task_name == "plot" else "diagram"
        style_guide = str(params.get("style_guide", "")).strip()
        return self._format_stylist_prompt(
            task_name=task_name,
            detailed_description=upstream_text or detailed_description,
            style_guide=style_guide,
            content=str(params.get("content") or context.get("content", "")),
            visual_intent=str(params.get("visual_intent") or context.get("visual_intent", "")),
        )

    @staticmethod
    def _format_stylist_prompt(
        *,
        task_name: str,
        detailed_description: str,
        style_guide: str,
        content: str,
        visual_intent: str,
    ) -> str:
        context_label = "Raw Data" if task_name == "plot" else "Methodology Section"
        visual_label = (
            "Visual Intent of the Desired Plot" if task_name == "plot" else "Diagram Caption"
        )
        return (
            f"Detailed Description: {detailed_description}\n"
            f"Style Guidelines: {style_guide}\n"
            f"{context_label}: {content}\n"
            f"{visual_label}: {visual_intent}\n"
            "Your Output:"
        )
