from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.paperviz.common import (
    extract_paperbanana_context,
    parse_json_or_text,
    parse_upstream_payload,
    split_root_and_upstream,
    task_name_from_params,
)
from app.logger import logger
from app.prompt.paperviz.critic import (
    DIAGRAM_CRITIC_SYSTEM_PROMPT,
    PLOT_CRITIC_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message


class PaperVizCriticAgent(BaseAgent):
    """PaperViz critic agent that emits structured continue/finish decisions for FSMFlow."""

    name: str = "paperviz_critic"
    description: str = "Critiques a generated figure and emits continue/finish with revision guidance."
    max_steps: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)

    async def step(self) -> str:
        system_prompt = self._system_prompt()
        logger.info(f"[PaperVizCriticAgent] Running node '{self.name}'")
        return await self.ask_with_trace(
            messages=self.messages,
            system_msgs=[Message.system_message(system_prompt)],
            stream=False,
        )

    def _system_prompt(self) -> str:
        return (
            PLOT_CRITIC_SYSTEM_PROMPT
            if task_name_from_params(self.agent_params) == "plot"
            else DIAGRAM_CRITIC_SYSTEM_PROMPT
        )

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        params = self.agent_params
        root_text = str(payload.get("root_text", ""))
        context = extract_paperbanana_context(root_text)
        task_name = str(params.get("task_name") or context.get("task_name") or task_name_from_params(params))
        task_name = "plot" if task_name == "plot" else "diagram"
        detailed_description = payload.get("detailed_description", payload.get("description", ""))
        render_output = payload.get("render_output", payload.get("code", payload.get("render_prompt", "")))
        return self._format_critic_prompt(
            task_name=task_name,
            render_output=str(render_output),
            detailed_description=str(detailed_description),
            content=str(params.get("content") or context.get("content", "")),
            visual_intent=str(params.get("visual_intent") or context.get("visual_intent", "")),
        )

    @staticmethod
    def _format_critic_prompt(
        *,
        task_name: str,
        render_output: str,
        detailed_description: str,
        content: str,
        visual_intent: str,
    ) -> str:
        context_label = "Raw Data" if task_name == "plot" else "Methodology Section"
        visual_label = "Visual Intent" if task_name == "plot" else "Figure Caption"
        target_label = "Target Plot" if task_name == "plot" else "Target Diagram"
        return (
            f"{target_label}: {render_output}\n"
            f"Detailed Description: {detailed_description}\n"
            f"{context_label}: {content}\n"
            f"{visual_label}: {visual_intent}\n"
            "Your Output:"
        )

    def _normalize_output(self, raw_response: str, request: Optional[str]) -> str:
        parsed = parse_json_or_text(raw_response)
        suggestions = str(parsed.get("critic_suggestions", "")).strip()
        revised = str(parsed.get("revised_description", "")).strip()
        fallback_text = str(parsed.get("text", "")).strip()

        if not suggestions and "No changes needed." in (raw_response or ""):
            suggestions = "No changes needed."
        if not revised:
            payload = parse_upstream_payload(request)
            if not payload:
                payload = parse_json_or_text(request)
            revised = str(payload.get("detailed_description", payload.get("description", ""))).strip()
        normalized_suggestions = suggestions.strip().strip('`"').strip()
        if not normalized_suggestions and fallback_text:
            normalized_suggestions = fallback_text.strip().strip('`"').strip()
        decision = (
            "finish"
            if normalized_suggestions.lower().startswith("no changes needed")
            else "continue"
        )
        return json.dumps(
            {
                "decision": decision,
                "critic_suggestions": normalized_suggestions,
                "revised_description": revised or "No changes needed.",
            },
            ensure_ascii=False,
            indent=2,
        )

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        root_text, _ = split_root_and_upstream(request)
        payload = parse_upstream_payload(request)
        if not payload:
            payload = parse_json_or_text(request)
        payload["root_text"] = root_text
        prompt = self._build_prompt(payload)
        self.update_memory("user", prompt, base64_image=payload.get("image_base64"))

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            raw = await self.step()
            result = self._normalize_output(raw, request)
            self.update_memory("assistant", result)
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result
