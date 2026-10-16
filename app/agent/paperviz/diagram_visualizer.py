from __future__ import annotations

import json
import base64
from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.paperviz.common import extract_visualizer_description
from app.logger import logger
from app.prompt.paperviz.visualizer import (
    DIAGRAM_VISUALIZER_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState
from app.tracer import tracer
from app.utils.image_generation import generate_image_bytes


class PaperVizDiagramVisualizerAgent(BaseAgent):
    """Diagram visualizer agent.

    This node owns the image-generation step directly and requires a working
    image backend for successful execution.
    """

    name: str = "paperviz_diagram_visualizer"
    description: str = "Uses an image-capable model to render a scientific diagram from a description."
    system_prompt: str = DIAGRAM_VISUALIZER_SYSTEM_PROMPT
    max_steps: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        detailed_description = extract_visualizer_description(request)
        prompt = self._format_diagram_visualizer_prompt(detailed_description)
        self.update_memory("user", prompt)

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result_payload = await self.step(detailed_description)
            result = json.dumps(result_payload, ensure_ascii=False, indent=2)
            self.update_memory(
                "assistant",
                result,
                base64_image=result_payload.get("image_base64"),
            )
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result

    async def step(self, detailed_description: str) -> Dict[str, Any]:
        logger.info(f"[PaperVizDiagramVisualizerAgent] Running node '{self.name}'")
        image_bytes = await self._try_generate_image(detailed_description)
        return {
            "mode": "diagram",
            "detailed_description": detailed_description,
            "image_base64_present": True,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        }

    async def _try_generate_image(self, detailed_description: str) -> bytes:
        size = str(self.agent_params.get("size", "1024x1024"))
        negative_prompt = self.agent_params.get("negative_prompt")
        watermark = bool(self.agent_params.get("watermark", False))
        prompt_extend = bool(self.agent_params.get("prompt_extend", True))
        timeout_s = float(self.agent_params.get("timeout_s", 180.0))
        with tracer.llm_call_scope(
            runtime="legacy",
            scope="agent",
            agent_id=self.name,
            agent_class=type(self).__name__,
            llm_config_name=self.llm_config_name,
            step=self.current_step,
            call_type=f"{self.name}.image_generate",
        ):
            return await generate_image_bytes(
                llm=self.llm,
                prompt=self._format_diagram_render_prompt(detailed_description),
                size=size,
                negative_prompt=negative_prompt,
                watermark=watermark,
                prompt_extend=prompt_extend,
                timeout_s=timeout_s,
            )

    @staticmethod
    def _format_diagram_visualizer_prompt(detailed_description: str) -> str:
        return (
            "Generate a publication-quality scientific diagram image based on the following detailed description.\n"
            "Do not include figure titles in the image.\n"
            "Return only the final diagram result.\n\n"
            f"Detailed Description:\n{detailed_description}\n"
        )

    @staticmethod
    def _format_diagram_render_prompt(detailed_description: str) -> str:
        return (
            "Create a clean academic scientific diagram. "
            "No title text in the canvas.\n\n"
            f"{detailed_description}"
        )
