from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.logger import logger
from app.prompt.image_generator import (
    IMAGE_GENERATION_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState
from app.tracer import tracer
from app.utils.image_generation import generate_image_bytes


class ImageGeneratorAgent(BaseAgent):
    """General text-to-image agent.

    The node owns image generation directly so PlanAgent can assign an image
    model to it for any task that needs a plain text-to-image capability.
    """

    name: str = "image_generator"
    description: str = "General-purpose text-to-image generation agent."
    system_prompt: str = SYSTEM_PROMPT
    max_steps: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        prompt = IMAGE_GENERATION_USER_PROMPT_TEMPLATE.format(
            request=(request or "").strip(),
        )
        self.update_memory("user", prompt)

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result_payload = await self.step(request or "")
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

    async def step(self, request_text: str) -> Dict[str, Any]:
        logger.info(f"[ImageGeneratorAgent] Running node '{self.name}'")
        image_bytes = await self._try_generate_image(request_text)
        return {
            "mode": "image",
            "request": request_text,
            "image_base64_present": True,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        }

    async def _try_generate_image(self, request_text: str) -> bytes:
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
                prompt=request_text,
                size=size,
                negative_prompt=negative_prompt,
                watermark=watermark,
                prompt_extend=prompt_extend,
                timeout_s=timeout_s,
            )
