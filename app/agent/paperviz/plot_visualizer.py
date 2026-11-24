from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from pydantic import Field

from app.agent.paperviz.common import extract_visualizer_description
from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.paperviz.visualizer import (
    PLOT_VISUALIZER_SYSTEM_PROMPT,
    TOOL_AWARE_PLOT_VISUALIZER_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message, ToolChoice
from app.tool import ToolCollection
from app.tool.paperviz.execute_plot_code import PaperVizExecutePlotCode
from app.tracer import tracer


class PaperVizPlotVisualizerAgent(ToolCallAgent):
    name: str = "paperviz_plot_visualizer"
    description: str = "Generates and optionally executes plot code from a visual description."
    system_prompt: str = TOOL_AWARE_PLOT_VISUALIZER_SYSTEM_PROMPT
    next_step_prompt: str = ""
    max_steps: int = 4
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PaperVizExecutePlotCode())
    )
    tool_choices: ToolChoice = ToolChoice.AUTO

    def _tools_enabled(self) -> bool:
        return bool(getattr(self, "allowed_tools_config", []))

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        detailed_description = extract_visualizer_description(request)
        prompt = self._format_plot_visualizer_prompt(detailed_description)
        self.update_memory("user", prompt)

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            code = await self._generate_code()
            execution_payload, image_base64 = await self._maybe_execute_code(code=code)
            result = json.dumps(
                {
                    "mode": "plot",
                    "detailed_description": detailed_description,
                    "code": code,
                    "execution": execution_payload,
                    "image_base64_present": bool(image_base64),
                    "image_base64": image_base64,
                },
                ensure_ascii=False,
                indent=2,
            )
            self.update_memory("assistant", result, base64_image=image_base64)
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result

    async def step(self) -> str:
        raise NotImplementedError("PaperVizPlotVisualizerAgent uses run() directly.")

    async def _generate_code(self) -> str:
        logger.info(f"[PaperVizPlotVisualizerAgent] Running node '{self.name}'")
        return await self.ask_with_trace(
            messages=self.messages,
            system_msgs=[Message.system_message(PLOT_VISUALIZER_SYSTEM_PROMPT)],
            stream=False,
        )

    async def _maybe_execute_code(self, *, code: str) -> tuple[Dict[str, Any], Optional[str]]:
        if not self._tools_enabled():
            return {}, None

        instruction = self._format_plot_execution_tool_prompt(code)
        messages = [Message.user_message(instruction)]
        think_start = time.time()
        response = await self.llm.ask_tool(
            messages=messages,
            system_msgs=[Message.system_message(self.system_prompt)],
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )
        think_end = time.time()

        tool_calls = response.tool_calls if response and response.tool_calls else []
        content = response.content if response and response.content else ""
        tracer.record_think(
            self.name,
            input_messages=messages,
            output_content=content,
            tool_calls_info=[
                {"name": tc.function.name, "arguments": tc.function.arguments[:500] if tc.function.arguments else ""}
                for tc in tool_calls
            ],
            start_time=think_start,
            end_time=think_end,
            input_tokens=getattr(self.llm, "last_input_tokens", 0),
            completion_tokens=getattr(self.llm, "last_completion_tokens", 0),
        )

        if not tool_calls:
            return {}, None

        for tool_call in tool_calls:
            if tool_call.function.name != "paperviz_execute_plot_code":
                continue
            args = json.loads(tool_call.function.arguments or "{}")
            if "code" not in args:
                logger.warning("[PaperVizPlotVisualizerAgent] Plot tool call missing required arg: code")
                return {}, None
            tool_start = time.time()
            result = await self.available_tools.execute(name=tool_call.function.name, tool_input=args)
            tool_end = time.time()
            tracer.record_tool_execution(
                self.name,
                tool_name=tool_call.function.name,
                tool_input=args,
                tool_output=str(result)[:3000] if result else "",
                start_time=tool_start,
                end_time=tool_end,
            )
            payload = json.loads(result.output or "{}") if getattr(result, "output", None) else {}
            return payload, getattr(result, "base64_image", None)

        return {}, None

    @staticmethod
    def _format_plot_visualizer_prompt(detailed_description: str) -> str:
        return (
            "Use python matplotlib to generate a statistical plot based on the following detailed description.\n"
            "Only provide the code without any explanations.\n\n"
            f"Detailed Description:\n{detailed_description}\n"
        )

    @staticmethod
    def _format_plot_execution_tool_prompt(code: str) -> str:
        return (
            "If a plot-execution tool is available, call it with the generated code and then stop.\n\n"
            f"Generated code:\n```python\n{code}\n```"
        )
