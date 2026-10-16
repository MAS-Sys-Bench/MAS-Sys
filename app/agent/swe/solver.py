from typing import List

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.swe.solver import SYSTEM_PROMPT
from app.tool import Bash, StrReplaceEditor, ToolCollection


class SWESolverAgent(ToolCallAgent):
    """SWE repair agent for generic FSM-based repair loops."""

    name: str = "swe_solver"
    description: str = "Focused code-repair agent for patch generation inside bounded SWE loops."
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(Bash(), StrReplaceEditor())
    )
    special_tool_names: List[str] = Field(default_factory=list)
    strict_tool_message_order: bool = True
    max_steps: int = 24
