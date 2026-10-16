from typing import List

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.swe.general import SYSTEM_PROMPT
from app.tool import Bash, StrReplaceEditor, ToolCollection


class SWEAgent(ToolCallAgent):
    """General code-repair agent for single-node software engineering tasks."""

    name: str = "swe"
    description: str = "General software-repair agent that inspects, edits, and iterates on repository fixes."
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(Bash(), StrReplaceEditor())
    )
    special_tool_names: List[str] = Field(default_factory=list)
    strict_tool_message_order: bool = True
    max_steps: int = 20
