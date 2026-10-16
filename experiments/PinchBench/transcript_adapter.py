"""PinchBench transcript adapter.

Converts MAS-Sys agent execution records (Message objects) into the
transcript format expected by PinchBench grading functions.

PinchBench graders expect a list of dicts in this shape::

    [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "..."},
                    {"type": "toolCall", "name": "tool_name", "params": {...}},
                ]
            }
        },
        ...
    ]

MAS-Sys uses ``Message`` objects with:
- role: "user" | "assistant" | "tool" | "system"
- content: Optional[str]
- tool_calls: Optional[List[ToolCall]]  (ToolCall has .function.name, .function.arguments)
"""

import json
from typing import Any, Dict, List

from app.schema import Message


def convert_messages_to_transcript(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert MAS-Sys Message list to PinchBench transcript format.

    Args:
        messages: List of MAS-Sys Message objects from agent memory.

    Returns:
        List of transcript entries compatible with PinchBench grade() functions.
    """
    transcript = []

    for msg in messages:
        content_items: List[Dict[str, Any]] = []

        # Add text content
        if msg.content:
            content_items.append({"type": "text", "text": msg.content})

        # Add tool calls (assistant messages)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    params = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    params = {"raw": tc.function.arguments}
                content_items.append(
                    {
                        "type": "toolCall",
                        "name": tc.function.name,
                        "params": params,
                    }
                )

        # Map tool role to toolResult type
        if msg.role == "tool":
            entry = {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": content_items,
                    "name": msg.name or "",
                    "tool_call_id": msg.tool_call_id or "",
                },
            }
        else:
            entry = {
                "type": "message",
                "message": {
                    "role": msg.role,
                    "content": content_items,
                },
            }

        transcript.append(entry)

    return transcript


def collect_all_messages(flow: Any) -> List[Message]:
    """Collect messages from all agents in a flow after execution.

    Walks the flow's agent registry and concatenates their message histories.

    Args:
        flow: An executed BaseFlow instance.

    Returns:
        Combined list of Message objects from all agents, in temporal order.
    """
    all_messages: List[Message] = []

    agents = getattr(flow, "agents", {})
    if isinstance(agents, dict):
        agent_list = list(agents.values())
    elif isinstance(agents, list):
        agent_list = agents
    else:
        return all_messages

    for agent in agent_list:
        memory = getattr(agent, "memory", None)
        if memory is None:
            continue
        messages = getattr(memory, "messages", [])
        all_messages.extend(messages)

    return all_messages
