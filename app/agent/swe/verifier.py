import json
import re
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.schema import AgentState, Message
from app.prompt.swe.verifier import SYSTEM_PROMPT
from app.tool import Bash, ToolCollection


class SWEVerifierAgent(ToolCallAgent):
    """Code-repair verifier agent for generic FSM-based repair loops."""

    name: str = "swe_verifier"
    description: str = "Runs validation commands and returns repair feedback without editing files."
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(Bash())
    )
    special_tool_names: List[str] = Field(default_factory=list)
    strict_tool_message_order: bool = True
    max_steps: int = 8
    verifier_response_format: ClassVar[Dict[str, Any]] = {
        "type": "json_schema",
        "json_schema": {
            "name": "swe_verifier_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["pass", "retry", "fail"]},
                    "summary": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "suggested_next_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "decision": {"type": "string", "enum": ["finish", "continue"]},
                },
                "required": [
                    "status",
                    "summary",
                    "evidence",
                    "suggested_next_actions",
                    "decision",
                ],
            },
        },
    }

    @staticmethod
    def _extract_json_dict(text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = text.strip()
        if not candidate:
            return None

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def _extract_json_object(cls, text: Optional[str]) -> Optional[str]:
        parsed = cls._extract_json_dict(text)
        if not parsed or "decision" not in parsed:
            return None
        required = {
            "status",
            "summary",
            "evidence",
            "suggested_next_actions",
            "decision",
        }
        missing = required - set(parsed)
        if missing:
            return None
        status = str(parsed.get("status", "")).strip().lower()
        decision = str(parsed.get("decision", "")).strip().lower()
        summary = parsed.get("summary")
        evidence = parsed.get("evidence")
        suggested_next_actions = parsed.get("suggested_next_actions")
        if status not in {"pass", "retry", "fail"}:
            return None
        if decision not in {"finish", "continue"}:
            return None
        if not isinstance(summary, str):
            return None
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) for item in evidence
        ):
            return None
        if not isinstance(suggested_next_actions, list) or not all(
            isinstance(item, str) for item in suggested_next_actions
        ):
            return None
        normalized = {
            "status": status,
            "summary": summary,
            "evidence": evidence,
            "suggested_next_actions": suggested_next_actions,
            "decision": decision,
        }
        return json.dumps(normalized, ensure_ascii=False)

    def _latest_assistant_json(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role != "assistant" or not msg.content:
                continue
            extracted = self._extract_json_object(msg.content)
            if extracted:
                return extracted
        return None

    async def _finalize_without_tools(self) -> str:
        final_prompt = """Stop calling tools and return the verifier verdict now.

Return ONLY this JSON object:
{"status":"pass|retry|fail","summary":"...","evidence":["..."],"suggested_next_actions":["..."],"decision":"finish|continue"}

Rules:
- Use pass/finish only if targeted validation evidence supports the fix.
- Use retry/continue if validation failed, timed out, was inconclusive, or found remaining issues.
- Use fail/finish only if validation cannot be completed meaningfully and no credible next step remains.
- Any failed command, timed-out command, or incomplete test output must be retry/continue.
"""
        response = await self.ask_with_trace(
            messages=self.messages + [Message.user_message(final_prompt)],
            system_msgs=[Message.system_message(self.system_prompt)],
            stream=False,
            response_format=self.verifier_response_format,
        )
        response = response.strip()
        self.memory.add_message(Message.assistant_message(response))
        self.state = AgentState.FINISHED
        extracted = self._extract_json_object(response)
        if extracted:
            return extracted
        raise ValueError(
            "[SWEVerifierAgent] Finalization did not produce a valid verifier JSON object "
            "with keys status, summary, evidence, suggested_next_actions, and decision."
        )

    async def run(self, request: Optional[str] = None) -> str:
        await super().run(request)
        final_json = self._latest_assistant_json()
        if final_json:
            return final_json
        return await self._finalize_without_tools()
