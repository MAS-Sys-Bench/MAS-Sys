"""PlanAgent: analyses a task prompt and produces a MASConfig via structured LLM output."""

import json
import time
from typing import Dict, Optional

from app.agent.agents_config import build_agents_info_for_prompt
from app.config import config
from app.flow.flow_config import build_topologies_info_for_prompt
from app.llm import LLM
from app.logger import logger
from app.mas.schema import MASConfig
from app.prompt.planning import SYSTEM_PROMPT as _SYSTEM_PROMPT
from app.prompt.planning import USER_PROMPT as _USER_PROMPT
from app.schema import Message
from app.tool.tools_config import build_tools_info_for_prompt
from app.tracer import tracer


class PlanAgent:
    """Single-shot MAS planner: task prompt → MASConfig."""

    def __init__(self, llm: Optional[LLM] = None, runtime: str = "legacy"):
        self.llm = llm or LLM("planner")
        self.runtime = runtime
        self.last_raw_output: Optional[str] = None
        self.last_plan_start_time: Optional[float] = None
        self.last_plan_end_time: Optional[float] = None
        self.last_plan_duration_s: Optional[float] = None
        self.last_input_tokens: int = 0
        self.last_completion_tokens: int = 0

    # LLM candidate info builder

    @staticmethod
    def build_llm_candidates_info(runtime: str = "legacy") -> str:
        """Build a human-readable summary of candidate LLMs from config.

        Scans config.llm for keys prefixed with 'lang.', 'image.', or
        'vision.' and formats their metadata (model, size, capabilities,
        cost) for the planning prompt.
        """
        groups: Dict[str, list] = {"lang": [], "image": [], "vision": []}
        allowed_orla_lang_backends = set()
        if runtime == "orla" and config.orla:
            allowed_orla_lang_backends = set(config.orla.backends.keys())
        for key, settings in config.llm.items():
            for prefix in groups:
                if key.startswith(f"{prefix}."):
                    if (
                        runtime == "orla"
                        and prefix == "lang"
                        and key not in allowed_orla_lang_backends
                    ):
                        continue
                    groups[prefix].append((key, settings))

        lines = []
        for group_name, entries in groups.items():
            if not entries:
                continue
            lines.append(f"{group_name} candidates:")
            for key, s in entries:
                lines.append(
                    f"  - {key}: model={s.model}, size={s.size}, "
                    f"capabilities={s.capabilities}, cost={s.cost}"
                )
        if runtime == "orla" and not groups["lang"]:
            raise ValueError(
                "ORLA runtime requested but no `[orla.backends]` entries match any "
                "`[llm.lang.*]` candidates."
            )
        return "\n".join(lines)

    # Pre-execution planning

    async def plan(self, task: str, runtime: Optional[str] = None) -> MASConfig:
        """Analyse task and return a MASConfig. Does not execute any flow.

        Args:
            task: Natural-language description of the task to accomplish.

        Returns:
            A validated MASConfig instance.

        Raises:
            ValueError: If the LLM output cannot be parsed as valid MASConfig JSON.
        """
        runtime_name = runtime or self.runtime
        system_msg = Message.system_message(
            _SYSTEM_PROMPT.format(
                topologies_info=build_topologies_info_for_prompt(),
                agents_info=build_agents_info_for_prompt(),
                tools_info=build_tools_info_for_prompt(),
                llm_candidates_info=self.build_llm_candidates_info(runtime=runtime_name),
            )
        )
        user_msg = Message.user_message(_USER_PROMPT.format(task=task))

        logger.info(f"[PlanAgent] Planning MAS for task: {task[:120]}...")
        _plan_start = time.time()
        raw = await self.llm.ask(
            messages=[user_msg],
            system_msgs=[system_msg],
            stream=False,
        )
        _plan_end = time.time()
        self.last_raw_output = raw
        self.last_plan_start_time = _plan_start
        self.last_plan_end_time = _plan_end
        self.last_plan_duration_s = round(_plan_end - _plan_start, 4)
        self.last_input_tokens = int(getattr(self.llm, 'last_input_tokens', 0) or 0)
        self.last_completion_tokens = int(getattr(self.llm, 'last_completion_tokens', 0) or 0)
        logger.debug(f"[PlanAgent] Raw LLM output:\n{raw}")

        tracer.record_planning(
            input_text=task,
            output_raw=raw,
            start_time=_plan_start,
            end_time=_plan_end,
            input_tokens=self.last_input_tokens,
            completion_tokens=self.last_completion_tokens,
        )

        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> MASConfig:
        """Extract and validate JSON from raw LLM response."""
        # Strip optional markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"[PlanAgent] LLM output is not valid JSON: {e}\n---\n{raw}") from e

        try:
            config = MASConfig(**data)
        except Exception as e:
            raise ValueError(f"[PlanAgent] JSON does not match MASConfig schema: {e}") from e

        logger.info(
            f"[PlanAgent] Config: topology={config.topology}, "
            f"agents={[n.instance_id for n in config.agents]}"
        )
        return config
