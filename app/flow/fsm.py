"""FSMFlow: explicit finite-state execution graph with configurable decision nodes.

Unlike DAGFlow, FSMFlow supports cycles. The runtime itself stays generic:
- it executes the current node,
- aggregates predecessor outputs when available,
- and consults MASConfig.fsm.decision_nodes only for nodes that are allowed to
  branch or terminate.

Domain logic stays outside the flow. A critic/verifier-style agent decides
whether to continue or finish by emitting either:
- raw text such as "continue" / "finish", or
- JSON containing a configurable decision field, e.g. {"decision": "continue"}.
"""

import json
import time
from typing import Any, Dict, Optional

from app.flow.base import BaseFlow
from app.logger import logger
from app.mas.schema import FSMDecisionNode
from app.tracer import tracer
from pydantic import Field


class FSMFlow(BaseFlow):
    """Generic finite-state execution flow with explicit decision-node routing."""

    last_run_outputs: Dict[str, str] = Field(default_factory=dict)
    last_terminal_node: Optional[str] = None
    last_terminal_output: Optional[str] = None
    last_run_rounds: int = 0

    async def execute(self, input_text: str) -> str:
        if not self.mas_config.fsm:
            raise ValueError("[FSMFlow] mas_config.fsm is required when topology='fsm'.")

        fsm = self.mas_config.fsm
        node_ids = {n.instance_id for n in self.mas_config.agents}
        if fsm.start not in node_ids:
            raise ValueError(f"[FSMFlow] Unknown start node '{fsm.start}'.")

        decision_nodes = {cfg.node: cfg for cfg in fsm.decision_nodes}
        tracer.set_topology("fsm")

        outputs: Dict[str, str] = {}
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0
        current_id = fsm.start
        rounds = 0
        decision_rounds = 0

        while True:
            node_input = await self._build_node_input(current_id, input_text, outputs)
            self._reset_agent(current_id)

            _start = time.time()
            output = await self._run_agent(current_id, node_input)
            _end = time.time()

            tracer.record_agent_run_phase(
                agent_id=current_id,
                start_time=_start,
                end_time=_end,
                parallel=False,
            )

            outputs[current_id] = output
            self.last_run_outputs = dict(outputs)
            self.last_terminal_node = current_id
            self.last_terminal_output = output
            self.last_run_rounds = rounds + 1
            rounds += 1
            logger.info(f"[FSMFlow] Round {rounds}: '{current_id}' produced {len(output)} chars")

            decision_cfg = decision_nodes.get(current_id)
            if decision_cfg is not None:
                decision_rounds += 1
                next_id = self._resolve_decision_next(output, decision_cfg)
                if next_id is None:
                    logger.info(f"[FSMFlow] Decision node '{current_id}' chose termination.")
                    self.last_run_outputs = dict(outputs)
                    self.last_terminal_node = current_id
                    self.last_terminal_output = output
                    self.last_run_rounds = rounds
                    return output
                if decision_rounds >= self.mas_config.max_rounds:
                    logger.info(
                        f"[FSMFlow] max_rounds={self.mas_config.max_rounds} decision rounds reached; "
                        f"terminating after node '{current_id}' emitted '{self._extract_decision_value(output, decision_cfg.decision_field)}'."
                    )
                    self.last_run_outputs = dict(outputs)
                    self.last_terminal_node = current_id
                    self.last_terminal_output = output
                    self.last_run_rounds = rounds
                    return output
                current_id = next_id
                continue

            successors = self.get_successors(current_id)
            if not successors:
                self.last_run_outputs = dict(outputs)
                self.last_terminal_node = current_id
                self.last_terminal_output = output
                self.last_run_rounds = rounds
                return output
            if len(successors) > 1:
                raise ValueError(
                    f"[FSMFlow] Non-decision node '{current_id}' has multiple successors: {successors}. "
                    "Use mas_config.fsm.decision_nodes to make branching explicit."
                )
            current_id = successors[0]

    async def _build_node_input(
        self,
        instance_id: str,
        root_input: str,
        outputs: Dict[str, str],
    ) -> str:
        preds = self.get_predecessors(instance_id)
        if not preds:
            return root_input

        pred_outputs = [outputs[p] for p in preds if p in outputs]
        if not pred_outputs:
            return root_input
        aggregated = await self._aggregate_inputs(instance_id, pred_outputs)
        if not root_input.strip():
            return aggregated
        if not aggregated.strip():
            return root_input
        return f"{root_input}\n\n[Upstream Outputs]\n{aggregated}"

    def _resolve_decision_next(
        self,
        output: str,
        decision_cfg: FSMDecisionNode,
    ) -> Optional[str]:
        decision_value = self._extract_decision_value(output, decision_cfg.decision_field)
        normalized = decision_value.strip().lower()

        for transition in decision_cfg.transitions:
            if normalized == transition.value.strip().lower():
                if transition.terminate:
                    return None
                if not transition.target:
                    raise ValueError(
                        f"[FSMFlow] Decision '{transition.value}' for node '{decision_cfg.node}' "
                        "is non-terminating but missing a target."
                    )
                return transition.target

        raise ValueError(
            f"[FSMFlow] Node '{decision_cfg.node}' emitted unmatched decision '{decision_value}' "
            "and no transition matches it."
        )

    def _extract_decision_value(self, output: str, decision_field: str) -> str:
        text = output.strip()
        if not text:
            return ""

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and decision_field in parsed:
                return str(parsed[decision_field])
        except Exception:
            pass

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            candidate = line
            if ": " in candidate and candidate.lower().startswith("step "):
                candidate = candidate.split(": ", 1)[1].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and decision_field in parsed:
                    return str(parsed[decision_field])
            except Exception:
                continue

        if len(lines) == 1:
            return lines[0]
        return text
