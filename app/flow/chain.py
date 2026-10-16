"""ChainFlow: linear A → B → C pipeline.

Execution order follows the route chain. If no routes are provided,
agents are executed in the order they appear in mas_config.agents.
Each agent's output becomes the next agent's input.

If an agent terminates with failure status, the chain short-circuits and
remaining agents are skipped.
"""

import time
from typing import List

from app.flow.base import BaseFlow
from app.logger import logger
from app.tracer import tracer

_FAILURE_MARKER = "The interaction has been completed with status: failure"


class ChainFlow(BaseFlow):
    """Execute agents in a linear sequence; each output feeds the next input."""

    async def execute(self, input_text: str) -> str:
        order = self._resolve_chain_order()
        logger.info(f"[ChainFlow] Execution order: {order}")
        tracer.set_topology("chain")
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        current_input = input_text
        for instance_id in order:
            self._reset_agent(instance_id)
            _agent_start = time.time()
            current_input = await self._run_agent(instance_id, current_input)
            _agent_end = time.time()
            self.last_run_outputs[instance_id] = current_input
            self.last_terminal_node = instance_id
            self.last_terminal_output = current_input
            logger.info(f"[ChainFlow] '{instance_id}' produced {len(current_input)} chars")

            tracer.record_agent_run_phase(
                agent_id=instance_id,
                start_time=_agent_start,
                end_time=_agent_end,
                parallel=False,
            )

            if self._agent_failed(instance_id, current_input):
                logger.warning(
                    f"[ChainFlow] Agent '{instance_id}' terminated with failure. "
                    f"Short-circuiting remaining chain: {order[order.index(instance_id) + 1:]}"
                )
                break

        return current_input

    def _agent_failed(self, instance_id: str, output: str) -> bool:
        """Check whether an agent terminated with an explicit failure status."""
        return _FAILURE_MARKER in output

    def _resolve_chain_order(self) -> List[str]:
        """Derive execution order from routes, or fall back to config declaration order."""
        # Filter out self-loop routes (source == target) which are always invalid
        valid_routes = [r for r in (self.mas_config.routes or []) if r.source != r.target]
        if not valid_routes:
            return [n.instance_id for n in self.mas_config.agents]

        # Build predecessor map to find the chain start (no incoming edges)
        all_targets = {r.target for r in valid_routes}
        all_sources = {r.source for r in valid_routes}
        all_ids = {n.instance_id for n in self.mas_config.agents}

        starts = [n for n in all_ids if n not in all_targets]
        if not starts:
            raise ValueError("[ChainFlow] Cannot find chain start: all nodes have predecessors (cycle?).")

        # Follow the chain from the start node
        adj = {r.source: r.target for r in valid_routes}
        order: List[str] = []
        current = starts[0]
        visited = set()
        while current and current not in visited:
            order.append(current)
            visited.add(current)
            current = adj.get(current)

        return order
