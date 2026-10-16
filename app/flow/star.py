"""StarFlow: fan-out/fan-in execution where workers run before the leader.

Execution model
---------------
1. All non-center agents are treated as workers and receive the shared input in parallel.
2. The center agent then receives the aggregated worker outputs and produces the final result.

topology_params required: {"center": "<instance_id>"}
"""

from __future__ import annotations

import time

from app.flow.base import BaseFlow
from app.logger import logger
from app.tracer import tracer


class StarFlow(BaseFlow):
    """Worker-first, leader-second star execution."""

    async def execute(self, input_text: str) -> str:
        center_id = self.mas_config.topology_params.get("center")
        if not center_id:
            raise ValueError("[StarFlow] topology_params must contain 'center'.")

        worker_ids = [
            n.instance_id for n in self.mas_config.agents if n.instance_id != center_id
        ]

        logger.info(f"[StarFlow] center='{center_id}', workers={worker_ids}")
        tracer.set_topology("star")
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        if not worker_ids:
            self._reset_agent(center_id)
            result = await self._run_agent(center_id, input_text)
            self.last_run_outputs[center_id] = result
            self.last_terminal_node = center_id
            self.last_terminal_output = result
            return result

        pairs = [(wid, input_text) for wid in worker_ids]
        worker_outputs_map = await self._run_agents_parallel(pairs)
        self.last_run_outputs.update(worker_outputs_map)

        ordered_outputs = [worker_outputs_map[wid] for wid in worker_ids if worker_outputs_map.get(wid)]
        if ordered_outputs:
            leader_input = f"{input_text}\n\n[Upstream Outputs]\n" + "\n\n".join(ordered_outputs)
        else:
            leader_input = input_text

        self._reset_agent(center_id)
        start = time.time()
        result = await self._run_agent(center_id, leader_input)
        end = time.time()
        tracer.record_agent_run_phase(
            agent_id=center_id,
            start_time=start,
            end_time=end,
            parallel=False,
        )
        self.last_run_outputs[center_id] = result
        self.last_terminal_node = center_id
        self.last_terminal_output = result
        return result
