"""DAGFlow: static directed acyclic graph topology.

Execution order is determined by topological generations (layered Kahn's algorithm).
Nodes within the same generation have no mutual dependencies and are executed
in parallel via asyncio.gather. Nodes with multiple predecessors wait for all
predecessors to complete, then aggregate their outputs via the node's
input_aggregation strategy.

Routes must form a valid DAG; a cycle raises ValueError at startup.
"""

import time
from typing import Dict, List

from app.flow.base import BaseFlow
from app.mas.topology import build_predecessors_map, topological_generations
from app.logger import logger
from app.tracer import tracer


class DAGFlow(BaseFlow):
    """Layered-parallel DAG execution with multi-predecessor input aggregation."""

    async def execute(self, input_text: str) -> str:
        node_ids = [n.instance_id for n in self.mas_config.agents]
        generations = topological_generations(node_ids, self.mas_config.routes)
        preds_map = build_predecessors_map(self.mas_config.routes)
        logger.info(
            f"[DAGFlow] {len(generations)} generation(s): "
            f"{[layer for layer in generations]}"
        )
        tracer.set_topology("dag")
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        outputs: Dict[str, str] = {}

        for gen_idx, layer in enumerate(generations):
            # Prepare inputs for every node in this layer
            pairs: List[tuple] = []
            for instance_id in layer:
                preds = preds_map.get(instance_id, [])
                if not preds:
                    node_input = input_text
                else:
                    pred_outputs = [outputs[p] for p in preds if p in outputs]
                    node_input = await self._aggregate_inputs(instance_id, pred_outputs)
                pairs.append((instance_id, node_input))

            # Execute all nodes in the same generation concurrently
            if len(pairs) == 1:
                # Single node: run directly (avoids gather overhead)
                iid, inp = pairs[0]
                self._reset_agent(iid)
                _single_start = time.time()
                outputs[iid] = await self._run_agent(iid, inp)
                _single_end = time.time()
                logger.info(f"[DAGFlow] gen {gen_idx}: '{iid}' produced {len(outputs[iid])} chars")
                tracer.record_agent_run_phase(
                    agent_id=iid,
                    start_time=_single_start,
                    end_time=_single_end,
                    parallel=False,
                )
            else:
                layer_outputs = await self._run_agents_parallel(pairs)
                outputs.update(layer_outputs)
                logger.info(f"[DAGFlow] gen {gen_idx}: parallel batch of {len(pairs)} agents done")

        # Return outputs of all sink nodes (no outgoing routes), aggregated
        flat_order = [n for layer in generations for n in layer]
        all_sources = {r.source for r in self.mas_config.routes}
        sinks = [n for n in flat_order if n not in all_sources]
        sink_outputs = [outputs[n] for n in sinks if n in outputs]
        self.last_run_outputs = dict(outputs)

        if not sink_outputs:
            terminal_id = flat_order[-1] if flat_order else None
            result = outputs.get(terminal_id, "") if terminal_id else ""
            self.last_terminal_node = terminal_id
            self.last_terminal_output = result
            return result
        if len(sink_outputs) == 1:
            self.last_terminal_node = sinks[0]
            self.last_terminal_output = sink_outputs[0]
            return sink_outputs[0]
        result = await self._aggregate_inputs(sinks[0], sink_outputs)
        self.last_terminal_node = sinks[0]
        self.last_terminal_output = result
        return result
