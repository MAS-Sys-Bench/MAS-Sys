"""TreeFlow: hierarchical tree topology with parent-node runtime coordination.

Execution model (aligned with MARBLE tree_coordinate)
-----------------------------------------------------
Each iteration:
  The tree is traversed recursively starting from the root.  For every
  **non-leaf** node the following three sub-steps occur (MARBLE pattern):

    1. **Plan** – The parent node uses its own LLM to decompose the current
       task into distinct subtasks, one per direct child.
    2. **Execute** – All children execute their subtasks *in parallel*
       (asyncio.gather), with each child recursively applying the same
       Plan-Execute-Synthesize logic if it has its own children.
    3. **Synthesize** – The parent collects all children's outputs, appends
       them to its own task context, and runs itself (act) to produce a
       synthesised result.

  **Leaf nodes** execute directly with no decomposition step.

After each full-tree traversal the root node summarises the tree output and
always performs an LLM-based convergence check.
The loop repeats up to ``max_rounds`` iterations.

topology_params required: {"root": "<instance_id>"}
Routes define parent → child edges.
"""

import asyncio
import time
from typing import Dict, List

from app.flow.base import BaseFlow
from app.logger import logger
from app.mas.topology import build_adjacency_list
from app.tracer import tracer


class TreeFlow(BaseFlow):
    """Recursive Plan-Execute-Synthesize tree with parent-node coordination."""

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute(self, input_text: str) -> str:  # noqa: C901
        root_id = self.mas_config.topology_params.get("root")
        if not root_id:
            raise ValueError("[TreeFlow] topology_params must contain 'root'.")

        max_rounds = self.mas_config.max_rounds
        use_llm_assign = True
        use_llm_check = True

        # Build parent→children adjacency map (static for the lifetime of this run)
        adj = build_adjacency_list(self.mas_config.routes)

        logger.info(
            f"[TreeFlow] root='{root_id}', max_rounds={max_rounds}, "
            f"llm_assign={use_llm_assign}, llm_check={use_llm_check}"
        )
        tracer.set_topology("tree")
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        root_output = ""
        progress_history: List[str] = []
        # task accumulates progress context across rounds (same pattern as StarFlow)
        task = input_text

        for round_num in range(1, max_rounds + 1):
            logger.info(f"[TreeFlow] === Round {round_num}/{max_rounds} ===")

            # Run the full tree recursively with the (possibly enriched) task
            root_output = await self._execute_recursive(
                node_id=root_id,
                task=task,
                adj=adj,
                use_llm_assign=use_llm_assign,
                progress_history=progress_history,
            )
            logger.info(
                f"[TreeFlow] Round {round_num} root output: {len(root_output)} chars"
            )

            # ── Summarise and check convergence ───────────────────────────────
            summary = await self._summarize_results_with_agent(
                coordinator_id=root_id,
                task=input_text,
                results={root_id: root_output},
            )
            progress_history.append(summary)
            self.last_run_rounds = round_num
            logger.info(f"[TreeFlow] Round {round_num} summary: {summary[:200]}")

            if use_llm_check:
                keep_going = await self._should_continue_with_agent(
                    coordinator_id=root_id,
                    task=input_text,
                    summary=summary,
                    round_num=round_num,
                    max_rounds=max_rounds,
                )
                if not keep_going:
                    logger.info(
                        f"[TreeFlow] LLM convergence: stopping after round {round_num}."
                    )
                    break
            # Enrich task with progress for next round
            task = f"{input_text}\n\n[Progress after round {round_num}]\n{summary}"

        self.last_terminal_node = root_id
        self.last_terminal_output = root_output
        self.last_run_outputs[root_id] = root_output
        return root_output

    # ── Recursive Plan-Execute-Synthesize ─────────────────────────────────────

    async def _execute_recursive(
        self,
        node_id: str,
        task: str,
        adj: Dict[str, List[str]],
        use_llm_assign: bool,
        progress_history: List[str],
    ) -> str:
        """Recursively execute a node using Plan-Execute-Synthesize.

        Args:
            node_id: Instance ID of the current node.
            task: Task text assigned to this node.
            adj: Parent→children adjacency list for the whole tree.
            use_llm_assign: Whether to use LLM to assign per-child subtasks.
            progress_history: Accumulated round summaries from previous rounds.

        Returns:
            The synthesised output string for this node.
        """
        children_ids: List[str] = adj.get(node_id, [])

        # ── Leaf node: execute directly ───────────────────────────────────────
        if not children_ids:
            logger.info(f"[TreeFlow] Leaf '{node_id}' executing directly.")
            self._reset_agent(node_id)
            _leaf_start = time.time()
            result = await self._run_agent(node_id, task)
            _leaf_end = time.time()
            self.last_run_outputs[node_id] = result
            tracer.record_agent_run_phase(
                agent_id=node_id,
                start_time=_leaf_start,
                end_time=_leaf_end,
                parallel=False,
            )
            return result

        # ── Non-leaf: Plan → Execute children → Synthesize ───────────────────

        # 1. PLAN: decompose task into per-child subtasks
        child_profiles: Dict[str, str] = {
            cid: next(
                (n.role_description for n in self.mas_config.agents if n.instance_id == cid),
                "",
            )
            for cid in children_ids
        }

        if use_llm_assign:
            child_subtasks = await self._assign_subtasks_with_agent(
                coordinator_id=node_id,
                task=task,
                agent_profiles=child_profiles,
                progress_history=progress_history,
            )
        else:
            # Fallback: every child receives the same parent task
            child_subtasks = {cid: task for cid in children_ids}

        logger.info(
            f"[TreeFlow] Non-leaf '{node_id}' dispatching to children: {children_ids}"
        )

        # 2. EXECUTE children in parallel (MARBLE-style asyncio.gather)
        _children_start = time.time()
        child_coros = [
            self._execute_recursive(
                node_id=cid,
                task=child_subtasks.get(cid, task),
                adj=adj,
                use_llm_assign=use_llm_assign,
                progress_history=progress_history,
            )
            for cid in children_ids
        ]
        child_outputs: List[str] = list(await asyncio.gather(*child_coros))
        _children_end = time.time()

        if len(children_ids) > 1:
            tracer.record_parallel_batch_phase(
                agent_ids=children_ids,
                start_time=_children_start,
                end_time=_children_end,
            )

        # Format children results for synthesis context
        children_results_text = "\n\n".join(
            f"[{cid}]:\n{out}" for cid, out in zip(children_ids, child_outputs)
        )

        # 3. SYNTHESIZE: parent acts on original task + children results
        synthesis_input = (
            f"{task}\n\n"
            f"[Children Results]\n{children_results_text}\n\n"
            "Based on your subtask and the children's results above, "
            "synthesise a final response. Continue working on the original task "
            "without repeating what children already covered."
        )

        logger.info(f"[TreeFlow] Non-leaf '{node_id}' synthesising results.")
        self._reset_agent(node_id)
        _synth_start = time.time()
        result = await self._run_agent(node_id, synthesis_input)
        _synth_end = time.time()
        self.last_run_outputs[node_id] = result
        tracer.record_agent_run_phase(
            agent_id=node_id,
            start_time=_synth_start,
            end_time=_synth_end,
            parallel=False,
        )
        return result

    # _reset_agent is inherited from BaseFlow (clears memory + resets state)
