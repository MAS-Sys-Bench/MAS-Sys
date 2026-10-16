"""BaseFlow: shared base for all MAS topology flows.

Responsibilities:
- Accept agents as a single agent, list, or dict.
- Inject mas_context + role_description into each agent's system_prompt at startup.
- Register per-agent memory permissions on the shared BaseMemory.
- Provide message broker helpers (get_successors / get_predecessors).
- Concatenate multiple predecessor outputs before delivering them downstream.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from app import continuum
from app.agent.base import BaseAgent
from app.logger import logger
from app.mas.schema import InputAggregation, MASConfig
from app.memory.base import BaseMemory
from app.memory.global_memory import GlobalMemory
from app.prompt.coordination import (
    ASSIGN_SUBTASKS_SYSTEM,
    ASSIGN_SUBTASKS_USER,
    CONTINUE_SYSTEM,
    CONTINUE_USER,
    SUMMARIZE_RESULTS_SYSTEM,
    SUMMARIZE_RESULTS_USER,
)
from app.prompt.flow import (
    LLM_MERGE_PROMPT_TEMPLATE,
    MAS_SYSTEM_PREFIX_TEMPLATE,
    ONLY_AGENT_DESCRIPTION,
)
from app.schema import AgentState, Message
from app.tracer import tracer


class BaseFlow(BaseModel, ABC):
    """Shared base for all MAS topology flows. Subclasses implement execute()."""

    agents: Dict[str, BaseAgent]
    tools: Optional[List] = None
    primary_agent_key: Optional[str] = None

    mas_config: MASConfig
    public_memory: BaseMemory = Field(default_factory=GlobalMemory)
    last_run_outputs: Dict[str, str] = Field(default_factory=dict)
    last_terminal_node: Optional[str] = None
    last_terminal_output: Optional[str] = None
    last_run_rounds: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Handle different ways of providing agents
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        # If primary agent not specified, use first agent
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        # Set the agents dictionary
        data["agents"] = agents_dict

        # Initialize using BaseModel's init
        super().__init__(**data)

        # MAS-specific initialization
        self._register_memory_permissions()
        self._enrich_agent_system_prompts()

    # ── Agent accessors ───────────────────────────────────────────────

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """Get the primary agent for the flow."""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """Get a specific agent by key."""
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """Add a new agent to the flow."""
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """Execute the flow with given input."""

    # Memory permissions 

    def _register_memory_permissions(self) -> None:
        """Register each agent's read/write permissions on the shared BaseMemory."""
        for node in self.mas_config.agents:
            self.public_memory.register_agent(
                instance_id=node.instance_id,
                can_read=node.memory_permission.can_read,
                can_write=node.memory_permission.can_write,
            )

    # System prompt enrichment

    def _build_peers_description(self, exclude_id: str) -> str:
        """Summarise all other agents for injection into one agent's system prompt."""
        peers = [
            f"- {n.instance_id}: {n.role_description or '(no description)'}"
            for n in self.mas_config.agents
            if n.instance_id != exclude_id
        ]
        return "\n".join(peers) if peers else ONLY_AGENT_DESCRIPTION

    def _enrich_agent_system_prompts(self) -> None:
        """Prepend MAS context + role info to every agent's system_prompt."""
        for node in self.mas_config.agents:
            agent = self.agents.get(node.instance_id)
            if not agent:
                logger.warning(
                    f"[BaseFlow] Agent '{node.instance_id}' not found in agents dict, "
                    f"skipping prompt enrichment."
                )
                continue

            routes_out = [r.target for r in self.mas_config.routes if r.source == node.instance_id]
            routes_in  = [r.source for r in self.mas_config.routes if r.target == node.instance_id]
            mem_perm   = (
                f"read={'yes' if node.memory_permission.can_read else 'no'}, "
                f"write={'yes' if node.memory_permission.can_write else 'no'}"
            )

            prefix = MAS_SYSTEM_PREFIX_TEMPLATE.format(
                mas_context=self.mas_config.mas_context,
                role_description=node.role_description,
                peers_description=self._build_peers_description(node.instance_id),
                routes_in=routes_in or "none",
                routes_out=routes_out or "none",
                memory_permission=mem_perm,
            )
            base = agent.system_prompt or ""
            agent.system_prompt = (prefix + ("\n\n" + base if base else "")).strip()

    # Routing helpers

    def get_successors(self, instance_id: str) -> List[str]:
        """Return instance_ids that should receive output from instance_id."""
        return [r.target for r in self.mas_config.routes if r.source == instance_id]

    def get_predecessors(self, instance_id: str) -> List[str]:
        """Return instance_ids whose output feeds into instance_id."""
        return [r.source for r in self.mas_config.routes if r.target == instance_id]

    # Input aggregation

    async def _aggregate_inputs(self, instance_id: str, inputs: List[str]) -> str:
        """Merge multiple predecessor outputs according to the node's strategy."""
        if not inputs:
            return ""
        if len(inputs) == 1:
            return inputs[0]

        node = next(
            (n for n in self.mas_config.agents if n.instance_id == instance_id), None
        )
        strategy = node.input_aggregation if node else InputAggregation.CONCAT

        if strategy == InputAggregation.CONCAT:
            return "\n\n".join(inputs)

        if strategy == InputAggregation.VOTE:
            return Counter(inputs).most_common(1)[0][0]

        if strategy == InputAggregation.LLM_MERGE:
            target_agent = self.agents.get(instance_id)
            if target_agent is None:
                raise ValueError(f"[BaseFlow] Cannot LLM-merge inputs for unknown agent '{instance_id}'.")
            numbered = "\n\n".join(
                f"[Input {i + 1}]\n{s}" for i, s in enumerate(inputs)
            )
            prompt = LLM_MERGE_PROMPT_TEMPLATE.format(numbered_inputs=numbered)
            start_time = time.time()
            with tracer.llm_call_scope(
                runtime="legacy",
                scope="coordination",
                topology=str(getattr(self.mas_config.topology, "value", self.mas_config.topology)),
                coordinator_id=instance_id,
                call_type=f"{instance_id}.llm_merge",
                llm_config_name=getattr(target_agent, "llm_config_name", ""),
                **self._continuum_coordination_context(
                    coordinator_id=instance_id,
                    call_type=f"{instance_id}.llm_merge",
                    is_last_step=True,
                ),
            ):
                merged = await target_agent.llm.ask(
                    [Message.user_message(prompt)], stream=False
                )
            end_time = time.time()
            tracer.record_coordination_call(
                coordinator_id=instance_id,
                call_type=f"{instance_id}.llm_merge",
                input_text=prompt,
                output_text=merged,
                start_time=start_time,
                end_time=end_time,
                input_tokens=getattr(target_agent.llm, "last_input_tokens", 0),
                completion_tokens=getattr(target_agent.llm, "last_completion_tokens", 0),
            )
            tracer.record_coordination_phase(
                coordinator_id=instance_id,
                call_type=f"{instance_id}.llm_merge",
                start_time=start_time,
                end_time=end_time,
            )
            return merged

        raise ValueError(f"[BaseFlow] Unsupported input aggregation strategy: {strategy}")

    # Agent execution helpers

    async def _run_agent(self, instance_id: str, input_text: str) -> str:
        """Run a single agent by instance_id and return its string output."""
        agent = self.agents.get(instance_id)
        if not agent:
            raise ValueError(f"Agent '{instance_id}' not found in flow.")
        logger.info(f"[BaseFlow] Running agent '{instance_id}'")
        self._prepare_agent_continuum_job(instance_id, agent)
        _agent_start = time.time()
        result = await agent.run(input_text)
        _agent_end = time.time()
        tracer.add_agent_duration(instance_id, _agent_end - _agent_start)
        return result or ""

    def _prepare_agent_continuum_job(
        self,
        instance_id: str,
        agent: BaseAgent,
    ) -> None:
        topology = str(
            getattr(self.mas_config.topology, "value", self.mas_config.topology)
        )
        agent.continuum_job_id = continuum.job_id(
            tracer.current_session_id(),
            topology,
            instance_id,
        )
        agent.continuum_llm_call_index = 0

    def _continuum_coordination_context(
        self,
        *,
        coordinator_id: str,
        call_type: str,
        is_last_step: bool = True,
    ) -> dict:
        topology = str(
            getattr(self.mas_config.topology, "value", self.mas_config.topology)
        )
        return {
            continuum.TRACE_JOB_ID: continuum.job_id(
                tracer.current_session_id(),
                topology,
                coordinator_id,
                "coord",
                call_type,
            ),
            continuum.TRACE_CALL_INDEX: 1,
            continuum.TRACE_IS_LAST_STEP: is_last_step,
        }

    async def _run_agents_parallel(
        self,
        id_input_pairs: List[Tuple[str, str]],
    ) -> Dict[str, str]:
        """Run multiple agents concurrently and return {instance_id: output}.

        Each agent is reset before execution. If any agent raises an exception,
        the exception is propagated to the caller.

        Args:
            id_input_pairs: List of (instance_id, input_text) tuples.

        Returns:
            Dict mapping each instance_id to its output string.
        """
        if not id_input_pairs:
            return {}

        # Reset all agents before launching
        for instance_id, _ in id_input_pairs:
            self._reset_agent(instance_id)

        async def _timed_run(
            instance_id: str, input_text: str
        ) -> Tuple[str, str, float, float]:
            _start = time.time()
            output = await self._run_agent(instance_id, input_text)
            _end = time.time()
            return instance_id, output, _start, _end

        _batch_start = time.time()
        tasks = [
            _timed_run(instance_id, input_text)
            for instance_id, input_text in id_input_pairs
        ]
        results = await asyncio.gather(*tasks)
        _batch_end = time.time()

        outputs = {instance_id: output for instance_id, output, _, _ in results}
        individual_times = {
            instance_id: {"start_time": s, "end_time": e}
            for instance_id, _, s, e in results
        }

        tracer.record_parallel_batch_phase(
            agent_ids=[iid for iid, _ in id_input_pairs],
            start_time=_batch_start,
            end_time=_batch_end,
            individual_times=individual_times,
        )

        logger.info(
            f"[BaseFlow] Parallel batch completed: "
            f"{[f'{iid}({len(out)} chars)' for iid, out in outputs.items()]}"
        )
        return outputs

    # Runtime coordination helpers

    @staticmethod
    def _format_progress_history(progress_history: List[str]) -> str:
        if not progress_history:
            return "No progress yet."
        return "\n".join(
            f"Round {index + 1}: {summary}"
            for index, summary in enumerate(progress_history)
        )

    async def _ask_coordinator(
        self,
        *,
        coordinator_id: str,
        system_prompt: str,
        user_prompt: str,
        call_type: str,
    ) -> str:
        coordinator = self.agents.get(coordinator_id)
        if coordinator is None:
            raise ValueError(f"[BaseFlow] Unknown coordinator agent '{coordinator_id}'.")

        messages = [Message.user_message(user_prompt)]
        system_msgs = []
        if coordinator.system_prompt:
            system_msgs.append(Message.system_message(coordinator.system_prompt))
        system_msgs.append(Message.system_message(system_prompt))

        start_time = time.time()
        with tracer.llm_call_scope(
            runtime="legacy",
            scope="coordination",
            topology=str(getattr(self.mas_config.topology, "value", self.mas_config.topology)),
            coordinator_id=coordinator_id,
            call_type=call_type,
            llm_config_name=getattr(coordinator, "llm_config_name", ""),
            **self._continuum_coordination_context(
                coordinator_id=coordinator_id,
                call_type=call_type,
                is_last_step=True,
            ),
        ):
            response = await coordinator.llm.ask(
                messages=messages,
                system_msgs=system_msgs,
                stream=False,
            )
        end_time = time.time()

        tracer.record_coordination_call(
            coordinator_id=coordinator_id,
            call_type=call_type,
            input_text=user_prompt,
            output_text=response,
            start_time=start_time,
            end_time=end_time,
            input_tokens=getattr(coordinator.llm, "last_input_tokens", 0),
            completion_tokens=getattr(coordinator.llm, "last_completion_tokens", 0),
        )
        tracer.record_coordination_phase(
            coordinator_id=coordinator_id,
            call_type=call_type,
            start_time=start_time,
            end_time=end_time,
        )
        return response

    async def _assign_subtasks_with_agent(
        self,
        *,
        coordinator_id: str,
        task: str,
        agent_profiles: Dict[str, str],
        progress_history: List[str],
    ) -> Dict[str, str]:
        profiles_text = "\n".join(
            f"  - {agent_id}: {profile}" for agent_id, profile in agent_profiles.items()
        )
        raw = await self._ask_coordinator(
            coordinator_id=coordinator_id,
            system_prompt=ASSIGN_SUBTASKS_SYSTEM,
            user_prompt=ASSIGN_SUBTASKS_USER.format(
                task=task,
                agent_profiles=profiles_text,
                progress=self._format_progress_history(progress_history),
            ),
            call_type=f"{coordinator_id}.assign_subtasks",
        )
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        mapping = json.loads(text)
        for agent_id in agent_profiles:
            if agent_id not in mapping:
                raise ValueError(
                    f"[BaseFlow] Coordinator '{coordinator_id}' missed subtask for '{agent_id}'."
                )
        return {str(k): str(v) for k, v in mapping.items()}

    async def _summarize_results_with_agent(
        self,
        *,
        coordinator_id: str,
        task: str,
        results: Dict[str, str],
    ) -> str:
        results_text = "\n".join(
            f"  [{agent_id}]: {output[:800]}" for agent_id, output in results.items()
        )
        return await self._ask_coordinator(
            coordinator_id=coordinator_id,
            system_prompt=SUMMARIZE_RESULTS_SYSTEM,
            user_prompt=SUMMARIZE_RESULTS_USER.format(task=task, results=results_text),
            call_type=f"{coordinator_id}.summarize_results",
        )

    async def _should_continue_with_agent(
        self,
        *,
        coordinator_id: str,
        task: str,
        summary: str,
        round_num: int,
        max_rounds: int,
    ) -> bool:
        if round_num >= max_rounds:
            return False
        raw = await self._ask_coordinator(
            coordinator_id=coordinator_id,
            system_prompt=CONTINUE_SYSTEM,
            user_prompt=CONTINUE_USER.format(
                task=task,
                summary=summary,
                round_num=round_num,
                max_rounds=max_rounds,
            ),
            call_type=f"{coordinator_id}.should_continue",
        )
        decision = (raw or "").strip().upper()
        return decision != "CONVERGED"

    def _reset_agent(self, instance_id: str) -> None:
        """Reset an agent to IDLE state and clear its message history.

        Handles both FINISHED and ERROR states so that transient failures in one
        flow step do not permanently brick the agent for subsequent steps.

        Clearing memory.messages is critical for multi-round flows (Star/Tree):
        LocalMemory uses a sliding window (max_messages=100).  Without clearing,
        messages accumulate across rounds and the window eventually truncates an
        ``assistant`` message that contains ``tool_calls`` while keeping the
        corresponding ``role:tool`` response — producing the OpenAI 400 error
        "messages with role 'tool' must be a response to a preceding message
        with 'tool_calls'".
        """
        agent = self.agents.get(instance_id)
        if agent and agent.state in (AgentState.FINISHED, AgentState.ERROR):
            agent.state = AgentState.IDLE
            agent.current_step = 0
            agent.continuum_llm_call_index = 0
            agent.memory.messages.clear()  # prevent cross-round tool-message orphans
