from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, PrivateAttr

from app.config import config
from app.flow.base import BaseFlow
from app.mas.schema import FSMDecisionNode, InputAggregation
from app.mas.topology import (
    build_adjacency_list,
    build_predecessors_map,
    topological_generations,
)
from app.orla.compiler import CompiledNode, CompiledSegment, OrlaCompiler
from app.orla.recipes import NodeRecipe, StageRecipe, build_node_recipe
from app.orla.runtime import (
    OrlaBackendRegistry,
    app_message_to_orla_message,
    orla_message_to_app_message,
    tool_collection_to_orla_tools,
)
from app.prompt.coordination import (
    ASSIGN_SUBTASKS_SYSTEM,
    ASSIGN_SUBTASKS_USER,
    CONTINUE_SYSTEM,
    CONTINUE_USER,
    SUMMARIZE_RESULTS_SYSTEM,
    SUMMARIZE_RESULTS_USER,
)
from app.prompt.flow import LLM_MERGE_PROMPT_TEMPLATE
from app.schema import AgentState, Message
from app.tracer import tracer


_FAILURE_MARKER = "The interaction has been completed with status: failure"

RequestTextResolver = Callable[[dict[str, object]], str]


@dataclass
class RuntimeStageSpec:
    stage_id: str
    logical_node_id: str
    backend_name: str
    stage_name: str
    build_messages: Callable[[dict[str, object]], list[Message]]
    dependencies: list[str] = field(default_factory=list)
    hidden: bool = False
    stage_recipe: Optional[StageRecipe] = None


@dataclass
class CoordinationMarker:
    stage_id: str
    coordinator_id: str
    call_type: str
    build_input_text: Callable[[dict[str, object]], str]


@dataclass
class TreeRoundPlan:
    stage_specs: list[RuntimeStageSpec] = field(default_factory=list)
    node_order: list[str] = field(default_factory=list)
    node_request_resolvers: dict[str, RequestTextResolver] = field(default_factory=dict)
    node_stage_ids: dict[str, list[str]] = field(default_factory=dict)
    stage_name_lookup: dict[str, dict[str, str]] = field(default_factory=dict)
    node_exit_stage_ids: dict[str, str] = field(default_factory=dict)
    coordination_markers: list[CoordinationMarker] = field(default_factory=list)
    summarize_stage_id: str = ""
    continue_stage_id: Optional[str] = None


class OrlaFlow(BaseFlow):
    """ORLA-backed runtime that preserves existing topology semantics."""

    last_run_outputs: Dict[str, str] = Field(default_factory=dict)
    last_terminal_node: Optional[str] = None
    last_terminal_output: Optional[str] = None
    last_run_rounds: int = 0

    _registry: OrlaBackendRegistry | None = PrivateAttr(default=None)
    _compiler: OrlaCompiler = PrivateAttr()
    _node_map: Dict[str, object] = PrivateAttr(default_factory=dict)
    _recipes: Dict[str, NodeRecipe] = PrivateAttr(default_factory=dict)
    _hidden_stage_counter: int = PrivateAttr(default=0)

    def model_post_init(self, __context) -> None:
        self._node_map = {node.instance_id: node for node in self.mas_config.agents}
        self._recipes = {
            node.instance_id: build_node_recipe(node, self.agents[node.instance_id])
            for node in self.mas_config.agents
            if node.instance_id in self.agents
        }
        self._compiler = OrlaCompiler(self.mas_config, self._recipes)
        self._registry = OrlaBackendRegistry()
        required_backends = {
            node.llm_config_name
            for node in self.mas_config.agents
            if self._recipes.get(node.instance_id, NodeRecipe(kind="external")).kind != "external"
        }
        self._registry.ensure_registered(required_backends)

    async def execute(self, input_text: str) -> str:
        topology = getattr(self.mas_config.topology, "value", self.mas_config.topology)
        if topology == "chain":
            return await self._execute_chain(input_text)
        if topology == "dag":
            return await self._execute_dag(input_text)
        if topology == "star":
            return await self._execute_star(input_text)
        if topology == "tree":
            return await self._execute_tree(input_text)
        if topology == "fsm":
            return await self._execute_fsm(input_text)
        raise ValueError(f"[OrlaFlow] Unsupported topology: {topology}")

    async def _aggregate_inputs(self, instance_id: str, inputs: List[str]) -> str:
        if not inputs:
            return ""
        if len(inputs) == 1:
            return inputs[0]

        node = self._node_map.get(instance_id)
        strategy = node.input_aggregation if node else InputAggregation.CONCAT
        recipe = self._recipes.get(instance_id, NodeRecipe(kind="external"))

        if strategy == InputAggregation.CONCAT:
            return "\n\n".join(inputs)
        if strategy == InputAggregation.VOTE:
            return Counter(inputs).most_common(1)[0][0]
        if strategy == InputAggregation.LLM_MERGE:
            if recipe.kind == "external":
                return await super()._aggregate_inputs(instance_id, inputs)
            numbered = "\n\n".join(f"[Input {i + 1}]\n{s}" for i, s in enumerate(inputs))
            prompt = LLM_MERGE_PROMPT_TEMPLATE.format(numbered_inputs=numbered)
            return await self._ask_hidden_stage(
                instance_id=instance_id,
                purpose="llm_merge",
                system_prompts=[],
                user_prompt=prompt,
                include_agent_system_prompt=False,
            )
        raise ValueError(f"[OrlaFlow] Unsupported input aggregation strategy: {strategy}")

    async def _ask_coordinator(
        self,
        *,
        coordinator_id: str,
        system_prompt: str,
        user_prompt: str,
        call_type: str,
    ) -> str:
        start_time = time.time()
        if self._recipes.get(coordinator_id, NodeRecipe(kind="external")).kind == "external":
            response = await super()._ask_coordinator(
                coordinator_id=coordinator_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                call_type=call_type,
            )
            return response

        response = await self._ask_hidden_stage(
            instance_id=coordinator_id,
            purpose=call_type.replace(".", "__"),
            system_prompts=[system_prompt],
            user_prompt=user_prompt,
        )
        end_time = time.time()
        tracer.record_coordination_call(
            coordinator_id=coordinator_id,
            call_type=call_type,
            input_text=user_prompt,
            output_text=response,
            start_time=start_time,
            end_time=end_time,
            input_tokens=0,
            completion_tokens=0,
        )
        tracer.record_coordination_phase(
            coordinator_id=coordinator_id,
            call_type=call_type,
            start_time=start_time,
            end_time=end_time,
        )
        return response

    async def _run_runtime_node(self, instance_id: str, input_text: str) -> str:
        recipe = self._recipes.get(instance_id, NodeRecipe(kind="external"))
        if recipe.kind == "external":
            return await self._run_agent(instance_id, input_text)
        if self._registry is None:
            raise RuntimeError("[OrlaFlow] ORLA registry is unavailable in dry-run mode.")

        agent = self.agents.get(instance_id)
        node = self._node_map.get(instance_id)
        if agent is None or node is None:
            raise ValueError(f"[OrlaFlow] Agent '{instance_id}' not found in flow.")

        backend = self._registry.get_backend(node.llm_config_name)
        llm_settings = config.llm[node.llm_config_name]
        workflow = self._registry.pyorla.Workflow(
            self._registry.client,
            tags={"runtime": "orla", "agent_id": instance_id},
        )

        stage_id_to_name: dict[str, str] = {}
        stage_objects: list[object] = []
        previous_stage_id: Optional[str] = None
        for stage_recipe in recipe.stages:
            stage_id = f"{instance_id}__{stage_recipe.name}"
            stage = self._registry.pyorla.Stage(stage_id, backend)
            stage.id = stage_id
            stage.tags = {
                "runtime": "orla",
                "agent_id": instance_id,
                "logical_node_id": instance_id,
                "stage_name": stage_recipe.name,
                "backend_name": node.llm_config_name,
                "hidden": "false",
                "execution_mode": stage_recipe.execution_mode,
            }
            stage.set_max_tokens(llm_settings.max_tokens)
            stage.set_temperature(llm_settings.temperature)
            if stage_recipe.execution_mode == "agent_loop":
                stage.set_execution_mode(self._registry.pyorla.EXECUTION_MODE_AGENT_LOOP)
                stage.set_max_turns(stage_recipe.max_turns)
                for tool in tool_collection_to_orla_tools(agent.available_tools):
                    stage.add_tool(tool)

            def _build_messages(
                dep_results,
                *,
                current_recipe=stage_recipe,
                id_map=stage_id_to_name,
                request_text=input_text,
            ):
                outputs = {
                    id_map[stage_key]: dep_results[stage_key].response.content
                    for stage_key in dep_results
                    if stage_key in id_map and dep_results[stage_key].response is not None
                }
                return current_recipe.build_messages(request_text, outputs)

            stage_spec = RuntimeStageSpec(
                stage_id=stage_id,
                logical_node_id=instance_id,
                backend_name=node.llm_config_name,
                stage_name=stage_recipe.name,
                build_messages=lambda dep_results, current_builder=_build_messages: current_builder(dep_results),
                dependencies=[previous_stage_id] if previous_stage_id is not None else [],
                hidden=False,
                stage_recipe=stage_recipe,
            )
            stage.set_messages_builder(self._wrap_orla_messages_builder(stage=stage, stage_spec=stage_spec))
            workflow.add_stage(stage)
            stage_objects.append(stage)
            stage_id_to_name[stage_id] = stage_recipe.name
            if previous_stage_id is not None:
                workflow.add_dependency(stage_id, previous_stage_id)
            previous_stage_id = stage_id

        start_time = time.time()
        results = await asyncio.to_thread(workflow.execute)
        end_time = time.time()
        tracer.add_agent_duration(instance_id, end_time - start_time)
        tracer.record_orla_workflow(
            workflow_id=getattr(stage_objects[0], "_workflow_id", "") if stage_objects else "",
            stage_ids=list(stage_id_to_name.keys()),
            backend_names=[node.llm_config_name],
            start_time=start_time,
            end_time=end_time,
            hidden=False,
        )

        last_stage_id = previous_stage_id
        return self._finalize_recipe_node(
            instance_id=instance_id,
            recipe=recipe,
            request_text=input_text,
            results=results,
            recipe_stage_ids=list(stage_id_to_name.keys()),
            stage_name_lookup=stage_id_to_name,
            fallback_stage_id=last_stage_id,
        )

    def _finalize_recipe_node(
        self,
        *,
        instance_id: str,
        recipe: NodeRecipe,
        request_text: str,
        results: dict[str, object],
        recipe_stage_ids: list[str],
        stage_name_lookup: dict[str, str],
        fallback_stage_id: Optional[str],
    ) -> str:
        agent = self.agents[instance_id]
        result_text = self._compute_recipe_result_text(
            instance_id=instance_id,
            recipe=recipe,
            request_text=request_text,
            results=results,
            recipe_stage_ids=recipe_stage_ids,
            stage_name_lookup=stage_name_lookup,
            fallback_stage_id=fallback_stage_id,
        )
        stage_outputs = self._build_recipe_stage_outputs(
            results=results,
            recipe_stage_ids=recipe_stage_ids,
            stage_name_lookup=stage_name_lookup,
        )

        if recipe.kind == "agent_loop" and fallback_stage_id and fallback_stage_id in results:
            agent.memory.messages.clear()
            for message in results[fallback_stage_id].messages:
                agent.memory.add_message(orla_message_to_app_message(message))
        else:
            agent.memory.messages.clear()
            if recipe.persist is not None:
                recipe.persist(agent, request_text, stage_outputs, result_text)

        agent.current_step = max(1, len(recipe.stages))
        agent.state = AgentState.FINISHED
        return result_text

    def _build_recipe_stage_outputs(
        self,
        *,
        results: dict[str, object],
        recipe_stage_ids: list[str],
        stage_name_lookup: dict[str, str],
    ) -> dict[str, str]:
        return {
            stage_name_lookup[stage_id]: (
                results[stage_id].response.content if results[stage_id].response else ""
            )
            for stage_id in recipe_stage_ids
            if stage_id in results and stage_id in stage_name_lookup
        }

    def _compute_recipe_result_text(
        self,
        *,
        instance_id: str,
        recipe: NodeRecipe,
        request_text: str,
        results: dict[str, object],
        recipe_stage_ids: list[str],
        stage_name_lookup: dict[str, str],
        fallback_stage_id: Optional[str],
    ) -> str:
        if recipe.kind == "agent_loop" and fallback_stage_id and fallback_stage_id in results:
            fallback_result = results[fallback_stage_id]
            return (
                fallback_result.response.content
                if fallback_result.response is not None
                else ""
            )

        stage_outputs = self._build_recipe_stage_outputs(
            results=results,
            recipe_stage_ids=recipe_stage_ids,
            stage_name_lookup=stage_name_lookup,
        )
        if recipe.finalize is not None:
            return recipe.finalize(self.agents[instance_id], request_text, stage_outputs)
        return stage_outputs.get(stage_name_lookup.get(fallback_stage_id or "", ""), "")

    async def _ask_hidden_stage(
        self,
        *,
        instance_id: str,
        purpose: str,
        system_prompts: list[str],
        user_prompt: str,
        include_agent_system_prompt: bool = True,
    ) -> str:
        node = self._node_map[instance_id]
        stage_id = self._next_hidden_stage_id(instance_id, purpose)
        stage_spec = RuntimeStageSpec(
            stage_id=stage_id,
            logical_node_id=instance_id,
            backend_name=node.llm_config_name,
            stage_name=purpose,
            hidden=True,
            build_messages=lambda _deps: self._build_hidden_stage_messages(
                instance_id=instance_id,
                system_prompts=system_prompts,
                user_prompt=user_prompt,
                include_agent_system_prompt=include_agent_system_prompt,
            ),
        )
        results, _, _, _ = await self._execute_stage_specs(
            workflow_tags={"runtime": "orla", "agent_id": instance_id, "hidden": purpose},
            stage_specs=[stage_spec],
            record_hidden=True,
        )
        stage_result = results[stage_id]
        return stage_result.response.content if stage_result.response is not None else ""

    def _next_hidden_stage_id(self, instance_id: str, purpose: str) -> str:
        self._hidden_stage_counter += 1
        return f"{instance_id}__coord__{purpose}__{self._hidden_stage_counter}"

    def _build_hidden_stage_messages(
        self,
        *,
        instance_id: str,
        system_prompts: list[str],
        user_prompt: str,
        include_agent_system_prompt: bool = True,
    ) -> list[Message]:
        agent = self.agents[instance_id]
        return (
            (
                [Message.system_message(agent.system_prompt)]
                if include_agent_system_prompt and agent.system_prompt
                else []
            )
            + [Message.system_message(prompt) for prompt in system_prompts if prompt]
            + [Message.user_message(user_prompt)]
        )

    @staticmethod
    def _orla_llm_identity(backend_name: str) -> tuple[str, str]:
        try:
            llm_settings = config.llm[backend_name]
        except Exception:  # noqa: BLE001
            return "", ""
        return llm_settings.model, llm_settings.base_url

    def _record_orla_llm_call(
        self,
        *,
        stage: object,
        backend_name: str,
        messages: list[Message],
        invocation: str,
        logical_node_id: str,
        stage_name: str,
        hidden: bool,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        model, base_url = self._orla_llm_identity(backend_name)
        tracer.record_llm_call(
            transport="orla",
            invocation=invocation,
            messages=messages,
            model=model,
            base_url=base_url,
            extra={
                "runtime": "orla",
                "workflow_id": getattr(stage, "_workflow_id", ""),
                "stage_id": getattr(stage, "id", ""),
                "logical_node_id": logical_node_id,
                "stage_name": stage_name,
                "backend_name": backend_name,
                "hidden": hidden,
                "tags": dict(getattr(stage, "tags", {}) or {}),
                **(extra or {}),
            },
        )

    def _wrap_orla_messages_builder(
        self,
        *,
        stage: object,
        stage_spec: RuntimeStageSpec,
    ):
        def _builder(dep_results):
            app_messages = stage_spec.build_messages(dep_results)
            if (
                stage_spec.stage_recipe is None
                or stage_spec.stage_recipe.execution_mode != "agent_loop"
            ):
                self._record_orla_llm_call(
                    stage=stage,
                    backend_name=stage_spec.backend_name,
                    messages=app_messages,
                    invocation="stage_execute",
                    logical_node_id=stage_spec.logical_node_id,
                    stage_name=stage_spec.stage_name,
                    hidden=stage_spec.hidden,
                    extra={
                        "dependency_count": len(stage_spec.dependencies),
                        "execution_mode": (
                            stage_spec.stage_recipe.execution_mode
                            if stage_spec.stage_recipe is not None
                            else "single_shot"
                        ),
                    },
                )
            return [app_message_to_orla_message(message) for message in app_messages]

        return _builder

    async def _execute_stage_specs(
        self,
        *,
        workflow_tags: dict[str, str],
        stage_specs: list[RuntimeStageSpec],
        record_hidden: bool,
    ) -> tuple[dict[str, object], float, float, str]:
        if self._registry is None:
            raise RuntimeError("[OrlaFlow] ORLA registry is unavailable in dry-run mode.")
        if not stage_specs:
            return {}, 0.0, 0.0, ""

        workflow = self._registry.pyorla.Workflow(self._registry.client, tags=workflow_tags)
        stage_objects: dict[str, object] = {}

        for stage_spec in stage_specs:
            backend = self._registry.get_backend(stage_spec.backend_name)
            llm_settings = config.llm[stage_spec.backend_name]
            stage = self._registry.pyorla.Stage(stage_spec.stage_id, backend)
            stage.id = stage_spec.stage_id
            stage.tags = {
                **{key: str(value) for key, value in workflow_tags.items()},
                "logical_node_id": stage_spec.logical_node_id,
                "stage_name": stage_spec.stage_name,
                "backend_name": stage_spec.backend_name,
                "hidden": str(stage_spec.hidden).lower(),
                "execution_mode": (
                    stage_spec.stage_recipe.execution_mode
                    if stage_spec.stage_recipe is not None
                    else "single_shot"
                ),
            }
            stage.set_max_tokens(llm_settings.max_tokens)
            stage.set_temperature(llm_settings.temperature)
            if (
                stage_spec.stage_recipe is not None
                and stage_spec.stage_recipe.execution_mode == "agent_loop"
            ):
                stage.set_execution_mode(self._registry.pyorla.EXECUTION_MODE_AGENT_LOOP)
                stage.set_max_turns(stage_spec.stage_recipe.max_turns)
                agent = self.agents[stage_spec.logical_node_id]
                for tool in tool_collection_to_orla_tools(agent.available_tools):
                    stage.add_tool(tool)
            stage.set_messages_builder(
                self._wrap_orla_messages_builder(stage=stage, stage_spec=stage_spec)
            )
            workflow.add_stage(stage)
            stage_objects[stage_spec.stage_id] = stage

        for stage_spec in stage_specs:
            for dep_stage_id in stage_spec.dependencies:
                workflow.add_dependency(stage_spec.stage_id, dep_stage_id)

        start_time = time.time()
        results = await asyncio.to_thread(workflow.execute)
        end_time = time.time()
        workflow_id = getattr(stage_objects[stage_specs[0].stage_id], "_workflow_id", "")
        tracer.record_orla_workflow(
            workflow_id=workflow_id,
            stage_ids=[stage_spec.stage_id for stage_spec in stage_specs],
            backend_names=list(dict.fromkeys(stage_spec.backend_name for stage_spec in stage_specs)),
            start_time=start_time,
            end_time=end_time,
            hidden=record_hidden,
        )
        return results, start_time, end_time, workflow_id

    def _record_coordination_markers(
        self,
        *,
        results: dict[str, object],
        markers: list[CoordinationMarker],
        anchor_time: float,
    ) -> None:
        for index, marker in enumerate(markers):
            timestamp = anchor_time + index * 1e-6
            stage_result = results.get(marker.stage_id)
            output_text = (
                stage_result.response.content
                if stage_result is not None and stage_result.response is not None
                else ""
            )
            tracer.record_coordination_call(
                coordinator_id=marker.coordinator_id,
                call_type=marker.call_type,
                input_text=marker.build_input_text(results),
                output_text=output_text,
                start_time=timestamp,
                end_time=timestamp,
                input_tokens=0,
                completion_tokens=0,
            )
            tracer.record_coordination_phase(
                coordinator_id=marker.coordinator_id,
                call_type=marker.call_type,
                start_time=timestamp,
                end_time=timestamp,
                count_toward_breakdown=False,
            )

    @staticmethod
    def _strip_fenced_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        return text

    def _parse_assignment_mapping(
        self,
        *,
        coordinator_id: str,
        raw: str,
        expected_agent_ids: list[str],
    ) -> dict[str, str]:
        mapping = json.loads(self._strip_fenced_json(raw))
        for agent_id in expected_agent_ids:
            if agent_id not in mapping:
                raise ValueError(
                    f"[OrlaFlow] Coordinator '{coordinator_id}' missed subtask for '{agent_id}'."
                )
        return {str(key): str(value) for key, value in mapping.items()}

    def _sync_fsm_inputs(
        self,
        *,
        instance_id: str,
        root_input: str,
        predecessor_outputs: list[str],
    ) -> str:
        if not predecessor_outputs:
            return root_input
        if len(predecessor_outputs) == 1:
            aggregated = predecessor_outputs[0]
        else:
            node = self._node_map.get(instance_id)
            strategy = node.input_aggregation if node else InputAggregation.CONCAT
            if strategy == InputAggregation.CONCAT:
                aggregated = "\n\n".join(predecessor_outputs)
            elif strategy == InputAggregation.VOTE:
                aggregated = Counter(predecessor_outputs).most_common(1)[0][0]
            else:
                raise ValueError(
                    f"[OrlaFlow] FSM span for '{instance_id}' requires async merge strategy."
                )
        return self._append_upstream_outputs(root_input=root_input, aggregated=aggregated)

    @staticmethod
    def _append_upstream_outputs(*, root_input: str, aggregated: str) -> str:
        if not root_input.strip():
            return aggregated
        if not aggregated.strip():
            return root_input
        return f"{root_input}\n\n[Upstream Outputs]\n{aggregated}"

    @staticmethod
    def _stage_response_content(results: dict[str, object], stage_id: str) -> str:
        stage_result = results.get(stage_id)
        if stage_result is None or stage_result.response is None:
            return ""
        return stage_result.response.content

    def _compiled_node_stage_name_lookup(self, compiled_node: CompiledNode) -> dict[str, str]:
        return {
            stage_id: stage_recipe.name
            for stage_id, stage_recipe in zip(
                compiled_node.recipe_stage_ids,
                compiled_node.recipe.stages,
            )
        }

    def _compute_compiled_node_output(
        self,
        *,
        node_id: str,
        segment: CompiledSegment,
        root_input: str,
        output_store: Dict[str, str],
        dep_results: dict[str, object],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> str:
        compiled_node = compiled_nodes[node_id]
        request_text = self._resolve_compiled_node_request(
            compiled_node=compiled_node,
            segment=segment,
            root_input=root_input,
            output_store=output_store,
            dep_results=dep_results,
            compiled_nodes=compiled_nodes,
        )
        return self._compute_recipe_result_text(
            instance_id=node_id,
            recipe=compiled_node.recipe,
            request_text=request_text,
            results=dep_results,
            recipe_stage_ids=compiled_node.recipe_stage_ids,
            stage_name_lookup=self._compiled_node_stage_name_lookup(compiled_node),
            fallback_stage_id=compiled_node.exit_stage_id,
        )

    def _compiled_node_summary(self, compiled_node: CompiledNode) -> dict:
        node = self._node_map[compiled_node.instance_id]
        return {
            "instance_id": compiled_node.instance_id,
            "agent_key": getattr(node, "agent_key", ""),
            "recipe_kind": compiled_node.recipe.kind,
            "backend_name": compiled_node.backend_name,
            "predecessors": list(compiled_node.predecessors),
            "stage_ids": list(compiled_node.stage_ids),
            "merge_stage_id": compiled_node.merge_stage_id,
            "external": compiled_node.is_external,
        }

    def _compiled_segment_summary(self, segment: CompiledSegment) -> dict:
        return {
            "segment_id": segment.segment_id,
            "node_ids": list(segment.node_ids),
            "stage_ids": list(segment.stage_ids),
            "backend_names": list(segment.backend_names),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "logical_node_id": stage.logical_node_id,
                    "kind": stage.kind,
                    "stage_name": stage.stage_name,
                    "backend_name": stage.backend_name,
                    "dependencies": list(stage.dependencies),
                    "hidden": stage.hidden,
                }
        for stage in (segment.stages[stage_id] for stage_id in segment.stage_ids)
            ],
        }

    def _reset_run_state(self) -> None:
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

    async def _build_node_input_from_outputs(
        self,
        *,
        instance_id: str,
        root_input: str,
        outputs: Dict[str, str],
        predecessor_map: Dict[str, List[str]],
    ) -> str:
        preds = predecessor_map.get(instance_id, [])
        if not preds:
            return root_input
        pred_outputs = [outputs[pred] for pred in preds if pred in outputs]
        if not pred_outputs:
            return root_input
        return await self._aggregate_inputs(instance_id, pred_outputs)

    def _segment_ready(
        self,
        *,
        segment: CompiledSegment,
        executed_nodes: set[str],
        predecessor_map: Dict[str, List[str]],
    ) -> bool:
        segment_nodes = set(segment.node_ids)
        for node_id in segment.node_ids:
            for pred in predecessor_map.get(node_id, []):
                if pred not in segment_nodes and pred not in executed_nodes:
                    return False
        return True

    def _collect_compiled_predecessor_outputs(
        self,
        *,
        compiled_node: CompiledNode,
        segment: CompiledSegment,
        root_input: str,
        output_store: Dict[str, str],
        dep_results: dict[str, object],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> list[str]:
        inputs: list[str] = []
        segment_node_ids = set(segment.node_ids)
        for pred in compiled_node.predecessors:
            pred_compiled_node = compiled_nodes[pred]
            pred_exit_stage_id = pred_compiled_node.exit_stage_id
            if pred in segment_node_ids and pred_exit_stage_id and pred_exit_stage_id in dep_results:
                inputs.append(
                    self._compute_compiled_node_output(
                        node_id=pred,
                        segment=segment,
                        root_input=root_input,
                        output_store=output_store,
                        dep_results=dep_results,
                        compiled_nodes=compiled_nodes,
                    )
                )
                continue
            if pred in output_store:
                inputs.append(output_store[pred])
                continue
            raise ValueError(
                f"[OrlaFlow] Missing predecessor output for '{compiled_node.instance_id}' <- '{pred}'."
            )
        return inputs

    def _resolve_compiled_node_request(
        self,
        *,
        compiled_node: CompiledNode,
        segment: CompiledSegment,
        root_input: str,
        output_store: Dict[str, str],
        dep_results: dict[str, object],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> str:
        if compiled_node.merge_stage_id:
            merge_result = dep_results.get(compiled_node.merge_stage_id)
            if merge_result is None:
                raise ValueError(
                    f"[OrlaFlow] Missing merge output for node '{compiled_node.instance_id}'."
                )
            return merge_result.response.content if merge_result.response else ""

        inputs = self._collect_compiled_predecessor_outputs(
            compiled_node=compiled_node,
            segment=segment,
            root_input=root_input,
            output_store=output_store,
            dep_results=dep_results,
            compiled_nodes=compiled_nodes,
        )
        if not inputs:
            return root_input
        if len(inputs) == 1:
            return inputs[0]
        if compiled_node.input_aggregation == InputAggregation.CONCAT:
            return "\n\n".join(inputs)
        if compiled_node.input_aggregation == InputAggregation.VOTE:
            return Counter(inputs).most_common(1)[0][0]
        raise ValueError(
            f"[OrlaFlow] Node '{compiled_node.instance_id}' requires an explicit merge stage."
        )

    def _build_segment_stage_messages(
        self,
        *,
        stage_spec,
        segment: CompiledSegment,
        root_input: str,
        output_store: Dict[str, str],
        compiled_nodes: Dict[str, CompiledNode],
    ):
        def _builder(dep_results):
            compiled_node = compiled_nodes[stage_spec.logical_node_id]
            if stage_spec.kind == "merge":
                inputs = self._collect_compiled_predecessor_outputs(
                    compiled_node=compiled_node,
                    segment=segment,
                    root_input=root_input,
                    output_store=output_store,
                    dep_results=dep_results,
                    compiled_nodes=compiled_nodes,
                )
                numbered = "\n\n".join(
                    f"[Input {index + 1}]\n{text}" for index, text in enumerate(inputs)
                )
                prompt = LLM_MERGE_PROMPT_TEMPLATE.format(numbered_inputs=numbered)
                return [Message.user_message(prompt)]

            request_text = self._resolve_compiled_node_request(
                compiled_node=compiled_node,
                segment=segment,
                root_input=root_input,
                output_store=output_store,
                dep_results=dep_results,
                compiled_nodes=compiled_nodes,
            )
            stage_outputs = self._build_recipe_stage_outputs(
                results=dep_results,
                recipe_stage_ids=compiled_node.recipe_stage_ids,
                stage_name_lookup=self._compiled_node_stage_name_lookup(compiled_node),
            )
            return stage_spec.stage_recipe.build_messages(request_text, stage_outputs)

        return _builder

    async def _execute_compiled_segment(
        self,
        *,
        segment: CompiledSegment,
        root_input: str,
        output_store: Dict[str, str],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> Dict[str, str]:
        stage_specs = [
            RuntimeStageSpec(
                stage_id=stage_id,
                logical_node_id=segment.stages[stage_id].logical_node_id,
                backend_name=segment.stages[stage_id].backend_name,
                stage_name=segment.stages[stage_id].stage_name,
                dependencies=list(segment.stages[stage_id].dependencies),
                hidden=segment.stages[stage_id].hidden,
                stage_recipe=segment.stages[stage_id].stage_recipe,
                build_messages=self._build_segment_stage_messages(
                    stage_spec=segment.stages[stage_id],
                    segment=segment,
                    root_input=root_input,
                    output_store=output_store,
                    compiled_nodes=compiled_nodes,
                ),
            )
            for stage_id in segment.stage_ids
        ]
        results, start_time, end_time, _ = await self._execute_stage_specs(
            workflow_tags={"runtime": "orla", "segment_id": segment.segment_id},
            stage_specs=stage_specs,
            record_hidden=False,
        )
        tracer.record_orla_segment_phase(
            agent_ids=segment.node_ids,
            segment_id=segment.segment_id,
            start_time=start_time,
            end_time=end_time,
        )

        node_outputs: Dict[str, str] = {}
        for node_id in segment.node_ids:
            compiled_node = compiled_nodes[node_id]
            request_text = self._resolve_compiled_node_request(
                compiled_node=compiled_node,
                segment=segment,
                root_input=root_input,
                output_store=output_store,
                dep_results=results,
                compiled_nodes=compiled_nodes,
            )
            result_text = self._finalize_recipe_node(
                instance_id=node_id,
                recipe=compiled_node.recipe,
                request_text=request_text,
                results=results,
                recipe_stage_ids=compiled_node.recipe_stage_ids,
                stage_name_lookup=self._compiled_node_stage_name_lookup(compiled_node),
                fallback_stage_id=compiled_node.exit_stage_id,
            )
            node_outputs[node_id] = result_text

        return node_outputs

    async def _run_external_node(
        self,
        *,
        instance_id: str,
        input_text: str,
    ) -> str:
        self._reset_agent(instance_id)
        start = time.time()
        output = await self._run_agent(instance_id, input_text)
        end = time.time()
        tracer.record_agent_run_phase(
            agent_id=instance_id,
            start_time=start,
            end_time=end,
            parallel=False,
        )
        return output

    async def _run_external_nodes_parallel(
        self,
        *,
        id_input_pairs: List[tuple[str, str]],
    ) -> Dict[str, str]:
        if not id_input_pairs:
            return {}

        for instance_id, _ in id_input_pairs:
            self._reset_agent(instance_id)

        async def _safe_run(instance_id: str, input_text: str):
            start = time.time()
            output = await self._run_agent(instance_id, input_text)
            end = time.time()
            return instance_id, output, start, end

        batch_start = time.time()
        results = await asyncio.gather(
            *[_safe_run(instance_id, input_text) for instance_id, input_text in id_input_pairs]
        )
        batch_end = time.time()
        tracer.record_parallel_batch_phase(
            agent_ids=[instance_id for instance_id, _ in id_input_pairs],
            start_time=batch_start,
            end_time=batch_end,
            individual_times={
                instance_id: {"start_time": start, "end_time": end}
                for instance_id, _, start, end in results
            },
        )
        return {instance_id: output for instance_id, output, _, _ in results}

    async def _execute_segmented_graph(
        self,
        *,
        root_input: str,
        node_order: List[str],
        predecessor_map: Dict[str, List[str]],
        compiled_nodes: Dict[str, CompiledNode],
        segments: List[CompiledSegment],
    ) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        executed_nodes: set[str] = set()
        completed_segments: set[str] = set()
        segment_by_node = {
            node_id: segment for segment in segments for node_id in segment.node_ids
        }

        while len(executed_nodes) < len(node_order):
            progress = False
            for node_id in node_order:
                if node_id in executed_nodes:
                    continue

                compiled_node = compiled_nodes[node_id]
                if compiled_node.is_external:
                    if not all(pred in executed_nodes for pred in predecessor_map.get(node_id, [])):
                        continue
                    node_input = await self._build_node_input_from_outputs(
                        instance_id=node_id,
                        root_input=root_input,
                        outputs=outputs,
                        predecessor_map=predecessor_map,
                    )
                    outputs[node_id] = await self._run_external_node(
                        instance_id=node_id,
                        input_text=node_input,
                    )
                    self.last_run_outputs[node_id] = outputs[node_id]
                    executed_nodes.add(node_id)
                    progress = True
                    continue

                segment = segment_by_node[node_id]
                if segment.segment_id in completed_segments:
                    executed_nodes.add(node_id)
                    progress = True
                    continue
                if not self._segment_ready(
                    segment=segment,
                    executed_nodes=executed_nodes,
                    predecessor_map=predecessor_map,
                ):
                    continue

                for segment_node_id in segment.node_ids:
                    self._reset_agent(segment_node_id)
                segment_outputs = await self._execute_compiled_segment(
                    segment=segment,
                    root_input=root_input,
                    output_store=outputs,
                    compiled_nodes=compiled_nodes,
                )
                outputs.update(segment_outputs)
                self.last_run_outputs.update(segment_outputs)
                executed_nodes.update(segment.node_ids)
                completed_segments.add(segment.segment_id)
                progress = True

            if not progress:
                raise RuntimeError("[OrlaFlow] Failed to make progress while executing segmented graph.")

        return outputs

    async def _execute_chain(self, input_text: str) -> str:
        order = self._resolve_chain_order()
        predecessor_map = {order[0]: []} if order else {}
        for prev, current in zip(order, order[1:]):
            predecessor_map[current] = [prev]
        tracer.set_topology("chain")
        self._reset_run_state()

        compiled_nodes = self._compiler.compile_nodes(predecessor_map)
        segments = self._compiler.build_connected_segments(
            node_order=order,
            predecessor_map=predecessor_map,
            compiled_nodes=compiled_nodes,
        )
        outputs = await self._execute_segmented_graph(
            root_input=input_text,
            node_order=order,
            predecessor_map=predecessor_map,
            compiled_nodes=compiled_nodes,
            segments=segments,
        )
        terminal_id = order[-1] if order else None
        self.last_terminal_node = terminal_id
        self.last_terminal_output = outputs.get(terminal_id, "") if terminal_id else ""
        return self.last_terminal_output or ""

    def _resolve_chain_order(self) -> List[str]:
        valid_routes = [r for r in (self.mas_config.routes or []) if r.source != r.target]
        if not valid_routes:
            return [n.instance_id for n in self.mas_config.agents]

        all_targets = {r.target for r in valid_routes}
        all_ids = {n.instance_id for n in self.mas_config.agents}
        starts = [n for n in all_ids if n not in all_targets]
        if not starts:
            raise ValueError("[OrlaFlow] Cannot find chain start.")
        adj = {r.source: r.target for r in valid_routes}
        order: List[str] = []
        current = starts[0]
        visited = set()
        while current and current not in visited:
            order.append(current)
            visited.add(current)
            current = adj.get(current)
        return order

    async def _execute_dag(self, input_text: str) -> str:
        node_ids = [n.instance_id for n in self.mas_config.agents]
        preds_map = build_predecessors_map(self.mas_config.routes)
        tracer.set_topology("dag")
        self._reset_run_state()
        generations = topological_generations(node_ids, self.mas_config.routes)
        flat_order = [n for layer in generations for n in layer]
        compiled_nodes = self._compiler.compile_nodes(preds_map)
        segments = self._compiler.build_connected_segments(
            node_order=flat_order,
            predecessor_map=preds_map,
            compiled_nodes=compiled_nodes,
        )
        outputs = await self._execute_segmented_graph(
            root_input=input_text,
            node_order=flat_order,
            predecessor_map=preds_map,
            compiled_nodes=compiled_nodes,
            segments=segments,
        )

        all_sources = {r.source for r in self.mas_config.routes}
        sinks = [n for n in flat_order if n not in all_sources]
        sink_outputs = [outputs[n] for n in sinks if n in outputs]
        if not sink_outputs:
            self.last_terminal_node = flat_order[-1] if flat_order else None
            self.last_terminal_output = outputs.get(flat_order[-1], "") if flat_order else None
            return outputs.get(flat_order[-1], "")
        if len(sink_outputs) == 1:
            self.last_terminal_node = sinks[0]
            self.last_terminal_output = sink_outputs[0]
            return sink_outputs[0]
        merged = await self._aggregate_inputs(sinks[0], sink_outputs)
        self.last_terminal_node = sinks[0]
        self.last_terminal_output = merged
        return merged

    async def _execute_star(self, input_text: str) -> str:
        center_id = self.mas_config.topology_params.get("center")
        if not center_id:
            raise ValueError("[OrlaFlow] topology_params must contain 'center'.")
        worker_ids = [
            n.instance_id for n in self.mas_config.agents if n.instance_id != center_id
        ]
        tracer.set_topology("star")
        self._reset_run_state()

        predecessor_map: Dict[str, List[str]] = {worker_id: [] for worker_id in worker_ids}
        predecessor_map[center_id] = list(worker_ids)
        node_order = list(worker_ids) + [center_id]
        compiled_nodes = self._compiler.compile_nodes(predecessor_map)

        outputs: Dict[str, str] = {}
        external_workers = [
            worker_id for worker_id in worker_ids if compiled_nodes[worker_id].is_external
        ]
        if external_workers:
            worker_outputs = await self._run_external_nodes_parallel(
                id_input_pairs=[(worker_id, input_text) for worker_id in external_workers]
            )
            outputs.update(worker_outputs)
            self.last_run_outputs.update(worker_outputs)

        orla_worker_ids = [
            worker_id for worker_id in worker_ids if not compiled_nodes[worker_id].is_external
        ]
        star_segment_node_ids = list(orla_worker_ids)
        if not compiled_nodes[center_id].is_external:
            star_segment_node_ids.append(center_id)

        if star_segment_node_ids:
            segment = self._compiler.build_explicit_segment(
                segment_id="star_segment",
                node_ids=star_segment_node_ids,
                predecessor_map=predecessor_map,
                compiled_nodes=compiled_nodes,
            )
            for node_id in star_segment_node_ids:
                self._reset_agent(node_id)
            segment_outputs = await self._execute_compiled_segment(
                segment=segment,
                root_input=input_text,
                output_store=outputs,
                compiled_nodes=compiled_nodes,
            )
            outputs.update(segment_outputs)
            self.last_run_outputs.update(segment_outputs)

        if compiled_nodes[center_id].is_external:
            ordered_outputs = [outputs[worker_id] for worker_id in worker_ids if worker_id in outputs]
            center_input = "\n\n".join(ordered_outputs)
            outputs[center_id] = await self._run_external_node(
                instance_id=center_id,
                input_text=center_input,
            )
            self.last_run_outputs[center_id] = outputs[center_id]

        self.last_terminal_node = center_id
        self.last_terminal_output = outputs.get(center_id, "")
        return self.last_terminal_output or ""

    def _collect_fsm_orla_span(
        self,
        *,
        start_node_id: str,
        decision_nodes: dict[str, FSMDecisionNode],
        compiled_nodes: Dict[str, CompiledNode],
        max_nodes: Optional[int] = None,
    ) -> list[str]:
        span = [start_node_id]
        current_id = start_node_id

        while True:
            if max_nodes is not None and len(span) >= max_nodes:
                break
            if current_id in decision_nodes:
                break
            successors = self.get_successors(current_id)
            if not successors:
                break
            if len(successors) > 1:
                raise ValueError(
                    f"[OrlaFlow] Non-decision node '{current_id}' has multiple successors: {successors}."
                )
            next_id = successors[0]
            if compiled_nodes[next_id].is_external:
                break
            if len(self.get_predecessors(next_id)) != 1:
                break
            span.append(next_id)
            current_id = next_id

        return span

    def _build_fsm_span_stage_specs(
        self,
        *,
        segment_id: str,
        span_node_ids: list[str],
        root_input: str,
        initial_request_text: str,
        output_store: Dict[str, str],
        predecessor_map: Dict[str, List[str]],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> list[RuntimeStageSpec]:
        span_node_set = set(span_node_ids)
        stage_specs: list[RuntimeStageSpec] = []

        def _resolve_request(node_id: str, dep_results: dict[str, object]) -> str:
            return self._compute_fsm_span_node_request(
                node_id=node_id,
                span_start_id=span_node_ids[0],
                initial_request_text=initial_request_text,
                root_input=root_input,
                output_store=output_store,
                dep_results=dep_results,
                predecessor_map=predecessor_map,
                compiled_nodes=compiled_nodes,
            )

        for node_id in span_node_ids:
            compiled_node = compiled_nodes[node_id]
            stage_name_lookup = {
                recipe_stage_id: recipe_stage.name
                for recipe_stage_id, recipe_stage in zip(
                    compiled_node.recipe_stage_ids,
                    compiled_node.recipe.stages,
                )
            }
            previous_stage_id: Optional[str] = None
            for stage_id, stage_recipe in zip(compiled_node.recipe_stage_ids, compiled_node.recipe.stages):
                dependencies: list[str] = []
                if previous_stage_id is not None:
                    dependencies.append(previous_stage_id)
                else:
                    for pred in predecessor_map.get(node_id, []):
                        if pred not in span_node_set:
                            continue
                        pred_exit_stage_id = compiled_nodes[pred].exit_stage_id
                        if pred_exit_stage_id:
                            dependencies.append(pred_exit_stage_id)

                def _build_messages(
                    dep_results,
                    *,
                    current_node_id=node_id,
                    current_stage_recipe=stage_recipe,
                    current_stage_name_lookup=stage_name_lookup,
                    current_recipe_stage_ids=list(compiled_node.recipe_stage_ids),
                ):
                    request_text = _resolve_request(current_node_id, dep_results)
                    stage_outputs = self._build_recipe_stage_outputs(
                        results=dep_results,
                        recipe_stage_ids=current_recipe_stage_ids,
                        stage_name_lookup=current_stage_name_lookup,
                    )
                    return current_stage_recipe.build_messages(request_text, stage_outputs)

                stage_specs.append(
                    RuntimeStageSpec(
                        stage_id=stage_id,
                        logical_node_id=node_id,
                        backend_name=compiled_node.backend_name or "",
                        stage_name=stage_recipe.name,
                        dependencies=dependencies,
                        hidden=False,
                        stage_recipe=stage_recipe,
                        build_messages=_build_messages,
                    )
                )
                previous_stage_id = stage_id

        tracer.record_orla_compilation(
            {
                "runtime": "orla",
                "topology": "fsm",
                "segment_id": segment_id,
                "node_ids": span_node_ids,
                "stage_ids": [stage_spec.stage_id for stage_spec in stage_specs],
            }
        )
        return stage_specs

    def _compute_fsm_span_node_request(
        self,
        *,
        node_id: str,
        span_start_id: str,
        initial_request_text: str,
        root_input: str,
        output_store: Dict[str, str],
        dep_results: dict[str, object],
        predecessor_map: Dict[str, List[str]],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> str:
        if node_id == span_start_id:
            return initial_request_text

        predecessor_outputs: list[str] = []
        for pred in predecessor_map.get(node_id, []):
            if pred in output_store:
                predecessor_outputs.append(output_store[pred])
                continue
            pred_compiled_node = compiled_nodes.get(pred)
            pred_exit_stage_id = pred_compiled_node.exit_stage_id if pred_compiled_node else None
            if pred_compiled_node is None or not pred_exit_stage_id or pred_exit_stage_id not in dep_results:
                continue
            pred_request_text = self._compute_fsm_span_node_request(
                node_id=pred,
                span_start_id=span_start_id,
                initial_request_text=initial_request_text,
                root_input=root_input,
                output_store=output_store,
                dep_results=dep_results,
                predecessor_map=predecessor_map,
                compiled_nodes=compiled_nodes,
            )
            predecessor_outputs.append(
                self._compute_recipe_result_text(
                    instance_id=pred,
                    recipe=pred_compiled_node.recipe,
                    request_text=pred_request_text,
                    results=dep_results,
                    recipe_stage_ids=pred_compiled_node.recipe_stage_ids,
                    stage_name_lookup=self._compiled_node_stage_name_lookup(pred_compiled_node),
                    fallback_stage_id=pred_compiled_node.exit_stage_id,
                )
            )

        return self._sync_fsm_inputs(
            instance_id=node_id,
            root_input=root_input,
            predecessor_outputs=predecessor_outputs,
        )

    def _append_tree_recipe_stages(
        self,
        *,
        node_id: str,
        request_text_resolver: RequestTextResolver,
        dependencies: list[str],
        plan: TreeRoundPlan,
    ) -> str:
        recipe = self._recipes.get(node_id, NodeRecipe(kind="external"))
        if recipe.kind == "external":
            raise ValueError(f"[OrlaFlow] Tree round builder cannot compile external node '{node_id}'.")

        stage_ids: list[str] = []
        stage_name_lookup: dict[str, str] = {}
        previous_stage_id: Optional[str] = None

        for stage_recipe in recipe.stages:
            stage_id = f"{node_id}__{stage_recipe.name}"
            stage_name_lookup[stage_id] = stage_recipe.name
            entry_dependencies = [previous_stage_id] if previous_stage_id else list(
                dict.fromkeys(dependencies)
            )

            def _build_messages(
                dep_results,
                *,
                current_stage_recipe=stage_recipe,
                current_request_resolver=request_text_resolver,
                current_stage_ids=list(stage_ids),
                current_stage_name_lookup=dict(stage_name_lookup),
            ):
                request_text = current_request_resolver(dep_results)
                stage_outputs = {
                    current_stage_name_lookup[recipe_stage_id]: self._stage_response_content(
                        dep_results,
                        recipe_stage_id,
                    )
                    for recipe_stage_id in current_stage_ids
                    if recipe_stage_id in dep_results and recipe_stage_id in current_stage_name_lookup
                }
                return current_stage_recipe.build_messages(request_text, stage_outputs)

            plan.stage_specs.append(
                RuntimeStageSpec(
                    stage_id=stage_id,
                    logical_node_id=node_id,
                    backend_name=self._node_map[node_id].llm_config_name,
                    stage_name=stage_recipe.name,
                    build_messages=_build_messages,
                    dependencies=entry_dependencies,
                    hidden=False,
                    stage_recipe=stage_recipe,
                )
            )
            stage_ids.append(stage_id)
            previous_stage_id = stage_id

        plan.node_request_resolvers[node_id] = request_text_resolver
        plan.node_stage_ids[node_id] = list(stage_ids)
        plan.stage_name_lookup[node_id] = stage_name_lookup
        plan.node_exit_stage_ids[node_id] = previous_stage_id or ""
        if node_id not in plan.node_order:
            plan.node_order.append(node_id)
        return previous_stage_id or ""

    def _compute_tree_round_node_output(
        self,
        node_id: str,
        dep_results: dict[str, object],
        plan: TreeRoundPlan,
    ) -> str:
        recipe = self._recipes[node_id]
        request_text = plan.node_request_resolvers[node_id](dep_results)
        return self._compute_recipe_result_text(
            instance_id=node_id,
            recipe=recipe,
            request_text=request_text,
            results=dep_results,
            recipe_stage_ids=plan.node_stage_ids[node_id],
            stage_name_lookup=plan.stage_name_lookup[node_id],
            fallback_stage_id=plan.node_exit_stage_ids[node_id],
        )

    def _append_tree_round_node(
        self,
        *,
        node_id: str,
        request_text_resolver: RequestTextResolver,
        request_dependencies: list[str],
        adj: Dict[str, List[str]],
        use_llm_assign: bool,
        progress_history: List[str],
        plan: TreeRoundPlan,
    ) -> str:
        children_ids = adj.get(node_id, [])
        if not children_ids:
            return self._append_tree_recipe_stages(
                node_id=node_id,
                request_text_resolver=request_text_resolver,
                dependencies=request_dependencies,
                plan=plan,
            )

        child_request_dependencies = list(request_dependencies)
        if use_llm_assign:
            profiles_text = "\n".join(
                f"  - {child_id}: {self._node_map[child_id].role_description}"
                for child_id in children_ids
            )
            assign_stage_id = self._next_hidden_stage_id(node_id, "assign_subtasks")

            def _assign_user_prompt(
                dep_results,
                *,
                current_request_resolver=request_text_resolver,
                current_profiles_text=profiles_text,
                current_progress_history=list(progress_history),
            ):
                return ASSIGN_SUBTASKS_USER.format(
                    task=current_request_resolver(dep_results),
                    agent_profiles=current_profiles_text,
                    progress=self._format_progress_history(current_progress_history),
                )

            plan.stage_specs.append(
                RuntimeStageSpec(
                    stage_id=assign_stage_id,
                    logical_node_id=node_id,
                    backend_name=self._node_map[node_id].llm_config_name,
                    stage_name="assign_subtasks",
                    dependencies=list(dict.fromkeys(request_dependencies)),
                    hidden=True,
                    build_messages=lambda dep_results, current_input_builder=_assign_user_prompt: self._build_hidden_stage_messages(
                        instance_id=node_id,
                        system_prompts=[ASSIGN_SUBTASKS_SYSTEM],
                        user_prompt=current_input_builder(dep_results),
                    ),
                )
            )
            plan.coordination_markers.append(
                CoordinationMarker(
                    stage_id=assign_stage_id,
                    coordinator_id=node_id,
                    call_type=f"{node_id}.assign_subtasks",
                    build_input_text=_assign_user_prompt,
                )
            )
            child_request_dependencies = [assign_stage_id]

            def _make_child_request_resolver(child_id: str) -> RequestTextResolver:
                def _resolver(
                    dep_results,
                    *,
                    current_child_id=child_id,
                    current_assign_stage_id=assign_stage_id,
                    current_request_resolver=request_text_resolver,
                    expected_agent_ids=list(children_ids),
                ):
                    parent_task = current_request_resolver(dep_results)
                    mapping = self._parse_assignment_mapping(
                        coordinator_id=node_id,
                        raw=self._stage_response_content(dep_results, current_assign_stage_id),
                        expected_agent_ids=expected_agent_ids,
                    )
                    return mapping.get(current_child_id, parent_task)

                return _resolver

            child_request_resolvers = {
                child_id: _make_child_request_resolver(child_id) for child_id in children_ids
            }
        else:
            child_request_resolvers = {
                child_id: request_text_resolver for child_id in children_ids
            }

        child_exit_stage_ids = [
            self._append_tree_round_node(
                node_id=child_id,
                request_text_resolver=child_request_resolvers[child_id],
                request_dependencies=child_request_dependencies,
                adj=adj,
                use_llm_assign=use_llm_assign,
                progress_history=progress_history,
                plan=plan,
            )
            for child_id in children_ids
        ]

        def _synthesis_request(
            dep_results,
            *,
            current_request_resolver=request_text_resolver,
            current_children_ids=list(children_ids),
            current_child_exit_stage_ids=dict(plan.node_exit_stage_ids),
        ):
            task_text = current_request_resolver(dep_results)
            children_results_text = "\n\n".join(
                f"[{child_id}]:\n{self._compute_tree_round_node_output(child_id, dep_results, plan)}"
                for child_id in current_children_ids
            )
            return (
                f"{task_text}\n\n"
                f"[Children Results]\n{children_results_text}\n\n"
                "Based on your subtask and the children's results above, "
                "synthesise a final response. Continue working on the original task "
                "without repeating what children already covered."
            )

        return self._append_tree_recipe_stages(
            node_id=node_id,
            request_text_resolver=_synthesis_request,
            dependencies=list(dict.fromkeys([*request_dependencies, *child_exit_stage_ids])),
            plan=plan,
        )

    def _build_tree_round_plan(
        self,
        *,
        root_id: str,
        task: str,
        input_text: str,
        round_num: int,
        max_rounds: int,
        adj: Dict[str, List[str]],
        use_llm_assign: bool,
        progress_history: List[str],
        use_llm_check: bool,
    ) -> TreeRoundPlan:
        plan = TreeRoundPlan()
        root_request_resolver: RequestTextResolver = lambda _deps, current_task=task: current_task
        root_exit_stage_id = self._append_tree_round_node(
            node_id=root_id,
            request_text_resolver=root_request_resolver,
            request_dependencies=[],
            adj=adj,
            use_llm_assign=use_llm_assign,
            progress_history=progress_history,
            plan=plan,
        )

        def _summary_prompt(dep_results, *, current_root_exit_stage_id=root_exit_stage_id):
            return SUMMARIZE_RESULTS_USER.format(
                task=input_text,
                results=f"  [{root_id}]: {self._compute_tree_round_node_output(root_id, dep_results, plan)[:800]}",
            )

        summarize_stage_id = self._next_hidden_stage_id(root_id, "summarize_results")
        plan.stage_specs.append(
            RuntimeStageSpec(
                stage_id=summarize_stage_id,
                logical_node_id=root_id,
                backend_name=self._node_map[root_id].llm_config_name,
                stage_name="summarize_results",
                dependencies=[root_exit_stage_id],
                hidden=True,
                build_messages=lambda dep_results, current_prompt_builder=_summary_prompt: self._build_hidden_stage_messages(
                    instance_id=root_id,
                    system_prompts=[SUMMARIZE_RESULTS_SYSTEM],
                    user_prompt=current_prompt_builder(dep_results),
                ),
            )
        )
        plan.coordination_markers.append(
            CoordinationMarker(
                stage_id=summarize_stage_id,
                coordinator_id=root_id,
                call_type=f"{root_id}.summarize_results",
                build_input_text=_summary_prompt,
            )
        )
        plan.summarize_stage_id = summarize_stage_id

        if use_llm_check and round_num < max_rounds:
            def _continue_prompt(
                dep_results,
                *,
                current_summary_stage_id=summarize_stage_id,
            ):
                return CONTINUE_USER.format(
                    task=input_text,
                    summary=self._stage_response_content(dep_results, current_summary_stage_id),
                    round_num=round_num,
                    max_rounds=max_rounds,
                )

            continue_stage_id = self._next_hidden_stage_id(root_id, "should_continue")
            plan.stage_specs.append(
                RuntimeStageSpec(
                    stage_id=continue_stage_id,
                    logical_node_id=root_id,
                    backend_name=self._node_map[root_id].llm_config_name,
                    stage_name="should_continue",
                    dependencies=[summarize_stage_id],
                    hidden=True,
                    build_messages=lambda dep_results, current_prompt_builder=_continue_prompt: self._build_hidden_stage_messages(
                        instance_id=root_id,
                        system_prompts=[CONTINUE_SYSTEM],
                        user_prompt=current_prompt_builder(dep_results),
                    ),
                )
            )
            plan.coordination_markers.append(
                CoordinationMarker(
                    stage_id=continue_stage_id,
                    coordinator_id=root_id,
                    call_type=f"{root_id}.should_continue",
                    build_input_text=_continue_prompt,
                )
            )
            plan.continue_stage_id = continue_stage_id

        tracer.record_orla_compilation(
            {
                "runtime": "orla",
                "topology": "tree",
                "round": round_num,
                "root": root_id,
                "stage_ids": [stage_spec.stage_id for stage_spec in plan.stage_specs],
                "stages": [
                    {
                        "stage_id": stage_spec.stage_id,
                        "logical_node_id": stage_spec.logical_node_id,
                        "stage_name": stage_spec.stage_name,
                        "backend_name": stage_spec.backend_name,
                        "dependencies": list(stage_spec.dependencies),
                        "hidden": stage_spec.hidden,
                    }
                    for stage_spec in plan.stage_specs
                ],
            }
        )
        return plan

    async def _execute_tree_round(
        self,
        *,
        root_id: str,
        task: str,
        input_text: str,
        round_num: int,
        max_rounds: int,
        adj: Dict[str, List[str]],
        use_llm_assign: bool,
        progress_history: List[str],
        use_llm_check: bool,
    ) -> tuple[str, str, bool, dict[str, str]]:
        plan = self._build_tree_round_plan(
            root_id=root_id,
            task=task,
            input_text=input_text,
            round_num=round_num,
            max_rounds=max_rounds,
            adj=adj,
            use_llm_assign=use_llm_assign,
            progress_history=progress_history,
            use_llm_check=use_llm_check,
        )
        for node_id in plan.node_order:
            self._reset_agent(node_id)

        results, start_time, end_time, _ = await self._execute_stage_specs(
            workflow_tags={"runtime": "orla", "topology": "tree", "round": str(round_num)},
            stage_specs=plan.stage_specs,
            record_hidden=False,
        )
        tracer.record_orla_segment_phase(
            agent_ids=plan.node_order,
            segment_id=f"tree_round_{round_num}",
            start_time=start_time,
            end_time=end_time,
        )
        self._record_coordination_markers(
            results=results,
            markers=plan.coordination_markers,
            anchor_time=start_time,
        )

        node_outputs: dict[str, str] = {}
        for node_id in plan.node_order:
            request_text = plan.node_request_resolvers[node_id](results)
            node_outputs[node_id] = self._finalize_recipe_node(
                instance_id=node_id,
                recipe=self._recipes[node_id],
                request_text=request_text,
                results=results,
                recipe_stage_ids=plan.node_stage_ids[node_id],
                stage_name_lookup=plan.stage_name_lookup[node_id],
                fallback_stage_id=plan.node_exit_stage_ids[node_id],
            )

        summary = self._stage_response_content(results, plan.summarize_stage_id)
        keep_going = True
        if plan.continue_stage_id is not None:
            keep_going = self._stage_response_content(results, plan.continue_stage_id).strip().upper() != "CONVERGED"
        return node_outputs[root_id], summary, keep_going, node_outputs

    async def _execute_tree(self, input_text: str) -> str:
        root_id = self.mas_config.topology_params.get("root")
        if not root_id:
            raise ValueError("[OrlaFlow] topology_params must contain 'root'.")

        convergence = self.mas_config.convergence
        max_rounds = convergence.max_rounds
        use_llm_assign = convergence.llm_assign_tasks
        use_llm_check = convergence.llm_check
        adj = build_adjacency_list(self.mas_config.routes)
        tracer.set_topology("tree")
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        root_output = ""
        progress_history: List[str] = []
        task = input_text
        can_batch_round = all(
            self._recipes.get(node.instance_id, NodeRecipe(kind="external")).kind != "external"
            for node in self.mas_config.agents
        )

        for round_num in range(1, max_rounds + 1):
            if can_batch_round:
                root_output, summary, keep_going, node_outputs = await self._execute_tree_round(
                    root_id=root_id,
                    task=task,
                    input_text=input_text,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    adj=adj,
                    use_llm_assign=use_llm_assign,
                    progress_history=progress_history,
                    use_llm_check=use_llm_check,
                )
                self.last_run_outputs.update(node_outputs)
            else:
                root_output = await self._execute_tree_recursive(
                    node_id=root_id,
                    task=task,
                    adj=adj,
                    use_llm_assign=use_llm_assign,
                    progress_history=progress_history,
                )
                summary = await self._summarize_results_with_agent(
                    coordinator_id=root_id,
                    task=input_text,
                    results={root_id: root_output},
                )
                keep_going = True
                if use_llm_check:
                    keep_going = await self._should_continue_with_agent(
                        coordinator_id=root_id,
                        task=input_text,
                        summary=summary,
                        round_num=round_num,
                        max_rounds=max_rounds,
                    )
            progress_history.append(summary)
            self.last_run_rounds = round_num
            if use_llm_check:
                if not keep_going:
                    break
            task = f"{input_text}\n\n[Progress after round {round_num}]\n{summary}"

        self.last_terminal_node = root_id
        self.last_terminal_output = root_output
        self.last_run_outputs[root_id] = root_output
        return root_output

    async def _execute_tree_recursive(
        self,
        *,
        node_id: str,
        task: str,
        adj: Dict[str, List[str]],
        use_llm_assign: bool,
        progress_history: List[str],
    ) -> str:
        children_ids = adj.get(node_id, [])
        if not children_ids:
            self._reset_agent(node_id)
            start = time.time()
            result = await self._run_runtime_node(node_id, task)
            end = time.time()
            tracer.record_agent_run_phase(
                agent_id=node_id,
                start_time=start,
                end_time=end,
                parallel=False,
            )
            self.last_run_outputs[node_id] = result
            return result

        child_profiles = {
            child_id: next(
                (
                    node.role_description
                    for node in self.mas_config.agents
                    if node.instance_id == child_id
                ),
                "",
            )
            for child_id in children_ids
        }
        if use_llm_assign:
            child_subtasks = await self._assign_subtasks_with_agent(
                coordinator_id=node_id,
                task=task,
                agent_profiles=child_profiles,
                progress_history=progress_history,
            )
        else:
            child_subtasks = {child_id: task for child_id in children_ids}

        children_start = time.time()
        child_outputs = await asyncio.gather(
            *[
                self._execute_tree_recursive(
                    node_id=child_id,
                    task=child_subtasks.get(child_id, task),
                    adj=adj,
                    use_llm_assign=use_llm_assign,
                    progress_history=progress_history,
                )
                for child_id in children_ids
            ]
        )
        children_end = time.time()
        if len(children_ids) > 1:
            tracer.record_parallel_batch_phase(
                agent_ids=children_ids,
                start_time=children_start,
                end_time=children_end,
            )

        children_results_text = "\n\n".join(
            f"[{child_id}]:\n{output}"
            for child_id, output in zip(children_ids, child_outputs)
        )
        synthesis_input = (
            f"{task}\n\n"
            f"[Children Results]\n{children_results_text}\n\n"
            "Based on your subtask and the children's results above, "
            "synthesise a final response. Continue working on the original task "
            "without repeating what children already covered."
        )
        self._reset_agent(node_id)
        start = time.time()
        result = await self._run_runtime_node(node_id, synthesis_input)
        end = time.time()
        tracer.record_agent_run_phase(
            agent_id=node_id,
            start_time=start,
            end_time=end,
            parallel=False,
        )
        self.last_run_outputs[node_id] = result
        return result

    async def _execute_fsm(self, input_text: str) -> str:
        if not self.mas_config.fsm:
            raise ValueError("[OrlaFlow] mas_config.fsm is required when topology='fsm'.")
        fsm = self.mas_config.fsm
        decision_nodes = {cfg.node: cfg for cfg in fsm.decision_nodes}
        predecessor_map = build_predecessors_map(self.mas_config.routes)
        compiled_nodes = self._compiler.compile_nodes(predecessor_map)
        tracer.set_topology("fsm")

        outputs: Dict[str, str] = {}
        current_id = fsm.start
        rounds = 0
        self.last_run_outputs = {}
        self.last_terminal_node = None
        self.last_terminal_output = None
        self.last_run_rounds = 0

        while True:
            if rounds >= self.mas_config.convergence.max_rounds:
                self.last_run_rounds = rounds
                return self.last_terminal_output or ""

            if compiled_nodes[current_id].is_external:
                node_input = await self._build_fsm_node_input(current_id, input_text, outputs)
                self._reset_agent(current_id)
                start = time.time()
                output = await self._run_runtime_node(current_id, node_input)
                end = time.time()
                tracer.record_agent_run_phase(
                    agent_id=current_id,
                    start_time=start,
                    end_time=end,
                    parallel=False,
                )
                outputs[current_id] = output
                terminal_id = current_id
            else:
                span_node_ids = self._collect_fsm_orla_span(
                    start_node_id=current_id,
                    decision_nodes=decision_nodes,
                    compiled_nodes=compiled_nodes,
                    max_nodes=self.mas_config.convergence.max_rounds - rounds,
                )
                span_input = await self._build_fsm_node_input(current_id, input_text, outputs)
                stage_specs = self._build_fsm_span_stage_specs(
                    segment_id=f"fsm_span_{rounds}",
                    span_node_ids=span_node_ids,
                    root_input=input_text,
                    initial_request_text=span_input,
                    output_store=outputs,
                    predecessor_map=predecessor_map,
                    compiled_nodes=compiled_nodes,
                )
                for node_id in span_node_ids:
                    self._reset_agent(node_id)
                results, start, end, _ = await self._execute_stage_specs(
                    workflow_tags={"runtime": "orla", "topology": "fsm", "segment_id": f"fsm_span_{rounds}"},
                    stage_specs=stage_specs,
                    record_hidden=False,
                )
                tracer.record_orla_segment_phase(
                    agent_ids=span_node_ids,
                    segment_id=f"fsm_span_{rounds}",
                    start_time=start,
                    end_time=end,
                )
                for node_id in span_node_ids:
                    compiled_node = compiled_nodes[node_id]
                    stage_name_lookup = {
                        stage_id: stage_recipe.name
                        for stage_id, stage_recipe in zip(
                            compiled_node.recipe_stage_ids,
                            compiled_node.recipe.stages,
                        )
                    }
                    request_text = self._compute_fsm_span_node_request(
                        node_id=node_id,
                        span_start_id=current_id,
                        initial_request_text=span_input,
                        root_input=input_text,
                        output_store=outputs,
                        dep_results=results,
                        predecessor_map=predecessor_map,
                        compiled_nodes=compiled_nodes,
                    )
                    outputs[node_id] = self._finalize_recipe_node(
                        instance_id=node_id,
                        recipe=compiled_node.recipe,
                        request_text=request_text,
                        results=results,
                        recipe_stage_ids=compiled_node.recipe_stage_ids,
                        stage_name_lookup=stage_name_lookup,
                        fallback_stage_id=compiled_node.exit_stage_id,
                    )
                terminal_id = span_node_ids[-1]
                output = outputs[terminal_id]

            self.last_run_outputs = dict(outputs)
            self.last_terminal_node = terminal_id
            self.last_terminal_output = output
            span_cost = 1 if compiled_nodes[terminal_id].is_external else len(span_node_ids)
            self.last_run_rounds = rounds + span_cost
            rounds += span_cost

            decision_cfg = decision_nodes.get(terminal_id)
            if decision_cfg is not None:
                next_id = self._resolve_decision_next(output, decision_cfg)
                if next_id is None:
                    return output
                current_id = next_id
                continue

            successors = self.get_successors(terminal_id)
            if not successors:
                return output
            if len(successors) > 1:
                raise ValueError(
                    f"[OrlaFlow] Non-decision node '{terminal_id}' has multiple successors: {successors}."
                )
            current_id = successors[0]

    async def _build_fsm_node_input(
        self,
        instance_id: str,
        root_input: str,
        outputs: Dict[str, str],
    ) -> str:
        preds = self.get_predecessors(instance_id)
        if not preds:
            return root_input
        pred_outputs = [outputs[pred] for pred in preds if pred in outputs]
        if not pred_outputs:
            return root_input
        aggregated = await self._aggregate_inputs(instance_id, pred_outputs)
        return self._append_upstream_outputs(root_input=root_input, aggregated=aggregated)

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
                        f"[OrlaFlow] Decision '{transition.value}' for node '{decision_cfg.node}' "
                        "is non-terminating but missing a target."
                    )
                return transition.target
        raise ValueError(
            f"[OrlaFlow] Node '{decision_cfg.node}' emitted unmatched decision '{decision_value}'."
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
