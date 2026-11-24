from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.mas.schema import InputAggregation, MASConfig
from app.orla.recipes import NodeRecipe, StageRecipe


@dataclass
class CompiledStage:
    stage_id: str
    logical_node_id: str
    backend_name: str
    kind: str
    stage_name: str
    stage_recipe: Optional[StageRecipe] = None
    dependencies: list[str] = field(default_factory=list)
    hidden: bool = False


@dataclass
class CompiledNode:
    instance_id: str
    backend_name: Optional[str]
    recipe: NodeRecipe
    predecessors: list[str]
    input_aggregation: InputAggregation
    stage_ids: list[str] = field(default_factory=list)
    recipe_stage_ids: list[str] = field(default_factory=list)
    entry_stage_id: Optional[str] = None
    exit_stage_id: Optional[str] = None
    merge_stage_id: Optional[str] = None

    @property
    def is_external(self) -> bool:
        return self.recipe.kind == "external"


@dataclass
class CompiledSegment:
    segment_id: str
    node_ids: list[str]
    stage_ids: list[str]
    stages: dict[str, CompiledStage]

    @property
    def backend_names(self) -> list[str]:
        seen: list[str] = []
        for stage_id in self.stage_ids:
            backend_name = self.stages[stage_id].backend_name
            if backend_name not in seen:
                seen.append(backend_name)
        return seen


class OrlaCompiler:
    """Compile MAS logical nodes into ORLA-executable stage graphs."""

    def __init__(self, mas_config: MASConfig, recipes: Dict[str, NodeRecipe]):
        self.mas_config = mas_config
        self.recipes = recipes
        self.node_map = {node.instance_id: node for node in mas_config.agents}

    def compile_nodes(self, predecessor_map: Dict[str, List[str]]) -> Dict[str, CompiledNode]:
        compiled: dict[str, CompiledNode] = {}
        for node in self.mas_config.agents:
            recipe = self.recipes.get(node.instance_id, NodeRecipe(kind="external"))
            compiled_node = CompiledNode(
                instance_id=node.instance_id,
                backend_name=node.llm_config_name if recipe.kind != "external" else None,
                recipe=recipe,
                predecessors=list(predecessor_map.get(node.instance_id, [])),
                input_aggregation=node.input_aggregation,
            )

            if recipe.kind == "external":
                compiled[node.instance_id] = compiled_node
                continue

            stages: list[CompiledStage] = []
            if (
                len(compiled_node.predecessors) > 1
                and compiled_node.input_aggregation == InputAggregation.LLM_MERGE
            ):
                merge_stage_id = f"{node.instance_id}__coord__llm_merge"
                stages.append(
                    CompiledStage(
                        stage_id=merge_stage_id,
                        logical_node_id=node.instance_id,
                        backend_name=node.llm_config_name,
                        kind="merge",
                        stage_name="llm_merge",
                        hidden=True,
                    )
                )
                compiled_node.merge_stage_id = merge_stage_id
                compiled_node.entry_stage_id = merge_stage_id

            previous_stage_id = compiled_node.merge_stage_id
            for stage_recipe in recipe.stages:
                stage_id = f"{node.instance_id}__{stage_recipe.name}"
                stage = CompiledStage(
                    stage_id=stage_id,
                    logical_node_id=node.instance_id,
                    backend_name=node.llm_config_name,
                    kind="recipe",
                    stage_name=stage_recipe.name,
                    stage_recipe=stage_recipe,
                    dependencies=[previous_stage_id] if previous_stage_id else [],
                )
                stages.append(stage)
                compiled_node.stage_ids.append(stage_id)
                compiled_node.recipe_stage_ids.append(stage_id)
                if compiled_node.entry_stage_id is None:
                    compiled_node.entry_stage_id = stage_id
                previous_stage_id = stage_id

            compiled_node.stage_ids = [stage.stage_id for stage in stages]
            compiled_node.exit_stage_id = previous_stage_id
            compiled[node.instance_id] = compiled_node

        return compiled

    def build_connected_segments(
        self,
        *,
        node_order: List[str],
        predecessor_map: Dict[str, List[str]],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> list[CompiledSegment]:
        orla_nodes = {
            node_id for node_id, compiled_node in compiled_nodes.items() if not compiled_node.is_external
        }
        if not orla_nodes:
            return []

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in orla_nodes}
        for node_id, preds in predecessor_map.items():
            if node_id not in orla_nodes:
                continue
            for pred in preds:
                if pred in orla_nodes:
                    adjacency[node_id].add(pred)
                    adjacency[pred].add(node_id)

        components: list[list[str]] = []
        visited: set[str] = set()
        for node_id in node_order:
            if node_id not in orla_nodes or node_id in visited:
                continue
            stack = [node_id]
            component: list[str] = []
            visited.add(node_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append([candidate for candidate in node_order if candidate in set(component)])

        return [
            self.build_explicit_segment(
                segment_id=f"segment_{index}",
                node_ids=component,
                predecessor_map=predecessor_map,
                compiled_nodes=compiled_nodes,
            )
            for index, component in enumerate(components)
        ]

    def build_explicit_segment(
        self,
        *,
        segment_id: str,
        node_ids: List[str],
        predecessor_map: Dict[str, List[str]],
        compiled_nodes: Dict[str, CompiledNode],
    ) -> CompiledSegment:
        node_id_set = set(node_ids)
        stages: dict[str, CompiledStage] = {}
        stage_ids: list[str] = []

        for node_id in node_ids:
            compiled_node = compiled_nodes[node_id]
            for stage_id in compiled_node.stage_ids:
                base_stage = self._find_stage(compiled_node, stage_id)
                stage = CompiledStage(
                    stage_id=base_stage.stage_id,
                    logical_node_id=base_stage.logical_node_id,
                    backend_name=base_stage.backend_name,
                    kind=base_stage.kind,
                    stage_name=base_stage.stage_name,
                    stage_recipe=base_stage.stage_recipe,
                    dependencies=list(base_stage.dependencies),
                    hidden=base_stage.hidden,
                )
                stages[stage_id] = stage
                stage_ids.append(stage_id)

        for node_id in node_ids:
            compiled_node = compiled_nodes[node_id]
            if compiled_node.is_external or not compiled_node.entry_stage_id:
                continue
            entry_stage_id = compiled_node.merge_stage_id or compiled_node.entry_stage_id
            for pred in predecessor_map.get(node_id, []):
                if pred not in node_id_set:
                    continue
                pred_exit_stage_id = compiled_nodes[pred].exit_stage_id
                if pred_exit_stage_id and pred_exit_stage_id not in stages[entry_stage_id].dependencies:
                    stages[entry_stage_id].dependencies.append(pred_exit_stage_id)

        return CompiledSegment(
            segment_id=segment_id,
            node_ids=list(node_ids),
            stage_ids=stage_ids,
            stages=stages,
        )

    @staticmethod
    def _find_stage(compiled_node: CompiledNode, stage_id: str) -> CompiledStage:
        stage_lookup: dict[str, CompiledStage] = {}
        previous_stage_id = compiled_node.merge_stage_id
        if compiled_node.merge_stage_id and compiled_node.backend_name is not None:
            stage_lookup[compiled_node.merge_stage_id] = CompiledStage(
                stage_id=compiled_node.merge_stage_id,
                logical_node_id=compiled_node.instance_id,
                backend_name=compiled_node.backend_name,
                kind="merge",
                stage_name="llm_merge",
                hidden=True,
            )
        for recipe_stage_id, stage_recipe in zip(compiled_node.recipe_stage_ids, compiled_node.recipe.stages):
            stage_lookup[recipe_stage_id] = CompiledStage(
                stage_id=recipe_stage_id,
                logical_node_id=compiled_node.instance_id,
                backend_name=compiled_node.backend_name or "",
                kind="recipe",
                stage_name=stage_recipe.name,
                stage_recipe=stage_recipe,
                dependencies=[previous_stage_id] if previous_stage_id else [],
            )
            previous_stage_id = recipe_stage_id
        return stage_lookup[stage_id]
