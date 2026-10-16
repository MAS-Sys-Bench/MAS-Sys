"""FlowFactory: assembles a runnable Flow from a MASConfig.

Steps:
1. For each AgentNodeConfig, import the agent class from AGENT_METADATA.
2. Instantiate with name=instance_id (supports homogeneous multi-instance).
3. Filter the agent's available_tools to exactly the PlanAgent-authorized intersection.
4. Build agents dict and instantiate the topology-appropriate Flow.
5. Register memory permissions (delegated to BaseFlow.__init__).
"""

import importlib
from typing import TYPE_CHECKING, Any, Dict

from app.agent.agents_config import AGENT_METADATA
from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.logger import logger
from app.mas.schema import AgentNodeConfig, MASConfig, TopologyType
from app.memory.base import BaseMemory
from app.memory.global_memory import GlobalMemory
from app.tool.tool_collection import ToolCollection

if TYPE_CHECKING:
    from app.mas.plan_agent import PlanAgent


_TOOL_CONTRACT_HEADER = "Runtime Tool Contract:"


def _load_agent_class(agent_key: str):
    """Dynamically import and return the agent class for agent_key."""
    meta = AGENT_METADATA.get(agent_key)
    if not meta:
        raise ValueError(f"Unknown agent_key '{agent_key}'. Not found in AGENT_METADATA.")
    module = importlib.import_module(meta["module"])
    return getattr(module, meta["class_name"])


def _sync_tool_contract_prompt(agent: BaseAgent) -> None:
    """Inject the exact runtime tool inventory into the agent prompt."""
    if not hasattr(agent, "available_tools") or not isinstance(agent.available_tools, ToolCollection):
        return

    tool_names = [tool.name for tool in agent.available_tools.tools]
    available = ", ".join(tool_names) if tool_names else "none"
    contract = (
        f"{_TOOL_CONTRACT_HEADER}\n"
        f"- Available tool names for this run: {available}\n"
        "- Call only these exact tool names.\n"
        "- If a capability is unavailable, explain that limitation instead of inventing another tool."
    )
    base_prompt = getattr(agent, "system_prompt", "") or ""
    if _TOOL_CONTRACT_HEADER in base_prompt:
        return
    agent.system_prompt = f"{contract}\n\n{base_prompt}".strip()


def _apply_tool_filter(agent: BaseAgent, allowed_tools: list[str]) -> None:
    """Restrict agent.available_tools to the intersection of allowed_tools and its current tools.

    If allowed_tools is empty, the agent receives no tools.
    """
    if not hasattr(agent, "available_tools") or not isinstance(agent.available_tools, ToolCollection):
        return

    allowed = set(allowed_tools or [])
    filtered = [t for t in agent.available_tools.tools if t.name in allowed]
    agent.available_tools = ToolCollection(*filtered)
    _sync_tool_contract_prompt(agent)
    logger.info(f"[FlowFactory] Agent '{agent.name}' tools restricted to: {[t.name for t in filtered]}")


def _instantiate_agent(
    node: AgentNodeConfig,
    *,
    skip_tool_filter: bool = False,
) -> BaseAgent:
    """Create an agent instance from AgentNodeConfig.

    Any extra keys in AGENT_METADATA beyond the standard fields
    (class_name, module, description, available_tools, llm_type) are forwarded
    as keyword arguments to the agent constructor.
    """
    meta = AGENT_METADATA.get(node.agent_key, {})
    _standard_keys = {"class_name", "module", "description", "available_tools", "llm_type"}
    extra_kwargs = {k: v for k, v in meta.items() if k not in _standard_keys}

    agent_cls = _load_agent_class(node.agent_key)
    agent = agent_cls(
        name=node.instance_id,
        llm_config_name=node.llm_config_name,
        **extra_kwargs,
    )
    agent.agent_params = dict(node.agent_params)
    agent.initialize_from_params()
    agent.allowed_tools_config = list(node.allowed_tools)
    if not skip_tool_filter:
        _apply_tool_filter(agent, node.allowed_tools)
    return agent


def _get_flow_class(topology: TopologyType):
    """Return the Flow class for the given topology type (lazy import to avoid circular deps)."""
    if topology == TopologyType.CHAIN:
        from app.flow.chain import ChainFlow

        return ChainFlow
    if topology == TopologyType.FSM:
        from app.flow.fsm import FSMFlow

        return FSMFlow
    if topology == TopologyType.STAR:
        from app.flow.star import StarFlow

        return StarFlow
    if topology == TopologyType.TREE:
        from app.flow.tree import TreeFlow

        return TreeFlow
    if topology == TopologyType.DAG:
        from app.flow.dag import DAGFlow

        return DAGFlow
    raise ValueError(f"No Flow class registered for topology '{topology}'.")


class FlowFactory:
    """Assemble a runnable MAS Flow from a MASConfig."""

    @staticmethod
    def from_config(
        mas_config: MASConfig,
        public_memory: BaseMemory | None = None,
        plan_agent: "PlanAgent | None" = None,
        runtime: str = "legacy",
    ) -> BaseFlow:
        """Build and return a fully configured Flow instance.

        Args:
            mas_config: Complete MAS description (produced by PlanAgent).
            public_memory: Shared memory instance. Defaults to GlobalMemory.
            plan_agent: Optional PlanAgent that may be shared by the caller for
                planning before this flow is constructed. Runtime coordination is
                handled by flow agents themselves.
        Returns:
            A BaseFlow subclass instance ready to call .execute(input_text).
        """
        # Build agents dict: instance_id → agent instance
        agents: Dict[str, BaseAgent] = {}
        for node in mas_config.agents:
            agent = _instantiate_agent(node, skip_tool_filter=False)
            agents[node.instance_id] = agent
            logger.info(f"[FlowFactory] Instantiated '{node.instance_id}' ({node.agent_key})")

        if runtime == "orla_graph":
            from app.flow.orla import OrlaFlow

            flow_cls = OrlaFlow
        else:
            flow_cls = _get_flow_class(mas_config.topology)
        memory = public_memory or GlobalMemory()

        flow = flow_cls(
            agents=agents,
            mas_config=mas_config,
            public_memory=memory,
        )
        logger.info(
            f"[FlowFactory] Created {flow_cls.__name__} with {len(agents)} agents "
            f"(runtime={runtime})."
        )

        return flow
