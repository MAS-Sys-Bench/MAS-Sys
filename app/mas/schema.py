from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class TopologyType(str, Enum):
    """Supported MAS topology structures."""
    CHAIN = "chain"      # A → B → C linear pipeline
    FSM = "fsm"          # explicit finite-state execution graph; supports cycles via decision nodes
    STAR = "star"        # center dispatches to N workers, aggregates results
    TREE = "tree"        # root output propagates downward via DFS
    DAG = "dag"          # static DAG, topologically sorted execution


class MemoryPermission(BaseModel):
    """Read/write access for one agent on the shared BaseMemory."""
    can_read: bool = False
    can_write: bool = False


class ConvergenceCriteria(BaseModel):
    """Termination settings for bounded multi-round flows."""

    max_rounds: int = Field(default=6)
    max_rounds_per_agent: Optional[int] = Field(default=None)
    llm_check: bool = Field(default=False)
    llm_check_prompt: Optional[str] = Field(default=None)
    llm_assign_tasks: bool = Field(default=True)


class InputAggregation(str, Enum):
    """How a node merges multiple predecessor outputs."""

    CONCAT = "concat"
    LLM_MERGE = "llm_merge"
    VOTE = "vote"


class AgentNodeConfig(BaseModel):
    """Configuration for one agent node in the MAS."""
    agent_key: str = Field(description="Type key in AGENT_METADATA (e.g. 'manus', 'swe').")
    instance_id: str = Field(description="Unique instance name within this MAS (e.g. 'debater_0').")
    llm_config_name: str = Field(description="LLM config key assigned by PlanAgent (e.g. 'lang.qwen2_5-14b', 'vision.qwen3-max', 'image.qwen-image').")
    role_description: str = Field(default="", description="Role description injected into system prompt.")
    agent_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Instance-specific parameters for this agent node, such as bio or papers.",
    )
    allowed_tools: List[str] = Field(default_factory=list, description="Tool whitelist; factory intersects with agent class tools.")
    memory_permission: MemoryPermission = Field(default_factory=MemoryPermission)
    input_aggregation: InputAggregation = Field(default=InputAggregation.CONCAT)


class MessageRoute(BaseModel):
    """Directed communication edge between two agent instances."""
    source: str = Field(description="Sender instance_id.")
    target: str = Field(description="Receiver instance_id.")


class FSMTransition(BaseModel):
    """One decision outcome in an FSM node."""
    value: str = Field(description="Decision token emitted by the node, e.g. 'continue' or 'finish'.")
    target: Optional[str] = Field(
        default=None,
        description="Next node for this decision. Leave null when terminate=true.",
    )
    terminate: bool = Field(
        default=False,
        description="Whether this decision ends the FSM execution.",
    )


class FSMDecisionNode(BaseModel):
    """How one node controls branching in an FSM topology."""
    node: str = Field(description="Decision-making node instance_id.")
    decision_field: str = Field(
        default="decision",
        description="JSON field to read from the node output.",
    )
    transitions: List[FSMTransition] = Field(
        default_factory=list,
        description="Decision-to-next-node mapping for this node.",
    )


class FSMConfig(BaseModel):
    """Execution control for the generic FSM topology."""
    start: str = Field(description="First node to execute.")
    decision_nodes: List[FSMDecisionNode] = Field(
        default_factory=list,
        description="Decision nodes that determine looping or termination.",
    )


class MASConfig(BaseModel):
    """Complete MAS description produced by PlanAgent and consumed by FlowFactory."""
    topology: TopologyType
    topology_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Topology-specific parameters; concrete requirements live in app.flow.flow_config metadata.",
    )
    agents: List[AgentNodeConfig]
    routes: List[MessageRoute] = Field(
        default_factory=list,
        description="Directed message routes between agent instances.",
    )
    fsm: Optional[FSMConfig] = Field(
        default=None,
        description="FSM-specific control config. Required when topology='fsm'.",
    )
    max_rounds: int = Field(
        default=6,
        description=(
            "Hard upper bound on rounds. For FSM, this counts decision-node evaluations "
            "(for example critic/verifier judgments), not raw node executions."
        ),
    )
    convergence: ConvergenceCriteria = Field(default_factory=ConvergenceCriteria)
    mas_context: str = Field(default="", description="Global mission context injected into every agent's system prompt.")

    @model_validator(mode="after")
    def _sync_convergence(self) -> "MASConfig":
        max_rounds_explicit = "max_rounds" in self.model_fields_set
        convergence_explicit = "convergence" in self.model_fields_set
        convergence_max_rounds_explicit = (
            self.convergence is not None
            and "max_rounds" in self.convergence.model_fields_set
        )

        if self.convergence is None:
            self.convergence = ConvergenceCriteria(max_rounds=self.max_rounds)
            return self

        if max_rounds_explicit and convergence_max_rounds_explicit:
            if self.max_rounds != self.convergence.max_rounds:
                raise ValueError(
                    "Conflicting max_rounds values were provided in MASConfig.max_rounds "
                    f"({self.max_rounds}) and MASConfig.convergence.max_rounds "
                    f"({self.convergence.max_rounds})."
                )
        elif max_rounds_explicit:
            self.convergence.max_rounds = self.max_rounds
        elif convergence_explicit and convergence_max_rounds_explicit:
            self.max_rounds = self.convergence.max_rounds
        else:
            # Keep both views synchronized when only defaults are present.
            self.convergence.max_rounds = self.max_rounds

        self.max_rounds = self.convergence.max_rounds
        return self
