"""Topology registry metadata exposed to PlanAgent."""

TOPOLOGY_METADATA = {
    "chain": (
        "linear pipeline A→B→C. Use routes for explicit order; if routes are "
        "empty, declaration order is used. Set fsm to null."
    ),
    "star": (
        "fan-out/fan-in star. Workers execute first in parallel on the shared input, "
        "then the center node aggregates their outputs. "
        'topology_params must contain {"center": "<instance_id>"}. Set fsm to null.'
    ),
    "tree": (
        "hierarchical root→children structure with delegated subtasks. "
        'topology_params must contain {"root": "<instance_id>"}. Set fsm to null.'
    ),
    "dag": (
        "static directed acyclic graph executed in topological generations; "
        "nodes in the same generation run in parallel. Routes must be acyclic. Set fsm to null."
    ),
    "fsm": (
        "explicit finite-state machine with decision nodes and bounded loops. "
        "Provide routes, allow cycles when needed, and include an fsm object with "
        "start plus decision_nodes/transitions. max_rounds must be "
        "large enough to reach the first terminal or decision node at least once; "
        "do not set it below the number of sequential node executions required "
        "for one full pass through the planned FSM; "
        "must include at least one critic/verifier-style decision node in fsm.decision_nodes "
        "that decides whether the workflow should continue or finish."
    ),
}


def build_topologies_info_for_prompt() -> str:
    return "\n".join(f"- {key}: {desc}" for key, desc in TOPOLOGY_METADATA.items())
