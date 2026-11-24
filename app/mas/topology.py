"""Graph algorithm utilities for MAS topology validation and execution ordering."""

from collections import deque
from typing import Dict, List

from app.mas.schema import MessageRoute


def build_adjacency_list(routes: List[MessageRoute]) -> Dict[str, List[str]]:
    """Build {source: [targets]} from route list."""
    adj: Dict[str, List[str]] = {}
    for r in routes:
        adj.setdefault(r.source, []).append(r.target)
    return adj


def build_predecessors_map(routes: List[MessageRoute]) -> Dict[str, List[str]]:
    """Build {target: [sources]} from route list."""
    preds: Dict[str, List[str]] = {}
    for r in routes:
        preds.setdefault(r.target, []).append(r.source)
    return preds


def topological_sort(node_ids: List[str], routes: List[MessageRoute]) -> List[str]:
    """Return nodes in topological order using Kahn's algorithm.

    Raises ValueError if a cycle is detected (i.e., routes do not form a DAG).
    """
    adj = build_adjacency_list(routes)
    in_degree: Dict[str, int] = {n: 0 for n in node_ids}
    for r in routes:
        if r.target in in_degree:
            in_degree[r.target] += 1

    queue = deque(n for n in node_ids if in_degree[n] == 0)
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for successor in adj.get(node, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(order) != len(node_ids):
        raise ValueError(
            f"Cycle detected in DAG routes. Only {len(order)}/{len(node_ids)} nodes reachable."
        )
    return order


def has_cycle(node_ids: List[str], routes: List[MessageRoute]) -> bool:
    """Return True if the given routes contain a directed cycle."""
    try:
        topological_sort(node_ids, routes)
        return False
    except ValueError:
        return True


def dfs_order(root: str, adj: Dict[str, List[str]]) -> List[str]:
    """Return nodes in DFS pre-order starting from root (for TreeFlow traversal)."""
    order: List[str] = []
    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for child in reversed(adj.get(node, [])):
            if child not in visited:
                stack.append(child)
    return order


def get_node_depths(root: str, adj: Dict[str, List[str]]) -> Dict[str, int]:
    """Return {node: depth} using BFS from root (depth of root = 0)."""
    depths: Dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in adj.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def topological_generations(node_ids: List[str], routes: List[MessageRoute]) -> List[List[str]]:
    """Return nodes grouped into layers (generations) by topological order.

    Each layer contains nodes whose predecessors all belong to earlier layers,
    so nodes within the same layer can be executed in parallel.

    Raises ValueError if a cycle is detected.
    """
    adj = build_adjacency_list(routes)
    in_degree: Dict[str, int] = {n: 0 for n in node_ids}
    for r in routes:
        if r.target in in_degree:
            in_degree[r.target] += 1

    current_layer = [n for n in node_ids if in_degree[n] == 0]
    generations: List[List[str]] = []
    visited_count = 0

    while current_layer:
        generations.append(current_layer)
        visited_count += len(current_layer)
        next_layer: List[str] = []
        for node in current_layer:
            for successor in adj.get(node, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    next_layer.append(successor)
        current_layer = next_layer

    if visited_count != len(node_ids):
        raise ValueError(
            f"Cycle detected in DAG routes. Only {visited_count}/{len(node_ids)} nodes reachable."
        )
    return generations
