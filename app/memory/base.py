from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemory(ABC):
    """Abstract shared memory for MAS agents with per-agent permission control.

    Subclasses can swap the backend (Redis, vector store, etc.) without
    touching any Flow or Agent code.
    """

    def __init__(self) -> None:
        self._permissions: Dict[str, Dict[str, bool]] = {}

    def register_agent(self, instance_id: str, can_read: bool = False, can_write: bool = False) -> None:
        """Register an agent's read/write permissions. Called by FlowFactory at startup."""
        self._permissions[instance_id] = {"can_read": can_read, "can_write": can_write}

    def _assert_read(self, agent_id: str) -> None:
        if not self._permissions.get(agent_id, {}).get("can_read", False):
            raise PermissionError(f"Agent '{agent_id}' has no read permission on BaseMemory.")

    def _assert_write(self, agent_id: str) -> None:
        if not self._permissions.get(agent_id, {}).get("can_write", False):
            raise PermissionError(f"Agent '{agent_id}' has no write permission on BaseMemory.")

    @abstractmethod
    def write(self, key: str, value: Any, agent_id: str, namespace: str = "default") -> None:
        """Write value. Raises PermissionError if agent lacks write permission."""

    @abstractmethod
    def read(self, key: str, agent_id: str, namespace: str = "default") -> Optional[Any]:
        """Read value. Returns None if key missing. Raises PermissionError if no read permission."""

    @abstractmethod
    def read_all(self, agent_id: str, namespace: str = "default") -> Dict[str, Any]:
        """Return shallow copy of all entries in namespace. Raises PermissionError if no read permission."""

    @abstractmethod
    def delete(self, key: str, agent_id: str, namespace: str = "default") -> bool:
        """Delete key. Returns True if existed. Raises PermissionError if no write permission."""

    @abstractmethod
    def keys(self, agent_id: str, namespace: str = "default") -> List[str]:
        """Return sorted key list. Raises PermissionError if no read permission."""

    def clear_namespace(self, namespace: str = "default") -> None:
        """Remove all entries in namespace (admin, no permission check)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement clear_namespace.")
