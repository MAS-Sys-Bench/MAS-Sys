import threading
from typing import Any, Dict, List, Optional

from app.memory.base import BaseMemory


class GlobalMemory(BaseMemory):
    """Thread-safe in-process global shared memory backed by a plain dict.

    Storage: _store[namespace][key] = value.
    One threading.Lock per namespace (sufficient for single-thread asyncio callers).
    Replace with a Redis/vector-store subclass for persistence or semantic retrieval.
    """

    def __init__(self) -> None:
        super().__init__()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def _get_lock(self, namespace: str) -> threading.Lock:
        if namespace not in self._locks:
            self._locks[namespace] = threading.Lock()
        return self._locks[namespace]

    def _get_ns(self, namespace: str) -> Dict[str, Any]:
        if namespace not in self._store:
            self._store[namespace] = {}
        return self._store[namespace]

    def write(self, key: str, value: Any, agent_id: str, namespace: str = "default") -> None:
        self._assert_write(agent_id)
        with self._get_lock(namespace):
            self._get_ns(namespace)[key] = value

    def read(self, key: str, agent_id: str, namespace: str = "default") -> Optional[Any]:
        self._assert_read(agent_id)
        with self._get_lock(namespace):
            return self._get_ns(namespace).get(key)

    def read_all(self, agent_id: str, namespace: str = "default") -> Dict[str, Any]:
        self._assert_read(agent_id)
        with self._get_lock(namespace):
            return self._get_ns(namespace).copy()

    def delete(self, key: str, agent_id: str, namespace: str = "default") -> bool:
        self._assert_write(agent_id)
        with self._get_lock(namespace):
            ns = self._get_ns(namespace)
            if key in ns:
                del ns[key]
                return True
            return False

    def keys(self, agent_id: str, namespace: str = "default") -> List[str]:
        self._assert_read(agent_id)
        with self._get_lock(namespace):
            return sorted(self._get_ns(namespace).keys())

    def clear_namespace(self, namespace: str = "default") -> None:
        with self._get_lock(namespace):
            self._store[namespace] = {}
