from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from weakref import WeakKeyDictionary

from app.config import config
from app.schema import Function, Message, ToolCall
from app.tracer import tracer


class OrlaRuntimeError(RuntimeError):
    """Raised when the ORLA runtime cannot be initialized or executed."""


@dataclass
class PyOrlaModules:
    """Compatibility bundle for whichever ORLA SDK implementation is active."""

    OrlaClient: Any
    OrlaError: Any
    Workflow: Any
    Stage: Any
    StageResultData: Any
    LLMBackend: Any
    CostModel: Any
    Message: Any
    new_tool: Any
    ToolResult: Any
    EXECUTION_MODE_AGENT_LOOP: str


@dataclass
class _ImportedPyOrla:
    CostModel: Any
    EXECUTION_MODE_AGENT_LOOP: Any
    LLMBackend: Any
    Message: Any
    OrlaClient: Any
    OrlaError: Any
    Stage: Any
    StageResultData: Any
    ToolResult: Any
    Workflow: Any
    new_tool: Any
    random_workflow_id: Any


class _AsyncToolRunnerBridge:
    """Run async tool coroutines on a shared background event loop."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="orla-tool-runner",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_DEFAULT_MAX_AGENT_LOOP_TURNS = 100


def _response_metrics_dict(response: Any) -> dict[str, Any]:
    metrics = getattr(response, "metrics", None)
    if metrics is None:
        return {}
    raw = getattr(metrics, "raw", None)
    if isinstance(raw, dict):
        payload: dict[str, Any] = dict(raw)
    else:
        payload = {}
    fields = (
        "ttft_ms",
        "tpot_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "queue_wait_ms",
        "scheduler_decision_ms",
        "dispatch_ms",
        "backend_latency_ms",
        "estimated_cost_usd",
        "backend_ttft_s",
        "queue_wait_s",
        "backend_latency_s",
        "backend_prefill_start_ts",
        "kv_cache_reuse_rate",
        "cache_reuse_rate",
        "cached_tokens",
        "num_cached_tokens",
    )
    for field in fields:
        value = getattr(metrics, field, None)
        if value is not None and field not in payload:
            payload[field] = value
    return payload


def _llm_identity_from_stage(stage: Any) -> tuple[str, str]:
    tags = dict(getattr(stage, "tags", {}) or {})
    backend_name = tags.get("backend_name", "")
    if not backend_name:
        return "", ""
    try:
        llm_settings = config.llm[backend_name]
    except Exception:  # noqa: BLE001
        return "", ""
    return llm_settings.model, llm_settings.base_url


def _serialize_stage_tools(stage: Any) -> list[dict[str, Any]]:
    tools = []
    for name, tool in dict(getattr(stage, "tools", {}) or {}).items():
        tools.append(
            {
                "name": name,
                "description": getattr(tool, "description", ""),
                "input_schema": getattr(tool, "input_schema", {}),
            }
        )
    return tools


def _patched_stage_class(raw_stage_cls: Any, raw_tool_result_cls: Any) -> Any:
    """Patch ORLA Stage with MAS-Sys tool-call tracing and error handling."""

    class CompatStage(raw_stage_cls):
        def run_tool_call(self, tool_call: Any) -> Any:
            call_start = time.time()
            trace_ctx = getattr(self, "_orla_trace_context", {}) or {}
            tool = self.tools.get(tool_call.name)
            if tool is None:
                result = raw_tool_result_cls(
                    id=tool_call.id,
                    name=tool_call.name,
                    error=f"unknown tool {tool_call.name!r}",
                    is_error=True,
                )
            elif tool.run is None:
                result = raw_tool_result_cls(
                    id=tool_call.id,
                    name=tool_call.name,
                    error=f"tool {tool_call.name!r} has no runner",
                    is_error=True,
                )
            else:
                try:
                    result = tool.run(tool_call.input_arguments)
                except Exception as exc:  # noqa: BLE001
                    result = raw_tool_result_cls(
                        id=tool_call.id,
                        name=tool_call.name,
                        error=str(exc),
                        is_error=True,
                    )
                else:
                    result.id = tool_call.id
                    result.name = tool_call.name

            call_end = time.time()
            output_values = getattr(result, "output_values", None)
            tracer.record_orla_tool_call(
                workflow_id=trace_ctx.get("workflow_id", getattr(self, "_workflow_id", "")),
                stage_id=getattr(self, "id", ""),
                turn_index=trace_ctx.get("turn_index"),
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                start_time=call_start,
                end_time=call_end,
                extra={
                    "is_error": bool(getattr(result, "is_error", False)),
                    "error": getattr(result, "error", ""),
                    "input_arg_keys": sorted((tool_call.input_arguments or {}).keys()),
                    "output_keys": (
                        sorted(output_values.keys()) if isinstance(output_values, dict) else []
                    ),
                },
            )
            return result

    return CompatStage


def _patched_workflow_class(raw_workflow_cls: Any, random_workflow_id: Any) -> Any:
    """Patch ORLA Workflow with MAS-Sys workflow tracing semantics."""

    class CompatWorkflow(raw_workflow_cls):
        def _execute_agent_loop(self, stage: Any, dep_results: dict[str, Any]) -> Any:
            pyorla = load_pyorla()
            messages: list[Any]

            if stage.messages_builder is not None:
                messages = stage.messages_builder(dep_results)
            else:
                prompt = stage.prompt
                if stage.prompt_builder is not None:
                    prompt = stage.prompt_builder(dep_results)
                if not prompt:
                    raise ValueError(f"stage {stage.name!r}: prompt is empty")
                messages = [pyorla.Message(role="user", content=prompt)]

            max_turns = (
                stage.max_turns
                if getattr(stage, "max_turns", 0) > 0
                else _DEFAULT_MAX_AGENT_LOOP_TURNS
            )
            last_resp: Any = None

            for turn_index in range(1, max_turns + 1):
                turn_start = time.time()
                message_count_before = len(messages)
                model, base_url = _llm_identity_from_stage(stage)
                tracer.record_llm_call(
                    transport="orla",
                    invocation="agent_loop_turn",
                    messages=messages,
                    model=model,
                    base_url=base_url,
                    tools=_serialize_stage_tools(stage),
                    extra={
                        "runtime": "orla",
                        "workflow_id": getattr(stage, "_workflow_id", ""),
                        "stage_id": getattr(stage, "id", ""),
                        "turn_index": turn_index,
                        "tags": dict(getattr(stage, "tags", {}) or {}),
                    },
                )
                llm_start = time.time()
                resp = stage.execute_with_messages(messages)
                llm_end = time.time()
                last_resp = resp

                tool_calls = list(getattr(resp, "tool_calls", []) or [])
                assistant_content = getattr(resp, "content", "") or ""
                messages.append(
                    pyorla.Message(
                        role="assistant",
                        content=assistant_content,
                        tool_calls=tool_calls,
                    )
                )

                tool_results: list[Any] = []
                tool_start: float | None = None
                tool_end: float | None = None
                if tool_calls:
                    tool_start = time.time()
                    previous_ctx = getattr(stage, "_orla_trace_context", None)
                    stage._orla_trace_context = {
                        "workflow_id": getattr(stage, "_workflow_id", ""),
                        "stage_id": getattr(stage, "id", ""),
                        "turn_index": turn_index,
                    }
                    try:
                        tool_results = stage.run_tool_calls_in_response(resp)
                    finally:
                        if previous_ctx is None:
                            try:
                                delattr(stage, "_orla_trace_context")
                            except AttributeError:
                                pass
                        else:
                            stage._orla_trace_context = previous_ctx
                    tool_end = time.time()
                    for tool_result in tool_results:
                        msg_dict = tool_result.to_message_dict()
                        messages.append(
                            pyorla.Message(
                                role="tool",
                                content=msg_dict["content"],
                                tool_call_id=msg_dict.get("tool_call_id", ""),
                                tool_name=msg_dict.get("tool_name", ""),
                            )
                        )

                turn_end = time.time()
                tracer.record_orla_agent_loop_turn(
                    workflow_id=getattr(stage, "_workflow_id", ""),
                    stage_id=getattr(stage, "id", ""),
                    turn_index=turn_index,
                    start_time=turn_start,
                    end_time=turn_end,
                    llm_start_time=llm_start,
                    llm_end_time=llm_end,
                    tool_start_time=tool_start,
                    tool_end_time=tool_end,
                    extra={
                        "message_count_before": message_count_before,
                        "message_count_after": len(messages),
                        "tool_call_count": len(tool_calls),
                        "tool_result_count": len(tool_results),
                        "assistant_content_len": len(assistant_content),
                        "assistant_content_preview": assistant_content[:400],
                        "response_metrics": _response_metrics_dict(resp),
                        "stop_reason": "tool_calls" if tool_calls else "completed",
                    },
                )

                if not tool_calls:
                    break

            return pyorla.StageResultData(response=last_resp, messages=messages)

        def execute(self) -> dict[str, Any]:
            if not self._stages:
                return {}

            workflow_id = random_workflow_id()
            for stage in self._stages.values():
                stage.set_workflow_id(workflow_id)
                if self.tags:
                    merged = dict(self.tags)
                    merged.update(stage.tags)
                    stage.tags = merged

            edges: list[tuple[str, str]] = []
            for stage_id, deps in self._dependencies.items():
                for dep_id in deps:
                    edges.append((dep_id, stage_id))

            try:
                # ORLA expects every request that carries workflow_id to have been
                # registered first, even when the DAG has zero edges.
                self.client.register_workflow(workflow_id, edges)
            except Exception:  # noqa: BLE001
                pass

            remaining_deps: dict[str, int] = {}
            dependents: dict[str, list[str]] = {}
            for stage_id in self._stages:
                deps = self._dependencies.get(stage_id, [])
                remaining_deps[stage_id] = len(deps)
                for dep_id in deps:
                    dependents.setdefault(dep_id, []).append(stage_id)

            results: dict[str, Any] = {}
            ready = [stage_id for stage_id, count in remaining_deps.items() if count == 0]
            trace_session_id = tracer.current_session_id()

            try:
                with ThreadPoolExecutor() as pool:
                    futures: dict[Any, str] = {}

                    def _submit(stage_id: str) -> None:
                        snapshot = dict(results)
                        submitted_at = time.time()

                        def _execute_logged_stage() -> tuple[str, Any]:
                            tracer.activate_session(trace_session_id)
                            stage_start = time.time()
                            executed_stage_id, result = self._execute_stage_in_dag(stage_id, snapshot)
                            stage_end = time.time()
                            response = getattr(result, "response", None)
                            response_content = getattr(response, "content", "") or ""
                            result_messages = getattr(result, "messages", None)
                            tracer.record_orla_stage_execution(
                                workflow_id=workflow_id,
                                stage_id=executed_stage_id,
                                ready_time=submitted_at,
                                start_time=stage_start,
                                end_time=stage_end,
                                tags=dict(getattr(self._stages[executed_stage_id], "tags", {})),
                                extra={
                                    "dependency_count": len(
                                        self._dependencies.get(executed_stage_id, [])
                                    ),
                                    "result_content_len": len(response_content),
                                    "result_message_count": (
                                        len(result_messages)
                                        if isinstance(result_messages, list)
                                        else None
                                    ),
                                    "response_metrics": _response_metrics_dict(response),
                                },
                            )
                            return executed_stage_id, result

                        future = pool.submit(_execute_logged_stage)
                        futures[future] = stage_id

                    for stage_id in ready:
                        _submit(stage_id)

                    while futures:
                        for future in as_completed(futures):
                            futures.pop(future)
                            stage_id, result = future.result()
                            results[stage_id] = result

                            for dependent_id in dependents.get(stage_id, []):
                                remaining_deps[dependent_id] -= 1
                                if remaining_deps[dependent_id] == 0:
                                    _submit(dependent_id)
                            break
            finally:
                try:
                    self._notify_workflow_complete(workflow_id)
                except Exception:  # noqa: BLE001
                    pass

            return results

    return CompatWorkflow


_PYORLA_MODULES: PyOrlaModules | None = None
_PYORLA_LOCK = threading.Lock()
_TOOL_COLLECTION_CACHE: WeakKeyDictionary[Any, list[Any]] = WeakKeyDictionary()
_TOOL_COLLECTION_CACHE_LOCK = threading.Lock()
_TOOL_RUNNER_BRIDGE: _AsyncToolRunnerBridge | None = None
_TOOL_RUNNER_BRIDGE_LOCK = threading.Lock()


def _load_pyorla_uncached() -> PyOrlaModules:
    """Load the ORLA SDK lazily so legacy runtime has no ORLA dependency."""
    sdk_mode = os.getenv("MAS_ORLA_SDK", "local").strip().lower()
    if sdk_mode not in {"local", "pyorla"}:
        raise OrlaRuntimeError(
            "MAS_ORLA_SDK must be either 'local' or 'pyorla' when set."
        )

    if sdk_mode == "pyorla":
        try:
            from pyorla import (
                CostModel,
                EXECUTION_MODE_AGENT_LOOP,
                LLMBackend,
                Message,
                OrlaClient,
                OrlaError,
                Stage as RawStage,
                ToolResult as RawToolResult,
                Workflow as RawWorkflow,
                new_tool,
            )
            from pyorla.stage import StageResultData
            from pyorla.workflow import _random_workflow_id
        except ImportError as exc:
            raise OrlaRuntimeError(
                "ORLA runtime requested with MAS_ORLA_SDK=pyorla, but "
                "`pyorla` is not installed. Install it with "
                "`pip install 'pyorla>=1.0.8'` or unset MAS_ORLA_SDK to "
                "use the bundled local SDK."
            ) from exc
    else:
        from app.orla.local_sdk import (
            CostModel,
            EXECUTION_MODE_AGENT_LOOP,
            LLMBackend,
            Message,
            OrlaClient,
            OrlaError,
            Stage as RawStage,
            StageResultData,
            ToolResult as RawToolResult,
            Workflow as RawWorkflow,
            _random_workflow_id,
            new_tool,
        )

    imported = _ImportedPyOrla(
        CostModel=CostModel,
        EXECUTION_MODE_AGENT_LOOP=EXECUTION_MODE_AGENT_LOOP,
        LLMBackend=LLMBackend,
        Message=Message,
        OrlaClient=OrlaClient,
        OrlaError=OrlaError,
        Stage=RawStage,
        StageResultData=StageResultData,
        ToolResult=RawToolResult,
        Workflow=RawWorkflow,
        new_tool=new_tool,
        random_workflow_id=_random_workflow_id,
    )
    CompatStage = _patched_stage_class(imported.Stage, imported.ToolResult)
    CompatWorkflow = _patched_workflow_class(imported.Workflow, imported.random_workflow_id)

    return PyOrlaModules(
        OrlaClient=imported.OrlaClient,
        OrlaError=imported.OrlaError,
        Workflow=CompatWorkflow,
        Stage=CompatStage,
        StageResultData=imported.StageResultData,
        LLMBackend=imported.LLMBackend,
        CostModel=imported.CostModel,
        Message=imported.Message,
        new_tool=imported.new_tool,
        ToolResult=imported.ToolResult,
        EXECUTION_MODE_AGENT_LOOP=imported.EXECUTION_MODE_AGENT_LOOP,
    )


def load_pyorla() -> PyOrlaModules:
    global _PYORLA_MODULES
    if _PYORLA_MODULES is not None:
        return _PYORLA_MODULES
    with _PYORLA_LOCK:
        if _PYORLA_MODULES is None:
            _PYORLA_MODULES = _load_pyorla_uncached()
    return _PYORLA_MODULES


def _get_tool_runner_bridge() -> _AsyncToolRunnerBridge:
    global _TOOL_RUNNER_BRIDGE
    if _TOOL_RUNNER_BRIDGE is not None:
        return _TOOL_RUNNER_BRIDGE
    with _TOOL_RUNNER_BRIDGE_LOCK:
        if _TOOL_RUNNER_BRIDGE is None:
            _TOOL_RUNNER_BRIDGE = _AsyncToolRunnerBridge()
    return _TOOL_RUNNER_BRIDGE


_REGISTERED_BACKENDS: set[tuple[str, str]] = set()
_REGISTER_BACKEND_LOCK_DIR = Path(
    os.environ.get("MAS_ORLA_LOCK_DIR", "/tmp/mas_sys_orla_locks")
)


@contextlib.contextmanager
def _orla_backend_registration_lock(orla_url: str, backend_name: str):
    safe = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in f"{orla_url}_{backend_name}"
    )
    _REGISTER_BACKEND_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _REGISTER_BACKEND_LOCK_DIR / f"{safe}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _remote_orla_backend_exists(orla_url: str, backend_name: str, timeout: float | None) -> bool:
    url = f"{orla_url.rstrip('/')}/api/v1/backends"
    try:
        with urllib.request.urlopen(url, timeout=timeout or 30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return False
    backends = payload.get("backends") if isinstance(payload, dict) else None
    if not isinstance(backends, list):
        return False
    for backend in backends:
        if isinstance(backend, dict) and backend.get("name") == backend_name:
            return True
    return False


class OrlaBackendRegistry:
    """Build and register ORLA backends from config on demand."""

    def __init__(self):
        self.pyorla = load_pyorla()
        orla_cfg = config.orla
        if orla_cfg is None:
            raise OrlaRuntimeError(
                "ORLA runtime requested but no `[orla]` section is configured."
            )
        self.config = orla_cfg
        self.client = self.pyorla.OrlaClient(orla_cfg.url, timeout=orla_cfg.timeout)
        self._backend_cache: dict[str, Any] = {}

    def ensure_registered(self, backend_names: Iterable[str]) -> None:
        for backend_name in backend_names:
            self.get_backend(backend_name)

    def get_backend(self, backend_name: str) -> Any:
        if backend_name in self._backend_cache:
            return self._backend_cache[backend_name]

        backend_cfg = self.config.backends.get(backend_name)
        if backend_cfg is None:
            raise OrlaRuntimeError(
                f"Missing ORLA backend config for llm_config_name={backend_name!r}."
            )

        backend = self.pyorla.LLMBackend(
            name=backend_name,
            endpoint=backend_cfg.endpoint,
            type=backend_cfg.type,
            model_id=backend_cfg.model_id,
            api_key_env_var=backend_cfg.api_key_env_var,
        )
        backend.set_max_concurrency(backend_cfg.max_concurrency)
        backend.set_queue_capacity(backend_cfg.queue_capacity)
        if backend_cfg.quality is not None:
            backend.set_quality(backend_cfg.quality)
        if backend_cfg.cost_model is not None:
            backend.set_cost_model(
                self.pyorla.CostModel(
                    input_cost_per_mtoken=backend_cfg.cost_model.input_cost_per_mtoken,
                    output_cost_per_mtoken=backend_cfg.cost_model.output_cost_per_mtoken,
                )
            )

        key = (self.config.url, backend_name)
        if key not in _REGISTERED_BACKENDS:
            with _orla_backend_registration_lock(self.config.url, backend_name):
                if key not in _REGISTERED_BACKENDS:
                    if not _remote_orla_backend_exists(
                        self.config.url,
                        backend_name,
                        self.config.timeout,
                    ):
                        try:
                            self.client.register_backend(backend)
                        except self.pyorla.OrlaError as exc:
                            if not _is_duplicate_backend_error(exc):
                                raise
                    _REGISTERED_BACKENDS.add(key)

        self._backend_cache[backend_name] = backend
        return backend


def app_message_to_orla_message(message: Message) -> Any:
    """Convert an app.schema.Message to an ORLA SDK message."""
    pyorla = load_pyorla()
    tool_calls: list[dict[str, Any]] = []
    for tool_call in message.tool_calls or []:
        args = tool_call.function.arguments
        try:
            parsed_args: Any = json.loads(args) if isinstance(args, str) else args
        except Exception:
            parsed_args = {"raw": args}
        tool_calls.append(
            {
                "id": tool_call.id,
                "method": "tools/call",
                "params": {
                    "name": tool_call.function.name,
                    "arguments": parsed_args,
                },
            }
        )
    return pyorla.Message(
        role=message.role,
        content=message.content or "",
        tool_call_id=message.tool_call_id or "",
        tool_name=message.name or "",
        tool_calls=tool_calls,
    )


def orla_message_to_app_message(message: Any) -> Message:
    """Convert an ORLA SDK message to app.schema.Message."""
    tool_calls: list[ToolCall] = []
    for raw_call in getattr(message, "tool_calls", []) or []:
        params = raw_call.get("params", raw_call.get("McpCallToolParams", {}))
        arguments = params.get("arguments", raw_call.get("arguments", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        tool_calls.append(
            ToolCall(
                id=raw_call.get("id", ""),
                function=Function(
                    name=params.get("name", raw_call.get("name", "")),
                    arguments=arguments,
                ),
            )
        )

    if getattr(message, "role", "") == "tool":
        content = getattr(message, "content", "") or ""
        base64_image = None
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                base64_image = payload.get("base64_image")
                if "output" in payload and isinstance(payload["output"], str):
                    content = payload["output"]
        except Exception:
            pass
        return Message.tool_message(
            content=content,
            name=getattr(message, "tool_name", "") or "",
            tool_call_id=getattr(message, "tool_call_id", "") or "",
            base64_image=base64_image,
        )

    if tool_calls:
        return Message.from_tool_calls(
            tool_calls=tool_calls,
            content=getattr(message, "content", "") or "",
        )

    if getattr(message, "role", "") == "system":
        return Message.system_message(getattr(message, "content", "") or "")
    if getattr(message, "role", "") == "user":
        return Message.user_message(getattr(message, "content", "") or "")
    return Message.assistant_message(getattr(message, "content", "") or "")


def tool_collection_to_orla_tools(tool_collection: Any) -> list[Any]:
    """Convert a ToolCollection into ORLA SDK tools with sync runners."""
    if tool_collection is None:
        return []
    with _TOOL_COLLECTION_CACHE_LOCK:
        cached_tools = _TOOL_COLLECTION_CACHE.get(tool_collection)
    if cached_tools is not None:
        return cached_tools

    pyorla = load_pyorla()
    bridge = _get_tool_runner_bridge()
    tools: list[Any] = []
    for tool in getattr(tool_collection, "tools", []) or []:
        schema = tool.to_param().get("function", {})
        input_schema = schema.get("parameters") or {"type": "object", "properties": {}}

        def _runner(
            input_args: dict[str, Any],
            *,
            tool_name: str = tool.name,
            collection: Any = tool_collection,
            async_bridge: _AsyncToolRunnerBridge = bridge,
        ) -> Any:
            result = async_bridge.run(
                collection.execute(name=tool_name, tool_input=input_args or {})
            )
            if getattr(result, "error", None):
                return pyorla.ToolResult(
                    error=str(result.error),
                    is_error=True,
                )
            payload = getattr(result, "output", None)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"output": payload}
            if payload is None:
                payload = {}
            elif not isinstance(payload, dict):
                payload = {"output": payload}
            if getattr(result, "base64_image", None):
                payload = {**payload, "base64_image": result.base64_image}
            return pyorla.ToolResult(output_values=payload or {})

        tools.append(
            pyorla.new_tool(
                name=tool.name,
                description=tool.description,
                input_schema=input_schema,
                run=_runner,
            )
        )
    with _TOOL_COLLECTION_CACHE_LOCK:
        try:
            _TOOL_COLLECTION_CACHE[tool_collection] = tools
        except TypeError:
            pass
    return tools


def _is_duplicate_backend_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 409:
        return True
    message = str(exc).lower()
    return "already exists" in message or "duplicate" in message
