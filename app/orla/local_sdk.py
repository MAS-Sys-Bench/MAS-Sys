"""Small ORLA daemon client used by MAS-Sys.

This module intentionally implements the subset of the upstream Python SDK
that MAS-Sys needs at runtime. Keeping it local avoids pulling the upstream
``pyorla`` dependency into the main app environment, where its transitive
dependencies currently conflict with the browser-use stack.
"""

from __future__ import annotations

import json
import os
import random
import string
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


EXECUTION_MODE_SINGLE_SHOT = "single_shot"
EXECUTION_MODE_AGENT_LOOP = "agent_loop"


@dataclass
class CostModel:
    input_cost_per_mtoken: float = 0.0
    output_cost_per_mtoken: float = 0.0


@dataclass
class LLMBackend:
    name: str
    endpoint: str
    type: str
    model_id: str
    api_key_env_var: str = ""
    max_concurrency: int = 1
    queue_capacity: int = 0
    cost_model: CostModel | None = None
    quality: float | None = None

    def set_max_concurrency(self, n: int) -> None:
        self.max_concurrency = n

    def set_queue_capacity(self, n: int) -> None:
        self.queue_capacity = n

    def set_cost_model(self, cm: CostModel) -> None:
        self.cost_model = cm

    def set_quality(self, q: float) -> None:
        self.quality = q


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str = ""
    tool_name: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StructuredOutputRequest:
    name: str
    schema: dict[str, Any]
    strict: bool = False


@dataclass
class SchedulingHints:
    priority: int | None = None
    request_priority: int | None = None


@dataclass
class CacheHints:
    preserve_threshold_tokens: int | None = None
    flush_on_complete: bool | None = None


@dataclass
class ExecuteRequest:
    backend: str
    stage_id: str = ""
    prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    response_format: StructuredOutputRequest | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    scheduling_policy: str = ""
    request_scheduling_policy: str = ""
    scheduling_hints: SchedulingHints | None = None
    workflow_id: str = ""
    cache_policy: str = ""
    cache_hints: CacheHints | None = None
    reasoning_effort: str = ""
    accuracy: float | None = None
    accuracy_policy: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    data_labels: list[str] = field(default_factory=list)
    skill_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"backend": self.backend}
        if self.stage_id:
            payload["stage_id"] = self.stage_id
        if self.prompt:
            payload["prompt"] = self.prompt
        if self.messages:
            payload["messages"] = [_message_to_dict(message) for message in self.messages]
        if self.tools:
            payload["tools"] = self.tools
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.stream:
            payload["stream"] = True
        if self.response_format is not None:
            payload["response_format"] = {
                "name": self.response_format.name,
                "schema": self.response_format.schema,
                "strict": self.response_format.strict,
            }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.scheduling_policy:
            payload["scheduling_policy"] = self.scheduling_policy
        if self.request_scheduling_policy:
            payload["request_scheduling_policy"] = self.request_scheduling_policy
        if self.scheduling_hints is not None:
            hints: dict[str, int] = {}
            if self.scheduling_hints.priority is not None:
                hints["priority"] = self.scheduling_hints.priority
            if self.scheduling_hints.request_priority is not None:
                hints["request_priority"] = self.scheduling_hints.request_priority
            if hints:
                payload["scheduling_hints"] = hints
        if self.workflow_id:
            payload["workflow_id"] = self.workflow_id
        if self.cache_policy:
            payload["cache_policy"] = self.cache_policy
        if self.cache_hints is not None:
            cache_hints: dict[str, Any] = {}
            if self.cache_hints.preserve_threshold_tokens is not None:
                cache_hints["preserve_threshold_tokens"] = (
                    self.cache_hints.preserve_threshold_tokens
                )
            if self.cache_hints.flush_on_complete is not None:
                cache_hints["flush_on_complete"] = self.cache_hints.flush_on_complete
            if cache_hints:
                payload["cache_hints"] = cache_hints
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.accuracy is not None:
            payload["accuracy"] = self.accuracy
        if self.accuracy_policy:
            payload["accuracy_policy"] = self.accuracy_policy
        if self.tags:
            payload["tags"] = self.tags
        if self.data_labels:
            payload["data_labels"] = self.data_labels
        if self.skill_id:
            payload["skill_id"] = self.skill_id
        return payload


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_name:
        payload["tool_name"] = message.tool_name
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


@dataclass
class InferenceResponseMetrics:
    ttft_ms: int | None = None
    tpot_ms: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queue_wait_ms: int = 0
    scheduler_decision_ms: int = 0
    dispatch_ms: int = 0
    backend_latency_ms: int | None = None
    estimated_cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    content: str
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: InferenceResponseMetrics | None = None


@dataclass
class StreamEvent:
    type: str
    content: str = ""
    thinking: str = ""
    tool_call: dict[str, Any] | None = None
    response: InferenceResponse | None = None


ToolRunner = Callable[[dict[str, Any]], "ToolResult"]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    run: ToolRunner | None = None

    def to_mcp(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema:
            payload["outputSchema"] = self.output_schema
        return payload


@dataclass
class ToolCall:
    id: str
    name: str
    input_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    id: str = ""
    name: str = ""
    output_values: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    is_error: bool = False

    def to_message_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.id,
            "tool_name": self.name,
        }
        if self.is_error:
            content = "tool execution error"
            if self.error:
                content += f": {self.error}"
            payload["content"] = content
        else:
            payload["content"] = json.dumps(self.output_values)
        return payload


def new_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    run: ToolRunner,
    output_schema: dict[str, Any] | None = None,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        run=run,
    )


def tool_call_from_raw(raw_call: dict[str, Any]) -> ToolCall:
    params = raw_call.get("McpCallToolParams", raw_call.get("params", {}))
    arguments = params.get("arguments", raw_call.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    elif not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(
        id=raw_call.get("id", ""),
        name=params.get("name", raw_call.get("name", "")),
        input_arguments=arguments or {},
    )


class OrlaError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


class OrlaClient:
    DEFAULT_TIMEOUT: float = 300

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        *,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._sync = httpx.Client(
            base_url=self.base_url,
            timeout=timeout if timeout is not None else self.DEFAULT_TIMEOUT,
        )

    @classmethod
    def from_env(
        cls,
        default: str = "http://localhost:8081",
        *,
        timeout: float | None = None,
    ) -> "OrlaClient":
        return cls(os.environ.get("ORLA_URL", default), timeout=timeout)

    def health(self) -> None:
        response = self._sync.get("/api/v1/health")
        _raise_http(response)

    def register_backend(self, backend: LLMBackend) -> None:
        response = self._sync.post("/api/v1/backends", json=_backend_to_dict(backend))
        _raise_http(response)
        data = response.json()
        if not data.get("success"):
            raise OrlaError(f"register backend failed: {data.get('error', 'unknown')}")

    def register_workflow(
        self,
        workflow_id: str,
        edges: list[tuple[str, str]],
        declassifications: dict[str, list[str]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "workflow_id": workflow_id,
            "edges": [{"from": from_id, "to": to_id} for from_id, to_id in edges],
        }
        if declassifications:
            payload["declassifications"] = [
                {"stage_id": stage_id, "labels": labels}
                for stage_id, labels in declassifications.items()
            ]
        response = self._sync.post("/api/v1/workflows", json=payload)
        _raise_http(response)

    def execute(self, request: ExecuteRequest) -> InferenceResponse:
        response = self._sync.post("/api/v1/execute", json=request.to_dict())
        _raise_http(response)
        return _parse_execute_response(response.json())

    def execute_stream(self, request: ExecuteRequest) -> Iterator[StreamEvent]:
        payload = request.to_dict()
        payload["stream"] = True
        with self._sync.stream("POST", "/api/v1/execute", json=payload) as response:
            _raise_http(response)
            yield from _iter_sse_events(response.iter_lines())

    def workflow_complete(self, workflow_id: str, backends: list[str]) -> None:
        response = self._sync.post(
            "/api/v1/workflow/complete",
            json={"workflow_id": workflow_id, "backends": backends},
        )
        _raise_http(response)

    def close(self) -> None:
        self._sync.close()


def _backend_to_dict(backend: LLMBackend) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": backend.name,
        "endpoint": backend.endpoint,
        "type": backend.type,
        "model_id": backend.model_id,
    }
    if backend.api_key_env_var:
        payload["api_key_env_var"] = backend.api_key_env_var
    if backend.max_concurrency > 1:
        payload["max_concurrency"] = backend.max_concurrency
    if backend.queue_capacity > 0:
        payload["queue_capacity"] = backend.queue_capacity
    if backend.cost_model is not None:
        payload["cost_model"] = {
            "input_cost_per_mtoken": backend.cost_model.input_cost_per_mtoken,
            "output_cost_per_mtoken": backend.cost_model.output_cost_per_mtoken,
        }
    if backend.quality is not None:
        payload["quality"] = backend.quality
    return payload


def _raise_http(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        data: Any = None
        request_id = None
        try:
            data = response.json()
            if isinstance(data, dict):
                request_id = data.get("request_id") or data.get("requestId")
        except Exception:
            pass
        error_text = data.get("error") if isinstance(data, dict) else None
        message = f"HTTP {response.status_code} {response.reason_phrase}"
        if error_text:
            message = f"{message}: {error_text}"
        elif response.text:
            message = f"{message}: {response.text[:2048]}"
        raise OrlaError(
            message,
            status_code=response.status_code,
            body=response.text[:2048] or None,
            request_id=request_id if isinstance(request_id, str) else None,
        ) from exc


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _merge_usage_metrics(metrics: dict[str, Any], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None and key not in metrics:
            metrics[key] = value
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
        if cached_tokens is not None and "cached_tokens" not in metrics:
            metrics["cached_tokens"] = cached_tokens


def _merge_backend_response_metrics(metrics: dict[str, Any], backend_response: Any) -> None:
    backend = _dict_or_empty(backend_response)
    if not backend:
        return
    for key in ("metrics", "meta_info"):
        nested = backend.get(key)
        if isinstance(nested, dict):
            for metric_key, value in nested.items():
                metrics.setdefault(metric_key, value)
    _merge_usage_metrics(metrics, backend.get("usage"))


def _collect_response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("metrics", "meta_info"):
        raw = response.get(key)
        if isinstance(raw, dict):
            metrics.update(raw)

    _merge_usage_metrics(metrics, response.get("usage"))

    for key in (
        "backend_metrics",
        "raw_backend_metrics",
        "upstream_metrics",
        "raw_upstream_metrics",
    ):
        raw = response.get(key)
        if isinstance(raw, dict):
            for metric_key, value in raw.items():
                metrics.setdefault(metric_key, value)

    for key in (
        "backend_response",
        "raw_backend_response",
        "upstream_response",
        "raw_upstream_response",
        "openai_response",
        "raw_openai_response",
    ):
        _merge_backend_response_metrics(metrics, response.get(key))

    return metrics


def _parse_execute_response(data: dict[str, Any]) -> InferenceResponse:
    if not data.get("success"):
        raise OrlaError(f"execution failed: {data.get('error', 'unknown')}")
    response = data.get("response", {})
    metrics = None
    raw_metrics = _collect_response_metrics(response)
    if raw_metrics:
        metrics = InferenceResponseMetrics(
            ttft_ms=raw_metrics.get("ttft_ms"),
            tpot_ms=raw_metrics.get("tpot_ms"),
            prompt_tokens=raw_metrics.get("prompt_tokens", 0),
            completion_tokens=raw_metrics.get("completion_tokens", 0),
            queue_wait_ms=raw_metrics.get("queue_wait_ms", 0),
            scheduler_decision_ms=raw_metrics.get("scheduler_decision_ms", 0),
            dispatch_ms=raw_metrics.get("dispatch_ms", 0),
            backend_latency_ms=raw_metrics.get("backend_latency_ms"),
            estimated_cost_usd=raw_metrics.get("estimated_cost_usd"),
            raw=dict(raw_metrics),
        )
    return InferenceResponse(
        content=response.get("content", ""),
        thinking=response.get("thinking", ""),
        tool_calls=response.get("tool_calls") or [],
        metrics=metrics,
    )


def _iter_sse_events(lines: Iterator[str]) -> Iterator[StreamEvent]:
    state: dict[str, str] = {}
    for line in lines:
        event = _parse_sse_line(line, state)
        if event is not None:
            yield event


def _parse_sse_line(line: str, state: dict[str, str]) -> StreamEvent | None:
    if line.startswith("event: "):
        state["event"] = line[7:]
        return None
    if line.startswith("data: "):
        state["data"] = line[6:]
        return None
    if line == "" and "event" in state and "data" in state:
        event = _build_stream_event(state["event"], state["data"])
        state.clear()
        return event
    return None


def _build_stream_event(event_type: str, data_str: str) -> StreamEvent | None:
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if event_type == "content":
        return StreamEvent(type="content", content=data.get("content", ""))
    if event_type == "thinking":
        return StreamEvent(type="thinking", thinking=data.get("thinking", ""))
    if event_type == "tool_call":
        return StreamEvent(
            type="tool_call",
            tool_call={"name": data.get("name", ""), "arguments": data.get("arguments", {})},
        )
    if event_type == "done" and data.get("success") and data.get("response"):
        return StreamEvent(type="done", response=_parse_execute_response(data))
    return None


@dataclass
class StageResultData:
    response: InferenceResponse | None = None
    messages: list[Message] = field(default_factory=list)


StageMessagesBuilder = Callable[[dict[str, StageResultData]], list[Message]]
StagePromptBuilder = Callable[[dict[str, StageResultData]], str]


def _random_stage_id() -> str:
    return (
        "".join(random.choices(string.ascii_lowercase, k=4))
        + "-"
        + "".join(random.choices(string.ascii_lowercase, k=4))
    )


class Stage:
    def __init__(self, name: str, backend: LLMBackend | None = None) -> None:
        self.id = _random_stage_id()
        self.name = name
        self.client: OrlaClient | None = None
        self.backend = backend
        self.tools: dict[str, Tool] = {}
        self.execution_mode = EXECUTION_MODE_SINGLE_SHOT
        self.max_turns = 0
        self.prompt = ""
        self.prompt_builder: StagePromptBuilder | None = None
        self.messages_builder: StageMessagesBuilder | None = None
        self.max_tokens: int | None = None
        self.temperature: float | None = None
        self.top_p: float | None = None
        self.response_format: StructuredOutputRequest | None = None
        self.chat_template_kwargs: dict[str, Any] | None = None
        self.reasoning_effort = ""
        self.stage_scheduling_policy = ""
        self.request_scheduling_policy = ""
        self.scheduling_hints: SchedulingHints | None = None
        self.cache_policy = ""
        self.cache_hints: CacheHints | None = None
        self.stream = False
        self.accuracy: float | None = None
        self.accuracy_policy = ""
        self._workflow_id = ""
        self.tags: dict[str, str] = {}
        self.data_labels: list[str] = []
        self.declassifies: list[str] = []
        self.skill_id = ""

    def set_max_tokens(self, n: int) -> None:
        self.max_tokens = n

    def set_temperature(self, value: float) -> None:
        self.temperature = value

    def set_top_p(self, value: float) -> None:
        self.top_p = value

    def set_response_format(self, value: StructuredOutputRequest) -> None:
        self.response_format = value

    def set_chat_template_kwargs(self, kwargs: dict[str, Any]) -> None:
        self.chat_template_kwargs = kwargs

    def set_reasoning_effort(self, effort: str) -> None:
        self.reasoning_effort = effort

    def set_scheduling_policy(self, policy: str) -> None:
        self.stage_scheduling_policy = policy

    def set_request_scheduling_policy(self, policy: str) -> None:
        self.request_scheduling_policy = policy

    def set_scheduling_hints(self, hints: SchedulingHints) -> None:
        self.scheduling_hints = hints

    def set_execution_mode(self, mode: str) -> None:
        self.execution_mode = mode

    def set_max_turns(self, n: int) -> None:
        self.max_turns = n

    def set_prompt_builder(self, builder: StagePromptBuilder) -> None:
        self.prompt_builder = builder

    def set_messages_builder(self, builder: StageMessagesBuilder) -> None:
        self.messages_builder = builder

    def set_cache_policy(self, policy: str) -> None:
        self.cache_policy = policy

    def set_cache_hints(self, hints: CacheHints) -> None:
        self.cache_hints = hints

    def set_stream(self, enabled: bool) -> None:
        self.stream = enabled

    def set_accuracy(self, score: float) -> None:
        self.accuracy = score

    def set_accuracy_policy(self, policy: str) -> None:
        self.accuracy_policy = policy

    def set_workflow_id(self, workflow_id: str) -> None:
        self._workflow_id = workflow_id

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags = tags

    def set_data_labels(self, labels: list[str]) -> None:
        self.data_labels = labels

    def set_declassifies(self, labels: list[str]) -> None:
        self.declassifies = labels

    def set_skill_id(self, skill_id: str) -> None:
        self.skill_id = skill_id

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def build_request(
        self,
        prompt: str = "",
        messages: list[Message] | None = None,
    ) -> ExecuteRequest:
        if self.backend is None:
            raise ValueError(f"stage {self.name!r}: backend is nil")
        request = ExecuteRequest(backend=self.backend.name, stage_id=self.id)
        if messages:
            request.messages = messages
        elif prompt:
            request.prompt = prompt
        request.max_tokens = self.max_tokens
        request.temperature = self.temperature
        request.top_p = self.top_p
        request.response_format = self.response_format
        request.chat_template_kwargs = self.chat_template_kwargs
        request.reasoning_effort = self.reasoning_effort
        request.scheduling_policy = self.stage_scheduling_policy
        request.request_scheduling_policy = self.request_scheduling_policy
        request.scheduling_hints = self.scheduling_hints
        request.workflow_id = self._workflow_id
        request.cache_policy = self.cache_policy
        request.cache_hints = self.cache_hints
        request.accuracy = self.accuracy
        request.accuracy_policy = self.accuracy_policy
        request.tags = self.tags
        request.data_labels = self.data_labels
        request.skill_id = self.skill_id
        if self.tools:
            request.tools = [tool.to_mcp() for tool in self.tools.values()]
        return request

    def execute(self, prompt: str) -> InferenceResponse:
        if self.client is None:
            raise ValueError(f"stage {self.name!r}: client is nil")
        return self.client.execute(self.build_request(prompt=prompt))

    def execute_with_messages(self, messages: list[Message]) -> InferenceResponse:
        if self.client is None:
            raise ValueError(f"stage {self.name!r}: client is nil")
        return self.client.execute(self.build_request(messages=messages))

    def execute_stream(self, prompt: str) -> Iterator[StreamEvent]:
        if self.client is None:
            raise ValueError(f"stage {self.name!r}: client is nil")
        return self.client.execute_stream(self.build_request(prompt=prompt))

    def execute_stream_with_messages(self, messages: list[Message]) -> Iterator[StreamEvent]:
        if self.client is None:
            raise ValueError(f"stage {self.name!r}: client is nil")
        return self.client.execute_stream(self.build_request(messages=messages))

    def consume_stream(self, stream: Iterator[StreamEvent]) -> InferenceResponse:
        response = InferenceResponse(content="")
        for event in stream:
            if event.type == "content":
                response.content += event.content
            elif event.type == "thinking":
                response.thinking += event.thinking
            elif event.type == "done" and event.response is not None:
                return event.response
        return response

    def run_tool_call(self, tool_call: ToolCall) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            raise ValueError(f"unknown tool {tool_call.name!r}")
        if tool.run is None:
            raise ValueError(f"tool {tool_call.name!r} has no runner")
        result = tool.run(tool_call.input_arguments)
        result.id = tool_call.id
        result.name = tool_call.name
        return result

    def run_tool_calls_in_response(self, response: InferenceResponse) -> list[ToolResult]:
        return [
            self.run_tool_call(tool_call_from_raw(raw_call))
            for raw_call in response.tool_calls
        ]


def _random_workflow_id() -> str:
    return (
        "".join(random.choices(string.ascii_lowercase, k=4))
        + "-"
        + "".join(random.choices(string.ascii_lowercase, k=4))
    )


class Workflow:
    def __init__(self, client: OrlaClient, *, tags: dict[str, str] | None = None) -> None:
        self.client = client
        self.tags = tags or {}
        self._stages: dict[str, Stage] = {}
        self._dependencies: dict[str, list[str]] = {}

    def add_stage(self, stage: Stage) -> None:
        if stage.id in self._stages:
            raise ValueError(f"stage {stage.id!r} already exists")
        stage.client = self.client
        self._stages[stage.id] = stage

    def add_dependency(self, stage_id: str, depends_on_stage_id: str) -> None:
        if stage_id not in self._stages:
            raise ValueError(f"stage {stage_id!r} not found")
        if depends_on_stage_id not in self._stages:
            raise ValueError(f"dependency stage {depends_on_stage_id!r} not found")
        self._dependencies.setdefault(stage_id, []).append(depends_on_stage_id)

    def execute(self) -> dict[str, StageResultData]:
        if not self._stages:
            return {}

        workflow_id = _random_workflow_id()
        for stage in self._stages.values():
            stage.set_workflow_id(workflow_id)
            if self.tags:
                merged = dict(self.tags)
                merged.update(stage.tags)
                stage.tags = merged

        edges = [
            (dependency_id, stage_id)
            for stage_id, dependencies in self._dependencies.items()
            for dependency_id in dependencies
        ]
        try:
            self.client.register_workflow(workflow_id, edges)
        except Exception:
            pass

        remaining_deps: dict[str, int] = {}
        dependents: dict[str, list[str]] = {}
        for stage_id in self._stages:
            dependencies = self._dependencies.get(stage_id, [])
            remaining_deps[stage_id] = len(dependencies)
            for dependency_id in dependencies:
                dependents.setdefault(dependency_id, []).append(stage_id)

        results: dict[str, StageResultData] = {}
        ready = [stage_id for stage_id, count in remaining_deps.items() if count == 0]

        try:
            with ThreadPoolExecutor() as pool:
                futures: dict[Future[tuple[str, StageResultData]], str] = {}

                def submit(stage_id: str) -> None:
                    snapshot = dict(results)
                    futures[pool.submit(self._execute_stage_in_dag, stage_id, snapshot)] = (
                        stage_id
                    )

                for stage_id in ready:
                    submit(stage_id)

                while futures:
                    for future in as_completed(futures):
                        futures.pop(future)
                        stage_id, result = future.result()
                        results[stage_id] = result
                        for dependent_id in dependents.get(stage_id, []):
                            remaining_deps[dependent_id] -= 1
                            if remaining_deps[dependent_id] == 0:
                                submit(dependent_id)
                        break
        finally:
            self._notify_workflow_complete(workflow_id)
        return results

    def _execute_stage_in_dag(
        self,
        stage_id: str,
        dep_results: dict[str, StageResultData],
    ) -> tuple[str, StageResultData]:
        stage = self._stages[stage_id]
        if stage.execution_mode == EXECUTION_MODE_AGENT_LOOP:
            return stage_id, self._execute_agent_loop(stage, dep_results)
        return stage_id, self._execute_single_shot(stage, dep_results)

    def _execute_single_shot(
        self,
        stage: Stage,
        dep_results: dict[str, StageResultData],
    ) -> StageResultData:
        if stage.messages_builder is not None:
            messages = stage.messages_builder(dep_results)
            if stage.stream:
                response = stage.consume_stream(stage.execute_stream_with_messages(messages))
            else:
                response = stage.execute_with_messages(messages)
            return StageResultData(response=response, messages=messages)

        prompt = stage.prompt_builder(dep_results) if stage.prompt_builder else stage.prompt
        if not prompt:
            raise ValueError(f"stage {stage.name!r}: prompt is empty")
        if stage.stream:
            return StageResultData(response=stage.consume_stream(stage.execute_stream(prompt)))
        return StageResultData(response=stage.execute(prompt))

    def _execute_agent_loop(
        self,
        stage: Stage,
        dep_results: dict[str, StageResultData],
    ) -> StageResultData:
        if stage.messages_builder is not None:
            messages = stage.messages_builder(dep_results)
        else:
            prompt = stage.prompt_builder(dep_results) if stage.prompt_builder else stage.prompt
            if not prompt:
                raise ValueError(f"stage {stage.name!r}: prompt is empty")
            messages = [Message(role="user", content=prompt)]

        max_turns = stage.max_turns if stage.max_turns > 0 else 100
        last_response: InferenceResponse | None = None
        for _ in range(max_turns):
            response = stage.execute_with_messages(messages)
            last_response = response
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                break
            for tool_result in stage.run_tool_calls_in_response(response):
                message = tool_result.to_message_dict()
                messages.append(
                    Message(
                        role="tool",
                        content=message["content"],
                        tool_call_id=message.get("tool_call_id", ""),
                        tool_name=message.get("tool_name", ""),
                    )
                )
        return StageResultData(response=last_response, messages=messages)

    def _notify_workflow_complete(self, workflow_id: str) -> None:
        backends = {
            stage.backend.name
            for stage in self._stages.values()
            if stage.backend is not None and stage.backend.name
        }
        if not backends:
            return
        try:
            self.client.workflow_complete(workflow_id, sorted(backends))
        except Exception:
            pass
