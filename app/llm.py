import asyncio
import dataclasses
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app import continuum
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)
from app.tracer import tracer


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


def _print_llm_stream_to_stdout() -> bool:
    return os.getenv("MAS_PRINT_LLM_STREAM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _orla_transport_enabled() -> bool:
    return os.getenv("MAS_LLM_TRANSPORT", "").strip().lower() == "orla"


_ORLA_BACKEND_REGISTRY: Any = None
_ORLA_BACKEND_REGISTRY_KEY: tuple[str, tuple[str, ...]] | None = None
_ORLA_BACKEND_REGISTRY_LOCK = threading.Lock()


def _guess_base64_image_mime(base64_data: str) -> str:
    stripped = (base64_data or "").strip()
    if stripped.startswith("iVBOR"):
        return "image/png"
    if stripped.startswith("/9j/"):
        return "image/jpeg"
    if stripped.startswith("R0lGOD"):
        return "image/gif"
    if stripped.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            self.config_name = config_name
            llm_config = llm_config or config.llm
            if isinstance(llm_config, dict):
                if config_name not in llm_config:
                    available = ", ".join(sorted(llm_config.keys()))
                    raise KeyError(
                        f"LLM config '{config_name}' is not defined. "
                        f"Available keys: {available}"
                    )
                llm_config = llm_config[config_name]
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_style = llm_config.api_style
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            self.backend = llm_config.backend

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0

            # Per-call info for tracer consumption
            self.last_input_tokens = 0
            self.last_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def _continuum_metadata_enabled(self) -> bool:
        return continuum.metadata_enabled(self.backend)

    def _continuum_extra_body(self) -> dict[str, Any]:
        if not self._continuum_metadata_enabled():
            return {}
        if self.api_type.lower() in {"azure", "aws"}:
            return {}

        return continuum.extra_body_from_trace_context(
            tracer.current_llm_call_context()
        )

    @staticmethod
    def _merge_extra_body(
        existing: Optional[dict[str, Any]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return continuum.merge_extra_body(existing, extra)

    def _apply_continuum_extra_body(self, params: dict[str, Any]) -> None:
        extra_body = self._continuum_extra_body()
        if not extra_body:
            return
        params["extra_body"] = self._merge_extra_body(
            params.get("extra_body"),
            extra_body,
        )

    def _apply_qwen_chat_template_options(self, params: dict[str, Any]) -> None:
        if "qwen" not in str(self.model).lower():
            return
        qwen_extra = {"chat_template_kwargs": {"enable_thinking": False}}
        params["extra_body"] = self._merge_extra_body(
            params.get("extra_body"),
            qwen_extra,
        )

    def _use_orla_transport(self) -> bool:
        return bool(
            _orla_transport_enabled()
            and config.orla is not None
            and self.config_name.startswith("lang.")
            and self.config_name in config.orla.backends
            and self.api_type.lower() not in {"azure", "aws"}
        )

    @staticmethod
    def _openai_tools_to_orla_tools(tools: Optional[List[dict]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            converted.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "inputSchema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return [tool for tool in converted if tool["name"]]

    @staticmethod
    def _orla_tool_calls_to_openai(tool_calls: Any) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for index, raw_call in enumerate(tool_calls or []):
            if not isinstance(raw_call, dict):
                continue
            params = raw_call.get("params") or raw_call.get("McpCallToolParams") or {}
            if not isinstance(params, dict):
                params = {}
            arguments = params.get("arguments", raw_call.get("arguments", {}))
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            converted.append(
                {
                    "id": raw_call.get("id") or f"call_{index + 1}",
                    "type": "function",
                    "function": {
                        "name": params.get("name") or raw_call.get("name") or "",
                        "arguments": arguments,
                    },
                }
            )
        return [
            call
            for call in converted
            if call["function"]["name"]
        ]

    @staticmethod
    def _openai_tool_calls_to_orla_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for index, raw_call in enumerate(tool_calls or []):
            if not isinstance(raw_call, dict):
                continue
            params = raw_call.get("params") or raw_call.get("McpCallToolParams")
            if isinstance(params, dict):
                arguments = params.get("arguments", raw_call.get("arguments", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except Exception:
                        arguments = {"raw": arguments}
                elif not isinstance(arguments, dict):
                    arguments = {}
                converted.append(
                    {
                        "id": raw_call.get("id") or f"call_{index + 1}",
                        "method": raw_call.get("method") or "tools/call",
                        "params": {
                            "name": params.get("name") or raw_call.get("name") or "",
                            "arguments": arguments,
                        },
                    }
                )
                continue

            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments: Any = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except Exception:
                    arguments = {"raw": arguments}
            elif not isinstance(arguments, dict):
                arguments = {}
            converted.append(
                {
                    "id": raw_call.get("id") or f"call_{index + 1}",
                    "method": "tools/call",
                    "params": {
                        "name": function.get("name") or raw_call.get("name") or "",
                        "arguments": arguments,
                    },
                }
            )
        return [
            call
            for call in converted
            if call["params"]["name"]
        ]

    @classmethod
    def _format_tool_calls_for_orla_history(cls, tool_calls: Any) -> str:
        lines: list[str] = []
        for call in cls._openai_tool_calls_to_orla_tool_calls(tool_calls):
            params = call.get("params") or {}
            arguments = params.get("arguments") or {}
            try:
                arguments_text = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                arguments_text = str(arguments)
            call_id = call.get("id") or ""
            suffix = f" [id={call_id}]" if call_id else ""
            lines.append(f"- {params.get('name', '')}({arguments_text}){suffix}")
        return "\n".join(lines)

    @classmethod
    def _openai_messages_to_orla_messages(cls, messages: list[dict]) -> list[Any]:
        converted: list[Any] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                formatted_tool_calls = cls._format_tool_calls_for_orla_history(tool_calls)
                if formatted_tool_calls:
                    content = (
                        f"{content}\n\n" if content else ""
                    ) + f"Assistant requested tool calls:\n{formatted_tool_calls}"
                converted.append(
                    SimpleNamespace(
                        role=role,
                        content=content,
                        tool_call_id="",
                        tool_name="",
                        tool_calls=[],
                    )
                )
                continue

            if role == "tool":
                tool_name = message.get("tool_name") or message.get("name") or "tool"
                tool_call_id = message.get("tool_call_id") or ""
                call_suffix = f" id={tool_call_id}" if tool_call_id else ""
                converted.append(
                    SimpleNamespace(
                        role="user",
                        content=(
                            f"Tool result from {tool_name}{call_suffix}:\n{content}"
                        ),
                        tool_call_id="",
                        tool_name="",
                        tool_calls=[],
                    )
                )
                continue

            converted.append(
                SimpleNamespace(
                    role=role,
                    content=content,
                    tool_call_id=message.get("tool_call_id", ""),
                    tool_name=message.get("tool_name") or message.get("name", ""),
                    tool_calls=[],
                )
            )
        return converted

    async def _orla_execute(
        self,
        *,
        messages: list[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        from app.orla.local_sdk import ExecuteRequest
        from app.orla.runtime import OrlaBackendRegistry

        orla_cfg = config.orla
        if orla_cfg is None:
            raise RuntimeError("MAS_LLM_TRANSPORT=orla requires an [orla] config section")
        if self.config_name not in orla_cfg.backends:
            raise RuntimeError(
                f"ORLA transport requested for {self.config_name!r}, but no "
                "matching [orla.backends] entry exists."
            )

        completion_limit = max_tokens if max_tokens is not None else self.max_tokens
        request = ExecuteRequest(
            backend=self.config_name,
            messages=self._openai_messages_to_orla_messages(messages),
            tools=self._openai_tools_to_orla_tools(tools),
            max_tokens=completion_limit,
            temperature=temperature if temperature is not None else self.temperature,
            tags={
                "runtime": "orla",
                "llm_config_name": self.config_name,
            },
        )
        if "qwen" in str(self.model).lower():
            request.chat_template_kwargs = {"enable_thinking": False}
        if response_format:
            schema = response_format.get("json_schema") if isinstance(response_format, dict) else None
            if isinstance(schema, dict):
                request.response_format = SimpleNamespace(
                    name=schema.get("name", "response"),
                    schema=schema.get("schema", {}),
                    strict=bool(schema.get("strict", False)),
                )

        global _ORLA_BACKEND_REGISTRY, _ORLA_BACKEND_REGISTRY_KEY
        registry_key = (orla_cfg.url, tuple(sorted(orla_cfg.backends.keys())))
        with _ORLA_BACKEND_REGISTRY_LOCK:
            if _ORLA_BACKEND_REGISTRY is None or _ORLA_BACKEND_REGISTRY_KEY != registry_key:
                _ORLA_BACKEND_REGISTRY = OrlaBackendRegistry()
                _ORLA_BACKEND_REGISTRY_KEY = registry_key
            _ORLA_BACKEND_REGISTRY.get_backend(self.config_name)
            client = _ORLA_BACKEND_REGISTRY.client
        return await asyncio.to_thread(client.execute, request)

    @staticmethod
    def _orla_response_usage(response: Any) -> Any:
        metrics = getattr(response, "metrics", None)
        raw = getattr(metrics, "raw", {}) if metrics is not None else {}
        if not isinstance(raw, dict):
            raw = {}
        prompt_tokens = int(raw.get("prompt_tokens") or getattr(metrics, "prompt_tokens", 0) or 0)
        completion_tokens = int(
            raw.get("completion_tokens") or getattr(metrics, "completion_tokens", 0) or 0
        )
        total_tokens = int(raw.get("total_tokens") or prompt_tokens + completion_tokens)
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _orla_response_as_chat_message(response: Any) -> ChatCompletionMessage:
        return ChatCompletionMessage.model_validate(
            {
                "role": "assistant",
                "content": getattr(response, "content", "") or "",
                "tool_calls": LLM._orla_tool_calls_to_openai(
                    getattr(response, "tool_calls", []) or []
                ),
            }
        )

    @staticmethod
    def _orla_response_for_tracing(response: Any) -> Any:
        return SimpleNamespace(
            metrics=getattr(response, "metrics", None),
            usage=LLM._orla_response_usage(response),
        )

    @staticmethod
    def _response_metrics(response: Any) -> Dict[str, Any]:
        raw: Any = None
        for attr in ("metrics", "meta_info"):
            value = getattr(response, attr, None)
            if value:
                raw = value
                break
        if raw is None and hasattr(response, "model_extra"):
            extra = getattr(response, "model_extra", None) or {}
            raw = extra.get("metrics") or extra.get("meta_info")
        if raw is None and hasattr(response, "model_dump"):
            dumped = response.model_dump()
            raw = dumped.get("metrics") or dumped.get("meta_info")
        if raw is None:
            return {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif dataclasses.is_dataclass(raw):
            raw = dataclasses.asdict(raw)
        elif hasattr(raw, "__dict__"):
            raw = vars(raw)
        if isinstance(raw, dict):
            nested_raw = raw.get("raw")
            if isinstance(nested_raw, dict):
                merged = dict(nested_raw)
                for key, value in raw.items():
                    if key != "raw":
                        merged.setdefault(key, value)
                return merged
            return raw
        return {}

    @staticmethod
    def _record_legacy_llm_call(
        *,
        invocation: str,
        messages: Any,
        model: str,
        base_url: str,
        start_time: float,
        end_time: float,
        prompt_ready_time: Optional[float] = None,
        tools: Any = None,
        tool_choice: Any = None,
        response: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "client_observed_latency_s": round(end_time - start_time, 4),
            **(extra or {}),
        }
        if response is not None:
            metrics = LLM._response_metrics(response)
            if metrics:
                payload["response_metrics"] = metrics
            usage = getattr(response, "usage", None)
            if usage is not None:
                for source_attr, output_key in (
                    ("prompt_tokens", "prompt_tokens"),
                    ("completion_tokens", "completion_tokens"),
                    ("total_tokens", "total_tokens"),
                ):
                    value = getattr(usage, source_attr, None)
                    if value is not None:
                        payload[output_key] = value
        if prompt_ready_time is not None:
            payload["prompt_ready_time"] = prompt_ready_time
        tracer.record_llm_call(
            transport="legacy",
            invocation=invocation,
            messages=messages,
            model=model,
            base_url=base_url,
            tools=tools,
            tool_choice=tool_choice,
            start_time=start_time,
            end_time=end_time,
            extra=payload,
        )

    @property
    def supports_images(self) -> bool:
        return self.model in MULTIMODAL_MODELS or self.config_name.startswith("vision.")

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        # Store per-call info for tracer consumption
        self.last_input_tokens = input_tokens
        self.last_completion_tokens = completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{_guess_base64_image_mime(message['base64_image'])};"
                                    f"base64,{message['base64_image']}"
                                )
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                elif not supports_images and message.get("base64_image"):
                    raise ValueError(
                        "The target model does not support image inputs, "
                        "but the message payload contains base64_image."
                    )

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.supports_images

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }
            if response_format:
                params["response_format"] = response_format

            completion_limit = max_tokens if max_tokens is not None else self.max_tokens
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = completion_limit
            else:
                params["max_tokens"] = completion_limit
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
            self._apply_qwen_chat_template_options(params)
            self._apply_continuum_extra_body(params)
            prompt_ready_time = time.time()

            if self._use_orla_transport():
                request_start = time.time()
                response = await self._orla_execute(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                request_end = time.time()
                trace_response = self._orla_response_for_tracing(response)
                self._record_legacy_llm_call(
                    invocation="ask",
                    messages=messages,
                    model=self.model,
                    base_url=config.orla.url if config.orla is not None else self.base_url,
                    start_time=request_start,
                    end_time=request_end,
                    prompt_ready_time=prompt_ready_time,
                    response=trace_response,
                    extra={
                        "config_name": self.config_name,
                        "stream": False,
                        "api_type": "orla",
                        "response_format": response_format,
                        "max_tokens": max_tokens,
                    },
                )
                usage = trace_response.usage
                self.update_token_count(usage.prompt_tokens, usage.completion_tokens)
                content = (getattr(response, "content", "") or "").strip()
                if not content:
                    raise ValueError("Empty or invalid response from ORLA LLM")
                return content

            if not stream:
                # Non-streaming request
                request_start = time.time()
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )
                request_end = time.time()
                self._record_legacy_llm_call(
                    invocation="ask",
                    messages=messages,
                    model=self.model,
                    base_url=self.base_url,
                    start_time=request_start,
                    end_time=request_end,
                    prompt_ready_time=prompt_ready_time,
                    response=response,
                    extra={
                        "config_name": self.config_name,
                        "stream": False,
                        "api_type": self.api_type,
                        "response_format": response_format,
                        "max_tokens": max_tokens,
                    },
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)
            params["stream_options"] = {"include_usage": True}

            request_start = time.time()
            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            first_chunk_time = None
            last_chunk = None
            print_stream = _print_llm_stream_to_stdout()
            async for chunk in response:
                last_chunk = chunk
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                if not getattr(chunk, "choices", None):
                    continue
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                if print_stream:
                    print(chunk_message, end="", flush=True)
            request_end = time.time()
            self._record_legacy_llm_call(
                invocation="ask",
                messages=messages,
                model=self.model,
                base_url=self.base_url,
                start_time=request_start,
                end_time=request_end,
                prompt_ready_time=prompt_ready_time,
                response=last_chunk,
                extra={
                    "config_name": self.config_name,
                    "stream": True,
                    "api_type": self.api_type,
                    "response_format": response_format,
                    "max_tokens": max_tokens,
                    "client_ttft_s": (
                        round(first_chunk_time - request_start, 4)
                        if first_chunk_time is not None
                        else None
                    ),
                },
            )

            if print_stream:
                print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if not self.supports_images:
                raise ValueError(
                    f"Model {self.model} ({self.config_name}) does not support images. "
                    f"Use a vision.* config or a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }
            if response_format:
                params["response_format"] = response_format

            # Add model-specific parameters
            completion_limit = max_tokens if max_tokens is not None else self.max_tokens
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = completion_limit
            else:
                params["max_tokens"] = completion_limit
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
            self._apply_qwen_chat_template_options(params)
            self._apply_continuum_extra_body(params)
            prompt_ready_time = time.time()

            # Handle non-streaming request
            if not stream:
                request_start = time.time()
                response = await self.client.chat.completions.create(**params)
                request_end = time.time()
                self._record_legacy_llm_call(
                    invocation="ask_with_images",
                    messages=all_messages,
                    model=self.model,
                    base_url=self.base_url,
                    start_time=request_start,
                    end_time=request_end,
                    prompt_ready_time=prompt_ready_time,
                    response=response,
                    extra={
                        "config_name": self.config_name,
                        "stream": False,
                        "api_type": self.api_type,
                        "response_format": response_format,
                        "max_tokens": max_tokens,
                    },
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            params["stream_options"] = {"include_usage": True}
            request_start = time.time()
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            first_chunk_time = None
            last_chunk = None
            print_stream = _print_llm_stream_to_stdout()
            async for chunk in response:
                last_chunk = chunk
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                if not getattr(chunk, "choices", None):
                    continue
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                if print_stream:
                    print(chunk_message, end="", flush=True)
            request_end = time.time()
            self._record_legacy_llm_call(
                invocation="ask_with_images",
                messages=all_messages,
                model=self.model,
                base_url=self.base_url,
                start_time=request_start,
                end_time=request_end,
                prompt_ready_time=prompt_ready_time,
                response=last_chunk,
                extra={
                    "config_name": self.config_name,
                    "stream": True,
                    "api_type": self.api_type,
                    "response_format": response_format,
                    "max_tokens": max_tokens,
                    "client_ttft_s": (
                        round(first_chunk_time - request_start, 4)
                        if first_chunk_time is not None
                        else None
                    ),
                },
            )

            if print_stream:
                print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.supports_images

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            completion_limit = max_tokens if max_tokens is not None else self.max_tokens
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = completion_limit
            else:
                params["max_tokens"] = completion_limit
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            self._apply_qwen_chat_template_options(params)
            self._apply_continuum_extra_body(params)
            prompt_ready_time = time.time()

            if self._use_orla_transport():
                request_start = time.time()
                response = await self._orla_execute(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                request_end = time.time()
                trace_response = self._orla_response_for_tracing(response)
                self._record_legacy_llm_call(
                    invocation="ask_tool",
                    messages=messages,
                    model=self.model,
                    base_url=config.orla.url if config.orla is not None else self.base_url,
                    tools=tools,
                    tool_choice=tool_choice,
                    start_time=request_start,
                    end_time=request_end,
                    prompt_ready_time=prompt_ready_time,
                    response=trace_response,
                    extra={
                        "config_name": self.config_name,
                        "api_type": "orla",
                        "timeout": timeout,
                        "max_tokens": max_tokens,
                    },
                )
                usage = trace_response.usage
                self.update_token_count(usage.prompt_tokens, usage.completion_tokens)
                return self._orla_response_as_chat_message(response)

            request_start = time.time()
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )
            request_end = time.time()
            self._record_legacy_llm_call(
                invocation="ask_tool",
                messages=messages,
                model=self.model,
                base_url=self.base_url,
                tools=tools,
                tool_choice=tool_choice,
                start_time=request_start,
                end_time=request_end,
                prompt_ready_time=prompt_ready_time,
                response=response,
                extra={
                    "config_name": self.config_name,
                    "api_type": self.api_type,
                    "timeout": timeout,
                    "max_tokens": max_tokens,
                },
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
