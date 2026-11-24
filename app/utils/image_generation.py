from __future__ import annotations

import base64
import time
from typing import Any, Iterable, Optional
from urllib.parse import urljoin

import httpx

from app.tracer import tracer

DASHSCOPE_PROMPT_CHAR_LIMIT = 800
DASHSCOPE_NEGATIVE_PROMPT_CHAR_LIMIT = 500


class ImageGenerationError(RuntimeError):
    """Raised when an image backend request cannot produce an image."""


def _response_metrics(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        raw = response.get("metrics") or response.get("meta_info")
        metrics = dict(raw) if isinstance(raw, dict) else {}
        if "backend_latency_s" not in metrics and response.get("inference_time_s") is not None:
            metrics["backend_latency_s"] = response.get("inference_time_s")
        return metrics

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
        if raw is None and dumped.get("inference_time_s") is not None:
            raw = {"backend_latency_s": dumped.get("inference_time_s")}
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    metrics = dict(raw) if isinstance(raw, dict) else {}

    top_level_inference_time = getattr(response, "inference_time_s", None)
    if "backend_latency_s" not in metrics and top_level_inference_time is not None:
        metrics["backend_latency_s"] = top_level_inference_time
    return metrics


def _record_image_call(
    *,
    llm: Any,
    invocation: str,
    prompt: str,
    start_time: float,
    end_time: float,
    response: Any = None,
    error: BaseException | None = None,
) -> None:
    extra: dict[str, Any] = {
        "client_observed_latency_s": round(end_time - start_time, 4),
        "config_name": getattr(llm, "config_name", ""),
        "backend": getattr(llm, "backend", ""),
        "api_style": getattr(llm, "api_style", ""),
    }
    if response is not None:
        metrics = _response_metrics(response)
        if metrics:
            extra["response_metrics"] = metrics
    if error is not None:
        extra["error"] = str(error)
        extra["error_type"] = type(error).__name__

    tracer.record_llm_call(
        transport="image",
        invocation=invocation,
        messages=[{"role": "user", "content": prompt}],
        model=getattr(llm, "model", ""),
        base_url=getattr(llm, "base_url", ""),
        start_time=start_time,
        end_time=end_time,
        extra=extra,
    )


def _normalize_openai_size(size: str) -> str:
    return str(size or "1024x1024").replace("*", "x")


def _normalize_dashscope_size(size: str) -> str:
    return str(size or "1024x1024").replace("x", "*")


def _derive_dashscope_api_url(base_url: str) -> str:
    normalized = (base_url or "").rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        root = normalized[: -len("/compatible-mode/v1")]
    elif normalized.endswith("/api/v1"):
        root = normalized[: -len("/api/v1")]
    elif normalized.endswith("/v1"):
        root = normalized[: -len("/v1")]
    else:
        root = normalized
    return root + "/api/v1/services/aigc/multimodal-generation/generation"


async def _download_image_bytes(url: str, timeout_s: float) -> bytes | None:
    if not url:
        return None

    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        if not encoded:
            return None
        return base64.b64decode(encoded)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _iter_nested_items(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_items(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_items(item)


async def _extract_image_bytes(
    payload: Any,
    timeout_s: float,
    *,
    base_url: str = "",
) -> bytes | None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    for item in _iter_nested_items(payload):
        if not isinstance(item, dict):
            continue

        b64_candidate = (
            item.get("b64_json")
            or item.get("image_base64")
            or item.get("base64")
            or item.get("base64_data")
        )
        if isinstance(b64_candidate, str) and b64_candidate:
            return base64.b64decode(b64_candidate)

        for key in ("image", "image_url", "url"):
            url_candidate = item.get(key)
            if not isinstance(url_candidate, str) or not url_candidate:
                continue
            if url_candidate.startswith("/"):
                root = str(base_url or "").rstrip("/")
                if root.endswith("/v1"):
                    root = root[: -len("/v1")]
                url_candidate = urljoin(root + "/", url_candidate.lstrip("/"))
            if (
                url_candidate.startswith("http://")
                or url_candidate.startswith("https://")
                or url_candidate.startswith("data:")
            ):
                return await _download_image_bytes(url_candidate, timeout_s)

    return None


def _truncate_text(value: Optional[str], limit: int) -> str:
    text = (value or "").strip()
    return text[:limit] if len(text) > limit else text


def _raise_http_error(response: httpx.Response, provider: str) -> None:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = (
                    error.get("message")
                    or error.get("code")
                    or str(error)
                )
            else:
                detail = body.get("message") or body.get("code") or str(body)
        else:
            detail = str(body)
    except Exception:
        detail = response.text.strip()

    status = f"{response.status_code} {response.reason_phrase}".strip()
    message = f"{provider} image generation failed with HTTP {status}"
    if detail:
        message += f": {detail}"
    raise ImageGenerationError(message)


async def _generate_via_openai_style(
    *,
    llm: Any,
    prompt: str,
    size: str,
    timeout_s: float,
) -> bytes:
    image_client = getattr(llm.client, "images", None)
    if image_client is None:
        raise ImageGenerationError(
            "Configured image backend uses openai-style image generation, "
            "but the client does not expose an images endpoint."
        )

    start_time = time.time()
    response: Any = None
    recorded = False
    try:
        response = await image_client.generate(
            model=llm.model,
            prompt=prompt,
            size=_normalize_openai_size(size),
            response_format="b64_json",
        )
        image_bytes = await _extract_image_bytes(
            response,
            timeout_s,
            base_url=getattr(llm, "base_url", ""),
        )
        if image_bytes is None:
            raise ImageGenerationError(
                "OpenAI-style image backend returned no image payload."
            )
        return image_bytes
    except Exception as exc:
        _record_image_call(
            llm=llm,
            invocation="images.generate",
            prompt=prompt,
            start_time=start_time,
            end_time=time.time(),
            response=response,
            error=exc,
        )
        recorded = True
        raise
    finally:
        if response is not None and not recorded:
            _record_image_call(
                llm=llm,
                invocation="images.generate",
                prompt=prompt,
                start_time=start_time,
                end_time=time.time(),
                response=response,
            )


async def _generate_via_dashscope_style(
    *,
    llm: Any,
    prompt: str,
    size: str,
    negative_prompt: Optional[str],
    watermark: bool,
    prompt_extend: bool,
    timeout_s: float,
) -> bytes:
    endpoint = _derive_dashscope_api_url(llm.base_url)
    clipped_prompt = _truncate_text(prompt, DASHSCOPE_PROMPT_CHAR_LIMIT)
    clipped_negative_prompt = _truncate_text(
        negative_prompt, DASHSCOPE_NEGATIVE_PROMPT_CHAR_LIMIT
    )
    payload = {
        "model": llm.model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": clipped_prompt}],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "watermark": watermark,
            "prompt_extend": prompt_extend,
            "size": _normalize_dashscope_size(size),
        },
    }
    if clipped_negative_prompt:
        payload["parameters"]["negative_prompt"] = clipped_negative_prompt

    headers = {
        "Authorization": f"Bearer {llm.api_key}",
        "Content-Type": "application/json",
    }
    start_time = time.time()
    body: Any = None
    recorded = False
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.is_error:
                _raise_http_error(response, "DashScope")
            body = response.json()

        status_code = body.get("status_code")
        if status_code not in (None, 200):
            raise ImageGenerationError(
                f"DashScope image generation failed: code={body.get('code')} "
                f"message={body.get('message')}"
            )
        image_bytes = await _extract_image_bytes(
            body,
            timeout_s,
            base_url=getattr(llm, "base_url", ""),
        )
        if image_bytes is None:
            raise ImageGenerationError("DashScope returned no image payload.")
        return image_bytes
    except Exception as exc:
        _record_image_call(
            llm=llm,
            invocation="dashscope.image.generate",
            prompt=prompt,
            start_time=start_time,
            end_time=time.time(),
            response=body,
            error=exc,
        )
        recorded = True
        raise
    finally:
        if body is not None and not recorded:
            _record_image_call(
                llm=llm,
                invocation="dashscope.image.generate",
                prompt=prompt,
                start_time=start_time,
                end_time=time.time(),
                response=body,
            )


async def generate_image_bytes(
    *,
    llm: Any,
    prompt: str,
    size: str = "1024x1024",
    negative_prompt: Optional[str] = None,
    watermark: bool = False,
    prompt_extend: bool = True,
    timeout_s: float = 180.0,
) -> bytes:
    api_style = (getattr(llm, "api_style", "") or "openai").strip().lower()
    if api_style == "dashscope":
        return await _generate_via_dashscope_style(
            llm=llm,
            prompt=prompt,
            size=size,
            negative_prompt=negative_prompt,
            watermark=watermark,
            prompt_extend=prompt_extend,
            timeout_s=timeout_s,
        )

    return await _generate_via_openai_style(
        llm=llm,
        prompt=prompt,
        size=size,
        timeout_s=timeout_s,
    )
