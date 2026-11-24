"""Continuum request metadata helpers.

MAS-Sys uses Continuum through the OpenAI-compatible serving API. These
helpers keep the scheduler metadata contract in one place so agents, flows,
and the LLM client do not each carry their own key strings.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


BACKEND_NAME = "continuum"
METADATA_ENV = "MAS_CONTINUUM_JOB_METADATA"

TRACE_JOB_ID = "continuum_job_id"
TRACE_CALL_INDEX = "continuum_call_index"
TRACE_IS_LAST_STEP = "continuum_is_last_step"
TRACE_LAST_FUNC_CALL = "continuum_last_func_call"
TRACE_THIS_FUNC_CALL = "continuum_this_func_call"
TRACE_LAST_TOOL_DURATION_S = "continuum_last_tool_duration_s"

XARGS_FIELD = "vllm_xargs"
XARG_JOB_ID = "job_id"
XARG_IS_LAST_STEP = "is_last_step"
XARG_LAST_FUNC_CALL = "last_func_call"
XARG_THIS_FUNC_CALL = "this_func_call"
XARG_LAST_TOOL_DURATION_S = "last_tool_duration_s"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def metadata_enabled(backend: str, *, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether MAS-Sys should attach Continuum scheduler metadata."""
    source = os.environ if environ is None else environ
    env_value = source.get(METADATA_ENV, "").strip().lower()
    if env_value in _TRUE_VALUES:
        return True
    if env_value in _FALSE_VALUES:
        return False
    return backend == BACKEND_NAME


def job_id(*parts: object) -> str:
    """Build a stable Continuum job id from non-empty components."""
    normalized: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part)
        if text:
            normalized.append(text)
    return ":".join(normalized)


def extra_body_from_trace_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Translate tracer LLM context into OpenAI ``extra_body`` payload."""
    raw_job_id = context.get(TRACE_JOB_ID)
    if not raw_job_id:
        return {}

    xargs: dict[str, Any] = {XARG_JOB_ID: str(raw_job_id)}
    if TRACE_IS_LAST_STEP in context:
        xargs[XARG_IS_LAST_STEP] = int(bool(context[TRACE_IS_LAST_STEP]))

    optional_fields = {
        TRACE_LAST_FUNC_CALL: XARG_LAST_FUNC_CALL,
        TRACE_THIS_FUNC_CALL: XARG_THIS_FUNC_CALL,
        TRACE_LAST_TOOL_DURATION_S: XARG_LAST_TOOL_DURATION_S,
    }
    for trace_key, xarg_key in optional_fields.items():
        value = context.get(trace_key)
        if value is not None:
            xargs[xarg_key] = value

    return {XARGS_FIELD: xargs}


def merge_extra_body(
    existing: dict[str, Any] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Merge Continuum metadata without clobbering caller-provided xargs."""
    if not existing:
        return dict(extra)

    merged = dict(existing)
    for key, value in extra.items():
        if (
            key == XARGS_FIELD
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
