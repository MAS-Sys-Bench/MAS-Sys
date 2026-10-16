from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_json_or_text(payload: str | None) -> Dict[str, Any]:
    text = (payload or "").strip()
    if not text:
        raise ValueError("[PaperViz] Empty payload cannot be parsed as JSON.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {"text": text}
    return {"text": text}


def parse_json_object(payload: str | None) -> Dict[str, Any]:
    text = (payload or "").strip()
    if not text:
        raise ValueError("[PaperViz] Empty payload cannot be parsed as JSON.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        raise ValueError("[PaperViz] Payload is not valid JSON.") from None
    if not isinstance(parsed, dict):
        raise ValueError("[PaperViz] Payload JSON must be an object.")
    return parsed


def json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def task_name_from_params(agent_params: Dict[str, Any]) -> str:
    task_name = str(agent_params.get("task_name", "diagram")).strip().lower()
    return "plot" if task_name == "plot" else "diagram"


def split_root_and_upstream(request: str | None) -> tuple[str, str]:
    text = (request or "").strip()
    marker = "\n\n[Upstream Outputs]\n"
    if marker not in text:
        return text, ""
    root, upstream = text.split(marker, 1)
    return root.strip(), upstream.strip()


def extract_paperbanana_context(text: str | None) -> Dict[str, str]:
    raw = (text or "").strip()
    if not raw:
        return {}

    task_match = re.search(r"Task type:\s*(diagram|plot)\b", raw, re.IGNORECASE)
    task_name = task_match.group(1).lower() if task_match else ""

    method_match = re.search(
        r"Methodology / Raw Data:\s*(.*?)\s*Figure caption / Visual intent:",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    visual_match = re.search(
        r"Figure caption / Visual intent:\s*(.*)$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )

    return {
        "task_name": task_name,
        "content": (method_match.group(1).strip() if method_match else ""),
        "visual_intent": (visual_match.group(1).strip() if visual_match else ""),
    }


def parse_upstream_payload(request: str | None) -> Dict[str, Any]:
    _, upstream = split_root_and_upstream(request)
    return parse_json_or_text(upstream)


def parse_upstream_payloads(request: str | None) -> list[Dict[str, Any]]:
    _, upstream = split_root_and_upstream(request)
    text = upstream.strip()
    if not text:
        return []

    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    payloads: list[Dict[str, Any]] = []
    for part in parts:
        parsed = parse_json_or_text(part)
        if parsed:
            payloads.append(parsed)
    return payloads


def extract_visualizer_description(request: str | None) -> str:
    """Pick the right description payload for a visualizer node.

    Priority:
    1. critic-provided revised_description
    2. upstream detailed_description / description
    3. raw upstream text
    4. raw request
    """
    payloads = parse_upstream_payloads(request)
    for payload in reversed(payloads):
        revised = str(payload.get("revised_description", "")).strip()
        if revised and revised.lower() != "no changes needed.":
            return revised

    for payload in reversed(payloads):
        detailed = str(
            payload.get("detailed_description", payload.get("description", ""))
        ).strip()
        if detailed:
            return detailed

    payload = parse_upstream_payload(request)
    if payload:
        revised = str(payload.get("revised_description", "")).strip()
        if revised and revised.lower() != "no changes needed.":
            return revised
        detailed = str(
            payload.get("detailed_description", payload.get("description", ""))
        ).strip()
        if detailed:
            return detailed

    _, upstream_text = split_root_and_upstream(request)
    if upstream_text.strip():
        return upstream_text.strip()
    return (request or "").strip()
