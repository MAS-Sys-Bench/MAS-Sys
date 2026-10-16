"""Grader for task_13_image_gen: AI Image Generation."""

import json
from pathlib import Path


def _load_raw_output(workspace: Path) -> dict:
    raw_output_path = workspace / "raw_output.json"
    if not raw_output_path.exists():
        return {}

    try:
        return json.loads(raw_output_path.read_text())
    except Exception:
        return {}


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {
        "used_image_tool": 0.0,
        "prompt_has_robot": 0.0,
        "prompt_has_cafe": 0.0,
        "prompt_has_book": 0.0,
        "file_saved": 0.0,
        "confirmed_generation": 0.0,
    }

    generated_prompt = ""
    used_tool = False

    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") == "assistant":
            for item in msg.get("content", []):
                if item.get("type") == "toolCall":
                    tool_name = item.get("name", "")
                    params = item.get("params", {})

                    if tool_name in [
                        "generate_image",
                        "generateImage",
                        "image_generation",
                        "paperviz_visualize",
                    ]:
                        used_tool = True
                        generated_prompt = params.get("prompt", params.get("description", "")).lower()
                        path = params.get("path", params.get("output_path", ""))
                        if "robot_cafe.png" in path or path.endswith("robot_cafe.png"):
                            scores["file_saved"] = 1.0

                if item.get("type") == "text":
                    text = item.get("text", "").lower()
                    if any(
                        phrase in text
                        for phrase in [
                            "generated",
                            "created the image",
                            "image has been",
                            "successfully created",
                            "saved the image",
                            "here's the image",
                            "image is ready",
                        ]
                    ):
                        scores["confirmed_generation"] = 1.0

    workspace = Path(workspace_path)
    raw_payload = _load_raw_output(workspace)
    image_agent_used = raw_payload.get("mode") == "image"
    image_generated = bool(raw_payload.get("image_base64_present"))
    if image_agent_used and not generated_prompt:
        generated_prompt = str(raw_payload.get("request", "")).lower()

    scores["used_image_tool"] = 1.0 if (used_tool or image_agent_used) else 0.0

    if generated_prompt:
        if "robot" in generated_prompt:
            scores["prompt_has_robot"] = 1.0
        if any(
            term in generated_prompt
            for term in ["coffee", "cafe", "café", "cozy", "shop"]
        ):
            scores["prompt_has_cafe"] = 1.0
        if any(term in generated_prompt for term in ["book", "reading", "read"]):
            scores["prompt_has_book"] = 1.0

    if (workspace / "robot_cafe.png").exists():
        scores["file_saved"] = 1.0
    if image_generated:
        scores["confirmed_generation"] = 1.0

    return scores
