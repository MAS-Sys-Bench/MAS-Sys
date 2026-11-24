"""Grader for task_10_workflow: Multi-step API Workflow."""

import ast
import re
from pathlib import Path


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}
    workspace = Path(workspace_path)

    read_config = False
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") == "assistant":
            for item in msg.get("content", []):
                if item.get("type") == "toolCall":
                    tool_name = item.get("name", "")
                    params = item.get("params", {})
                    if tool_name.lower() in ["read_file", "readfile", "read", "str_replace_editor"]:
                        files = params.get("files", [])
                        path_val = str(params.get("path", params.get("file_path", "")))
                        if any("config.json" in str(f) for f in files) or "config.json" in path_val:
                            read_config = True
                            break
    scores["read_config"] = 1.0 if read_config else 0.0

    py_files = list(workspace.glob("*.py"))
    if not py_files:
        scores["script_created"] = 0.0
        scores["valid_syntax"] = 0.0
        scores["parses_json"] = 0.0
        scores["has_http_request"] = 0.0
    else:
        scores["script_created"] = 1.0
        script_content = py_files[0].read_text()

        try:
            ast.parse(script_content)
            scores["valid_syntax"] = 1.0
        except SyntaxError:
            scores["valid_syntax"] = 0.0
            scores["parses_json"] = 0.0
            scores["has_http_request"] = 0.0
            scores["notes_created"] = 0.0
            return scores

        json_patterns = [
            r"import\s+json",
            r"from\s+json\s+import",
            r"json\.load",
            r"json\.loads",
        ]
        scores["parses_json"] = (
            1.0 if any(re.search(p, script_content) for p in json_patterns) else 0.0
        )

        http_patterns = [
            r"import\s+requests",
            r"from\s+requests\s+import",
            r"import\s+urllib",
            r"from\s+urllib",
            r"requests\.get",
            r"requests\.post",
            r"urllib\.request",
            r"urlopen",
        ]
        scores["has_http_request"] = (
            1.0
            if any(re.search(p, script_content, re.IGNORECASE) for p in http_patterns)
            else 0.0
        )

    notes_file = workspace / "NOTES.md"
    scores["notes_created"] = 1.0 if notes_file.exists() else 0.0

    return scores
