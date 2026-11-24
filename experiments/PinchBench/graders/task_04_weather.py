"""Grader for task_04_weather: Weather Script Creation."""

import ast
import re
from pathlib import Path


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}
    workspace = Path(workspace_path)

    script_file = workspace / "weather.py"
    if not script_file.exists():
        return {
            "file_created": 0.0,
            "valid_python": 0.0,
            "has_http_request": 0.0,
            "references_location": 0.0,
            "has_error_handling": 0.0,
            "has_output": 0.0,
            "executable_structure": 0.0,
        }

    scores["file_created"] = 1.0
    content = script_file.read_text()

    try:
        ast.parse(content)
        scores["valid_python"] = 1.0
    except SyntaxError:
        return {
            "file_created": 1.0,
            "valid_python": 0.0,
            "has_http_request": 0.0,
            "references_location": 0.0,
            "has_error_handling": 0.0,
            "has_output": 0.0,
            "executable_structure": 0.0,
        }

    http_patterns = [
        r"import\s+requests",
        r"from\s+requests\s+import",
        r"import\s+urllib",
        r"from\s+urllib",
        r"import\s+http",
        r"requests\.get",
        r"urllib\.request",
        r"urlopen",
    ]
    scores["has_http_request"] = (
        1.0
        if any(re.search(p, content, re.IGNORECASE) for p in http_patterns)
        else 0.0
    )

    sf_patterns = [r"San.Francisco", r"San_Francisco", r"SF", r"san.francisco"]
    scores["references_location"] = (
        1.0
        if any(re.search(p, content, re.IGNORECASE) for p in sf_patterns)
        else 0.0
    )

    error_patterns = [r"try\s*:", r"except\s+", r"except:", r"Exception"]
    scores["has_error_handling"] = (
        1.0 if any(re.search(p, content) for p in error_patterns) else 0.0
    )

    output_patterns = [r"print\s*\(", r"sys\.stdout"]
    scores["has_output"] = (
        1.0 if any(re.search(p, content) for p in output_patterns) else 0.0
    )

    structure_patterns = [
        r"def\s+\w+\s*\(",
        r"if\s+__name__\s*==\s*[\"']__main__[\"']",
    ]
    if any(re.search(p, content) for p in structure_patterns):
        scores["executable_structure"] = 1.0
    else:
        scores["executable_structure"] = 0.5

    return scores
