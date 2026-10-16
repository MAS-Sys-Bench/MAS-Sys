"""Tool registry metadata exposed to PlanAgent."""

from __future__ import annotations


TOOL_METADATA = {
    "bash": {
        "class_name": "Bash",
        "module": "app.tool.bash",
        "description": (
            "Run shell commands for repository inspection, tests, and local automation. "
            "Use only when command execution is needed."
        ),
    },
    "browser_use": {
        "class_name": "BrowserUseTool",
        "module": "app.tool.browser_use_tool",
        "description": (
            "Browser automation tool for navigating websites, clicking elements, typing, "
            "scrolling, switching tabs, searching the web, and returning page content "
            "snippets for the calling agent to analyze. Maintains browser session state "
            "across calls; use element indices from the current browser state."
        ),
    },
    "crawl4ai": {
        "class_name": "Crawl4aiTool",
        "module": "app.tool.crawl4ai",
        "description": "Extract clean AI-ready content from web pages.",
    },
    "python_execute": {
        "class_name": "PythonExecute",
        "module": "app.tool.python_execute",
        "description": "Execute Python code strings; printed output is visible to the calling agent.",
    },
    "str_replace_editor": {
        "class_name": "StrReplaceEditor",
        "module": "app.tool.str_replace_editor",
        "description": "Inspect and edit local text files with string-replacement editor operations.",
    },
    "web_search": {
        "class_name": "WebSearch",
        "module": "app.tool.web_search",
        "description": "Search the web for current or external information.",
    },
    "planning": {
        "class_name": "PlanningTool",
        "module": "app.tool.planning",
        "description": "Maintain and update an execution plan during an agent run.",
    },
    "paperviz_execute_plot_code": {
        "class_name": "PaperVizExecutePlotCode",
        "module": "app.tool.paperviz.execute_plot_code",
        "description": "Execute matplotlib plot code and return stdout plus a rendered JPEG image as base64.",
    },
}


def get_tool_description(tool_name: str) -> str:
    return TOOL_METADATA[tool_name]["description"]


def build_tools_info_for_prompt() -> str:
    lines = []
    for key, meta in TOOL_METADATA.items():
        lines.append(f"- {key}: {meta['description']}")
    return "\n".join(lines)
