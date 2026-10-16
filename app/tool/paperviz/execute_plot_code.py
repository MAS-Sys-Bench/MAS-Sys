from __future__ import annotations

import json
import sys
import subprocess
from textwrap import dedent

from app.tool.base import BaseTool, ToolResult


class PaperVizExecutePlotCode(BaseTool):
    name: str = "paperviz_execute_plot_code"
    description: str = "Execute matplotlib plot code and return stdout plus a rendered JPEG image as base64."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["code"],
    }

    async def execute(self, code: str, timeout: int = 30) -> ToolResult:
        wrapper = dedent(
            f"""
            import base64
            import io
            import json
            import re
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            raw = {code!r}
            match = re.search(r"```python(.*?)```", raw, re.DOTALL)
            code_clean = match.group(1).strip() if match else raw.strip()

            plt.close("all")
            plt.rcdefaults()

            payload = {{"success": False, "observation": "", "image_base64": None}}
            try:
                exec_globals = {{}}
                exec(code_clean, exec_globals, exec_globals)
                if plt.get_fignums():
                    buf = io.BytesIO()
                    plt.savefig(buf, format="jpeg", bbox_inches="tight", dpi=300)
                    plt.close("all")
                    buf.seek(0)
                    payload["success"] = True
                    payload["observation"] = "Plot rendered successfully."
                    payload["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")
                else:
                    payload["observation"] = "No matplotlib figure was produced."
            except Exception as e:
                payload["observation"] = str(e)

            print(json.dumps(payload))
            """
        )

        result = self._execute_locally(wrapper=wrapper, timeout=timeout)
        observation = result.get("observation", "") if isinstance(result, dict) else ""
        success = bool(result.get("success")) if isinstance(result, dict) else False

        parsed = None
        try:
            parsed = json.loads(observation.strip().splitlines()[-1])
        except Exception:
            parsed = None

        if not parsed:
            return ToolResult(
                output=observation or "Failed to parse plot execution result.",
                error=None if success else "plot_execution_failed",
            )

        return ToolResult(
            output=json.dumps(
                {
                    "success": parsed.get("success", False),
                    "observation": parsed.get("observation", ""),
                },
                ensure_ascii=False,
                indent=2,
            ),
            error=None if parsed.get("success", False) else "plot_execution_failed",
            base64_image=parsed.get("image_base64"),
        )

    def _execute_locally(self, wrapper: str, timeout: int) -> dict:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "observation": f"Execution timeout after {timeout} seconds",
                "success": False,
            }
        if completed.returncode == 0:
            return {
                "observation": completed.stdout,
                "success": True,
            }
        return {
            "observation": completed.stderr or completed.stdout or "Plot execution failed.",
            "success": False,
        }
