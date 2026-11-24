import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.config import config
from app.sandbox.client import SANDBOX_CLIENT
from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    _sandbox_initialized: bool = False

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def _ensure_sandbox(self) -> None:
        """Ensure the Docker sandbox is created (lazy init)."""
        if not self._sandbox_initialized:
            sandbox = SANDBOX_CLIENT.sandbox
            if not sandbox or not getattr(sandbox, "terminal", None):
                await SANDBOX_CLIENT.create(config=config.sandbox)
            self._sandbox_initialized = True

    async def _execute_in_sandbox(self, code: str, timeout: int) -> Dict:
        """Execute Python code inside the Docker sandbox container."""
        await self._ensure_sandbox()
        # Write code to a temp file in the container to avoid shell quoting issues.
        script_path = "/tmp/_mas_sys_exec.py"
        try:
            await SANDBOX_CLIENT.write_file(script_path, code)
            output = await SANDBOX_CLIENT.run_command(
                f"python3 {script_path}", timeout=timeout
            )
            return {"observation": output or "", "success": True}
        except TimeoutError:
            return {
                "observation": f"Execution timeout after {timeout} seconds",
                "success": False,
            }
        except Exception as e:
            return {"observation": str(e), "success": False}

    async def execute(
        self,
        code: str,
        timeout: int = 10,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        # Sandbox mode: execute inside Docker container
        if config.sandbox.use_sandbox:
            return await self._execute_in_sandbox(code, timeout)

        # Local mode: use multiprocessing
        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
