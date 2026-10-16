import asyncio
import os
import shutil
import subprocess
import sys
from typing import Optional

from app.config import config
from app.exceptions import ToolError
from app.sandbox.client import SANDBOX_CLIENT
from app.sandbox.core.exceptions import SandboxTimeoutError
from app.tool.base import BaseTool, CLIResult

_IS_WINDOWS = sys.platform == "win32"


def _build_shell_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    return env


def _find_shell() -> tuple[str, bool]:
    """Find the best shell for this platform. Returns (executable, is_bash).

    On Windows, prefers Git Bash over WSL bash (which may be broken) over cmd.exe.
    """
    if not _IS_WINDOWS:
        return "/bin/bash", True

    # 1. Look for Git Bash in common install locations
    candidate_roots = [
        os.environ.get("GIT_INSTALL_ROOT", ""),
        os.path.join(os.environ.get("ProgramFiles", ""), "Git"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Git"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git"),
    ]
    # Also infer Git root from git.exe on PATH (e.g. D:\Program Files\Git\cmd\git.exe)
    git_exe = shutil.which("git")
    if git_exe:
        git_cmd_dir = os.path.dirname(os.path.realpath(git_exe))
        candidate_roots.append(os.path.dirname(git_cmd_dir))

    for git_root in candidate_roots:
        if git_root:
            candidate = os.path.join(git_root, "bin", "bash.exe")
            if os.path.isfile(candidate):
                return candidate, True

    # 2. Try bash on PATH, but verify it actually works (WSL bash may be broken)
    bash_path = shutil.which("bash")
    if bash_path:
        try:
            result = subprocess.run(
                [bash_path, "-c", "echo ok"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return bash_path, True
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 3. Fallback to cmd.exe
    return "cmd.exe", False

_BASH_DESCRIPTION = """Execute a shell command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""


class _BashSession:
    """A persistent shell session. Uses bash on all platforms when available,
    falls back to cmd.exe on Windows if bash is not found."""

    _started: bool
    _process: asyncio.subprocess.Process
    _is_bash: bool

    _output_delay: float = 0.2  # seconds
    _timeout: float = 30.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False
        self._shell_cmd, self._is_bash = _find_shell()

    async def start(self):
        if self._started:
            return

        if _IS_WINDOWS:
            if self._is_bash:
                # Git Bash or WSL bash on Windows
                self._process = await asyncio.create_subprocess_exec(
                    self._shell_cmd,
                    env=_build_shell_env(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                # cmd.exe fallback on Windows
                self._process = await asyncio.create_subprocess_exec(
                    "cmd.exe",
                    "/Q",
                    env=_build_shell_env(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
        else:
            # Linux/macOS: use /bin/bash with setsid for process group control
            self._process = await asyncio.create_subprocess_exec(
                self._shell_cmd,
                env=_build_shell_env(),
                preexec_fn=os.setsid,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

        # Clear startup noise (banners, prompts) by sending a sentinel
        # and discarding everything before it
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr
        if self._is_bash:
            self._process.stdin.write(
                f"echo '{self._sentinel}'\n".encode()
            )
        else:
            # cmd.exe: also set UTF-8 code page
            self._process.stdin.write(
                f"chcp 65001 >nul & echo {self._sentinel}\r\n".encode()
            )
        await self._process.stdin.drain()
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    buf = self._process.stdout._buffer.decode(errors="replace")  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in buf:
                        break
        except asyncio.TimeoutError:
            pass
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

    def stop(self):
        """Terminate the shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return CLIResult(
                system="tool must be restarted",
                error=f"shell has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: shell has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command with sentinel to detect end of output
        if self._is_bash:
            self._process.stdin.write(
                command.encode() + f"; echo '{self._sentinel}'\n".encode()
            )
        else:
            # cmd.exe: use & to chain, echo sentinel
            self._process.stdin.write(
                (command + f" & echo {self._sentinel}\r\n").encode()
            )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode(errors="replace")
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: shell has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode(errors="replace")
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """A tool for executing shell commands (bash on Linux/macOS, cmd on Windows)."""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None
    _sandbox_initialized: bool = False

    async def _ensure_sandbox(self) -> None:
        """Ensure the Docker sandbox is created (lazy init)."""
        sandbox = SANDBOX_CLIENT.sandbox
        if sandbox and getattr(sandbox, "terminal", None):
            self._sandbox_initialized = True
            return

        await SANDBOX_CLIENT.create(
            config=config.sandbox,
            volume_bindings=SANDBOX_CLIENT._volume_bindings,
        )
        self._sandbox_initialized = True

    async def _execute_in_sandbox(self, command: str) -> CLIResult:
        """Execute a command inside the Docker sandbox container."""
        await self._ensure_sandbox()
        try:
            output = await SANDBOX_CLIENT.run_command(
                command, timeout=int(self._session._timeout if self._session else 30)
            )
            return CLIResult(output=output or "")
        except SandboxTimeoutError:
            return CLIResult(
                error="Command timed out in sandbox; terminal session was restarted",
            )
        except Exception as e:
            # If sandbox state went stale, recreate it once and retry.
            if "Sandbox not initialized" in str(e):
                self._sandbox_initialized = False
                await self._ensure_sandbox()
                try:
                    output = await SANDBOX_CLIENT.run_command(
                        command, timeout=int(self._session._timeout if self._session else 30)
                    )
                    return CLIResult(output=output or "")
                except Exception as retry_exc:
                    return CLIResult(error=str(retry_exc))
            return CLIResult(error=str(e))

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if command is None:
            raise ToolError("no command provided.")

        # Sandbox mode: execute inside Docker container
        if config.sandbox.use_sandbox:
            if restart:
                # Recreate the sandbox container
                await SANDBOX_CLIENT.cleanup()
                self._sandbox_initialized = False
                await self._ensure_sandbox()
                return CLIResult(system="tool has been restarted (sandbox).")
            return await self._execute_in_sandbox(command)

        # Local mode: use persistent shell session
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()
            return CLIResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        return await self._session.run(command)


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("echo hello world"))
    print(rst)
