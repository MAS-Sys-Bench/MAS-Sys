"""File operation interfaces and implementations for local and sandbox environments."""

import asyncio
import shlex
from pathlib import Path
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from app.config import SandboxSettings
from app.exceptions import ToolError
from app.sandbox.client import SANDBOX_CLIENT
from app.sandbox.core.exceptions import SandboxTimeoutError


PathLike = Union[str, Path]


@runtime_checkable
class FileOperator(Protocol):
    """Interface for file operations in different environments."""

    async def read_file(self, path: PathLike) -> str:
        """Read content from a file."""
        ...

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a file."""
        ...

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory."""
        ...

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists."""
        ...

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a shell command and return (return_code, stdout, stderr)."""
        ...


class LocalFileOperator(FileOperator):
    """File operations implementation for local filesystem."""

    encoding: str = "utf-8"

    async def read_file(self, path: PathLike) -> str:
        """Read content from a local file."""
        try:
            return Path(path).read_text(encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"Failed to read {path}: {str(e)}") from None

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a local file."""
        try:
            Path(path).write_text(content, encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"Failed to write to {path}: {str(e)}") from None

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory."""
        return Path(path).is_dir()

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists."""
        return Path(path).exists()

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a shell command locally."""
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            return (
                process.returncode or 0,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            raise TimeoutError(
                f"Command '{cmd}' timed out after {timeout} seconds"
            ) from exc


class SandboxFileOperator(FileOperator):
    """File operations implementation for sandbox environment."""

    @property
    def sandbox_client(self):
        """Always use the live global sandbox client.

        Tool instances may be copied by Pydantic during agent construction.
        Looking up the singleton dynamically avoids holding a stale cloned
        client whose sandbox state diverges from the one seeded by generation.
        """
        return SANDBOX_CLIENT

    async def _ensure_sandbox_initialized(self):
        """Ensure sandbox is initialized and the container is running."""
        from app.config import config

        sandbox_client = self.sandbox_client
        sandbox = sandbox_client.sandbox
        if not sandbox or not getattr(sandbox, "terminal", None):
            # Sandbox object missing or container not started — (re)create.
            # Reuse previously stored volume_bindings so mounts survive.
            await sandbox_client.create(
                config=config.sandbox,
                volume_bindings=sandbox_client._volume_bindings,
            )

    async def read_file(self, path: PathLike) -> str:
        """Read content from a file in sandbox."""
        await self._ensure_sandbox_initialized()
        try:
            return await self.sandbox_client.read_file(str(path))
        except Exception as e:
            raise ToolError(f"Failed to read {path} in sandbox: {str(e)}") from None

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a file in sandbox."""
        await self._ensure_sandbox_initialized()
        try:
            await self.sandbox_client.write_file(str(path), content)
        except Exception as e:
            raise ToolError(f"Failed to write to {path} in sandbox: {str(e)}") from None

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory in sandbox."""
        await self._ensure_sandbox_initialized()
        quoted_path = shlex.quote(str(path))
        sandbox_client = self.sandbox_client
        try:
            exit_code, _ = await sandbox_client.exec_simple(
                f"test -d {quoted_path}"
            )
            return exit_code == 0
        except Exception as e:
            raise ToolError(
                f"Failed to inspect directory status for {path} in sandbox: {str(e)}"
            ) from None

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists in sandbox."""
        await self._ensure_sandbox_initialized()
        quoted_path = shlex.quote(str(path))
        sandbox_client = self.sandbox_client
        try:
            exit_code, _ = await sandbox_client.exec_simple(
                f"test -e {quoted_path}"
            )
            return exit_code == 0
        except Exception as e:
            raise ToolError(
                f"Failed to inspect path {path} in sandbox: {str(e)}"
            ) from None

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a command in sandbox environment."""
        await self._ensure_sandbox_initialized()
        sandbox_client = self.sandbox_client
        try:
            stdout = await sandbox_client.run_command(
                cmd, timeout=int(timeout) if timeout else None
            )
            return (
                0,  # Always return 0 since we don't have explicit return code from sandbox
                stdout,
                "",  # No stderr capture in the current sandbox implementation
            )
        except (TimeoutError, SandboxTimeoutError) as exc:
            raise TimeoutError(
                f"Command '{cmd}' timed out after {timeout} seconds in sandbox"
            ) from exc
        except Exception as exc:
            return 1, "", f"Error executing command in sandbox: {str(exc)}"
