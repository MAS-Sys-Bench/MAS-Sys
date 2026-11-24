"""
Asynchronous Docker Terminal

This module provides asynchronous terminal functionality for Docker containers,
allowing interactive command execution with timeout control.
"""

import asyncio
import re
import socket
from typing import Dict, Optional, Tuple, Union

import docker
from docker import APIClient
from docker.errors import APIError
from docker.models.containers import Container


class DockerSession:
    def __init__(self, container_id: str) -> None:
        """Initializes a Docker session.

        Args:
            container_id: ID of the Docker container.
        """
        self.api = APIClient()
        self.container_id = container_id
        self.exec_id = None
        self.socket = None

    @staticmethod
    def _extract_socket(socket_data):
        """Extract a usable raw socket from Docker exec_start response.

        Handles platform differences: Linux returns an object with ``_sock``,
        while Windows named-pipe transport may return the socket-like object
        directly or wrapped in different layers.
        """
        # 1. Direct _sock attribute (Linux TCP socket)
        if hasattr(socket_data, "_sock"):
            return socket_data._sock

        # 2. The object itself is already socket-like (e.g. NpipeSocket on Windows)
        if hasattr(socket_data, "recv") and hasattr(socket_data, "sendall"):
            return socket_data

        # 3. Try common wrapper chains
        for attrs in [("_sock",), ("raw", "_sock"), ("_fp", "_sock")]:
            obj = socket_data
            try:
                for a in attrs:
                    obj = getattr(obj, a)
                if hasattr(obj, "recv") and hasattr(obj, "sendall"):
                    return obj
            except AttributeError:
                continue

        return None

    async def create(self, working_dir: str, env_vars: Dict[str, str]) -> None:
        """Creates an interactive session with the container.

        Args:
            working_dir: Working directory inside the container.
            env_vars: Environment variables to set.

        Raises:
            RuntimeError: If socket connection fails.
        """
        startup_command = [
            "bash",
            "-c",
            f"cd {working_dir} && "
            "if [ -f /root/.bashrc ]; then "
            "source /root/.bashrc >/dev/null 2>&1; "
            "fi && "
            "export PROMPT_COMMAND='' && "
            "export PS1='$ ' && "
            "exec bash --norc --noprofile",
        ]

        exec_data = self.api.exec_create(
            self.container_id,
            startup_command,
            stdin=True,
            tty=True,
            stdout=True,
            stderr=True,
            privileged=True,
            user="root",
            environment={**env_vars, "TERM": "dumb", "PS1": "$ ", "PROMPT_COMMAND": ""},
        )
        self.exec_id = exec_data["Id"]

        socket_data = self.api.exec_start(
            self.exec_id, socket=True, tty=True, stream=True, demux=True
        )

        self.socket = self._extract_socket(socket_data)
        if self.socket is None:
            raise RuntimeError("Failed to get socket connection")
        if hasattr(self.socket, "setblocking"):
            self.socket.setblocking(False)

        await self._read_until_prompt()

    async def close(self) -> None:
        """Cleans up session resources.

        1. Sends exit command
        2. Closes socket connection
        3. Checks and cleans up exec instance
        """
        try:
            if self.socket:
                # Send exit command to close bash session
                try:
                    self.socket.sendall(b"exit\n")
                    # Allow time for command execution
                    await asyncio.sleep(0.1)
                except:
                    pass  # Ignore sending errors, continue cleanup

                # Close socket connection
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass  # Some platforms may not support shutdown

                self.socket.close()
                self.socket = None

            if self.exec_id:
                try:
                    # Check exec instance status
                    exec_inspect = self.api.exec_inspect(self.exec_id)
                    if exec_inspect.get("Running", False):
                        # If still running, wait for it to complete
                        await asyncio.sleep(0.5)
                except:
                    pass  # Ignore inspection errors, continue cleanup

                self.exec_id = None

        except Exception as e:
            # Log error but don't raise, ensure cleanup continues
            print(f"Warning: Error during session cleanup: {e}")

    async def _read_until_prompt(self) -> str:
        """Reads output until prompt is found.

        Returns:
            String containing output up to the prompt.

        Raises:
            socket.error: If socket communication fails.
        """
        buffer = b""
        while b"$ " not in buffer:
            try:
                chunk = self.socket.recv(4096)
                if chunk:
                    buffer += chunk
            except socket.error as e:
                if e.errno == socket.EWOULDBLOCK:
                    await asyncio.sleep(0.1)
                    continue
                raise
        return buffer.decode("utf-8")

    async def execute(self, command: str, timeout: Optional[int] = None) -> str:
        """Executes a command and returns cleaned output.

        Args:
            command: Shell command to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Command output as string with prompt markers removed.

        Raises:
            RuntimeError: If session not initialized or execution fails.
            TimeoutError: If command execution exceeds timeout.
        """
        if not self.socket:
            raise RuntimeError("Session not initialized")

        try:
            # Sanitize command to prevent shell injection
            sanitized_command = self._sanitize_command(command)
            full_command = f"{sanitized_command}\necho $?\n"
            self.socket.sendall(full_command.encode())

            async def read_output() -> str:
                buffer = b""
                result_lines = []
                command_sent = False

                while True:
                    try:
                        chunk = self.socket.recv(4096)
                        if not chunk:
                            # On Windows, NpipeSocket.recv() returns b""
                            # when no data is available instead of raising
                            # EWOULDBLOCK.  Sleep and retry (the outer
                            # asyncio.wait_for provides the real timeout).
                            await asyncio.sleep(0.1)
                            continue

                        buffer += chunk
                        lines = buffer.split(b"\n")

                        buffer = lines[-1]
                        lines = lines[:-1]

                        for line in lines:
                            line = line.rstrip(b"\r")

                            if not command_sent:
                                command_sent = True
                                continue

                            if line.strip() == b"echo $?" or line.strip().isdigit():
                                continue

                            if line.strip():
                                result_lines.append(line)

                        if buffer.endswith(b"$ "):
                            break

                    except socket.error as e:
                        if e.errno == socket.EWOULDBLOCK:
                            await asyncio.sleep(0.1)
                            continue
                        raise

                output = b"\n".join(result_lines).decode("utf-8")
                output = re.sub(r"\n\$ echo \$\$?.*$", "", output)

                return output

            if timeout:
                result = await asyncio.wait_for(read_output(), timeout)
            else:
                result = await read_output()

            return result.strip()

        except asyncio.TimeoutError:
            raise TimeoutError(f"Command execution timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")

    def _sanitize_command(self, command: str) -> str:
        """Sanitizes the command string to prevent shell injection.

        Args:
            command: Raw command string.

        Returns:
            Sanitized command string.

        Raises:
            ValueError: If command contains potentially dangerous patterns.
        """

        # Additional checks for specific risky commands
        risky_commands = [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "chown -R",
        ]

        for risky in risky_commands:
            if risky in command.lower():
                raise ValueError(
                    f"Command contains potentially dangerous operation: {risky}"
                )

        return command


class AsyncDockerizedTerminal:
    def __init__(
        self,
        container: Union[str, Container],
        working_dir: str = "/workspace",
        env_vars: Optional[Dict[str, str]] = None,
        default_timeout: int = 60,
    ) -> None:
        """Initializes an asynchronous terminal for Docker containers.

        Args:
            container: Docker container ID or Container object.
            working_dir: Working directory inside the container.
            env_vars: Environment variables to set.
            default_timeout: Default command execution timeout in seconds.
        """
        self.client = docker.from_env()
        self.container = (
            container
            if isinstance(container, Container)
            else self.client.containers.get(container)
        )
        self.working_dir = working_dir
        self.env_vars = env_vars or {}
        self.default_timeout = default_timeout
        self.session = None

    async def init(self) -> None:
        """Initializes the terminal environment.

        Ensures working directory exists and creates an interactive session.

        Raises:
            RuntimeError: If initialization fails.
        """
        await self._ensure_workdir()

        self.session = DockerSession(self.container.id)
        await self.session.create(self.working_dir, self.env_vars)

    async def _ensure_workdir(self) -> None:
        """Ensures working directory exists in container.

        Raises:
            RuntimeError: If directory creation fails.
        """
        try:
            await self._exec_simple(f"mkdir -p {self.working_dir}")
        except APIError as e:
            raise RuntimeError(f"Failed to create working directory: {e}")

    async def _exec_simple(self, cmd: str) -> Tuple[int, str]:
        """Executes a simple command using Docker's exec_run.

        Args:
            cmd: Command to execute.

        Returns:
            Tuple of (exit_code, output).
        """
        result = await asyncio.to_thread(
            self.container.exec_run, cmd, environment=self.env_vars
        )
        return result.exit_code, result.output.decode("utf-8")

    async def run_command(self, cmd: str, timeout: Optional[int] = None) -> str:
        """Runs a command in the container with timeout.

        Args:
            cmd: Shell command to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Command output as string.

        Raises:
            RuntimeError: If terminal not initialized.
        """
        if not self.session:
            raise RuntimeError("Terminal not initialized")

        return await self.session.execute(cmd, timeout=timeout or self.default_timeout)

    async def close(self) -> None:
        """Closes the terminal session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def restart(self) -> None:
        """Restarts the interactive terminal session without recreating the container."""
        await self.close()
        await self.init()

    async def __aenter__(self) -> "AsyncDockerizedTerminal":
        """Async context manager entry."""
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
