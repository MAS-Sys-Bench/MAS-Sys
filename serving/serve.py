#!/usr/bin/env python3
"""Manage local serving backends from config.toml.

This script only considers nested deployable entries under:
  - llm.lang.*
  - llm.vision.*
  - llm.image.*

Each deployable entry must define backend / model_path (or model) / port / gpus.
Execution uses conda environments only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONF = PROJECT_ROOT / "config" / "config.toml"
LOG_DIR = SCRIPT_DIR / "logs"
PID_DIR = SCRIPT_DIR / "pids"

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"

DEPLOY_GROUPS = ("lang", "vision", "image")
BACKEND_ENVS = {
    "vllm": "mas-vllm",
    "continuum": "mas-vllm",
    "sglang": "mas-sglang",
    "vllm-omni": "mas-vllm-omni",
}


def info(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}")


def error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


@dataclass(init=False)
class Service:
    group: str
    name: str
    backend: str
    model_path: str
    port: int
    gpus: str
    model: str
    lmcache_cpu_gb: float

    def __init__(
        self,
        *,
        group: str | None = None,
        name: str,
        backend: str,
        model_path: str,
        port: int,
        gpus: str,
        model: str,
        lmcache_cpu_gb: float = 0.0,
    ) -> None:
        if group is None:
            if "." in name:
                group, short_name = name.split(".", 1)
                self.group = group
                self.name = name
            else:
                self.group = ""
                self.name = name
        else:
            self.group = group
            self.name = f"{group}.{name}"
        self.backend = backend
        self.model_path = model_path
        self.port = port
        self.gpus = gpus
        self.model = model
        self.lmcache_cpu_gb = float(lmcache_cpu_gb or 0.0)

    @property
    def key(self) -> str:
        return self.name

    @property
    def tp(self) -> int:
        return len([gpu for gpu in self.gpus.split(",") if gpu.strip()]) or 1

    @property
    def conda_env(self) -> str:
        try:
            return BACKEND_ENVS[self.backend]
        except KeyError as exc:
            raise ValueError(f"Unsupported backend: {self.backend}") from exc


def parse_services(conf_path: Path) -> list[Service]:
    with conf_path.open("rb") as f:
        raw = tomllib.load(f)

    llm = raw.get("llm", {})
    services: list[Service] = []
    for group in DEPLOY_GROUPS:
        group_cfg = llm.get(group, {})
        if not isinstance(group_cfg, dict):
            continue
        for name, cfg in group_cfg.items():
            if not isinstance(cfg, dict):
                continue
            if any(isinstance(v, dict) for v in cfg.values()) and "backend" not in cfg:
                raise ValueError(
                    f"Invalid config shape under [llm.{group}.{name}]. "
                    f"Use a single table entry like [llm.{group}.{name}] with backend/port/gpus/model_path."
                )
            if not cfg.get("backend"):
                continue
            services.append(
                Service(
                    group=group,
                    name=name,
                    backend=str(cfg["backend"]),
                    model_path=str(cfg.get("model_path") or cfg.get("model") or ""),
                    port=int(cfg.get("port", 0)),
                    gpus=str(cfg.get("gpus", "")),
                    model=str(cfg.get("model", "")),
                    lmcache_cpu_gb=float(cfg.get("lmcache_cpu_gb", 0.0) or 0.0),
                )
            )
    return services


def validate_service(svc: Service) -> list[str]:
    issues: list[str] = []
    if svc.backend not in BACKEND_ENVS:
        issues.append(f"unsupported backend={svc.backend}")
    if not svc.model_path:
        issues.append("missing model_path/model")
    if svc.port <= 0:
        issues.append("missing/invalid port")
    if not svc.gpus:
        issues.append("missing gpus")
    return issues


def find_conda() -> str:
    for candidate in (
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniforge3" / "bin" / "conda"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("conda not found in PATH and ~/miniforge3/bin/conda does not exist")


def _vllm_tool_args_for_service(svc: Service) -> str:
    model_path = svc.model_path.lower()
    if "qwen3" in model_path:
        return "--enable-auto-tool-choice --tool-call-parser qwen3_xml "
    if "qwen" in model_path:
        return "--enable-auto-tool-choice --tool-call-parser hermes "
    return ""


def _continuum_tool_args_for_service(svc: Service) -> str:
    model_path = svc.model_path.lower()
    if "qwen3" in model_path and "coder" in model_path:
        return "--enable-auto-tool-choice --tool-call-parser qwen3_coder "
    if "qwen3" in model_path:
        return "--enable-auto-tool-choice --tool-call-parser qwen3_xml "
    if "qwen" in model_path:
        return "--enable-auto-tool-choice --tool-call-parser hermes "
    return ""


def _sglang_tool_args_for_service(svc: Service) -> str:
    model_path = svc.model_path.lower()
    if "qwen3" in model_path:
        return "--reasoning-parser qwen3 --tool-call-parser qwen3_coder "
    if "qwen" in model_path:
        return "--tool-call-parser qwen "
    return ""


def _sglang_model_override_args_for_service(svc: Service) -> str:
    model_path = svc.model_path.lower()
    if "qwen3" in model_path and "coder" in model_path:
        # Transformers >= 5 stores Qwen3-Coder's RoPE base under
        # rope_scaling.rope_theta, while this SGLang version reads the legacy
        # top-level rope_theta field and otherwise falls back to 10,000.
        override = json.dumps({"rope_theta": 10_000_000}, separators=(",", ":"))
        return f"--json-model-override-args {shlex.quote(override)} "
    return ""


def _continuum_kv_transfer_config() -> str:
    return json.dumps(
        {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
        separators=(",", ":"),
    )


def _continuum_overlay_dir(svc: Service) -> Path:
    safe_key = svc.key.replace(".", "_").replace("/", "_")
    return Path("/tmp") / f"mas_continuum_overlay_{safe_key}"


def _prepare_continuum_overlay(svc: Service) -> Path:
    overlay_dir = _continuum_overlay_dir(svc)
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_vllm = overlay_dir / "vllm"
    shutil.copytree(PROJECT_ROOT / "serving" / "continuum" / "vllm", overlay_vllm)

    source_vllm = PROJECT_ROOT / "serving" / "vllm" / "vllm"
    runtime_dirs = (
        Path("vllm_flash_attn"),
        Path("third_party/deep_gemm"),
        Path("third_party/triton_kernels"),
    )
    for relative_name in runtime_dirs:
        source_path = source_vllm / relative_name
        target_path = overlay_vllm / relative_name
        if not source_path.exists():
            continue
        if target_path.is_symlink() or target_path.is_file():
            target_path.unlink()
        elif target_path.exists():
            shutil.rmtree(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    for so_path in source_vllm.rglob("*.so"):
        relative = so_path.relative_to(source_vllm)
        if any(relative == runtime_dir or runtime_dir in relative.parents for runtime_dir in runtime_dirs):
            continue
        target = overlay_vllm / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(so_path)
    return overlay_dir


def _build_cmd(svc: Service) -> str:
    if svc.backend == "vllm":
        return (
            "python -m vllm.entrypoints.openai.api_server "
            f"--model {shlex.quote(svc.model_path)} "
            "--host 127.0.0.1 "
            f"--port {svc.port} "
            f"--tensor-parallel-size {svc.tp} "
            "--trust-remote-code "
            "--enable-prefix-caching "
            f"{_vllm_tool_args_for_service(svc)}"
        ).strip()

    if svc.backend == "continuum":
        kv_transfer_args = ""
        if svc.lmcache_cpu_gb > 0:
            kv_transfer_args = (
                f"--kv-transfer-config {shlex.quote(_continuum_kv_transfer_config())} "
            )
        return (
            "python -m vllm.entrypoints.openai.api_server "
            f"--model {shlex.quote(svc.model_path)} "
            "--host 127.0.0.1 "
            f"{kv_transfer_args}"
            "--scheduling-policy continuum "
            f"--port {svc.port} "
            f"--tensor-parallel-size {svc.tp} "
            "--trust-remote-code "
            "--enable-prefix-caching "
            f"{_continuum_tool_args_for_service(svc)}"
        ).strip()

    if svc.backend == "sglang":
        if svc.group == "image":
            return (
                "sglang serve "
                f"--model-path {shlex.quote(svc.model_path)} "
                "--host 127.0.0.1 "
                f"--port {svc.port}"
            )
        return (
            "sglang serve "
            f"--model-path {shlex.quote(svc.model_path)} "
            "--host 127.0.0.1 "
            f"--port {svc.port} "
            f"--tensor-parallel-size {svc.tp} "
            "--trust-remote-code "
            f"{_sglang_model_override_args_for_service(svc)}"
            f"{_sglang_tool_args_for_service(svc)}"
            "--enable-metrics "
            "--disable-piecewise-cuda-graph"
        )

    if svc.backend == "vllm-omni":
        return (
            "python -m vllm_omni.entrypoints.cli.main serve "
            f"{shlex.quote(svc.model_path)} "
            "--omni "
            "--host 127.0.0.1 "
            f"--port {svc.port} "
            f"--num-gpus {svc.tp} "
            "--trust-remote-code"
        )

    raise ValueError(f"Unsupported backend: {svc.backend}")


def build_command(svc: Service) -> list[str]:
    return shlex.split(_build_cmd(svc))


def _build_env(svc: Service, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = svc.gpus
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if svc.backend == "continuum" and svc.lmcache_cpu_gb > 0:
        lmcache = svc.lmcache_cpu_gb
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = (
            str(int(lmcache)) if float(lmcache).is_integer() else str(lmcache)
        )
    if svc.backend == "continuum":
        overlay_dir = _continuum_overlay_dir(svc)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{overlay_dir}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath else str(overlay_dir)
        )
        env.setdefault("VLLM_USE_V1", "1")
        env.setdefault("VLLM_PLUGINS", "lora_filesystem_resolver")
    if svc.backend in {"vllm", "continuum"}:
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if svc.backend == "vllm-omni":
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def runtime_env(svc: Service) -> dict[str, str]:
    return _build_env(svc)


def pid_file(key: str) -> Path:
    return PID_DIR / f"{key.replace('.', '_')}.pid"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _signal_process_group(pid: int, sig: int) -> bool:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def _read_proc_children(pid: int) -> list[int]:
    children_file = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        raw = children_file.read_text().strip()
    except OSError:
        return []
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _collect_descendants(pid: int) -> set[int]:
    seen: set[int] = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in _read_proc_children(current):
            if child in seen:
                continue
            seen.add(child)
            stack.append(child)
    return seen


def _signal_pid_tree(pid: int, sig: int) -> set[int]:
    targets = _collect_descendants(pid)
    ordered = sorted(targets, reverse=True)
    if is_running(pid):
        ordered.append(pid)
    signaled: set[int] = set()
    for target in ordered:
        try:
            os.kill(target, sig)
            signaled.add(target)
        except ProcessLookupError:
            continue
    return signaled


def _pids_listening_on_port(port: int) -> set[int]:
    pids: set[int] = set()
    if shutil.which("lsof"):
        try:
            proc = subprocess.run(
                ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pids.add(int(line))
                except ValueError:
                    continue
        except OSError:
            pass
    if pids or not shutil.which("ss"):
        return pids

    try:
        proc = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return pids

    for match in re.finditer(r"pid=(\d+)", proc.stdout):
        try:
            pids.add(int(match.group(1)))
        except ValueError:
            continue
    return pids


def _wait_for_exit(pids: set[int], timeout_s: float) -> set[int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alive = {pid for pid in pids if is_running(pid)}
        if not alive:
            return set()
        time.sleep(0.25)
    return {pid for pid in pids if is_running(pid)}


def start_service(conda_exe: str, svc: Service) -> bool:
    pf = pid_file(svc.key)
    if pf.exists():
        pid = int(pf.read_text().strip())
        if is_running(pid):
            warn(f"{svc.key} already running (PID {pid})")
            return True
        pf.unlink(missing_ok=True)

    issues = validate_service(svc)
    if issues:
        error(f"{svc.key}: " + "; ".join(issues))
        return False
    if svc.backend == "continuum":
        _prepare_continuum_overlay(svc)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{svc.key.replace('.', '_')}.log"

    cmd = [conda_exe, "run", "--no-capture-output", "-n", svc.conda_env, *build_command(svc)]
    info(f"Starting {svc.key} on :{svc.port} using {svc.backend} [{svc.gpus}]")
    info("  " + " ".join(shlex.quote(x) for x in cmd))

    with log_file.open("w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=runtime_env(svc),
            start_new_session=True,
        )

    pf.write_text(str(proc.pid))
    time.sleep(2)
    if is_running(proc.pid):
        info(f"  started (PID {proc.pid})")
        return True

    error(f"  failed, check {log_file}")
    pf.unlink(missing_ok=True)
    return False


def stop_service(svc: Service) -> bool:
    pf = pid_file(svc.key)
    pid_from_file: int | None = None
    if pf.exists():
        try:
            pid_from_file = int(pf.read_text().strip())
        except ValueError:
            warn(f"{svc.key}: invalid PID file, ignoring")

    candidates: set[int] = set()
    if pid_from_file is not None:
        candidates.add(pid_from_file)
    candidates.update(_pids_listening_on_port(svc.port))

    live_candidates = {pid for pid in candidates if is_running(pid)}
    if not live_candidates:
        if pid_from_file is None:
            warn(f"{svc.key}: no PID file and nothing listening on :{svc.port}")
        else:
            warn(f"{svc.key}: tracked process already dead and nothing listening on :{svc.port}")
        pf.unlink(missing_ok=True)
        return True

    info(
        f"Stopping {svc.key} on :{svc.port} "
        f"(targets: {', '.join(str(pid) for pid in sorted(live_candidates))})"
    )
    signaled: set[int] = set()
    for pid in sorted(live_candidates):
        if not _signal_process_group(pid, signal.SIGTERM):
            _signal_pid_tree(pid, signal.SIGTERM)
        signaled.add(pid)
        signaled.update(_collect_descendants(pid))

    survivors = _wait_for_exit(signaled, 10.0)
    if survivors:
        warn(
            f"{svc.key}: forcing kill for "
            f"{', '.join(str(pid) for pid in sorted(survivors))}"
        )
        for pid in sorted(survivors):
            if not _signal_process_group(pid, signal.SIGKILL):
                _signal_pid_tree(pid, signal.SIGKILL)
        survivors = _wait_for_exit(survivors, 2.0)

    remaining_listeners = {
        pid for pid in _pids_listening_on_port(svc.port) if is_running(pid)
    }
    if remaining_listeners:
        warn(
            f"{svc.key}: processes still listening on :{svc.port}: "
            f"{', '.join(str(pid) for pid in sorted(remaining_listeners))}"
        )
    pf.unlink(missing_ok=True)
    return not remaining_listeners


def list_services(services: list[Service]) -> None:
    print(f"\n{CYAN}Deployable services{NC}\n")
    print(f"  {'NAME':<24} {'BACKEND':<12} {'PORT':<6} {'GPUS':<8} {'MODEL':<42}")
    print(f"  {'----':<24} {'-------':<12} {'----':<6} {'----':<8} {'-----':<42}")
    for svc in services:
        print(f"  {svc.key:<24} {svc.backend:<12} {svc.port:<6} {svc.gpus:<8} {svc.model_path:<42}")
    print()


def show_status(services: list[Service]) -> None:
    print(f"\n{CYAN}Service status{NC}\n")
    print(f"  {'NAME':<24} {'BACKEND':<12} {'PORT':<6} {'STATUS':<10} {'PID':<8}")
    print(f"  {'----':<24} {'-------':<12} {'----':<6} {'------':<10} {'---':<8}")
    for svc in services:
        pf = pid_file(svc.key)
        pid = pf.read_text().strip() if pf.exists() else "-"
        running = pid != "-" and is_running(int(pid))
        status = f"{GREEN}running{NC}" if running else f"{YELLOW}stopped{NC}"
        print(f"  {svc.key:<24} {svc.backend:<12} {svc.port:<6} {status:<19} {pid:<8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local MAS-Sys serving backends")
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start")
    p_start.add_argument("--all", action="store_true")
    p_start.add_argument("names", nargs="*")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--all", action="store_true")
    p_stop.add_argument("names", nargs="*")

    sub.add_parser("list")
    sub.add_parser("status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if not args.conf.exists():
        error(f"Config file not found: {args.conf}")
        sys.exit(1)

    services = parse_services(args.conf)
    svc_map = {svc.key: svc for svc in services}

    if args.command == "list":
        list_services(services)
        return
    if args.command == "status":
        show_status(services)
        return

    if args.command == "stop":
        names = list(svc_map) if args.all else (args.names or [])
        if not names:
            error("No services specified")
            sys.exit(1)
        for name in names:
            svc = svc_map.get(name)
            if svc is None:
                warn(f"Unknown service: {name}")
                continue
            stop_service(svc)
        return

    conda_exe = find_conda()
    targets = list(svc_map.values()) if args.all else [svc_map[n] for n in args.names if n in svc_map]
    missing = [n for n in args.names if n not in svc_map]
    for name in missing:
        warn(f"Unknown service: {name}")
    if not targets:
        error("No valid services specified")
        sys.exit(1)

    ok = 0
    fail = 0
    for svc in targets:
        if start_service(conda_exe, svc):
            ok += 1
        else:
            fail += 1
    info(f"Done. Started: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
