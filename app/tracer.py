"""Tracer: global singleton for collecting execution traces and metadata.

Produces two JSON files after each MAS run:
  - trace.json:    Per-agent ReAct loop details (think/act inputs & outputs)
  - metadata.json: Timing breakdown, token usage, parallel markers, E2E total
"""

import contextvars
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracer:
    """Global singleton that collects trace events and metadata during a MAS run."""

    _instance: Optional["Tracer"] = None

    def __new__(cls) -> "Tracer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "tracer_session_id",
            default="default",
        )
        self._llm_call_context_var: contextvars.ContextVar[Dict[str, Any]] = (
            contextvars.ContextVar(
                "tracer_llm_call_context",
                default={},
            )
        )
        self._sessions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.reset()

    def _new_trace_data(self) -> Dict[str, Any]:
        return {
            "plan_agent": {},
            "coordination": [],
            "llm_calls": [],
            "orla": {
                "stage_executions": [],
                "agent_loop_turns": [],
                "tool_calls": [],
            },
            "agents": {},
        }

    def _new_meta_data(self) -> Dict[str, Any]:
        return {
            "e2e_start_time": None,
            "e2e_end_time": None,
            "e2e_duration_s": None,
            "planning": {},
            "orla": {
                "workflows": [],
                "compilations": [],
                "stage_executions": [],
                "agent_loop_turns": [],
                "tool_calls": [],
            },
            "flow": {
                "topology": None,
                "phases": [],
            },
            "agents": {},
            "coordination_calls": [],
            "llm_calls": {
                "captured": False,
                "count": 0,
                "prompt_capture_enabled": False,
                "client_latency_s": [],
                "ttft_s": [],
                "backend_ttft_s": [],
                "backend_queue_wait_s": [],
                "backend_latency_s": [],
                "kv_cache_reuse_rate": [],
                "orla_ttft_s": [],
                "orla_tpot_s": [],
                "orla_queue_wait_s": [],
                "orla_scheduler_decision_s": [],
                "orla_dispatch_s": [],
                "orla_backend_request_latency_s": [],
                "orla_estimated_cost_usd": [],
            },
            "token_summary": {
                "total_input_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _get_session(self, session_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        current_session_id = session_id or self._session_id_var.get()
        if current_session_id not in self._sessions:
            self._sessions[current_session_id] = {
                "trace": self._new_trace_data(),
                "meta": self._new_meta_data(),
            }
        return self._sessions[current_session_id]

    @property
    def _trace(self) -> Dict[str, Any]:
        return self._get_session()["trace"]

    @property
    def _meta(self) -> Dict[str, Any]:
        return self._get_session()["meta"]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def activate_session(self, session_id: str, reset: bool = False) -> None:
        self._session_id_var.set(session_id)
        if reset or session_id not in self._sessions:
            self._sessions[session_id] = {
                "trace": self._new_trace_data(),
                "meta": self._new_meta_data(),
            }

    def clear_session(self, session_id: Optional[str] = None) -> None:
        current_session_id = session_id or self._session_id_var.get()
        self._sessions.pop(current_session_id, None)
        if current_session_id == self._session_id_var.get():
            self._session_id_var.set("default")

    def current_session_id(self) -> str:
        return self._session_id_var.get()

    def prompt_capture_enabled(self) -> bool:
        value = os.getenv("MAS_TRACE_LLM_PROMPTS", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @contextmanager
    def llm_call_scope(self, **context: Any):
        current = dict(self._llm_call_context_var.get())
        current.update({key: value for key, value in context.items() if value is not None})
        token = self._llm_call_context_var.set(current)
        try:
            yield
        finally:
            self._llm_call_context_var.reset(token)

    def current_llm_call_context(self) -> Dict[str, Any]:
        return dict(self._llm_call_context_var.get())

    def reset(self) -> None:
        """Clear all collected data. Call at the start of each run."""
        session = self._get_session()
        session["trace"] = self._new_trace_data()
        session["meta"] = self._new_meta_data()

    # ── E2E timing ────────────────────────────────────────────────────────────

    def start_e2e(self) -> None:
        self._meta["e2e_start_time"] = time.time()

    def end_e2e(self) -> None:
        self._meta["e2e_end_time"] = time.time()
        if self._meta["e2e_start_time"] is not None:
            self._meta["e2e_duration_s"] = round(
                self._meta["e2e_end_time"] - self._meta["e2e_start_time"], 4
            )

    # ── Planning phase ────────────────────────────────────────────────────────

    def record_planning(
        self,
        *,
        input_text: str,
        output_raw: str,
        start_time: float,
        end_time: float,
        input_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        duration = round(end_time - start_time, 4)
        self._trace["plan_agent"]["plan"] = {
            "input": input_text[:2000],
            "output": output_raw[:5000],
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
        }
        self._meta["planning"] = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
            "input_tokens": input_tokens,
            "completion_tokens": completion_tokens,
        }
        self._add_phase(
            phase_type="planning",
            agent_ids=["plan_agent"],
            start_time=start_time,
            end_time=end_time,
            parallel=False,
        )
        self._accumulate_tokens(input_tokens, completion_tokens)

    # ── Coordination calls (assign_subtasks / summarize / should_continue) ───

    def record_coordination_call(
        self,
        *,
        coordinator_id: str,
        call_type: str,
        input_text: str,
        output_text: str,
        start_time: float,
        end_time: float,
        input_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        duration = round(end_time - start_time, 4)
        entry = {
            "call_type": call_type,
            "input": input_text[:2000],
            "output": output_text[:5000],
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
            "input_tokens": input_tokens,
            "completion_tokens": completion_tokens,
        }
        self._trace["coordination"].append(
            {
                "coordinator_id": coordinator_id,
                **entry,
            }
        )
        # metadata
        self._meta["coordination_calls"].append(
            {
                "coordinator_id": coordinator_id,
                **entry,
            }
        )
        self._accumulate_tokens(input_tokens, completion_tokens)

    # ── Flow topology ─────────────────────────────────────────────────────────

    def set_topology(self, topology: str) -> None:
        self._meta["flow"]["topology"] = topology

    def _add_phase(
        self,
        *,
        phase_type: str,
        agent_ids: List[str],
        start_time: float,
        end_time: float,
        parallel: bool,
        sub_phases: Optional[List[Dict]] = None,
        extra: Optional[Dict[str, Any]] = None,
        count_toward_breakdown: bool = True,
    ) -> None:
        phase = {
            "type": phase_type,
            "agent_ids": agent_ids,
            "parallel": parallel,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": round(end_time - start_time, 4),
            "sub_phases": sub_phases or [],
            "count_toward_breakdown": count_toward_breakdown,
        }
        if extra:
            phase.update(extra)
        self._meta["flow"]["phases"].append(phase)

    def record_wait_phase(
        self,
        *,
        wait_type: str,
        start_time: float,
        end_time: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._add_phase(
            phase_type=f"{wait_type}_wait",
            agent_ids=[],
            start_time=start_time,
            end_time=end_time,
            parallel=False,
            extra=details,
        )

    def record_agent_run_phase(
        self,
        *,
        agent_id: str,
        start_time: float,
        end_time: float,
        parallel: bool = False,
    ) -> None:
        self._add_phase(
            phase_type="agent_run",
            agent_ids=[agent_id],
            start_time=start_time,
            end_time=end_time,
            parallel=parallel,
        )

    def record_parallel_batch_phase(
        self,
        *,
        agent_ids: List[str],
        start_time: float,
        end_time: float,
        individual_times: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        sub_phases = []
        if individual_times:
            for aid, times in individual_times.items():
                sub_phases.append(
                    {
                        "type": "agent_run",
                        "agent_id": aid,
                        "start_time": times["start_time"],
                        "end_time": times["end_time"],
                        "duration_s": round(
                            times["end_time"] - times["start_time"], 4
                        ),
                    }
                )
        self._add_phase(
            phase_type="parallel_batch",
            agent_ids=agent_ids,
            start_time=start_time,
            end_time=end_time,
            parallel=True,
            sub_phases=sub_phases,
        )

    def record_coordination_phase(
        self,
        *,
        coordinator_id: str,
        call_type: str,
        start_time: float,
        end_time: float,
        count_toward_breakdown: bool = True,
    ) -> None:
        self._add_phase(
            phase_type="coordination",
            agent_ids=[coordinator_id],
            start_time=start_time,
            end_time=end_time,
            parallel=False,
            count_toward_breakdown=count_toward_breakdown,
            extra={"call_type": call_type},
        )

    def record_orla_workflow(
        self,
        *,
        workflow_id: str,
        stage_ids: List[str],
        backend_names: List[str],
        start_time: float,
        end_time: float,
        hidden: bool = False,
    ) -> None:
        self._meta["orla"]["workflows"].append(
            {
                "workflow_id": workflow_id,
                "stage_ids": stage_ids,
                "backend_names": backend_names,
                "hidden": hidden,
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": round(end_time - start_time, 4),
            }
        )

    def record_orla_segment_phase(
        self,
        *,
        agent_ids: List[str],
        segment_id: str,
        start_time: float,
        end_time: float,
    ) -> None:
        self._add_phase(
            phase_type="orla_segment",
            agent_ids=agent_ids,
            start_time=start_time,
            end_time=end_time,
            parallel=False,
            extra={"segment_id": segment_id},
        )

    def record_orla_compilation(self, compilation: Dict[str, Any]) -> None:
        self._meta["orla"]["compilations"].append(compilation)

    def record_orla_stage_execution(
        self,
        *,
        workflow_id: str,
        stage_id: str,
        start_time: float,
        end_time: float,
        ready_time: Optional[float] = None,
        tags: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "workflow_id": workflow_id,
            "stage_id": stage_id,
            "ready_time": ready_time,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": round(end_time - start_time, 4),
            "queue_delay_s": (
                round(start_time - ready_time, 4) if ready_time is not None else None
            ),
            "tags": tags or {},
        }
        if extra:
            entry.update(extra)
        self._trace["orla"]["stage_executions"].append(entry)
        self._meta["orla"]["stage_executions"].append(entry)
        if (entry.get("tags") or {}).get("execution_mode") != "agent_loop":
            self._record_llm_metric_summary(entry)
        self._add_phase(
            phase_type="orla_stage",
            agent_ids=[],
            start_time=start_time,
            end_time=end_time,
            parallel=False,
            count_toward_breakdown=False,
            extra={
                "workflow_id": workflow_id,
                "stage_id": stage_id,
                "ready_time": ready_time,
                "queue_delay_s": entry["queue_delay_s"],
                "tags": tags or {},
                **(extra or {}),
            },
        )

    def record_orla_agent_loop_turn(
        self,
        *,
        workflow_id: str,
        stage_id: str,
        turn_index: int,
        start_time: float,
        end_time: float,
        llm_start_time: float,
        llm_end_time: float,
        tool_start_time: float | None = None,
        tool_end_time: float | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "workflow_id": workflow_id,
            "stage_id": stage_id,
            "turn_index": turn_index,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": round(end_time - start_time, 4),
            "llm_start_time": llm_start_time,
            "llm_end_time": llm_end_time,
            "llm_duration_s": round(llm_end_time - llm_start_time, 4),
            "tool_start_time": tool_start_time,
            "tool_end_time": tool_end_time,
            "tool_duration_s": (
                round(tool_end_time - tool_start_time, 4)
                if tool_start_time is not None and tool_end_time is not None
                else None
            ),
        }
        if extra:
            entry.update(extra)
        self._trace["orla"]["agent_loop_turns"].append(entry)
        self._meta["orla"]["agent_loop_turns"].append(entry)
        self._record_llm_metric_summary(entry)

    def record_orla_tool_call(
        self,
        *,
        workflow_id: str,
        stage_id: str,
        turn_index: int | None,
        tool_name: str,
        tool_call_id: str,
        start_time: float,
        end_time: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "workflow_id": workflow_id,
            "stage_id": stage_id,
            "turn_index": turn_index,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": round(end_time - start_time, 4),
        }
        if extra:
            entry.update(extra)
        self._trace["orla"]["tool_calls"].append(entry)
        self._meta["orla"]["tool_calls"].append(entry)

    def record_llm_call(
        self,
        *,
        transport: str,
        invocation: str,
        messages: Any,
        model: str = "",
        base_url: str = "",
        tools: Any = None,
        tool_choice: Any = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        prompt_capture = self.prompt_capture_enabled()

        entry: Dict[str, Any] = {
            "index": len(self._trace["llm_calls"]) + 1,
            "transport": transport,
            "invocation": invocation,
            "model": model,
            "base_url": base_url,
        }
        if prompt_capture:
            entry["messages"] = _serialize_messages(messages, truncate=False)
            entry["tools"] = _safe_serialize(tools, truncate=False)
            entry["tool_choice"] = _safe_serialize(tool_choice, truncate=False)
        else:
            entry["message_count"] = len(messages) if isinstance(messages, list) else None
            entry["has_tools"] = bool(tools)
            entry["tool_choice"] = _safe_serialize(tool_choice, truncate=True)
        if start_time is not None:
            entry["start_time"] = start_time
        if end_time is not None:
            entry["end_time"] = end_time
        if start_time is not None and end_time is not None:
            entry["duration_s"] = round(end_time - start_time, 4)
        entry.update(self.current_llm_call_context())
        if extra:
            entry.update(_safe_serialize(extra, truncate=False))

        self._trace["llm_calls"].append(entry)
        self._meta["llm_calls"]["captured"] = True
        self._meta["llm_calls"]["prompt_capture_enabled"] = prompt_capture
        self._meta["llm_calls"]["count"] = len(self._trace["llm_calls"])
        self._record_llm_metric_summary(entry)

    def _record_llm_metric_summary(self, entry: Dict[str, Any]) -> None:
        metrics = entry.get("response_metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        observed_latency = entry.get("client_observed_latency_s")
        if observed_latency is None:
            observed_latency = entry.get("duration_s")
        self._append_numeric_llm_metric("client_latency_s", observed_latency)
        self._append_numeric_llm_metric("ttft_s", entry.get("client_ttft_s"))

        backend_ttft_s = _metric_seconds(metrics, "backend_ttft_s")
        queue_wait_s = _metric_seconds(metrics, "queue_wait_s")
        backend_latency_s = _metric_seconds(metrics, "backend_latency_s")
        kv_cache_reuse_rate = _kv_cache_reuse_rate(metrics, entry)

        self._append_numeric_llm_metric("backend_ttft_s", backend_ttft_s)
        self._append_numeric_llm_metric("backend_queue_wait_s", queue_wait_s)
        self._append_numeric_llm_metric("backend_latency_s", backend_latency_s)
        self._append_numeric_llm_metric("kv_cache_reuse_rate", kv_cache_reuse_rate)
        self._append_numeric_llm_metric(
            "orla_ttft_s",
            _metric_seconds(metrics, "orla_ttft_s")
            or _metric_seconds(metrics, "ttft_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_tpot_s",
            _metric_seconds(metrics, "orla_tpot_s")
            or _metric_seconds(metrics, "tpot_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_queue_wait_s",
            _metric_seconds(metrics, "orla_queue_wait_s")
            or _metric_seconds(metrics, "queue_wait_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_scheduler_decision_s",
            _metric_seconds(metrics, "orla_scheduler_decision_s")
            or _metric_seconds(metrics, "scheduler_decision_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_dispatch_s",
            _metric_seconds(metrics, "orla_dispatch_s")
            or _metric_seconds(metrics, "dispatch_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_backend_request_latency_s",
            _metric_seconds(metrics, "orla_backend_request_latency_s")
            or _metric_seconds(metrics, "backend_latency_ms", divisor=1000.0),
        )
        self._append_numeric_llm_metric(
            "orla_estimated_cost_usd",
            _metric_number(metrics, "orla_estimated_cost_usd", "estimated_cost_usd"),
        )

    def _append_numeric_llm_metric(self, key: str, value: Any) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        self._meta["llm_calls"].setdefault(key, []).append(round(numeric, 4))

    # ── Agent-level trace recording ───────────────────────────────────────────

    def _ensure_agent(self, agent_id: str) -> Dict:
        if agent_id not in self._trace["agents"]:
            self._trace["agents"][agent_id] = {"steps": []}
        if agent_id not in self._meta["agents"]:
            self._meta["agents"][agent_id] = {
                "total_steps": 0,
                "input_tokens": 0,
                "completion_tokens": 0,
                "total_duration_s": 0.0,
                "tool_calls": [],
            }
        return self._trace["agents"][agent_id]

    def start_agent_step(self, agent_id: str, step_number: int) -> Dict:
        agent_trace = self._ensure_agent(agent_id)
        step_record = {
            "step": step_number,
            "start_time": time.time(),
            "end_time": None,
            "duration_s": None,
            "think": None,
            "act": None,
        }
        agent_trace["steps"].append(step_record)
        return step_record

    def end_agent_step(self, agent_id: str) -> None:
        agent_trace = self._ensure_agent(agent_id)
        if agent_trace["steps"]:
            step = agent_trace["steps"][-1]
            step["end_time"] = time.time()
            step["duration_s"] = round(step["end_time"] - step["start_time"], 4)
            self._meta["agents"][agent_id]["total_steps"] = len(agent_trace["steps"])

    def record_think(
        self,
        agent_id: str,
        *,
        input_messages: Any,
        output_content: str,
        tool_calls_info: Optional[List[Dict]] = None,
        start_time: float,
        end_time: float,
        input_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        agent_trace = self._ensure_agent(agent_id)
        duration = round(end_time - start_time, 4)

        think_record = {
            "input": _summarize_messages(input_messages),
            "output": output_content[:3000],
            "tool_calls": tool_calls_info or [],
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
            "input_tokens": input_tokens,
            "completion_tokens": completion_tokens,
        }

        if agent_trace["steps"]:
            agent_trace["steps"][-1]["think"] = think_record

        meta_agent = self._meta["agents"][agent_id]
        meta_agent["input_tokens"] += input_tokens
        meta_agent["completion_tokens"] += completion_tokens
        self._accumulate_tokens(input_tokens, completion_tokens)

    def record_tool_execution(
        self,
        agent_id: str,
        *,
        tool_name: str,
        tool_input: Any,
        tool_output: str,
        start_time: float,
        end_time: float,
    ) -> None:
        agent_trace = self._ensure_agent(agent_id)
        duration = round(end_time - start_time, 4)

        tool_record = {
            "tool_name": tool_name,
            "input": _safe_serialize(tool_input),
            "output": tool_output[:3000],
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
        }

        # Add to current step's act section
        if agent_trace["steps"]:
            step = agent_trace["steps"][-1]
            if step["act"] is None:
                step["act"] = {
                    "tool_calls": [],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_s": duration,
                }
            step["act"]["tool_calls"].append(tool_record)
            # Extend the act end_time to the latest tool
            step["act"]["end_time"] = end_time
            step["act"]["duration_s"] = round(
                end_time - step["act"]["start_time"], 4
            )

        # Also record in metadata agent summary
        self._meta["agents"][agent_id]["tool_calls"].append(
            {
                "tool_name": tool_name,
                "duration_s": duration,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    def add_agent_duration(self, agent_id: str, duration: float) -> None:
        self._ensure_agent(agent_id)
        self._meta["agents"][agent_id]["total_duration_s"] = round(
            self._meta["agents"][agent_id]["total_duration_s"] + duration, 4
        )

    # ── Token helpers ─────────────────────────────────────────────────────────

    def _accumulate_tokens(self, input_tokens: int, completion_tokens: int) -> None:
        self._meta["token_summary"]["total_input_tokens"] += input_tokens
        self._meta["token_summary"]["total_completion_tokens"] += completion_tokens
        self._meta["token_summary"]["total_tokens"] += input_tokens + completion_tokens

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path, prefix: str = "") -> None:
        """Write trace.json and metadata.json to *output_dir*.

        Args:
            output_dir: Directory to write JSON files into.
            prefix: Optional filename prefix, e.g. ``"2024-01-15_"`` produces
                    ``2024-01-15_trace.json`` and ``2024-01-15_metadata.json``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute time breakdown verification
        phases = self._meta["flow"]["phases"]
        # Only sum top-level wall-clock durations (parallel phases counted once)
        phases_total = sum(
            p["duration_s"] for p in phases if p.get("count_toward_breakdown", True)
        )
        waiting_total = sum(
            p["duration_s"]
            for p in phases
            if p.get("count_toward_breakdown", True)
            and str(p.get("type", "")).endswith("_wait")
        )
        self._meta["time_breakdown"] = {
            "phases_sum_s": round(phases_total, 4),
            "waiting_sum_s": round(waiting_total, 4),
            "active_execution_s": round(phases_total - waiting_total, 4),
            "e2e_duration_s": self._meta["e2e_duration_s"],
            "match": (
                abs(phases_total - (self._meta["e2e_duration_s"] or 0)) < 1.0
                if self._meta["e2e_duration_s"] is not None
                else False
            ),
        }

        trace_path = output_dir / f"{prefix}trace.json"
        meta_path = output_dir / f"{prefix}metadata.json"
        llm_calls_path = output_dir / f"{prefix}llm_calls.json"

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(self._trace, f, ensure_ascii=False, indent=2, default=str)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2, default=str)

        if self._trace["llm_calls"]:
            with open(llm_calls_path, "w", encoding="utf-8") as f:
                json.dump(self._trace["llm_calls"], f, ensure_ascii=False, indent=2, default=str)


# ── Module-level singleton accessor ──────────────────────────────────────────

tracer = Tracer()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _serialize_message(message: Any, *, truncate: bool) -> Any:
    """Convert a single message-like object into a JSON-serializable payload."""
    if hasattr(message, "to_dict"):
        payload = message.to_dict()
    elif isinstance(message, dict):
        payload = dict(message)
    else:
        payload = {}
        for attr in ("role", "content", "tool_call_id", "tool_name", "name", "tool_calls"):
            value = getattr(message, attr, None)
            if value not in (None, "", []):
                payload[attr] = value
        if not payload:
            payload = {"content": str(message)}

    if "tool_name" in payload and "name" not in payload:
        payload["name"] = payload["tool_name"]

    if truncate and isinstance(payload.get("content"), str) and len(payload["content"]) > 2000:
        payload = {**payload, "content": payload["content"][:2000] + "...(truncated)"}

    if "tool_calls" in payload:
        payload["tool_calls"] = _safe_serialize(payload["tool_calls"], truncate=truncate)

    return payload


def _serialize_messages(messages: Any, *, truncate: bool) -> Any:
    """Convert message list to a JSON-serializable payload."""
    if not messages:
        return []
    result = []
    for msg in messages:
        result.append(_serialize_message(msg, truncate=truncate))
    return result


def _summarize_messages(messages: Any) -> Any:
    return _serialize_messages(messages, truncate=True)


def _safe_serialize(obj: Any, *, truncate: bool = True) -> Any:
    """Make an object JSON-serializable, truncating large strings."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v, truncate=truncate) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, truncate=truncate) for v in obj]
    if truncate and isinstance(obj, str) and len(obj) > 2000:
        return obj[:2000] + "...(truncated)"
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)[:1000]


def _metric_seconds(
    metrics: Dict[str, Any],
    *names: str,
    divisor: float = 1.0,
) -> Optional[float]:
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return float(value) / divisor
        except (TypeError, ValueError):
            continue
    return None


def _metric_number(source: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = source.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _kv_cache_reuse_rate(
    metrics: Dict[str, Any],
    entry: Dict[str, Any],
) -> Optional[float]:
    direct = _metric_number(metrics, "kv_cache_reuse_rate", "cache_reuse_rate")
    if direct is not None:
        return direct

    cached_tokens = _metric_number(
        metrics,
        "cached_tokens",
        "num_cached_tokens",
    )
    prompt_tokens = _metric_number(metrics, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _metric_number(entry, "prompt_tokens")
    if cached_tokens is None or prompt_tokens is None or prompt_tokens <= 0:
        return None
    return max(0.0, min(1.0, cached_tokens / prompt_tokens))
