import json
import os
import re
import time
from typing import Any, Optional

from vllm.logger import init_logger
try:
    from vllm.tokenizers import TokenizerLike, get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import AnyTokenizer as TokenizerLike
    from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.request import Request

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = float(os.getenv("CONTINUUM_DEFAULT_PIN_SECONDS", "10.0"))
MAX_PIN_SECONDS_CONTINUUM = float(os.getenv("CONTINUUM_MAX_PIN_SECONDS", "60.0"))
PIN_MARGIN_SECONDS_CONTINUUM = float(os.getenv("CONTINUUM_PIN_MARGIN_SECONDS", "5.0"))

class ContinuumRecorder:
    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing
        self.scheduling_times = []  # List of {start_time, end_time, duration}

    def print_history(self):
        import os
        import json

        # Per-run output directory (set by launcher); fallback to default
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")
        os.makedirs(output_dir, exist_ok=True)

        # Atomic write to avoid partial reads by other processes
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(tmp_path, final_path)

    def request_arrives(self, request: Request):
        if request.job_id not in self.job_id_to_history:
            self.job_id_to_history[request.job_id] = []
        self.job_id_to_history[request.job_id].append({
            "Request_arrival_time": time.time(),
            "last_func_call": request.last_func_call,
            "this_func_call": request.this_func_call,
            "last_tool_duration_s": request.last_tool_duration_s,
            "is_last_step": request.is_last_step,
        })

    def request_finished(self, request: Request):
        self.job_id_to_history[request.job_id].append({
            "Request_departure_time": time.time(),
            "func_call": request.this_func_call,
            "last_func_call": request.last_func_call,
            "is_last_step": request.is_last_step,
        })

    def request_evicted_from_running_queue(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request, length_of_pin: float):
        self.job_id_to_history[request.job_id].append({
            "pinned_time": time.time(),
            "pin_seconds": length_of_pin,
            "func_call": request.this_func_call,
        })

    def request_unpinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int, hit_length: int = 0):
        self.job_id_to_history[request.job_id].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

    def request_evicted_to_running(self, request: Request, prompt_length: int, hit_length: int):
        self.job_id_to_history[request.job_id].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

class ToolCallParser:
    """Parser for extracting function calls from LLM output.

    Uses the same parsing logic as mini-swe-agent to extract bash commands
    from markdown code blocks and identify the function call.

    This can be extended for other datasets with different parsing logic.
    """

    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Args:
            text: Output text from the LLM

        Returns:
            The function call name (e.g., "ls", "cd", "git"), or None if not found
        """
        if not text:
            return None

        # Hermes/Qwen-style OpenAI tool-call payloads.
        tool_call_matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL,
        )
        for raw_call in tool_call_matches:
            func = self._parse_json_tool_name(raw_call)
            if func:
                return func

        # Required-tool mode may decode to a raw list of function definitions.
        func = self._parse_json_tool_name(text)
        if func:
            return func

        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if name_match:
            return name_match.group(1)

        # Same regex pattern as mini-swe-agent: r"```bash\s*\n(.*?)\n```"
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)

        if len(actions) == 1:
            bash_action = actions[0].strip()
            # Extract the first word (command) from the action
            words = bash_action.split()
            if words:
                return words[0]

        return None

    @staticmethod
    def _parse_json_tool_name(text: str) -> Optional[str]:
        try:
            payload = json.loads(text.strip())
        except Exception:
            return None
        candidates = payload if isinstance(payload, list) else [payload]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name:
                return name
            function = item.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    return name
        return None

class ToolCallEstimator:
    def __init__(
        self,
        tokenizer: Optional[TokenizerLike] = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}

        self.job_to_history: dict[str, list[dict[str, Any]]] = {}

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            try:
                self.tokenizer = get_tokenizer(
                    tokenizer_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                )
                logger.info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize parser (can be customized for different datasets)
        self.parser = parser if parser is not None else ToolCallParser()

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]

    def _record_func_call_exec_time(self, func: str, exec_time: float) -> None:
        if exec_time < 0:
            return
        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)
        samples = self.record_func_call_to_exec_time[func]
        self.func_call_to_exec_time[func] = sum(samples) / len(samples)

    def update_func_call_exec_time(
        self,
        job_id: str,
        *,
        func: Optional[str] = None,
        exec_time: Optional[float] = None,
    ) -> None:
        history = self.job_to_history.get(job_id)
        if not history and (func is None or exec_time is None):
            return

        if func is not None and exec_time is not None:
            self._record_func_call_exec_time(func, exec_time)
            return

        last_event = history[-1]
        last_departure_time = last_event.get("departure_time")
        func = func or last_event.get("func_call")
        if last_departure_time is None or func is None:
            return

        self._record_func_call_exec_time(func, time.time() - last_departure_time)
        return

    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return FIXED_THRESHOLD_CONTINUUM

        this_func_call_exec_time = self.get_func_call_exec_time(request.this_func_call)
        if this_func_call_exec_time is None:
            return FIXED_THRESHOLD_CONTINUUM

        return min(
            MAX_PIN_SECONDS_CONTINUUM,
            max(
                FIXED_THRESHOLD_CONTINUUM,
                this_func_call_exec_time + PIN_MARGIN_SECONDS_CONTINUUM,
            ),
        )

    def request_arrives(self, request: Request) -> None:
        logger.debug(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        if request.job_id not in self.job_to_history:
            self.job_to_history[request.job_id] = []
            self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
            if request.last_func_call and request.last_tool_duration_s is not None:
                self.update_func_call_exec_time(
                    request.job_id,
                    func=request.last_func_call,
                    exec_time=float(request.last_tool_duration_s),
                )
            return

        last_event = self.job_to_history[request.job_id][-1]
        history_last_func_call = last_event.get("func_call")
        client_last_func_call = request.last_func_call
        request.last_func_call = client_last_func_call or history_last_func_call
        logger.debug(f"Request job id: {request.job_id}, last func call: {request.last_func_call}")

        if client_last_func_call and request.last_tool_duration_s is not None:
            self.update_func_call_exec_time(
                request.job_id,
                func=client_last_func_call,
                exec_time=float(request.last_tool_duration_s),
            )
        else:
            self.update_func_call_exec_time(
                request.job_id,
                func=history_last_func_call,
            )

        self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
        return

    def request_finished(self, request: Request) -> None:
        logger.debug(f"Request job id finishing: {request.job_id}, time is {time.time()}")

        # Detokenize output and parse function call
        this_func_call = request.this_func_call
        if this_func_call is None and self.tokenizer is not None and len(request.output_token_ids) > 0:
            try:
                # Detokenize the output tokens
                output_text = self.tokenizer.decode(
                    request.output_token_ids,
                    skip_special_tokens=True
                )

                # Parse function call using the parser
                this_func_call = self.parser.parse(output_text)

                if this_func_call:
                    logger.debug(f"Extracted func_call: {this_func_call} from output")
                else:
                    logger.debug(f"No function call found in output: {output_text[:200]}")
            except Exception as e:
                logger.warning(f"Error detokenizing/parsing output for request {request.request_id}: {e}")

        request.this_func_call = this_func_call
        if request.job_id not in self.job_to_history:
            self.job_to_history[request.job_id] = []
        self.job_to_history[request.job_id].append({
            "departure_time": time.time(),
            "func_call": request.this_func_call
        })
        return


# Backward-compatible name used by older Continuum scheduler patches.
Continuum_Recorder = ContinuumRecorder
