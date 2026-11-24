from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import time
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from app import continuum
from app.llm import LLM
from app.logger import logger
from app.memory.long_term_memory import LongTermMemory
from app.memory.short_term_memory import ShortTermMemory
from app.prompt.base import STUCK_STATE_PROMPT
from app.schema import ROLE_TYPE, AgentState, Message
from app.tracer import tracer


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    llm_config_name: Optional[str] = Field(
        None,
        description="Explicit LLM config key assigned by PlanAgent (e.g. 'lang.qwen2_5-14b').",
    )

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: Optional[LLM] = Field(default=None, description="Language model instance")
    memory: ShortTermMemory = Field(default_factory=ShortTermMemory, description="Agent's short-term memory")
    long_term_memory: Optional[LongTermMemory] = Field(
        default=None, description="Optional long-term memory for semantic retrieval"
    )
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2
    continuum_job_id: Optional[str] = Field(
        default=None,
        description="Stable serving job id for Continuum scheduler metadata.",
        exclude=True,
    )
    continuum_expected_llm_calls: Optional[int] = Field(
        default=None,
        description="Expected LLM calls in this agent run when known.",
        exclude=True,
    )
    continuum_llm_call_index: int = Field(
        default=0,
        description="Per-agent-run LLM call counter for Continuum metadata.",
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with its PlanAgent-assigned LLM config."""
        if not self.llm_config_name:
            raise ValueError(
                f"Agent '{self.name}' is missing llm_config_name. "
                "Agents must be instantiated from a MASConfig produced by PlanAgent."
            )
        self.llm = LLM(config_name=self.llm_config_name)
        if not isinstance(self.memory, ShortTermMemory):
            self.memory = ShortTermMemory()
        # Wire up LLM and LongTermMemory to ShortTermMemory
        self.memory.attach_llm(self.llm)
        if self.long_term_memory is not None:
            self.memory.attach_long_term_memory(self.long_term_memory)
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            if self.state not in (AgentState.FINISHED, AgentState.ERROR):
                self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")

                # Trace: start step
                tracer.start_agent_step(self.name, self.current_step)

                step_result = await self.step()

                # Trace: end step
                tracer.end_agent_step(self.name)

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")
                if self.state == AgentState.FINISHED:
                    logger.info(f"Agent finished execution after step {self.current_step}")
                    break

            if self.current_step >= self.max_steps:
                self.current_step = 0
                if self.state != AgentState.FINISHED:
                    self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        return "\n".join(results) if results else "No steps executed"

    def start_single_step_trace(self) -> None:
        """Start a trace record for agents that override ``run()`` with one logical step."""
        tracer.start_agent_step(self.name, self.current_step)

    def end_single_step_trace(self) -> None:
        """Finish a trace record for agents that override ``run()`` with one logical step."""
        tracer.end_agent_step(self.name)

    def record_simple_llm_trace(
        self,
        *,
        input_messages: Any,
        output_content: str,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record a plain LLM call for non-tool ReAct agents."""
        tracer.record_think(
            self.name,
            input_messages=input_messages,
            output_content=output_content,
            tool_calls_info=[],
            start_time=start_time,
            end_time=end_time,
            input_tokens=getattr(self.llm, "last_input_tokens", 0),
            completion_tokens=getattr(self.llm, "last_completion_tokens", 0),
        )

    def initialize_from_params(self) -> None:
        """Finalize any instance state derived from agent_params after construction."""
        return None

    def next_continuum_llm_context(self) -> dict[str, Any]:
        """Return per-call Continuum metadata for the next LLM request."""
        self.continuum_llm_call_index += 1
        expected_calls = self.continuum_expected_llm_calls
        if expected_calls is not None:
            is_last_step = self.continuum_llm_call_index >= expected_calls
        else:
            is_last_step = self.current_step >= self.max_steps

        job_id = self.continuum_job_id or (
            continuum.job_id(tracer.current_session_id(), "agent", self.name)
        )
        return {
            continuum.TRACE_JOB_ID: job_id,
            continuum.TRACE_CALL_INDEX: self.continuum_llm_call_index,
            continuum.TRACE_IS_LAST_STEP: is_last_step,
        }

    async def ask_with_trace(
        self,
        *,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        stream: bool = True,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Wrapper around ``llm.ask`` that records think/token trace for simple agents."""
        start_time = time.time()
        continuum_context = self.next_continuum_llm_context()
        with tracer.llm_call_scope(
            runtime="legacy",
            scope="agent",
            agent_id=self.name,
            agent_class=type(self).__name__,
            llm_config_name=self.llm_config_name,
            step=self.current_step,
            **continuum_context,
        ):
            response = await self.llm.ask(
                messages=messages,
                system_msgs=system_msgs,
                stream=stream,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        end_time = time.time()
        self.record_simple_llm_trace(
            input_messages=messages,
            output_content=response,
            start_time=start_time,
            end_time=end_time,
        )
        return response

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        self.next_step_prompt = f"{STUCK_STATE_PROMPT}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {STUCK_STATE_PROMPT}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
