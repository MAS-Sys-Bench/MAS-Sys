"""Short-term memory with LLM summarization for overflow compression.

When the message count exceeds ``max_messages``, the oldest messages are
removed and compressed into a running summary via an LLM call.  The summary
is prepended as a system message when the agent builds its prompt, so that
earlier context is never completely lost.

Design references:
- MARBLE ``ShortTermMemory``: capacity limit + LLM summarization of oldest entries.
- MetaGPT ``RoleZeroLongTermMemory``: ``memory_k`` threshold + automatic overflow transfer.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional

from pydantic import Field, PrivateAttr

from app.memory.local_memory import LocalMemory
from app.schema import Message

try:
    from app.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.memory.long_term_memory import LongTermMemory


class ShortTermMemory(LocalMemory):
    """Agent-level short-term memory with automatic LLM summarization.

    Extends :class:`LocalMemory` with two capabilities:

    1. **Summarization** – when the message list exceeds *max_messages*, the
       oldest messages are popped, summarised by an LLM, and the summary is
       accumulated in :pyattr:`summary`.
    2. **Long-term memory offloading** – if a :class:`LongTermMemory` instance
       is attached, evicted messages are also stored there for later semantic
       retrieval.
    """

    # configuration
    summary: Optional[str] = Field(
        default=None,
        description="Running LLM-generated summary of evicted messages.",
    )
    summary_max_tokens: int = Field(
        default=200,
        description="Approximate max token budget for the summary text.",
    )
    enable_summarization: bool = Field(
        default=True,
        description="If False, overflow messages are silently discarded (same as LocalMemory).",
    )

    # Number of oldest messages to evict at once when overflow is triggered.
    _evict_count: int = PrivateAttr(default=2)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    # internal references (not serialised)
    _long_term_memory: Optional["LongTermMemory"] = PrivateAttr(default=None)
    _llm_instance: Optional[object] = PrivateAttr(default=None)  # app.llm.LLM

    # public API: attach helpers

    def attach_llm(self, llm: object) -> None:
        """Attach an :class:`~app.llm.LLM` instance used for summarization."""
        self._llm_instance = llm

    def attach_long_term_memory(self, ltm: "LongTermMemory") -> None:
        """Attach a :class:`LongTermMemory` to receive evicted messages."""
        self._long_term_memory = ltm

    # overridden mutators

    def add_message(self, message: Message) -> None:
        """Add a message. If overflow, trigger async summarization + offload."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._handle_overflow()

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages. Overflow handling fires once at the end."""
        self.messages.extend(messages)
        if len(self.messages) > self.max_messages:
            self._handle_overflow()

    # context construction

    def get_context_messages(self) -> List[Message]:
        """Return messages suitable for LLM prompt construction.

        If a summary exists it is prepended as a system message so the LLM
        retains awareness of earlier conversation turns.
        """
        msgs: List[Message] = []
        if self.summary:
            msgs.append(
                Message.system_message(
                    f"[Earlier conversation summary]\n{self.summary}"
                )
            )
        msgs.extend(self.messages)
        return msgs

    # internals

    def _handle_overflow(self) -> None:
        """Evict oldest messages, offload to LTM, and trigger summarization."""
        overflow_count = len(self.messages) - self.max_messages
        # Evict at least _evict_count to avoid thrashing
        evict_n = max(overflow_count, self._evict_count)
        evict_n = min(evict_n, len(self.messages))

        evicted = self.messages[:evict_n]
        self.messages = self.messages[evict_n:]

        # Offload to long-term memory (fire-and-forget async)
        if self._long_term_memory is not None:
            self._offload_to_ltm(evicted, self._long_term_memory)

        # Summarize evicted messages
        if self.enable_summarization:
            self._schedule_summarization(evicted)

    def _offload_to_ltm(self, evicted: List[Message], ltm: "LongTermMemory") -> None:
        """Store evicted messages in long-term memory."""
        for msg in evicted:
            if msg.content:
                metadata = {"role": msg.role}
                if msg.name:
                    metadata["name"] = msg.name
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ltm.store(msg.content, metadata=metadata))
                except RuntimeError:
                    # No running loop – skip async offload
                    logger.debug("No running event loop; skipping LTM offload.")

    def _schedule_summarization(self, evicted: List[Message]) -> None:
        """Schedule an async summarization task for *evicted* messages."""
        if self._llm_instance is None:
            logger.debug("No LLM attached; skipping summarization of evicted messages.")
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._summarize(evicted, self._llm_instance))
        except RuntimeError:
            logger.debug("No running event loop; skipping summarization.")

    async def _summarize(self, evicted: List[Message], llm: object) -> None:
        """Use the attached LLM to compress *evicted* messages into a summary.

        The new summary is merged with any existing summary.
        """
        # Build a textual representation of evicted messages
        lines: List[str] = []
        for msg in evicted:
            role = msg.role or "unknown"
            content = msg.content or ""
            if content:
                lines.append(f"[{role}] {content}")

        if not lines:
            return

        evicted_text = "\n".join(lines)

        prompt_parts: List[str] = []
        if self.summary:
            prompt_parts.append(f"Existing summary:\n{self.summary}\n")
        prompt_parts.append(f"New messages to incorporate:\n{evicted_text}\n")
        prompt_parts.append(
            f"Please produce a concise updated summary (max ~{self.summary_max_tokens} tokens) "
            "that captures all important information, decisions, and context from "
            "both the existing summary and the new messages. "
            "Focus on facts, decisions, and action items. "
            "Output ONLY the summary text, no preamble."
        )
        user_content = "\n".join(prompt_parts)

        try:
            summary_result = await llm.ask(
                messages=[Message.user_message(user_content)],
                system_msgs=[
                    Message.system_message(
                        "You are a helpful assistant that summarizes conversation history. "
                        "Be concise and preserve key information."
                    )
                ],
                stream=False,
            )
            if summary_result:
                self.summary = summary_result.strip()
                logger.debug(f"ShortTermMemory summary updated ({len(self.summary)} chars).")
        except Exception as e:
            logger.warning(f"Failed to summarize evicted messages: {e}")
