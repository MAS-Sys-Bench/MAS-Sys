"""Prompt templates for generic MAS runtime coordination."""

ASSIGN_SUBTASKS_SYSTEM = (
    "You are a task coordinator for a multi-agent system. "
    "Given the overall task, a list of agent profiles, and the current progress, "
    "assign a specific, actionable subtask to EACH agent. "
    "Return ONLY a JSON object mapping agent_id to subtask string, e.g. "
    '{"agent_a": "Subtask for agent_a ...", "agent_b": "Subtask for agent_b ..."}'
)

ASSIGN_SUBTASKS_USER = (
    "Overall task:\n{task}\n\n"
    "Agent profiles:\n{agent_profiles}\n\n"
    "Progress so far:\n{progress}\n\n"
    "Assign a distinct subtask to each agent. "
    "Return ONLY valid JSON (no markdown fences)."
)

SUMMARIZE_RESULTS_SYSTEM = (
    "You are a synthesis assistant for a multi-agent system. "
    "Given the overall task and the individual results produced by each agent, "
    "write a concise, structured summary of what has been accomplished and what "
    "still remains."
)

SUMMARIZE_RESULTS_USER = (
    "Overall task:\n{task}\n\n"
    "Agent results this round:\n{results}\n\n"
    "Write a summary of progress. Be concise."
)

CONTINUE_SYSTEM = (
    "You are a termination judge for a multi-agent task loop. "
    "Reply with exactly one word: CONVERGED if the task is fully complete, "
    "or CONTINUE if more work is needed."
)

CONTINUE_USER = (
    "Overall task:\n{task}\n\n"
    "Latest progress summary:\n{summary}\n\n"
    "Round {round_num} of {max_rounds} completed. "
    "Is the task fully accomplished? Reply CONVERGED or CONTINUE."
)
