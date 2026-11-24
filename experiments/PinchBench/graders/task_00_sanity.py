"""Grader for task_00_sanity: Sanity Check."""


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}

    has_response = False
    for entry in transcript:
        if entry.get("type") == "message":
            message = entry.get("message", {})
            if message.get("role") == "assistant":
                content = message.get("content", [])
                if content and len(content) > 0:
                    has_response = True
                    break

    scores["agent_responded"] = 1.0 if has_response else 0.0
    return scores
