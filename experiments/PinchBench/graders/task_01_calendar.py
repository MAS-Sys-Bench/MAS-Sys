"""Grader for task_01_calendar: Calendar Event Creation."""

import re
from datetime import datetime, timedelta
from pathlib import Path


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}
    workspace = Path(workspace_path)

    ics_files = list(workspace.glob("*.ics"))

    if not ics_files:
        return {
            "file_created": 0.0,
            "date_correct": 0.0,
            "time_correct": 0.0,
            "attendee_present": 0.0,
            "title_correct": 0.0,
            "description_present": 0.0,
        }

    scores["file_created"] = 1.0
    ics_content = ics_files[0].read_text()

    scores["attendee_present"] = (
        1.0
        if re.search(r"ATTENDEE.*john@example\.com", ics_content, re.IGNORECASE)
        else 0.0
    )
    scores["title_correct"] = (
        1.0
        if re.search(r"SUMMARY.*Project Sync", ics_content, re.IGNORECASE)
        else 0.0
    )
    scores["description_present"] = (
        1.0
        if re.search(r"DESCRIPTION.*roadmap", ics_content, re.IGNORECASE)
        else 0.0
    )

    if re.search(r"DTSTART.*T15\d{4}", ics_content) or re.search(
        r"DTSTART.*T030000", ics_content
    ):
        scores["time_correct"] = 1.0
    else:
        scores["time_correct"] = 0.0

    today = datetime.now()
    days_ahead = (1 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_tuesday = today + timedelta(days=days_ahead)
    expected_date = next_tuesday.strftime("%Y%m%d")
    scores["date_correct"] = 1.0 if expected_date in ics_content else 0.0

    return scores
