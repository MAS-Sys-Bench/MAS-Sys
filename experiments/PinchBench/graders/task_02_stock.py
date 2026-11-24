"""Grader for task_02_stock: Stock Price Research."""

import re
from pathlib import Path


def grade(transcript: list, workspace_path: str) -> dict:
    scores = {}
    workspace = Path(workspace_path)

    report_file = workspace / "stock_report.txt"
    if not report_file.exists():
        return {
            "file_created": 0.0,
            "ticker_present": 0.0,
            "price_present": 0.0,
            "date_present": 0.0,
            "summary_present": 0.0,
            "well_formatted": 0.0,
        }

    scores["file_created"] = 1.0
    content = report_file.read_text()

    scores["ticker_present"] = (
        1.0 if re.search(r"\bAAPL\b", content, re.IGNORECASE) else 0.0
    )

    price_patterns = [r"\$\s*\d+\.?\d*", r"\d+\.\d{2}", r"price.*\d+"]
    scores["price_present"] = (
        1.0
        if any(re.search(p, content, re.IGNORECASE) for p in price_patterns)
        else 0.0
    )

    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"(January|February|March|April|May|June|July|August|September|October|November|December)",
        r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    ]
    scores["date_present"] = (
        1.0
        if any(re.search(p, content, re.IGNORECASE) for p in date_patterns)
        else 0.0
    )

    content_stripped = re.sub(r"\s+", " ", content).strip()
    scores["summary_present"] = 1.0 if len(content_stripped) >= 50 else 0.0

    lines = content.split("\n")
    if len(lines) >= 3 and len(content_stripped) >= 50:
        scores["well_formatted"] = 1.0
    else:
        scores["well_formatted"] = 0.5 if len(lines) >= 2 else 0.0

    return scores
