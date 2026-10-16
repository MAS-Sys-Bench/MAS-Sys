TASK_PROMPT_TEMPLATE = """\
Review the following research paper and produce reviewer reports plus a final meta-review decision.

[Submission]
{paper_content}

[Reviewer Profiles]
{reviewer_profiles_text}

[Cited Paper Abstracts]
{citations_text}
"""
