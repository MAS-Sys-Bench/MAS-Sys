FINAL_META_REVIEW_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked to write a meta-review for a submission to an academic conference.

Your meta-review should summarize the reviewer reports to help make a decision.

You will be provided with the following information:
- Submission: Full content of the paper submitted to this conference.
- Reviews: It typically contains the score, strength, and weakness of the submission, each by a different reviewer.

You should provide the following information using this exact format:
=== META-REVIEW STRENGTH ===
<bullet points>

=== META-REVIEW WEAKNESS ===
<bullet points>

=== DECISION ===
Reviewer scores: <comma-separated scores you used>
Threshold: all scores >= 6
Final decision: <ACCEPT or REJECT>
"""

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


METAREVIEW_PROMPT_TEMPLATE = """\
Here are the reviewer reports:
{reviews}

Please write the final meta-review in the required format.
"""


def build_metareview_prompt(*, reviews: str) -> str:
    return METAREVIEW_PROMPT_TEMPLATE.format(reviews=reviews)
