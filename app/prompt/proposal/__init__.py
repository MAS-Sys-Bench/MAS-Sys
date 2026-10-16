from app.prompt.proposal.proposal_writer import (
    CANDIDATE_PROMPT_TEMPLATE,
    FEWSHOT_EXAMPLES as PROPOSAL_WRITER_FEWSHOT_EXAMPLES,
    GENERATE_PROFILE_SYSTEM_PROMPT,
    PAPER_WRITER_SYSTEM_TEMPLATE,
    PROFILE_PROMPT_TEMPLATE,
    build_candidate_prompt,
    build_profile_prompt,
)
from app.prompt.proposal.proposal_aggregator import (
    AGGREGATION_PROMPT_TEMPLATE,
    FINAL_PROPOSAL_SYSTEM_PROMPT,
)

__all__ = [
    "GENERATE_PROFILE_SYSTEM_PROMPT",
    "PROPOSAL_WRITER_FEWSHOT_EXAMPLES",
    "PAPER_WRITER_SYSTEM_TEMPLATE",
    "PROFILE_PROMPT_TEMPLATE",
    "CANDIDATE_PROMPT_TEMPLATE",
    "build_profile_prompt",
    "build_candidate_prompt",
    "FINAL_PROPOSAL_SYSTEM_PROMPT",
    "AGGREGATION_PROMPT_TEMPLATE",
]
