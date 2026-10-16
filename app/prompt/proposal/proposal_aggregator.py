FINAL_PROPOSAL_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked with summarizing multiple proposal candidates into one final proposal.
"""

AGGREGATION_PROMPT_TEMPLATE = """\
Here is a high-level, summarized insight of a research field: Machine Learning. Here are the five core questions to consider:
[Question 1]- What is the problem?
[Question 2]- Why is it interesting and important?
[Question 3]- Why is it hard?
[Question 4]- Why hasn't it been solved before?
[Question 5]- What are the key components of my approach and results?

Multiple papers have been generated for the above questions:
{proposal_candidates}

Your task is to summarize and select the key insights that are suitable from these proposals.
1. Identify shared themes and common points among the proposals.
2. Highlight and select any valuable perspectives or contrasting elements and combine them into one proposal.
3. Provide a concise proposal for each question based on the proposal candidates.

Output the result in the provided five-question format. Ensure the generated paper is clear, concise, and avoids repeating full proposals verbatim.
"""
