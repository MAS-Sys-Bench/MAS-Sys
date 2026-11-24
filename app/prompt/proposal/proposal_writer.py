GENERATE_PROFILE_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked with writing the first-person persona of a researcher based on the researcher's identity and publications.

You will be provided with the following information:
- Researcher identity: The researcher's name and affiliation.
- Publications: A list of paper abstracts written by the researcher that you will be writing about.

You should provide the following information:
- Persona: A comprehensive first-person persona. You should focus more on recent publications, which reflect the researcher's recent persona.

You should be concise and clear. The persona should range from 100 to 300 words.
"""

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


PAPER_WRITER_SYSTEM_TEMPLATE = """\
You are an autonomous intelligent agent participating in a ResearchTown-style paper-writing simulation.

The original paper prompt used a prior "Who are you?" exchange. In this chat-template adaptation, the generated researcher persona is provided directly in the system message.

Here is your profile:
{profile}
"""

PROFILE_PROMPT_TEMPLATE = """\
Here is the researcher identity:
Name: {author_name}
Affiliation: {affiliation}

Here are the publications of the researcher:
{publications}

Please write a first-person persona for this researcher.
"""


CANDIDATE_PROMPT_TEMPLATE = """\
Here is the content collected from related papers:
{related_papers}

You need to write a research proposal for a paper in the field of Machine Learning based on these related papers. The research proposal should rely more on the cited papers rather than your own research experience. Your research experience should be utilized to select the most useful and valuable papers from the related papers for proposal writing.

Here is a high-level, summarized insight into the Machine Learning research field. Here are the five core questions:

[Question 1]- What is the problem?
Formulate the specific research question you aim to address. Only output one question and do not include any more information.

[Question 2]- Why is it interesting and important?
Explain the broader implications of solving this problem for the research community. Discuss how the paper will affect future research. Discuss how addressing this question could advance knowledge or lead to practical applications.

[Question 3]- Why is it hard?
Discuss the challenges and complexities involved in solving this problem. Explain why naive or straightforward approaches may fail. Identify any technical, theoretical, or practical obstacles that need to be overcome. MAKE IT CLEAR.

[Question 4]- Why hasn't it been solved before?
Identify gaps or limitations in previous research or existing solutions. Discuss any barriers that have prevented this problem from being solved until now. Explain how your approach differs from or improves upon prior work. MAKE IT CLEAR.

[Question 5]- What are the key components of my approach and results?
Outline your proposed methodology in detail, including the method, dataset, and metric that you plan to use. But you must include these in one paragraph and not use subtitles. Describe the expected outcomes. MAKE IT CLEAR.

Please brainstorm the following proposal with the given format.
"""


def build_profile_prompt(*, author_name: str, affiliation: str, publications: str) -> str:
    return PROFILE_PROMPT_TEMPLATE.format(
        author_name=author_name or "Unknown",
        affiliation=affiliation or "Unknown",
        publications=publications,
    )


def build_candidate_prompt(*, related_papers: str) -> str:
    return CANDIDATE_PROMPT_TEMPLATE.format(related_papers=related_papers)
