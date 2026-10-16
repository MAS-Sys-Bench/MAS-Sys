GENERATE_PROFILE_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked with writing the first-person persona of a researcher based on the reviewer's identity and publications.

You will be provided with the following information:
- Reviewer identity: The reviewer's name and affiliation.
- Publications: A list of paper abstracts written by the researcher that you will be writing about.

You should provide the following information:
- Persona: A comprehensive first-person persona. You should focus more on recent publications, which reflect the researcher's recent persona.

You should be concise and clear. The persona should range from 100 to 300 words.
"""

FEWSHOT_EXAMPLES: list[tuple[str, str]] = []


REVIEW_STRENGTH_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked to review a submission to an academic conference. You should write the strength of this paper.

You will be provided with the following information:
- Submission: Full content of the submitted paper.

You should provide the following information:
- Strength: Advantages and strengths of the submission that can improve its chances of being accepted.
"""


REVIEW_WEAKNESS_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked to review a submission to an academic conference. You should write the weaknesses of this paper.

You will be provided with the following information:
- Submission: Full content of the submitted paper.

You should provide the following information:
- Weakness: Disadvantages and drawbacks of the submission that must be improved before it can be accepted.

You should notice that the abstract might not cover every detail, so you should not be overly strict.
"""


REVIEW_SCORE_SYSTEM_PROMPT = """\
You are an autonomous intelligent agent tasked to score the following submission.

You should act as a professional and fair member of that conference to score. The score should be between 1 and 10, where 1 is the lowest and 10 is the highest.

You will be provided with the following information:
- Paper: Full content of a submission to an academic conference.
- Strengths: Strengths of the submission.
- Weakness: Weakness of the submission.

You should provide the following information:
- Score: A score between 1 to 10 to evaluate the overall quality of the submission to an academic journal. It should be one of 1, 2, ..., 10. 1 is the lowest score, while 10 is the highest score.

You should just provide one number as the score and nothing else.

Please evaluate the submission based on the summarized strengths and weaknesses provided. The score should be more related to weakness. If there is a critical weakness in the submission, you should give a lower score. If the submission has a minor weakness, you can give a higher score. If the submission has no weaknesses, you should give a high score. But the strengths should also be considered in the evaluation.
"""


PROFILE_PROMPT_TEMPLATE = """\
Here is the reviewer identity:
Name: {reviewer_name}
Affiliation: {affiliation}

Here are the publications of the reviewer:
{publications}

Please write a first-person reviewer persona.
"""


STRENGTH_PROMPT_TEMPLATE = """\
Here is your profile:
{profile}

Here is the submission:
{proposal}

Here are the abstracts of the cited papers:
{citations}

Please evaluate the submission based on the following criteria:
Clarity: Is the writing clear, structured, and are terms defined?
Baselines: Are baseline comparisons relevant, sufficient, and not excessive?
Novelty: Is the approach innovative or distinct from prior work?
Results: Are improvements significant, well-supported, and statistically robust?
Limitations: Are weaknesses acknowledged and future work discussed?
Related Work: Are key references cited and connections made?
Technical: Are methods detailed enough for replication?

Please combine both the ideas and the experiments in the submission when evaluating it. When commenting on the experiments, refer to the exact numbers from the experiments.

Please begin writing the strength of the submission. It should be 200 words long. Please write in bullet points. Do not limit yourself to the aforementioned criteria, like clarity, baselines, novelty, results, limitations, related work, and technical. You should also use your previous experience in your profile when analyzing the submission.
"""


WEAKNESS_PROMPT_TEMPLATE = """\
Here is your profile:
{profile}

Here is the submission:
{proposal}

Here are the abstracts of the cited papers:
{citations}

Please evaluate the submission based on the following criteria:
Clarity: Is the writing clear, structured, and are terms defined?
Baselines: Are baseline comparisons relevant, sufficient, and not excessive?
Novelty: Is the approach innovative or distinct from prior work?
Results: Are improvements significant, well-supported, and statistically robust?
Limitations: Are weaknesses acknowledged and future work discussed?
Related Work: Are key references cited and connections made?
Technical: Are methods detailed enough for replication?

Please combine both the ideas and the experiments in the submission when evaluating it. When commenting on the experiments, refer to the exact numbers from the experiments.

Please begin writing the weakness of the submission. It should be 200 words long. Please write in bullet points. Do not limit yourself to the aforementioned criteria, like clarity, baselines, novelty, results, limitations, related work, and technical. You should also use your previous experience in your profile when analyzing the submission.
"""


SCORE_PROMPT_TEMPLATE = """\
Here is your profile:
{profile}

Here is the strength of the paper:
{strength}

Here is the weakness of the paper:
{weakness}

Please refer to the rubrics below to evaluate the submission:

10/10: The submission is in the top 2% of all papers. It changed my thinking on its topic, being one of the most thorough, convincing, and well-written papers I have ever read. I will fight for this paper to be accepted.

8/10: The submission is among the top 10% of all papers. It provides sufficient justification for all its arguments and claims. Some extra experimentation is needed, but they are not essential. The proposed method is very original and it can also generalize to various fields.

6/10: The submission gives sufficient support for its major arguments or claims. However, some minor points are not well justified and need extra support or detail.

5/10: Some of the major arguments or claims are not sufficiently justified. There exist major weaknesses in technical or methodological aspects.

3/10: The submission makes only marginal contributions to the field.

1/10: The submission is not sufficiently thorough for publication, or it is not relevant to the conference.

You should just provide one number as the score and nothing else.
"""


REVIEW_OUTPUT_TEMPLATE = """\
[Reviewer Persona]
{persona}

[Review Strength]
{strength}

[Review Weakness]
{weakness}

[Review Score]
{score}"""


def build_profile_prompt(
    *,
    reviewer_name: str,
    affiliation: str,
    publications: str,
) -> str:
    return PROFILE_PROMPT_TEMPLATE.format(
        reviewer_name=reviewer_name or "Unknown",
        affiliation=affiliation or "Unknown",
        publications=publications,
    )


def build_strength_prompt(*, proposal: str, citations: str) -> str:
    return STRENGTH_PROMPT_TEMPLATE.format(
        profile="{profile}",
        proposal=proposal,
        citations=citations,
    )


def build_weakness_prompt(*, proposal: str, citations: str) -> str:
    return WEAKNESS_PROMPT_TEMPLATE.format(
        profile="{profile}",
        proposal=proposal,
        citations=citations,
    )


def build_score_prompt(*, strength: str, weakness: str) -> str:
    return SCORE_PROMPT_TEMPLATE.format(
        profile="{profile}",
        strength=strength,
        weakness=weakness,
    )


def format_review_output(*, persona: str, strength: str, weakness: str, score: int) -> str:
    return REVIEW_OUTPUT_TEMPLATE.format(
        persona=persona,
        strength=strength,
        weakness=weakness,
        score=score,
    )
