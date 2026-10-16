import re
from pathlib import Path
from typing import Any, Dict, List

import nltk
import numpy as np
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from rouge_score import rouge_scorer

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

# Initialize NLTK resources
nltk.download('punkt_tab')


# Embedding configuration from config.toml

def _load_config() -> Dict[str, Dict[str, Any]]:
    """Load [embedding] and [llm] sections from config/config.toml."""
    config_path = Path(__file__).resolve().parents[2] / 'config' / 'config.toml'
    if not config_path.exists():
        raise FileNotFoundError(
            f'config.toml not found at {config_path}. '
            'Please create it with [embedding] and [llm] sections.'
        )
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)
    return cfg


_CFG: Dict[str, Any] = {}


def _get_config() -> Dict[str, Any]:
    """Lazy-load and cache the full config."""
    global _CFG
    if not _CFG:
        _CFG = _load_config()
    return _CFG


def _get_openai_client() -> OpenAI:
    """Build an OpenAI client from the [embedding] section (shared base_url/api_key)."""
    cfg = _get_config().get('embedding', {})
    return OpenAI(
        api_key=cfg.get('api_key', ''),
        base_url=cfg.get('base_url', ''),
    )


def _get_llm_client() -> OpenAI:
    """Build an OpenAI client from the [llm] section."""
    cfg = _get_config().get('llm', {})
    return OpenAI(
        api_key=cfg.get('api_key', ''),
        base_url=cfg.get('base_url', ''),
    )


def compute_bleu(reference: str, hypothesis: str) -> float:
    try:
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(
            [reference_tokens], hypothesis_tokens, smoothing_function=smoothie
        )
        return float(bleu_score)
    except Exception as e:
        print(f'Error computing BLEU score: {e}')
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        rouge_l_f1 = scores['rougeL'].fmeasure
        return float(rouge_l_f1)
    except Exception as e:
        print(f'Error computing ROUGE-L score: {e}')
        return 0.0


def compute_bertscore(reference: str, hypothesis: str) -> float:
    try:
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference], lang='en', rescale_with_baseline=True
        )
        return float(F1.mean().item())
    except Exception as e:
        print(f'Error computing BERTScore: {e}')
        return 0.0


def compute_proposal_gpt_metric(reference: str, generation: str) -> float:
    try:
        client = _get_llm_client()
        llm_cfg = _get_config().get('llm', {})
        model = llm_cfg.get('model', 'qwen3-max')
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        'Evaluate the alignment between the following two sets of paragraphs, with a particular emphasis on their objectives, methodologies, and expected outcomes.\n\n'
                        'Alignment Criteria Definitions:\n'
                        '1. **Objectives**: Do both sets of questions aim to address the same or complementary research goals?\n'
                        '2. **Methodologies**: Are the proposed methods similar, compatible, or capable of being effectively integrated?\n'
                        '3. **Expected Outcomes**: Are the anticipated research results and impacts consistent or mutually supportive?\n\n'
                        'Reference context:\n'
                        f'{reference}\n\n'
                        'Proposed context:\n'
                        f'{generation}\n\n'
                        'Based on the above alignment criteria, especially focusing on the methodologies, please provide a similarity score: **1** indicates alignment, and **0** indicates no alignment. **Only output the score without any additional information.**'
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=16,
        )
        raw = resp.choices[0].message.content.strip()
        val = float(raw)
        return max(0.0, min(1.0, val))
    except Exception as e:
        print(f'Error computing GPT metric: {e}')
        return -1.0


def compute_review_gpt_metric(reference: str, generation: str) -> float:
    try:
        client = _get_llm_client()
        llm_cfg = _get_config().get('llm', {})
        model = llm_cfg.get('model', 'qwen3-max')
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        'Evaluate the alignment between the following two reviews, focusing on whether they identify similar strengths, weaknesses, and overall assessment of the paper.\n\n'
                        'Reference review:'
                        f'\n{reference}\n\n'
                        'Generated review:'
                        f'\n{generation}\n\n'
                        'Provide a similarity score: **1** indicates alignment, and **0** indicates no alignment. **Only output the score without any additional information.**'
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=16,
        )
        raw = resp.choices[0].message.content.strip()
        val = float(raw)
        return max(0.0, min(1.0, val))
    except Exception as e:
        print(f'Error computing review GPT metric: {e}')
        return -1.0


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call the configured LLM and return the raw text response."""
    client = _get_llm_client()
    llm_cfg = _get_config().get('llm', {})
    model = llm_cfg.get('model', 'qwen3-max')
    resp = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_scores_from_text(text: str, keys: List[str]) -> Dict[str, float]:
    """Parse 'Key: number' pairs from LLM output text.

    Falls back to extracting all numbers in order if key-based parsing fails.
    """
    scores: Dict[str, float] = {}
    for key in keys:
        pattern = re.compile(
            rf'{re.escape(key)}\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            scores[key] = float(match.group(1))
    if len(scores) == len(keys):
        return scores
    # Fallback: extract all numbers
    nums = re.findall(r'\b([0-9]+(?:\.[0-9]+)?)\b', text)
    nums = [float(n) for n in nums if 0 <= float(n) <= 10]
    if len(nums) >= len(keys):
        return {k: nums[i] for i, k in enumerate(keys)}
    return {k: scores.get(k, -1.0) for k in keys}


def compute_llm_evaluation(
    reference: str, generation: str,
) -> Dict[str, float]:
    """LLM-based evaluation using the exact prompt from Table 17 of the ResearchTown paper.

    Evaluates both fine-grained similarity (6 dimensions) and intrinsic quality
    (novelty & feasibility for both reference and generated proposals) in a
    single LLM call, returning 10 scores.
    """
    all_keys = [
        'Topic Consistency',
        'Method Consistency',
        'Factual Consistency',
        'Claim Consistency',
        'Application Context Consistency',
        'Overall Semantic Similarity',
        'Novelty of Reference Proposal',
        'Feasibility of Reference Proposal',
        'Novelty of Generated Proposal',
        'Feasibility of Generated Proposal',
    ]

    prompt = (
        'You are an expert evaluator of research proposals. Given a reference and a generated '
        'proposal, rate their similarity from 1 (completely different) to 10 (nearly identical) for each '
        'of the following dimensions:\n\n'
        '- Topic Consistency: Do both proposals address the same research topic or problem area?\n'
        '- Method Consistency: Are the research methods and approaches used in both proposals similar?\n'
        '- Factual Consistency: Are the datasets, metrics, and models mentioned consistent between '
        'the two proposals?\n'
        '- Claim Consistency: Do both proposals present similar conclusions or findings?\n'
        '- Application Context Consistency: Are the application domains or use-cases targeted by '
        'both proposals the same?\n'
        '- Overall Semantic Similarity: Do the two proposals convey the same overall idea or message?\n\n'
        'Additionally, assess the following for each proposal individually:\n'
        '- Novelty of Reference Proposal: Does the reference proposal introduce new ideas or perspectives?\n'
        '- Feasibility of Reference Proposal: How realistic and implementable is the reference '
        'proposal based on its described methods and objectives?\n'
        '- Novelty of Generated Proposal: Does the generated proposal introduce new ideas or perspectives?\n'
        '- Feasibility of Generated Proposal: How realistic and implementable is the generated '
        'proposal based on its described methods and objectives?\n\n'
        'Respond strictly in JSON format like:\n'
        '{\n'
        '"Topic Consistency": X,\n'
        '"Method Consistency": X,\n'
        '"Factual Consistency": X,\n'
        '"Claim Consistency": X,\n'
        '"Application Context Consistency": X,\n'
        '"Overall Semantic Similarity": X,\n'
        '"Novelty of Reference Proposal": X,\n'
        '"Feasibility of Reference Proposal": X,\n'
        '"Novelty of Generated Proposal": X,\n'
        '"Feasibility of Generated Proposal": X\n'
        '}\n\n'
        f'Reference proposal:\n{reference}\n\n'
        f'Generated proposal:\n{generation}'
    )
    try:
        raw = _call_llm(prompt, max_tokens=512)
        # Try JSON parsing first
        import json
        # Extract JSON block if wrapped in markdown code fence
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            scores = {}
            for key in all_keys:
                val = parsed.get(key, -1.0)
                scores[key] = max(0.0, min(10.0, float(val)))
            return scores
        # Fallback to text parsing
        scores = _parse_scores_from_text(raw, all_keys)
        return {k: max(0.0, min(10.0, v)) for k, v in scores.items()}
    except Exception as e:
        print(f'Error computing LLM evaluation: {e}')
        return {k: -1.0 for k in all_keys}


def compute_embedding_similarity(reference: str, hypothesis: str) -> float:
    """Compute cosine similarity using the DashScope embedding API via OpenAI SDK."""
    try:
        client = _get_openai_client()
        emb_cfg = _get_config().get('embedding', {})
        model = emb_cfg.get('model', 'text-embedding-v3')
        # Strip litellm 'openai/' prefix if present
        if model.startswith('openai/'):
            model = model[len('openai/'):]
        dimensions = emb_cfg.get('dimensions', 1024)

        resp_ref = client.embeddings.create(model=model, input=[reference], dimensions=dimensions)
        resp_hyp = client.embeddings.create(model=model, input=[hypothesis], dimensions=dimensions)

        vec_ref = np.array(resp_ref.data[0].embedding)
        vec_hyp = np.array(resp_hyp.data[0].embedding)

        cosine_sim = np.dot(vec_ref, vec_hyp) / (
            np.linalg.norm(vec_ref) * np.linalg.norm(vec_hyp)
        )
        return float(cosine_sim)
    except Exception as e:
        print(f'Error computing embedding similarity: {e}')
        return 0.0


def extract_and_clean_question_content(
    text: str, question_prompts: List[str]
) -> List[str]:
    """Extract per-question content from proposal text.

    Supports two formats:
      - Legacy labels: "What is the problem?", "Why is it interesting ...", etc.
      - Numbered labels: "[Question 1]:", "[Question 2]:", etc.
    """
    if '\u2019' in text:
        text = text.replace('\u2019', "'")

    # --- Try legacy literal labels first ---
    question_contents: List[str] = []
    all_found = True
    for i, prompt in enumerate(question_prompts):
        start = text.find(prompt)
        if start == -1:
            all_found = False
            break
        end = (
            text.find(question_prompts[i + 1])
            if i + 1 < len(question_prompts)
            else len(text)
        )
        content = text[start + len(prompt) : end].strip()
        content = re.sub(r'[\*\#]+', '', content).strip()
        question_contents.append(content)

    if all_found and question_contents:
        return question_contents

    # --- Fallback: numbered [Question N]: pattern ---
    numbered_re = re.compile(
        r'\[Question\s*(\d)\]\s*[-:]?\s*',
        re.IGNORECASE,
    )
    splits = list(numbered_re.finditer(text))
    if len(splits) >= 5:
        result: List[str] = []
        for idx in range(5):
            start = splits[idx].end()
            end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
            content = text[start:end].strip()
            content = re.sub(r'[\*\#]+', '', content).strip()
            result.append(content)
        return result

    # Nothing matched
    return [''] * len(question_prompts)


def compute_embedding_similarity_per_question(
    reference: str, hypothesis: str
) -> List[float]:
    try:
        questions = [
            'What is the problem?',
            'Why is it interesting and important?',
            'Why is it hard?',
            "Why hasn't it been solved before?",
            'What are the key components of my approach and results?',
        ]

        ref_questions = extract_and_clean_question_content(reference, questions)
        hyp_questions = extract_and_clean_question_content(hypothesis, questions)

        similarities = []

        for ref_text, hyp_text in zip(ref_questions, hyp_questions):
            if not ref_text or not hyp_text:
                print(f'Empty question: {ref_text} vs {hyp_text}')
                similarities.append(0.0)
                continue

            cosine_sim = compute_embedding_similarity(ref_text, hyp_text)
            similarities.append(float(cosine_sim))

        return similarities

    except Exception as e:
        print(f'Error computing embedding similarity per question: {e}')
        return [0.0] * 5


def compute_proposal_metrics(reference: str, generation: str) -> Dict[str, float]:
    bleu = compute_bleu(reference, generation)
    rouge_l = compute_rouge_l(reference, generation)
    bert_score = compute_bertscore(reference, generation)
    gpt_metric = compute_proposal_gpt_metric(reference, generation)
    emb_sim = compute_embedding_similarity(reference, generation)
    emb_sim_per_question = compute_embedding_similarity_per_question(
        reference, generation
    )

    # LLM-based evaluation: fine-grained similarity + novelty/feasibility (Table 17)
    llm_eval = compute_llm_evaluation(reference, generation)

    return {
        'bleu': bleu,
        'rouge_l': rouge_l,
        'gpt_metric_score': gpt_metric,
        'bert_score': bert_score,
        'embedding_sim': emb_sim,
        'embedding_sim_q1': emb_sim_per_question[0],
        'embedding_sim_q2': emb_sim_per_question[1],
        'embedding_sim_q3': emb_sim_per_question[2],
        'embedding_sim_q4': emb_sim_per_question[3],
        'embedding_sim_q5': emb_sim_per_question[4],
        # Fine-grained similarity dimensions
        'fg_topic_consistency': llm_eval.get('Topic Consistency', -1.0),
        'fg_method_consistency': llm_eval.get('Method Consistency', -1.0),
        'fg_factual_consistency': llm_eval.get('Factual Consistency', -1.0),
        'fg_claim_consistency': llm_eval.get('Claim Consistency', -1.0),
        'fg_app_context_consistency': llm_eval.get('Application Context Consistency', -1.0),
        'fg_overall_similarity': llm_eval.get('Overall Semantic Similarity', -1.0),
        # Intrinsic quality of generated proposal
        'quality_novelty_gen': llm_eval.get('Novelty of Generated Proposal', -1.0),
        'quality_feasibility_gen': llm_eval.get('Feasibility of Generated Proposal', -1.0),
        # Intrinsic quality of reference (for comparison)
        'quality_novelty_ref': llm_eval.get('Novelty of Reference Proposal', -1.0),
        'quality_feasibility_ref': llm_eval.get('Feasibility of Reference Proposal', -1.0),
    }


def _embed_text(text: str) -> List[float]:
    """Embed a single text using the DashScope embedding API via OpenAI SDK."""
    client = _get_openai_client()
    emb_cfg = _get_config().get('embedding', {})
    model = emb_cfg.get('model', 'text-embedding-v3')
    if model.startswith('openai/'):
        model = model[len('openai/'):]
    dimensions = emb_cfg.get('dimensions', 1024)
    resp = client.embeddings.create(model=model, input=[text], dimensions=dimensions)
    return resp.data[0].embedding


def compute_review_metrics(
    strengths: List[str],
    weaknesses: List[str],
    generated_strength: str,
    generated_weakness: str,
) -> Dict[str, Any]:
    metrics_raw: Dict[str, Any] = {
        'embedding_strength': 0.0,
        'embedding_weakness': 0.0,
        'embedding_sim_strengths': [],
        'embedding_sim_weaknesses': [],
        'gt_strength_matchings': [],
        'gt_weakness_matchings': [],
    }

    # Clean out empty strings
    strengths = [s for s in strengths if s]
    weaknesses = [w for w in weaknesses if w]

    # Split generated lists
    gen_strengths = [s for s in generated_strength.split('\n') if s]
    gen_weaknesses = [w for w in generated_weakness.split('\n') if w]

    # Embed all ground-truth and generated items once
    gt_str_embs = [_embed_text(s) for s in strengths]
    gen_str_embs = [_embed_text(s) for s in gen_strengths]
    gt_wk_embs = [_embed_text(w) for w in weaknesses]
    gen_wk_embs = [_embed_text(w) for w in gen_weaknesses]

    # For each GT strength, find best generated strength
    for i, gt_emb in enumerate(gt_str_embs):
        sims = [
            np.dot(gt_emb, gen_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(gen_emb))
            for gen_emb in gen_str_embs
        ]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        metrics_raw['embedding_sim_strengths'].append(best_sim)
        metrics_raw['gt_strength_matchings'].append(
            {
                'gt': strengths[i],
                'matched_generated': gen_strengths[best_idx],
                'similarity': best_sim,
            }
        )

    # For each GT weakness, find best generated weakness
    for i, gt_emb in enumerate(gt_wk_embs):
        sims = [
            np.dot(gt_emb, gen_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(gen_emb))
            for gen_emb in gen_wk_embs
        ]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        metrics_raw['embedding_sim_weaknesses'].append(best_sim)
        metrics_raw['gt_weakness_matchings'].append(
            {
                'gt': weaknesses[i],
                'matched_generated': gen_weaknesses[best_idx],
                'similarity': best_sim,
            }
        )

    # Compute averages over ground-truth items
    if metrics_raw['embedding_sim_strengths']:
        metrics_raw['embedding_strength'] = float(np.mean(metrics_raw['embedding_sim_strengths']))
    if metrics_raw['embedding_sim_weaknesses']:
        metrics_raw['embedding_weakness'] = float(np.mean(metrics_raw['embedding_sim_weaknesses']))

    return metrics_raw
