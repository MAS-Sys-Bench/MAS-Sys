"""Evaluation toolkit for PaperBananaBench.

Adapted from PaperBanana's ``utils/eval_toolkits.py`` to work inside the
MAS-Sys project. Instead of using Gemini / Claude / OpenAI clients directly,
this module reads the **[llm.vision]** section of ``config/config.toml`` and
calls the model through the OpenAI-compatible endpoint.

Current evaluation dimensions:
  - instruction_faithfulness
  - conciseness
  - readability
  - aesthetics
  - overall_score
"""

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from typing import Any, Dict

import json_repair
from openai import AsyncOpenAI

# Eval prompts — copied verbatim from the original PaperBanana project
PROMPT_MAP = {
    "diagram": {
        "instruction_faithfulness": """
# Role
You are an expert judge in academic visual design.

# Task
Evaluate the generated diagram on **instruction faithfulness** from 0 to 10.

# Inputs
1. Methodology Section: [content]
2. Diagram Caption: [content]
3. Model-generated Diagram: [image]

# Scoring Standard
- 0 means completely unfaithful, fabricated, or unusable.
- 5 means partially correct but with notable omissions, mismatches, or distortions.
- 10 means fully faithful to the method section and caption, with correct logic flow, scope, and labels.

# Focus
Judge whether the diagram faithfully reflects the method content and the caption. Penalize hallucinated modules, wrong flow direction, scope mismatch, gibberish labels, or fabricated content.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "conciseness": """
# Role
You are an expert judge in academic visual design.

# Task
Evaluate the generated diagram on **conciseness** from 0 to 10.

# Inputs
1. Methodology Section: [content]
2. Diagram Caption: [content]
3. Model-generated Diagram: [image]

# Scoring Standard
- 0 means extremely cluttered, verbose, or mostly a box-ified copy of the text.
- 5 means somewhat abstracted but still noticeably overloaded or inefficient.
- 10 means clean, high-signal visual abstraction with minimal redundancy.

# Focus
Judge whether the diagram distills the method into compact visual structure instead of heavy text or raw equations.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "readability": """
# Role
You are an expert judge in academic visual design.

# Task
Evaluate the generated diagram on **readability** from 0 to 10.

# Inputs
1. Diagram Caption: [content]
2. Model-generated Diagram: [image]

# Scoring Standard
- 0 means very hard to read due to overlap, tiny text, chaotic routing, or low contrast.
- 5 means readable in parts but with noticeable layout or legibility problems.
- 10 means immediately understandable, legible, and easy to navigate.

# Focus
Judge clarity of visual flow, legibility, spacing, layout efficiency, and absence of distracting artifacts.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "aesthetics": """
# Role
You are an expert judge in academic visual design.

# Task
Evaluate the generated diagram on **aesthetics** from 0 to 10.

# Inputs
1. Diagram Caption: [content]
2. Model-generated Diagram: [image]

# Scoring Standard
- 0 means visually poor, amateurish, or clearly below academic publication quality.
- 5 means acceptable but unpolished.
- 10 means polished, harmonious, and publication-quality.

# Focus
Judge visual polish, typography consistency, color harmony, rendering quality, and professional presentation.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
    },
    "plot": {
        "instruction_faithfulness": """
# Role
You are an expert judge in academic data visualization.

# Task
Evaluate the generated plot on **instruction faithfulness** from 0 to 10.

# Inputs
1. Raw Data: [content]
2. Plot Brief Description: [content]
3. Model-generated Plot: [image]

# Scoring Standard
- 0 means the plot seriously misrepresents the data or intent.
- 5 means partially correct but with notable omissions, wrong labels, or chart-choice issues.
- 10 means accurate, complete, and well-aligned with the raw data and plot brief.

# Focus
Judge data accuracy, correct labels, suitable chart type, and whether the intended visual message is preserved.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "conciseness": """
# Role
You are an expert judge in academic data visualization.

# Task
Evaluate the generated plot on **conciseness** from 0 to 10.

# Inputs
1. Raw Data: [content]
2. Plot Brief Description: [content]
3. Model-generated Plot: [image]

# Scoring Standard
- 0 means heavily cluttered or redundantly labeled.
- 5 means acceptable but somewhat overloaded.
- 10 means the plot communicates the data with minimal redundancy and no unnecessary clutter.

# Focus
Judge whether the plot shows only the information needed to communicate the data effectively.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "readability": """
# Role
You are an expert judge in academic data visualization.

# Task
Evaluate the generated plot on **readability** from 0 to 10.

# Inputs
1. Plot Brief Description: [content]
2. Model-generated Plot: [image]

# Scoring Standard
- 0 means hard to interpret because labels, legend, data, or contrast are unusable.
- 5 means readable overall but with noticeable issues.
- 10 means easy to interpret immediately.

# Focus
Judge axis labels, legend clarity, text legibility, contrast, and visibility of the plotted data.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
        "aesthetics": """
# Role
You are an expert judge in academic data visualization.

# Task
Evaluate the generated plot on **aesthetics** from 0 to 10.

# Inputs
1. Plot Brief Description: [content]
2. Model-generated Plot: [image]

# Scoring Standard
- 0 means clearly unprofessional or visually poor.
- 5 means acceptable but plain or inconsistent.
- 10 means polished and publication-quality.

# Focus
Judge rendering quality, color choices, typography consistency, and overall professional polish.

# Output
Return strict JSON:
{
  "reasoning": "...",
  "score": 0-10
}
""",
    },
}

TASK_CONFIG = {
    "diagram": {
        "visual_intent_label": "Diagram Caption",
        "raw_content_label": "Methodology Section",
        "model_label": "Model-Generated Diagram",
    },
    "plot": {
        "visual_intent_label": "Visual Intent of the Desired Plot",
        "raw_content_label": "Raw Data",
        "model_label": "Model-Generated Plot",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _guess_data_uri_mime(base64_str: str) -> str:
    stripped = (base64_str or "").lstrip()
    if stripped.startswith("iVBOR"):
        return "image/png"
    if stripped.startswith("/9j/"):
        return "image/jpeg"
    if stripped.startswith("R0lGOD"):
        return "image/gif"
    if stripped.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _try_regex_extract_score(text: str) -> float | None:
    patterns = [
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'\*\*score\*\*\s*:\s*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).strip())
            except ValueError:
                return None
    return None


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, score))


def _looks_like_placeholder_base64(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("<base64:") and value.endswith(" chars>")


def _reasoning_mentions_missing_images(text: str) -> bool:
    lowered = (text or "").lower()
    markers = [
        "not provided",
        "not supplied",
        "not available",
        "image is missing",
        "image missing",
        "diagram is not provided",
        "plot is not provided",
        "neither diagram is available",
        "neither plot is available",
    ]
    return any(marker in lowered for marker in markers)


def _build_eval_messages(
    *,
    sys_prompt: str,
    cfg: dict[str, str],
    eval_dim: str,
    raw_content: str,
    visual_intent: str,
    model_image_b64: str,
) -> list[dict[str, Any]]:
    model_mime = _guess_data_uri_mime(model_image_b64)

    if eval_dim in ("readability", "aesthetics"):
        input_text = f"{cfg['visual_intent_label']}: {visual_intent}"
    else:
        input_text = (
            f"{cfg['raw_content_label']}: {raw_content}\n"
            f"{cfg['visual_intent_label']}: {visual_intent}"
        )

    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text + f"\n{cfg['model_label']}: "},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{model_mime};base64,{model_image_b64}", "detail": "high"},
                },
            ],
        },
    ]


# ── Single-dimension evaluation ───────────────────────────────────────────────


async def _run_single_eval(
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    task_name: str,
    eval_dim: str,
    raw_content: str,
    visual_intent: str,
    model_image_b64: str,
) -> tuple[str, dict]:
    """Call the vision LLM to evaluate one dimension."""
    sys_prompt = PROMPT_MAP[task_name][eval_dim]
    cfg = TASK_CONFIG[task_name]
    clean = ""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=_build_eval_messages(
                sys_prompt=sys_prompt,
                cfg=cfg,
                eval_dim=eval_dim,
                raw_content=raw_content,
                visual_intent=visual_intent,
                model_image_b64=model_image_b64,
            ),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw_text = response.choices[0].message.content or ""
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json_repair.loads(clean)

        if not isinstance(parsed, dict):
            parsed = {"reasoning": clean, "score": _try_regex_extract_score(clean)}

        score = _coerce_score(parsed.get("score"))
        if score is None:
            score = _try_regex_extract_score(clean)
            score = _coerce_score(score)
        if score is None:
            raise RuntimeError(f"{eval_dim}: evaluation returned no valid score")

        reasoning = str(parsed.get("reasoning") or parsed.get("comparison_reasoning") or clean)
        return eval_dim, {"reasoning": reasoning, "score": score}

    except Exception as e:
        print(f"❌ {eval_dim}: Evaluation failed — {str(e)[:120]}")
        return eval_dim, {"reasoning": str(e), "score": 0.0}


# ── Public API ────────────────────────────────────────────────────────────────


async def evaluate_sample(
    sample: Dict[str, Any],
    task_name: str,
    data_dir: Path,
    vision_base_url: str,
    vision_api_key: str,
    vision_model: str,
    vision_max_tokens: int = 8192,
) -> Dict[str, Any]:
    """Evaluate a single benchmark sample using the vision LLM.
    Returns the sample dict **augmented** with ``{dim}_score``,
    ``{dim}_reasoning``, and ``overall_score`` / ``overall_reasoning``.
    """

    # ── Load model image ──────────────────────────────────────────
    eval_field = sample.get("eval_image_field", "")
    model_b64 = sample.get(eval_field, "") if eval_field else ""
    if _looks_like_placeholder_base64(model_b64):
        model_b64 = ""
    if not model_b64:
        nested_b64 = (
            sample.get("visualizer_output", {}) or {}
        ).get("image_base64", "")
        if isinstance(nested_b64, str) and nested_b64 and not _looks_like_placeholder_base64(nested_b64):
            model_b64 = nested_b64

    if not model_b64:
        print(f"⚠️  Model image field '{eval_field}' empty — assigning zero scores.")
        for dim in ("instruction_faithfulness", "conciseness", "readability", "aesthetics"):
            sample[f"{dim}_reasoning"] = "Model failed to generate image"
            sample[f"{dim}_score"] = 0.0
        sample["overall_reasoning"] = "Model failed to generate image"
        sample["overall_score"] = 0.0
        return sample

    # ── Run 4-dimension evaluation in parallel ────────────────────
    client = AsyncOpenAI(api_key=vision_api_key, base_url=vision_base_url)
    dims = ["instruction_faithfulness", "conciseness", "readability", "aesthetics"]
    tasks = [
        _run_single_eval(
            client, vision_model, vision_max_tokens,
            task_name, dim,
            sample.get("content", ""),
            sample.get("visual_intent", ""),
            model_b64,
        )
        for dim in dims
    ]
    results = await asyncio.gather(*tasks)

    for dim, res_obj in results:
        sample[f"{dim}_reasoning"] = res_obj.get("reasoning", "")
        sample[f"{dim}_score"] = res_obj.get("score", 0.0)

    overall_score = round(
        sum(float(sample.get(f"{dim}_score", 0.0)) for dim in dims) / len(dims),
        4,
    )
    sample["overall_score"] = overall_score
    sample["overall_reasoning"] = (
        "Average of instruction_faithfulness, conciseness, readability, and aesthetics."
    )
    return sample
