from __future__ import annotations

import base64
from collections import Counter
import difflib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
import re
import unicodedata

from pydantic import Field, PrivateAttr

from app.agent.base import BaseAgent
from app.agent.paperviz.common import (
    extract_paperbanana_context,
    split_root_and_upstream,
    task_name_from_params,
)
from app.logger import logger
from app.prompt.paperviz.planner import (
    DIAGRAM_PLANNER_SYSTEM_PROMPT,
    PLOT_PLANNER_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_VISUAL_KEYWORDS = {
    "diagram",
    "influence",
    "framework",
    "pipeline",
    "workflow",
    "architecture",
    "module",
    "graph",
    "tree",
    "chart",
    "plot",
    "bar",
    "line",
    "scatter",
    "heatmap",
}


class PaperVizPlannerAgent(BaseAgent):
    """PaperViz planner agent that turns context and retrieved examples into a detailed figure description."""

    name: str = "paperviz_planner"
    description: str = "Turns source paper content and retrieved examples into a detailed figure plan."
    max_steps: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    _runtime_top_k: int = PrivateAttr(default=10)
    _runtime_candidate_pool_limit: int | None = PrivateAttr(default=None)
    _runtime_examples: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def initialize_from_params(self) -> None:
        params = self.agent_params
        top_k = params.get("top_k", 10)
        try:
            self._runtime_top_k = max(1, int(top_k))
        except Exception as exc:
            raise ValueError(f"Invalid paperviz planner top_k: {top_k!r}") from exc
        pool_limit = params.get("candidate_pool_limit")
        if pool_limit is None:
            self._runtime_candidate_pool_limit = None
        else:
            try:
                self._runtime_candidate_pool_limit = max(1, int(pool_limit))
            except Exception as exc:
                raise ValueError(
                    f"Invalid paperviz planner candidate_pool_limit: {pool_limit!r}"
                ) from exc

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        self._runtime_examples = self._retrieve_examples(request)
        prompt = self._build_prompt(request)
        self._build_messages(request, prompt)

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result = await self.step()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        system_prompt = self._system_prompt()
        logger.info(f"[PaperVizPlannerAgent] Running node '{self.name}'")
        response = await self.ask_with_trace(
            messages=self.messages,
            system_msgs=[Message.system_message(system_prompt)],
            stream=False,
        )
        self.update_memory("assistant", response)
        self.state = AgentState.FINISHED
        return response

    def _system_prompt(self) -> str:
        return (
            PLOT_PLANNER_SYSTEM_PROMPT
            if task_name_from_params(self.agent_params) == "plot"
            else DIAGRAM_PLANNER_SYSTEM_PROMPT
        )

    def _build_prompt(self, request: Optional[str]) -> str:
        params = self.agent_params
        root_text, _ = split_root_and_upstream(request)
        context = extract_paperbanana_context(root_text)
        task_name = str(params.get("task_name") or context.get("task_name") or task_name_from_params(params))
        task_name = "plot" if task_name == "plot" else "diagram"
        content = params.get("content") or context.get("content", "")
        visual_intent = params.get("visual_intent") or context.get("visual_intent", "")
        return self._format_planner_prompt(
            task_name=task_name,
            content=content,
            visual_intent=visual_intent,
            examples=self._runtime_examples,
        )

    def _build_messages(self, request: Optional[str], fallback_prompt: str) -> None:
        params = self.agent_params
        root_text, _ = split_root_and_upstream(request)
        context = extract_paperbanana_context(root_text)
        task_name = str(
            params.get("task_name") or context.get("task_name") or task_name_from_params(params)
        )
        task_name = "plot" if task_name == "plot" else "diagram"
        content = params.get("content") or context.get("content", "")
        visual_intent = params.get("visual_intent") or context.get("visual_intent", "")
        examples = self._runtime_examples

        if not isinstance(examples, list) or not examples:
            self.update_memory("user", fallback_prompt)
            return

        for idx, item in enumerate(examples, start=1):
            item_content = item.get("content", "")
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content, ensure_ascii=False)
            example_prompt = self._format_planner_example_prompt(
                task_name=task_name,
                example_index=idx,
                item_content=str(item_content),
                visual_intent=str(item.get("visual_intent", "")),
            )
            image_b64 = self._load_reference_image_base64(task_name=task_name, item=item)
            self.memory.add_message(
                Message.user_message(example_prompt, base64_image=image_b64)
            )

        target_prompt = self._format_planner_target_prompt(
            task_name=task_name,
            content=str(content),
            visual_intent=str(visual_intent),
        )
        self.update_memory("user", target_prompt)

    @staticmethod
    def _labels_for_task(task_name: str) -> tuple[str, str, str]:
        content_label = "Plot Raw Data" if task_name == "plot" else "Methodology Section"
        visual_label = (
            "Visual Intent of the Desired Plot" if task_name == "plot" else "Diagram Caption"
        )
        example_label = "Reference Plot" if task_name == "plot" else "Reference Diagram"
        return content_label, visual_label, example_label

    @classmethod
    def _format_planner_prompt(
        cls,
        *,
        task_name: str,
        content: str,
        visual_intent: str,
        examples: list[dict],
    ) -> str:
        content_label, visual_label, example_label = cls._labels_for_task(task_name)
        lines = []
        for idx, item in enumerate(examples, start=1):
            item_content = item.get("content", "")
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content, ensure_ascii=False)
            lines.extend(
                [
                    f"Example {idx}:",
                    f"{content_label}: {item_content}",
                    f"{visual_label}: {item.get('visual_intent', '')}",
                    f"{example_label}: {item.get('id', '')}",
                    "",
                ]
            )
        lines.extend(cls._format_planner_target_lines(
            task_name=task_name,
            content=str(content),
            visual_intent=str(visual_intent),
        ))
        return "\n".join(lines)

    @classmethod
    def _format_planner_example_prompt(
        cls,
        *,
        task_name: str,
        example_index: int,
        item_content: str,
        visual_intent: str,
    ) -> str:
        content_label, visual_label, example_label = cls._labels_for_task(task_name)
        return "\n".join(
            [
                f"Example {example_index}:",
                f"{content_label}: {item_content}",
                f"{visual_label}: {visual_intent}",
                f"{example_label}:",
            ]
        )

    @classmethod
    def _format_planner_target_prompt(
        cls,
        *,
        task_name: str,
        content: str,
        visual_intent: str,
    ) -> str:
        return "\n".join(cls._format_planner_target_lines(
            task_name=task_name,
            content=content,
            visual_intent=visual_intent,
        ))

    @classmethod
    def _format_planner_target_lines(
        cls,
        *,
        task_name: str,
        content: str,
        visual_intent: str,
    ) -> list[str]:
        content_label, visual_label, _ = cls._labels_for_task(task_name)
        return [
            f"Now, based on the following {content_label.lower()} and {visual_label.lower()}, provide a detailed description for the figure to be generated.",
            f"{content_label}: {content}",
            f"{visual_label}: {visual_intent}",
            "Detailed description of the target figure to be generated:"
            if task_name == "plot"
            else "Detailed description of the target figure to be generated (do not include figure titles):",
        ]

    def _retrieve_examples(self, request: Optional[str]) -> list[dict[str, Any]]:
        params = self.agent_params
        root_text, _ = split_root_and_upstream(request)
        context = extract_paperbanana_context(root_text)
        task_name = str(params.get("task_name") or context.get("task_name") or task_name_from_params(params))
        task_name = "plot" if task_name == "plot" else "diagram"
        content = str(params.get("content") or context.get("content", "")).strip()
        visual_intent = str(params.get("visual_intent") or context.get("visual_intent", "")).strip()
        candidate_pool = self._load_candidate_pool(task_name=task_name)
        if not candidate_pool:
            return []

        ranked = self._rank_candidates(
            task_name=task_name,
            content=content,
            visual_intent=visual_intent,
            candidate_pool=candidate_pool,
        )
        top_k = min(self._runtime_top_k, len(ranked))
        selected = ranked[:top_k]
        logger.info(
            "[PaperVizPlannerAgent] Retrieved %s reference examples for task=%s (pool=%s, top_k=%s)",
            len(selected),
            task_name,
            len(candidate_pool),
            self._runtime_top_k,
        )
        return selected

    def _load_candidate_pool(self, *, task_name: str) -> list[dict[str, Any]]:
        project_root = Path(__file__).resolve().parents[3]
        ref_path = (
            project_root / "experiments" / "PaperBananaBench" / "data" / task_name / "ref.json"
        )
        if not ref_path.exists():
            logger.warning(f"[PaperVizPlannerAgent] Candidate pool not found: {ref_path}")
            return []
        pool = json.loads(ref_path.read_text(encoding="utf-8"))
        limit = self._runtime_candidate_pool_limit
        if limit is None and task_name == "diagram":
            limit = 200
        if limit is not None and limit >= 0:
            pool = pool[:limit]
        return pool

    def _rank_candidates(
        self,
        *,
        task_name: str,
        content: str,
        visual_intent: str,
        candidate_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query_text = (
            f"Visual intent:\n{visual_intent}\n\nRaw data:\n{content}"
            if task_name == "plot"
            else f"Caption:\n{visual_intent}\n\nMethodology section:\n{content}"
        )
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return candidate_pool

        query_counts = Counter(query_tokens)
        query_vocab = set(query_counts)
        doc_freq: Counter[str] = Counter()
        indexed: list[dict[str, Any]] = []

        for item in candidate_pool:
            item_visual_intent = str(item.get("visual_intent", ""))
            item_content = str(item.get("content", ""))
            visual_tokens = self._tokenize(item_visual_intent)
            content_tokens = self._tokenize(item_content)
            all_tokens = set(visual_tokens) | set(content_tokens)
            for token in query_vocab & all_tokens:
                doc_freq[token] += 1
            indexed.append(
                {
                    "item": item,
                    "visual_tokens": Counter(visual_tokens),
                    "content_tokens": Counter(content_tokens),
                }
            )

        total_docs = max(1, len(indexed))
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in indexed:
            visual_counts = record["visual_tokens"]
            content_counts = record["content_tokens"]
            lexical_score = 0.0
            for token, qtf in query_counts.items():
                idf = math.log((1 + total_docs) / (1 + doc_freq.get(token, 0))) + 1.0
                lexical_score += qtf * idf * (
                    2.0 * min(visual_counts.get(token, 0), 3)
                    + 1.0 * min(content_counts.get(token, 0), 3)
                )

            query_visual_hits = query_vocab & _VISUAL_KEYWORDS
            doc_visual_hits = (set(visual_counts) | set(content_counts)) & _VISUAL_KEYWORDS
            structure_bonus = 3.0 * len(query_visual_hits & doc_visual_hits)
            scored.append((lexical_score + structure_bonus, record["item"]))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("id", ""))))
        return [item for _, item in scored]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1]

    def _load_reference_image_base64(
        self,
        *,
        task_name: str,
        item: Dict[str, Any],
    ) -> Optional[str]:
        path_to_gt_image = str(item.get("path_to_gt_image", "")).strip()
        if not path_to_gt_image:
            return None

        project_root = Path(__file__).resolve().parents[3]
        image_path = (
            project_root
            / "experiments"
            / "PaperBananaBench"
            / "data"
            / task_name
            / path_to_gt_image
        )
        image_path = self._resolve_reference_image_path(image_path)
        if image_path is None or not image_path.exists():
            logger.warning(
                f"[PaperVizPlannerAgent] Reference image not found for example '{item.get('id', '')}': "
                f"{project_root / 'experiments' / 'PaperBananaBench' / 'data' / task_name / path_to_gt_image}"
            )
            return None

        try:
            return base64.b64encode(image_path.read_bytes()).decode("utf-8")
        except Exception as exc:
            logger.warning(
                f"[PaperVizPlannerAgent] Failed to read reference image '{image_path}': {exc}"
            )
            return None

    def _resolve_reference_image_path(self, image_path: Path) -> Optional[Path]:
        if image_path.exists():
            return image_path

        parent = image_path.parent
        if not parent.exists():
            return None

        target_name = image_path.name
        target_stem = self._canonicalize_filename(image_path.stem)
        if not target_stem:
            return None

        candidates: list[tuple[str, Path]] = []
        for child in parent.iterdir():
            if not child.is_file():
                continue
            if child.suffix.lower() != image_path.suffix.lower():
                continue
            child_stem = self._canonicalize_filename(child.stem)
            if child_stem:
                candidates.append((child_stem, child))

        for child_stem, child in candidates:
            if child_stem == target_stem:
                logger.info(
                    f"[PaperVizPlannerAgent] Resolved reference image by canonical filename match: "
                    f"'{target_name}' -> '{child.name}'"
                )
                return child

        names = [name for name, _ in candidates]
        match = difflib.get_close_matches(target_stem, names, n=1, cutoff=0.75)
        if match:
            matched_name = match[0]
            for child_stem, child in candidates:
                if child_stem == matched_name:
                    logger.info(
                        f"[PaperVizPlannerAgent] Resolved reference image by fuzzy filename match: "
                        f"'{target_name}' -> '{child.name}'"
                    )
                    return child

        return None

    @staticmethod
    def _canonicalize_filename(text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
