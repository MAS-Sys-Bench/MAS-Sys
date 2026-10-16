from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.agent.agents_config import AGENT_METADATA
from app.agent.fund.analyst import FinancialAnalystAgent
from app.agent.fund.manager import FundManagerAgent
from app.agent.proposal.proposal_aggregator import ProposalAggregatorAgent
from app.agent.proposal.proposal_writer import ProposalWriterAgent
from app.agent.review.metareview_chair import MetaReviewChairAgent
from app.agent.review.review_writer import ReviewWriterAgent
from app.agent.toolcall import ToolCallAgent
from app.prompt.fund.manager import (
    MANAGER_DECISION_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    RISK_CONTROL_PROMPT,
    RISK_CONTROL_SYSTEM_PROMPT,
)
from app.prompt.proposal import PROPOSAL_WRITER_FEWSHOT_EXAMPLES
from app.prompt.proposal.proposal_writer import GENERATE_PROFILE_SYSTEM_PROMPT, PAPER_WRITER_SYSTEM_TEMPLATE
from app.prompt.review.metareview_chair import FEWSHOT_EXAMPLES as METAREVIEW_FEWSHOT_EXAMPLES
from app.prompt.review.metareview_chair import FINAL_META_REVIEW_SYSTEM_PROMPT, build_metareview_prompt
from app.prompt.review.review_writer import (
    FEWSHOT_EXAMPLES as REVIEW_FEWSHOT_EXAMPLES,
    GENERATE_PROFILE_SYSTEM_PROMPT as REVIEW_GENERATE_PROFILE_SYSTEM_PROMPT,
    REVIEW_SCORE_SYSTEM_PROMPT,
    REVIEW_STRENGTH_SYSTEM_PROMPT,
    REVIEW_WEAKNESS_SYSTEM_PROMPT,
    build_profile_prompt,
    build_score_prompt,
    build_strength_prompt,
    build_weakness_prompt,
    format_review_output,
)
from app.schema import Message


StageMessagesBuilder = Callable[[str, dict[str, str]], list[Message]]
NodeFinalizer = Callable[[object, str, dict[str, str]], str]
NodePersister = Callable[[object, str, dict[str, str], str], None]


@dataclass
class StageRecipe:
    name: str
    build_messages: StageMessagesBuilder
    execution_mode: str = "single_shot"
    max_turns: int = 0
    include_tools: bool = False


@dataclass
class NodeRecipe:
    kind: str
    stages: list[StageRecipe] = field(default_factory=list)
    finalize: Optional[NodeFinalizer] = None
    persist: Optional[NodePersister] = None


def _fewshot_messages(pairs: list[tuple[str, str]]) -> list[Message]:
    messages: list[Message] = []
    for user_text, assistant_text in pairs:
        messages.append(Message.user_message(user_text))
        messages.append(Message.assistant_message(assistant_text))
    return messages


def _persist_user_and_assistant(
    agent: object,
    prompt: str,
    _outputs: dict[str, str],
    result: str,
) -> None:
    if prompt:
        agent.update_memory("user", prompt)
    agent.update_memory("assistant", result)


def _persist_assistant_only(
    agent: object,
    _request: str,
    _outputs: dict[str, str],
    result: str,
) -> None:
    agent.update_memory("assistant", result)


def _generic_lang_single_recipe(agent: object) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        messages: list[Message] = []
        if getattr(agent, "system_prompt", None):
            messages.append(Message.system_message(agent.system_prompt))
        messages.append(Message.user_message(request_text))
        return messages

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="single", build_messages=_messages)],
        finalize=lambda _agent, _request, outputs: outputs["single"],
        persist=_persist_user_and_assistant,
    )


def _proposal_aggregator_recipe(agent: ProposalAggregatorAgent) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        prompt = agent.build_request(proposal_candidates=request_text)
        return (
            [Message.system_message(agent.system_prompt)]
            + _fewshot_messages(PROPOSAL_WRITER_FEWSHOT_EXAMPLES)
            + [Message.user_message(prompt)]
        )

    def _persist(
        recipe_agent: ProposalAggregatorAgent,
        request_text: str,
        outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory(
            "user", recipe_agent.build_request(proposal_candidates=request_text)
        )
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="aggregate", build_messages=_messages)],
        finalize=lambda _agent, _request, outputs: outputs["aggregate"],
        persist=_persist,
    )


def _metareview_chair_recipe(agent: MetaReviewChairAgent) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        return (
            [Message.system_message(FINAL_META_REVIEW_SYSTEM_PROMPT)]
            + _fewshot_messages(METAREVIEW_FEWSHOT_EXAMPLES)
            + [Message.user_message(build_metareview_prompt(reviews=request_text))]
        )

    def _persist(
        recipe_agent: MetaReviewChairAgent,
        request_text: str,
        _outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory("user", build_metareview_prompt(reviews=request_text))
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="metareview", build_messages=_messages)],
        finalize=lambda _agent, _request, outputs: outputs["metareview"],
        persist=_persist,
    )


def _financial_analyst_recipe(agent: FinancialAnalystAgent) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        prompt = agent._build_prompt(request=request_text)
        return [
            Message.system_message(agent._analysis_system_prompt),
            Message.user_message(prompt),
        ]

    def _persist(
        recipe_agent: FinancialAnalystAgent,
        request_text: str,
        _outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory("user", recipe_agent._build_prompt(request=request_text))
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="analysis", build_messages=_messages)],
        finalize=lambda _agent, _request, outputs: outputs["analysis"],
        persist=_persist,
    )


def _paperviz_retriever_recipe(agent: object) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        prompt = agent._build_prompt(request_text)
        return [Message.system_message(agent._system_prompt()), Message.user_message(prompt)]

    def _finalize(recipe_agent: object, _request: str, outputs: dict[str, str]) -> str:
        return recipe_agent._build_result(outputs["retrieve"])

    def _persist(
        recipe_agent: object,
        request_text: str,
        outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory("user", recipe_agent._build_prompt(request_text))
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="retrieve", build_messages=_messages)],
        finalize=_finalize,
        persist=_persist,
    )


def _paperviz_stylist_recipe(agent: object) -> NodeRecipe:
    def _messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        prompt = agent._build_prompt(request_text)
        return [Message.system_message(agent._system_prompt()), Message.user_message(prompt)]

    def _persist(
        recipe_agent: object,
        request_text: str,
        _outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory("user", recipe_agent._build_prompt(request_text))
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="single_stage",
        stages=[StageRecipe(name="style", build_messages=_messages)],
        finalize=lambda _agent, _request, outputs: outputs["style"],
        persist=_persist,
    )


def _proposal_writer_recipe(agent: ProposalWriterAgent) -> NodeRecipe:
    def _persona_messages(_request_text: str, _outputs: dict[str, str]) -> list[Message]:
        return (
            [Message.system_message(GENERATE_PROFILE_SYSTEM_PROMPT)]
            + _fewshot_messages(PROPOSAL_WRITER_FEWSHOT_EXAMPLES)
            + [
                Message.user_message(
                    agent.build_profile_request(
                        author_name=str(agent.agent_params.get("author_name", "")),
                        affiliation=str(agent.agent_params.get("affiliation", "")),
                        publications=str(agent.agent_params.get("publications", "")),
                    )
                )
            ]
        )

    def _candidate_messages(request_text: str, outputs: dict[str, str]) -> list[Message]:
        persona = outputs["persona"]
        return (
            [Message.system_message(PAPER_WRITER_SYSTEM_TEMPLATE.format(profile=persona))]
            + _fewshot_messages(PROPOSAL_WRITER_FEWSHOT_EXAMPLES)
            + [
                Message.user_message(
                    agent.build_candidate_request(related_papers=request_text)
                )
            ]
        )

    return NodeRecipe(
        kind="multi_stage",
        stages=[
            StageRecipe(name="persona", build_messages=_persona_messages),
            StageRecipe(name="candidate", build_messages=_candidate_messages),
        ],
        finalize=lambda _agent, _request, outputs: outputs["candidate"],
        persist=_persist_assistant_only,
    )


def _review_writer_recipe(agent: ReviewWriterAgent) -> NodeRecipe:
    def _with_fewshots(prompt: str, system_prompt: str) -> list[Message]:
        return [Message.system_message(system_prompt)] + _fewshot_messages(REVIEW_FEWSHOT_EXAMPLES) + [
            Message.user_message(prompt)
        ]

    def _split_request(request_text: str) -> tuple[str, str]:
        return agent._split_shared_request(request_text or "")

    def _persona_messages(_request_text: str, _outputs: dict[str, str]) -> list[Message]:
        return _with_fewshots(
            build_profile_prompt(
                reviewer_name=str(agent.agent_params.get("reviewer_name", "")),
                affiliation=str(agent.agent_params.get("affiliation", "")),
                publications=str(agent.agent_params.get("publications", "")),
            ),
            REVIEW_GENERATE_PROFILE_SYSTEM_PROMPT,
        )

    def _strength_messages(request_text: str, outputs: dict[str, str]) -> list[Message]:
        proposal, shared_citations = _split_request(request_text)
        citations = shared_citations or str(agent.agent_params.get("citations", ""))
        prompt = build_strength_prompt(proposal=proposal, citations=citations).replace(
            "{profile}", outputs["persona"]
        )
        return _with_fewshots(prompt, REVIEW_STRENGTH_SYSTEM_PROMPT)

    def _weakness_messages(request_text: str, outputs: dict[str, str]) -> list[Message]:
        proposal, shared_citations = _split_request(request_text)
        citations = shared_citations or str(agent.agent_params.get("citations", ""))
        prompt = build_weakness_prompt(proposal=proposal, citations=citations).replace(
            "{profile}", outputs["persona"]
        )
        return _with_fewshots(prompt, REVIEW_WEAKNESS_SYSTEM_PROMPT)

    def _score_messages(_request_text: str, outputs: dict[str, str]) -> list[Message]:
        prompt = build_score_prompt(
            strength=outputs["strength"],
            weakness=outputs["weakness"],
        ).replace("{profile}", outputs["persona"])
        return _with_fewshots(prompt, REVIEW_SCORE_SYSTEM_PROMPT)

    def _finalize(_agent: ReviewWriterAgent, _request: str, outputs: dict[str, str]) -> str:
        score = agent._extract_score(outputs["score"])
        return format_review_output(
            persona=outputs["persona"],
            strength=outputs["strength"],
            weakness=outputs["weakness"],
            score=score,
        )

    return NodeRecipe(
        kind="multi_stage",
        stages=[
            StageRecipe(name="persona", build_messages=_persona_messages),
            StageRecipe(name="strength", build_messages=_strength_messages),
            StageRecipe(name="weakness", build_messages=_weakness_messages),
            StageRecipe(name="score", build_messages=_score_messages),
        ],
        finalize=_finalize,
        persist=_persist_assistant_only,
    )


def _fund_manager_recipe(agent: FundManagerAgent) -> NodeRecipe:
    def _risk_messages(request_text: str, _outputs: dict[str, str]) -> list[Message]:
        ticker_signals = agent.agent_params.get("ticker_signals") or request_text or ""
        portfolio_state = agent.agent_params.get(
            "portfolio_state", agent.agent_params.get("portfolio", {})
        )
        max_position_ratio = float(agent.agent_params.get("max_position_ratio", 1.0))
        prompt = RISK_CONTROL_PROMPT.format(
            ticker_signals=str(ticker_signals or ""),
            portfolio=json.dumps(portfolio_state, ensure_ascii=False, indent=2),
            max_position_ratio=max_position_ratio,
        )
        return [Message.system_message(RISK_CONTROL_SYSTEM_PROMPT), Message.user_message(prompt)]

    def _decision_messages(_request_text: str, outputs: dict[str, str]) -> list[Message]:
        max_position_ratio = float(agent.agent_params.get("max_position_ratio", 1.0))
        optimal_ratio, risk_justification = agent._parse_risk_response(
            outputs["risk_control"],
            max_position_ratio=max_position_ratio,
        )
        current_price = float(agent.agent_params.get("current_price", 0.0) or 0.0)
        portfolio_state = agent.agent_params.get(
            "portfolio_state", agent.agent_params.get("portfolio", {})
        )
        current_shares, tradable_shares = agent._calculate_ticker_shares(
            portfolio_state=portfolio_state,
            current_price=current_price,
            ticker=str(agent.agent_params.get("ticker", "")),
            optimal_position_ratio=optimal_ratio,
        )
        prompt = MANAGER_DECISION_PROMPT.format(
            decision_memory=json.dumps(
                agent.agent_params.get("decision_memory", []),
                ensure_ascii=False,
                indent=2,
            ),
            current_price=current_price,
            current_shares=current_shares,
            tradable_shares=tradable_shares,
            optimal_position_ratio=optimal_ratio,
            risk_justification=risk_justification,
        )
        return [Message.system_message(MANAGER_SYSTEM_PROMPT), Message.user_message(prompt)]

    def _finalize(_agent: FundManagerAgent, _request: str, outputs: dict[str, str]) -> str:
        max_position_ratio = float(agent.agent_params.get("max_position_ratio", 1.0))
        optimal_ratio, risk_justification = agent._parse_risk_response(
            outputs["risk_control"],
            max_position_ratio=max_position_ratio,
        )
        current_price = float(agent.agent_params.get("current_price", 0.0) or 0.0)
        portfolio_state = agent.agent_params.get(
            "portfolio_state", agent.agent_params.get("portfolio", {})
        )
        _, tradable_shares = agent._calculate_ticker_shares(
            portfolio_state=portfolio_state,
            current_price=current_price,
            ticker=str(agent.agent_params.get("ticker", "")),
            optimal_position_ratio=optimal_ratio,
        )
        decision = agent._parse_decision_response(
            outputs["decision"],
            current_price=current_price,
            tradable_shares=tradable_shares,
            risk_justification=risk_justification,
        )
        return json.dumps(decision, ensure_ascii=False, indent=2)

    def _persist(
        recipe_agent: FundManagerAgent,
        request_text: str,
        _outputs: dict[str, str],
        result: str,
    ) -> None:
        recipe_agent.update_memory("user", request_text or "")
        recipe_agent.update_memory("assistant", result)

    return NodeRecipe(
        kind="multi_stage",
        stages=[
            StageRecipe(name="risk_control", build_messages=_risk_messages),
            StageRecipe(name="decision", build_messages=_decision_messages),
        ],
        finalize=_finalize,
        persist=_persist,
    )


_SINGLE_STAGE_BUILDERS: dict[str, Callable[[object], NodeRecipe]] = {
    "research_proposal_aggregator": _proposal_aggregator_recipe,
    "research_metareview_chair": _metareview_chair_recipe,
    "fund_technical": _financial_analyst_recipe,
    "fund_fundamental": _financial_analyst_recipe,
    "fund_insider": _financial_analyst_recipe,
    "fund_news": _financial_analyst_recipe,
    "fund_macro": _financial_analyst_recipe,
    "fund_policy": _financial_analyst_recipe,
    "paperviz_retriever": _paperviz_retriever_recipe,
    "paperviz_stylist": _paperviz_stylist_recipe,
}

_MULTI_STAGE_BUILDERS: dict[str, Callable[[object], NodeRecipe]] = {
    "research_proposal_writer": _proposal_writer_recipe,
    "research_review_writer": _review_writer_recipe,
    "fund_manager": _fund_manager_recipe,
}


def is_standard_tool_loop(agent: object) -> bool:
    return isinstance(agent, ToolCallAgent) and type(agent).run is ToolCallAgent.run


def build_node_recipe(node: object, agent: object) -> NodeRecipe:
    meta = AGENT_METADATA.get(getattr(node, "agent_key", ""), {})
    llm_type = meta.get("llm_type", "lang")
    if llm_type != "lang":
        return NodeRecipe(kind="external")

    if getattr(node, "agent_key", "") in _MULTI_STAGE_BUILDERS:
        return _MULTI_STAGE_BUILDERS[node.agent_key](agent)

    if is_standard_tool_loop(agent):
        return NodeRecipe(
            kind="agent_loop",
            stages=[
                StageRecipe(
                    name="agent_loop",
                    build_messages=lambda request_text, _outputs: (
                        ([Message.system_message(agent.system_prompt)] if agent.system_prompt else [])
                        + ([Message.user_message(request_text)] if request_text else [])
                    ),
                    execution_mode="agent_loop",
                    max_turns=int(getattr(agent, "max_steps", 0) or 0),
                    include_tools=True,
                )
            ],
        )

    if getattr(node, "agent_key", "") in _SINGLE_STAGE_BUILDERS:
        return _SINGLE_STAGE_BUILDERS[node.agent_key](agent)

    if llm_type == "lang" and not isinstance(agent, ToolCallAgent):
        return _generic_lang_single_recipe(agent)

    return NodeRecipe(kind="external")
