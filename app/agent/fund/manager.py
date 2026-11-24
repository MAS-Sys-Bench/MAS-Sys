from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import Field, PrivateAttr

from app.agent.base import BaseAgent
from app.agent.fund.analyst import _load_current_price
from app.logger import logger
from app.prompt.fund.manager import (
    MANAGER_DECISION_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    RISK_CONTROL_PROMPT,
    RISK_CONTROL_SYSTEM_PROMPT,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message


class FundManagerAgent(BaseAgent):
    name: str = "fund_manager"
    description: str = "Portfolio manager that performs risk control and emits the final trading decision."
    max_steps: int = 4
    continuum_expected_llm_calls: int = 2
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    _runtime_current_price: float = PrivateAttr(default=0.0)

    def initialize_from_params(self) -> None:
        params = self.agent_params
        ticker = str(params.get("ticker", "")).upper()
        trading_date = str(params.get("trading_date", ""))
        self._runtime_current_price = _load_current_price(ticker, trading_date)

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        self.update_memory("user", request or "")
        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            result = await self.step()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED
        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        ticker_signals = self.agent_params.get("ticker_signals")
        if not ticker_signals and self.messages:
            ticker_signals = self.messages[-1].content or ""
        portfolio_state = self.agent_params.get("portfolio_state", self.agent_params.get("portfolio", {}))
        max_position_ratio = float(self.agent_params.get("max_position_ratio", 1.0))
        risk_prompt = RISK_CONTROL_PROMPT.format(
            ticker_signals=str(ticker_signals or ""),
            portfolio=json.dumps(portfolio_state, ensure_ascii=False, indent=2),
            max_position_ratio=max_position_ratio,
        )
        risk_response = await self._ask_json(
            prompt=risk_prompt,
            system_prompt=RISK_CONTROL_SYSTEM_PROMPT,
        )
        optimal_ratio, risk_justification = self._parse_risk_response(
            risk_response,
            max_position_ratio=max_position_ratio,
        )

        current_price = float(self._runtime_current_price or 0.0)
        current_shares, tradable_shares = self._calculate_ticker_shares(
            portfolio_state=portfolio_state,
            current_price=current_price,
            ticker=str(self.agent_params.get("ticker", "")),
            optimal_position_ratio=optimal_ratio,
        )

        decision_prompt = MANAGER_DECISION_PROMPT.format(
            decision_memory=json.dumps(
                self.agent_params.get("decision_memory", []),
                ensure_ascii=False,
                indent=2,
            ),
            current_price=current_price,
            current_shares=current_shares,
            tradable_shares=tradable_shares,
            optimal_position_ratio=optimal_ratio,
            risk_justification=risk_justification,
        )
        decision_response = await self._ask_json(
            prompt=decision_prompt,
            system_prompt=MANAGER_SYSTEM_PROMPT,
        )
        decision = self._parse_decision_response(
            decision_response,
            current_price=current_price,
            tradable_shares=tradable_shares,
            risk_justification=risk_justification,
        )
        result = json.dumps(decision, ensure_ascii=False, indent=2)
        self.update_memory("assistant", result)
        self.state = AgentState.FINISHED
        return result

    async def _ask(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response_format: Optional[dict] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        logger.info(f"[FundManagerAgent] Running node '{self.name}'")
        return await self.ask_with_trace(
            messages=[Message.user_message(prompt)],
            system_msgs=[Message.system_message(system_prompt)],
            stream=stream,
            response_format=response_format,
            max_tokens=max_tokens,
        )

    async def _ask_json(self, *, prompt: str, system_prompt: str) -> str:
        return await self._ask(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            stream=True,
            max_tokens=512,
        )

    @staticmethod
    def _extract_json_dict(response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if not text:
            raise ValueError("empty response")

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise ValueError("no JSON object found in response")

    @staticmethod
    def _parse_risk_response(response: str, *, max_position_ratio: float) -> tuple[float, str]:
        data = FundManagerAgent._extract_json_dict(response)
        if "optimal_position_ratio" not in data:
            raise ValueError("Fund manager risk JSON missing 'optimal_position_ratio'")
        if "justification" not in data:
            raise ValueError("Fund manager risk JSON missing 'justification'")
        ratio = float(data["optimal_position_ratio"])
        justification = str(data["justification"]).strip()
        if not justification:
            raise ValueError("Fund manager risk JSON has empty 'justification'")
        ratio = max(0.0, min(ratio, max_position_ratio))
        return ratio, justification

    @staticmethod
    def _calculate_ticker_shares(
        *,
        portfolio_state: Dict[str, Any],
        current_price: float,
        ticker: str,
        optimal_position_ratio: float,
    ) -> tuple[int, int]:
        if current_price <= 0:
            return 0, 0

        positions = portfolio_state.get("positions", {}) or {}
        ticker_position = positions.get(ticker, {}) or {}
        current_shares = int(ticker_position.get("shares", 0) or 0)
        current_value = current_shares * current_price
        total_portfolio_value = float(portfolio_state.get("cashflow", 0.0) or 0.0) + sum(
            float((pos or {}).get("value", 0.0) or 0.0) for pos in positions.values()
        )
        position_limit = total_portfolio_value * optimal_position_ratio
        position_value_gap = position_limit - current_value

        if position_value_gap > 0:
            tradable_shares = int(
                min(position_value_gap, float(portfolio_state.get("cashflow", 0.0) or 0.0)) // current_price
            )
        else:
            tradable_shares = int(max(position_value_gap // current_price, -current_shares))

        return current_shares, tradable_shares

    @staticmethod
    def _parse_decision_response(
        response: str,
        *,
        current_price: float,
        tradable_shares: int,
        risk_justification: str,
    ) -> Dict[str, Any]:
        parsed = FundManagerAgent._extract_json_dict(response)
        missing = {"action", "shares", "price", "justification"} - set(parsed)
        if missing:
            raise ValueError(f"Fund manager decision JSON missing fields: {sorted(missing)}")

        action = str(parsed["action"]).title()
        if action not in {"Buy", "Sell", "Hold"}:
            raise ValueError(f"Fund manager decision JSON has invalid action: {parsed.get('action')!r}")
        try:
            shares = int(parsed["shares"])
        except Exception as exc:
            raise ValueError(f"Fund manager decision JSON has invalid shares: {parsed.get('shares')!r}") from exc
        if action == "Hold":
            shares = 0
        elif action == "Buy":
            shares = max(0, shares)
        else:
            shares = abs(shares)
        justification = str(parsed["justification"] or "").strip()
        if not justification:
            raise ValueError("Fund manager decision JSON has empty 'justification'")
        try:
            price = float(parsed["price"])
        except Exception as exc:
            raise ValueError(f"Fund manager decision JSON has invalid price: {parsed.get('price')!r}") from exc

        return {
            "action": action,
            "shares": shares,
            "price": price,
            "justification": justification,
        }
