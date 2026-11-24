from __future__ import annotations

import json
from datetime import datetime
import math
from typing import Any, Dict, Literal, Optional

import pandas as pd
from pydantic import Field, PrivateAttr, model_validator

from app.agent.base import BaseAgent
from app.logger import logger
from app.prompt.fund.analyst import (
    ROLE_PROMPTS,
)
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Message
from app.utils.fund_util import filter_before_trading_date, resolve_data_dir, safe_read_json

FinancialRole = Literal["technical", "fundamental", "insider", "news", "macro", "policy"]


def _compute_technical_analysis(prices_df: pd.DataFrame) -> dict[str, str]:
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    ema_short = _ema(prices_df["close"], 8)
    ema_medium = _ema(prices_df["close"], 21)
    ema_long = _ema(prices_df["close"], 55)
    if ema_short.iloc[-1] > ema_medium.iloc[-1] and ema_medium.iloc[-1] > ema_long.iloc[-1]:
        trend = "Bullish"
    elif ema_short.iloc[-1] < ema_medium.iloc[-1] and ema_medium.iloc[-1] < ema_long.iloc[-1]:
        trend = "Bearish"
    else:
        trend = "Neutral"

    sma20 = prices_df["close"].rolling(20).mean()
    std20 = prices_df["close"].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    ma50 = prices_df["close"].rolling(50).mean()
    std50 = prices_df["close"].rolling(50).std()
    z_score = (prices_df["close"] - ma50) / std50
    band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    bb_pos = 0.5 if not band_width or pd.isna(band_width) else (
        (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / band_width
    )
    if z_score.iloc[-1] < -2.0 and bb_pos < 0.2:
        mean_reversion = "Bullish"
    elif z_score.iloc[-1] > 2.0 and bb_pos > 0.8:
        mean_reversion = "Bearish"
    else:
        mean_reversion = "Neutral"

    delta = prices_df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_value = (100 - 100 / (1 + rs)).iloc[-1]
    if rsi_value > 70:
        rsi = "Bearish"
    elif rsi_value < 30:
        rsi = "Bullish"
    else:
        rsi = "Neutral"

    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z = (hist_vol - vol_ma) / hist_vol.rolling(63).std()
    current_regime = vol_regime.iloc[-1] if not pd.isna(vol_regime.iloc[-1]) else 1.0
    current_vol_z = vol_z.iloc[-1] if not pd.isna(vol_z.iloc[-1]) else 0.0
    if current_regime < 0.8 and current_vol_z < -1:
        volatility = "Bullish"
    elif current_regime > 1.2 and current_vol_z > 1:
        volatility = "Bearish"
    else:
        volatility = "Neutral"

    vol_ma20 = prices_df["volume"].rolling(20).mean()
    volume_text = (
        f"- Volume trend: {'Bullish' if prices_df['volume'].iloc[-1] > vol_ma20.iloc[-1] else 'Bearish'}\n"
        f"- Price-volume correlation: {prices_df['close'].rolling(20).corr(prices_df['volume']).iloc[-1]:.4f}\n"
        f"- Unusual volume: {bool(prices_df['volume'].iloc[-1] > vol_ma20.iloc[-1] * 2.0)}"
    )

    price_data = prices_df["close"]
    current_price = price_data.iloc[-1]
    supports: list[float] = []
    resistances: list[float] = []
    lookback = min(20, max(0, len(price_data) - 10))
    for i in range(lookback, len(price_data)):
        start = max(0, i - 5)
        end = min(len(price_data), i + 6)
        window = price_data.iloc[start:end]
        if price_data.iloc[i] == window.min():
            supports.append(float(price_data.iloc[i]))
        elif price_data.iloc[i] == window.max():
            resistances.append(float(price_data.iloc[i]))
    nearest_support = max((s for s in supports if s < current_price), default=None)
    nearest_resistance = min((r for r in resistances if r > current_price), default=None)
    price_levels = (
        f"Current price: {current_price:.2f}\n"
        f"Nearest support: {nearest_support}\n"
        f"Nearest resistance: {nearest_resistance}"
    )

    return {
        "trend": trend,
        "mean_reversion": mean_reversion,
        "rsi": rsi,
        "volatility": volatility,
        "volume": volume_text,
        "price_levels": price_levels,
    }


def _load_technical_text(ticker: str, trading_date: str) -> str:
    base = resolve_data_dir()
    csv_path = base / "daily" / f"{ticker.upper()}.csv"
    if not csv_path.exists():
        return f"No OHLCV data file found at {csv_path}"

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(trading_date)
    df = df[df["date"] <= cutoff].sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        return f"Insufficient price history for {ticker} before {trading_date}."

    df.set_index("date", inplace=True)
    analysis = _compute_technical_analysis(df)
    return (
        f"Price Trend Analysis:\n"
        f"- Trend Following: {analysis['trend']}\n\n"
        f"Mean Reversion and Momentum:\n"
        f"- Mean Reversion: {analysis['mean_reversion']}\n"
        f"- RSI: {analysis['rsi']}\n"
        f"- Volatility: {analysis['volatility']}\n\n"
        f"Volume Analysis:\n"
        f"{analysis['volume']}\n\n"
        f"Support and Resistance Levels:\n"
        f"{analysis['price_levels']}"
    )


def _load_fundamentals_text(ticker: str) -> str:
    base = resolve_data_dir()
    path = base / "fundamentals" / f"{ticker.upper()}.json"
    data = safe_read_json(path)
    if data is None:
        return f"No fundamentals data found at {path}"
    return json.dumps(data, ensure_ascii=False, indent=2)


def _load_insider_text(ticker: str, trading_date: str, limit: int = 10) -> str:
    base = resolve_data_dir()
    path = base / "insider" / f"{ticker.upper()}.json"
    trades = safe_read_json(path)
    if trades is None:
        return f"No insider trades data found at {path}"

    cutoff = datetime.strptime(trading_date, "%Y-%m-%d")
    filtered = []
    for trade in trades:
        raw = trade.get("transaction_date") or trade.get("Start Date") or ""
        if not raw:
            continue
        try:
            normalized = raw.replace("Z", "+00:00") if "Z" in raw else raw
            trade_date = datetime.fromisoformat(normalized).replace(tzinfo=None)
        except (TypeError, ValueError):
            try:
                trade_date = datetime.strptime(raw, "%Y-%m-%d")
            except (TypeError, ValueError):
                continue
        if trade_date < cutoff:
            filtered.append(trade)
    return json.dumps(filtered[:limit], ensure_ascii=False, indent=2)


def _load_news_text(ticker: str, trading_date: str = "", limit: int = 20) -> str:
    base = resolve_data_dir()
    path = base / "news" / f"{ticker.upper()}.json"
    articles = safe_read_json(path)
    if articles is None:
        return f"No company news data found at {path}"
    if trading_date:
        articles = filter_before_trading_date(articles, trading_date, "publish_time")
    return json.dumps(articles[:limit], ensure_ascii=False, indent=2)


def _load_macro_text() -> str:
    base = resolve_data_dir()
    path = base / "macro" / "indicators.json"
    indicators = safe_read_json(path)
    if indicators is None:
        return f"No macro indicators found at {path}"
    return json.dumps(indicators, ensure_ascii=False, indent=2)


def _load_policy_payload(trading_date: str = "", limit: int = 20) -> Dict[str, Any]:
    base = resolve_data_dir()
    policy_dir = base / "policy"
    fiscal = safe_read_json(policy_dir / "fiscal.json") or []
    monetary = safe_read_json(policy_dir / "monetary.json") or []
    if trading_date:
        fiscal = filter_before_trading_date(fiscal, trading_date, "publish_time")
        monetary = filter_before_trading_date(monetary, trading_date, "publish_time")
    return {
        "fiscal_policy": fiscal[:limit],
        "monetary_policy": monetary[:limit],
    }


def _load_current_price(ticker: str, trading_date: str) -> float:
    base = resolve_data_dir()
    csv_path = base / "daily" / f"{ticker.upper()}.csv"
    if not csv_path.exists():
        return 0.0

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(trading_date)
    df = df[df["date"] <= cutoff].sort_values("date").reset_index(drop=True)
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


class FinancialAnalystAgent(BaseAgent):
    name: str = "financial_analyst"
    description: str = "Role-specialized financial analyst that loads local market data and returns a final trading signal."
    role: FinancialRole = "technical"
    max_steps: int = 1
    continuum_expected_llm_calls: int = 1
    agent_params: Dict[str, Any] = Field(default_factory=dict)
    _analysis_system_prompt: str = ""
    _analysis_template: str = ""
    _runtime_data: str = PrivateAttr(default="")
    _runtime_fiscal_policy: str = PrivateAttr(default="[]")
    _runtime_monetary_policy: str = PrivateAttr(default="[]")

    @model_validator(mode="after")
    def _configure_prompt(self) -> "FinancialAnalystAgent":
        system_prompt, template = ROLE_PROMPTS[self.role]
        self._analysis_system_prompt = system_prompt
        self._analysis_template = template
        return self

    def initialize_from_params(self) -> None:
        params = self.agent_params
        ticker = str(params.get("ticker", "")).upper()
        trading_date = str(params.get("trading_date", ""))

        if self.role == "technical":
            self._runtime_data = _load_technical_text(ticker, trading_date)
        elif self.role == "fundamental":
            self._runtime_data = _load_fundamentals_text(ticker)
        elif self.role == "insider":
            self._runtime_data = _load_insider_text(ticker, trading_date)
        elif self.role == "news":
            self._runtime_data = _load_news_text(ticker, trading_date)
        elif self.role == "macro":
            self._runtime_data = _load_macro_text()
        elif self.role == "policy":
            payload = _load_policy_payload(trading_date)
            self._runtime_fiscal_policy = json.dumps(
                payload.get("fiscal_policy", []),
                ensure_ascii=False,
                indent=2,
            )
            self._runtime_monetary_policy = json.dumps(
                payload.get("monetary_policy", []),
                ensure_ascii=False,
                indent=2,
            )

    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        async with self.state_context(AgentState.RUNNING):
            self.current_step = 1
            self.start_single_step_trace()
            prompt = self._build_prompt(request=request)
            self.update_memory("user", prompt)
            result = await self._run_plain()
            self.end_single_step_trace()
            self.state = AgentState.FINISHED

        await SANDBOX_CLIENT.cleanup()
        return result or ""

    async def step(self) -> str:
        raise NotImplementedError("FinancialAnalystAgent uses run() directly.")

    async def _run_plain(self) -> str:
        logger.info(f"[FinancialAnalystAgent] Running node '{self.name}' role={self.role}")
        response = await self.ask_with_trace(
            messages=self.messages,
            system_msgs=[Message.system_message(self._analysis_system_prompt)],
            stream=True,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        response = self._normalize_response(response)
        self.update_memory("assistant", response)
        self.state = AgentState.FINISHED
        return response

    def _normalize_response(self, response: str) -> str:
        try:
            parsed = json.loads((response or "").strip())
            if isinstance(parsed, dict) and parsed.get("signal"):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return response

    def _build_prompt(self, *, request: Optional[str]) -> str:
        injected_data = self._runtime_data
        context_suffix = f"\n\nAdditional task context:\n{request}" if request else ""
        if self.role == "technical":
            return self._analysis_template.format(
                analysis=injected_data or "No technical data was loaded."
            ) + context_suffix
        if self.role == "fundamental":
            return self._analysis_template.format(
                fundamentals=injected_data or "No fundamentals data was loaded."
            ) + context_suffix
        if self.role == "insider":
            try:
                trades_payload = json.loads(injected_data) if injected_data else []
            except json.JSONDecodeError:
                trades_payload = []
            return self._analysis_template.format(
                num_trades=len(trades_payload),
                trades=injected_data or "[]",
            ) + context_suffix
        if self.role == "news":
            return self._analysis_template.format(
                news=injected_data or "No company-news data was loaded."
            ) + context_suffix
        if self.role == "macro":
            return self._analysis_template.format(
                economic_indicators=injected_data or "No macroeconomic data was loaded."
            ) + context_suffix
        if self.role == "policy":
            return self._analysis_template.format(
                fiscal_policy=self._runtime_fiscal_policy,
                monetary_policy=self._runtime_monetary_policy,
            ) + context_suffix
        return context_suffix.strip()
