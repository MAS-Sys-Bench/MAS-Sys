from __future__ import annotations

from typing import Dict, Tuple


ANALYST_OUTPUT_FORMAT = """
You must provide your analysis as a structured output with the following fields:
- signal: One of ["Bullish", "Bearish", "Neutral"]
- justification: A brief explanation of your analysis

Return ONLY one JSON object. Do not use markdown fences. Do not include any extra prose before or after the JSON.
The JSON schema is:
{{
  "signal": "Bullish" | "Bearish" | "Neutral",
  "justification": "..."
}}

Your response should be well-reasoned and consider all aspects of the analysis.
""".strip()

FUNDAMENTAL_SYSTEM_PROMPT = "You are a financial analyst evaluating a ticker based on fundamental analysis."
TECHNICAL_SYSTEM_PROMPT = "You are a technical analyst evaluating a ticker using multiple technical analysis strategies."
INSIDER_SYSTEM_PROMPT = (
    "You are an insider trading analyst evaluating a ticker based on company insider trades, "
    "including stock buys and sales of insiders like CEOs, CFOs, and Directors."
)
COMPANY_NEWS_SYSTEM_PROMPT = "You are a company news analyst evaluating a ticker based on recent news."
MACRO_SYSTEM_PROMPT = (
    "You are a senior macroeconomic analyst conducting a comprehensive evaluation of current macroeconomic conditions."
)
POLICY_SYSTEM_PROMPT = (
    "You are a policy analyst. Evaluate fiscal and monetary policy developments and classify their short-term economic impact."
)

ROLE_PROMPTS: Dict[str, Tuple[str, str]] = {
    "fundamental": (
        FUNDAMENTAL_SYSTEM_PROMPT,
        """
The following fundamentals have been generated from our analysis:
{fundamentals}

""" + ANALYST_OUTPUT_FORMAT,
    ),
    "technical": (
        TECHNICAL_SYSTEM_PROMPT,
        """
The following signals have been generated from our analysis:
{analysis}

""" + ANALYST_OUTPUT_FORMAT,
    ),
    "insider": (
        INSIDER_SYSTEM_PROMPT,
        """
Here are recent {num_trades} insider trades:
{trades}

""" + ANALYST_OUTPUT_FORMAT,
    ),
    "news": (
        COMPANY_NEWS_SYSTEM_PROMPT,
        """
Here are recent company news items:
{news}

""" + ANALYST_OUTPUT_FORMAT,
    ),
    "macro": (
        MACRO_SYSTEM_PROMPT,
        """
Here are the macroeconomic indicators of past periods:
{economic_indicators}

""" + ANALYST_OUTPUT_FORMAT,
    ),
    "policy": (
        POLICY_SYSTEM_PROMPT,
        """
Here are the fiscal policy updates:
{fiscal_policy}

Here are the monetary policy updates:
{monetary_policy}

""" + ANALYST_OUTPUT_FORMAT,
    ),
}


FUND_ANALYST_SELECTION_GUIDANCE = """
For fund tasks, mirror the original DeepFund planner logic inside PlanAgent:
- select a subset of analyst types based on the ticker(s) and task needs
- candidate analyst types are: technical, fundamental, insider, news, macro, policy
- use analyst capability descriptions when deciding which subset to instantiate
- prefer technical + fundamental as a baseline, and add insider/news/macro/policy only when they materially help
""".strip()


FUND_ANALYST_DESCRIPTIONS = {
    "fund_technical": "Technical analysis specialist using price trend, mean reversion, RSI, volatility, volume, and support/resistance signals.",
    "fund_fundamental": "Fundamental analysis specialist focusing on company financial health and valuation.",
    "fund_insider": "Insider trading specialist analyzing insider activity patterns.",
    "fund_news": "Company news specialist analyzing company news and media coverage.",
    "fund_macro": "Macroeconomic analysis specialist focusing on economic indicators, interest rates, inflation, and market trends.",
    "fund_policy": "Policy analysis specialist focusing on fiscal and monetary policy.",
}
