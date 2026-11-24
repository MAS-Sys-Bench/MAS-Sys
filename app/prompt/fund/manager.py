from __future__ import annotations


RISK_CONTROL_SYSTEM_PROMPT = (
    "You are a professional risk control analyst. Evaluate the risk of the ticker and set the optimal position ratio "
    "based on analyst signals and portfolio state."
)
MANAGER_SYSTEM_PROMPT = (
    "You are a portfolio manager making final trading decisions based on decision memory and the provided optimal position ratio."
)

RISK_CONTROL_PROMPT = """
Here are the analyst signals:
{ticker_signals}

Here is the portfolio state:
{portfolio}

The position ratio range: [0, {max_position_ratio}], and the minimum step is 0.05.
If you observe more bullish signals, you can set a larger position ratio.
If you observe more bearish signals, you can set a smaller position ratio.

You must provide your control recommendation as a structured output with the following fields:
- optimal_position_ratio: The optimal ratio of the position value to the total portfolio value
- justification: A brief explanation of your recommendation

Return ONLY one JSON object. Do not use markdown fences. Do not include any extra prose before or after the JSON.
The JSON schema is:
{{
  "optimal_position_ratio": 0.0,
  "justification": "..."
}}

Your response should be well-reasoned and consider all aspects of the analysis.
""".strip()

MANAGER_DECISION_PROMPT = """
Here is the decision memory:
{decision_memory}

Current Price: {current_price}
Holding Shares: {current_shares}
Tradable Shares: {tradable_shares}
Optimal Position Ratio: {optimal_position_ratio}
Risk Control Justification: {risk_justification}

If the value of tradable shares is positive, you can buy more shares.
If the value of tradable shares is negative, you can sell some shares.
If the value of tradable shares is close to 0, you can hold.

You must provide your decision as a structured output with the following fields:
- action: One of ["Buy", "Sell", "Hold"]
- shares: Number of shares to buy or sell, set 0 for hold
- price: The current price of the ticker
- justification: A brief explanation of your decision

Return ONLY one JSON object. Do not use markdown fences. Do not include any extra prose before or after the JSON.
The JSON schema is:
{{
  "action": "Buy" | "Sell" | "Hold",
  "shares": 0,
  "price": 0.0,
  "justification": "..."
}}

Your response should be well-reasoned and consider all aspects of the analysis.
""".strip()
