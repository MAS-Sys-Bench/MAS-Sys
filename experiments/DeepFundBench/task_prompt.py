TASK_PROMPT_TEMPLATE = """\
Run a DeepFund-style stock trading analysis for ticker {ticker}.
Trading date: {trading_date}
Current portfolio cash: ${cashflow:,.2f}

Current portfolio snapshot:
{portfolio_text}

Recent decision memory for {ticker}:
{memory_text}
"""
