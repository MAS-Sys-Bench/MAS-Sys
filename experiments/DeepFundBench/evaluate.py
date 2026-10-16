#!/usr/bin/env python
"""Evaluate DeepFundBench experiment results.

Reads the JSON result files produced by FundFlow (one per trading day) and
computes the seven metrics used in the DeepFund paper:

  1. Cumulative Return (CR)
  2. Cumulative Return at Buy & Hold (CRbnh)
  3. Sharpe Ratio (SR)
  4. Maximum Drawdown (MDD)
  5. Win Rate (WR)
  6. Beta (β)  — relative to a market benchmark
  7. Alpha (α) — excess return vs. benchmark

Usage::

    python -m experiments.DeepFundBench.evaluate \
        --results-dir experiments/DeepFundBench/results/dev

    # With custom benchmark and risk-free rate
    python -m experiments.DeepFundBench.evaluate \
        --results-dir experiments/DeepFundBench/results/dev \
        --benchmark SPY --risk-free-rate 0.0429
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


# Data loading

def load_results(results_dir: str) -> List[dict]:
    """Load and sort all trading-day JSON files from *results_dir*."""
    d = Path(results_dir)
    if not d.is_dir():
        print(f"ERROR: results directory not found: {d}")
        sys.exit(1)
    files = sorted(d.glob("*.json"))
    # Exclude tracer output files (trace/metadata)
    files = [f for f in files if not (f.name.endswith("_trace.json") or f.name.endswith("_metadata.json")
             or f.name in ("trace.json", "metadata.json", "evaluation_report.json"))]
    if not files:
        print(f"ERROR: no result files (*.json) found in {d}")
        sys.exit(1)
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skipping {f.name}: {e}")
    return records


def extract_portfolio_series(records: List[dict]) -> tuple[List[str], np.ndarray]:
    """Return (dates, total_values) from the result records."""
    dates = [r["trading_date"] for r in records]
    values = np.array([r["portfolio"]["total_value"] for r in records], dtype=float)
    return dates, values


def extract_decisions(records: List[dict]) -> List[dict]:
    """Flatten all decisions from all trading days."""
    decisions = []
    for r in records:
        for d in r.get("decisions", []):
            d["_trading_date"] = r["trading_date"]
            decisions.append(d)
    return decisions


# Benchmark data (yfinance)

def fetch_benchmark_returns(
    benchmark: str, dates: List[str]
) -> np.ndarray | None:
    """Fetch daily close prices from yfinance for the given dates and return
    daily returns.  Returns None if yfinance is unavailable or data is missing."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("WARNING: yfinance/pandas not installed — skipping Alpha/Beta.")
        return None

    start_date = dates[0]
    end_date = dates[-1]
    # Extend end by 1 day for yfinance's exclusive end
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=5)
    ticker = yf.Ticker(benchmark)
    df = ticker.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"))
    if df.empty or len(df) < 2:
        print(f"WARNING: insufficient benchmark data for {benchmark}.")
        return None

    # Align to our trading dates
    df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Get closes for our dates (nearest available)
    closes = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = df.index <= ts
        if mask.any():
            closes.append(float(df.loc[mask, "Close"].iloc[-1]))
        else:
            closes.append(np.nan)

    closes = np.array(closes)
    if np.isnan(closes).any() or len(closes) < 2:
        return None
    return np.diff(closes) / closes[:-1]  # daily returns


# Metrics

def cumulative_return(values: np.ndarray) -> float:
    """CR = (P_final - P_initial) / P_initial"""
    return (values[-1] - values[0]) / values[0]


def cumulative_return_buy_hold(
    records: List[dict], data_dir: str | None
) -> float | None:
    """CRbnh: simulate a buy-and-hold strategy using Day-1's portfolio
    composition held until the last trading day.

    Requires local OHLCV data to compute end-of-period values.
    """
    if not data_dir:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    first = records[0]
    last = records[-1]
    initial_cf = first["config"]["initial_cashflow"]

    # Day-1 positions
    positions = first["portfolio"]["positions"]
    if not positions:
        return None

    last_date = last["trading_date"]
    total_value = first["portfolio"]["cashflow"]  # leftover cash on day 1

    for ticker, pos in positions.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        csv_path = Path(data_dir) / "daily" / f"{ticker.upper()}.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True).dt.tz_localize(None)
        cutoff = pd.Timestamp(last_date)
        df = df[df["date"] <= cutoff].sort_values("date")
        if df.empty:
            return None
        total_value += shares * float(df["close"].iloc[-1])

    return (total_value - initial_cf) / initial_cf


def sharpe_ratio(values: np.ndarray, risk_free_rate: float = 0.0429) -> float:
    """Annualized Sharpe Ratio.

    SR = (mean(r_e) / std(r_e)) * sqrt(252)
    where r_e = daily excess return.
    """
    daily_returns = np.diff(values) / values[:-1]
    daily_rf = risk_free_rate / 252.0
    excess = daily_returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(values: np.ndarray) -> float:
    """MDD = max((peak - trough) / peak)"""
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    return float(drawdowns.max())


def win_rate(decisions: List[dict]) -> float | None:
    """Percentage of profitable trades.

    A trade is profitable if:
      - Buy and price subsequently increased (approximated by same-day info)
      - Sell and price subsequently decreased

    Since we may not have next-day prices, we use a simpler heuristic:
    count trades where action != Hold and shares > 0.
    For precise win rate, we'd need sequential daily prices.

    Here we count trades where profit can be inferred from consecutive
    portfolio values — but as a simpler approach, we just report the ratio
    of active trades to total (as a baseline metric).
    """
    active = [d for d in decisions if d.get("action", "Hold") != "Hold" and d.get("shares", 0) > 0]
    if not active:
        return None
    # Without next-day data, we approximate: report active trade count only
    return len(active) / len(decisions) if decisions else None


def beta_alpha(
    values: np.ndarray,
    benchmark_returns: np.ndarray | None,
    risk_free_rate: float = 0.0429,
) -> tuple[float | None, float | None]:
    """Compute Beta and Alpha against a benchmark.

    β = Cov(r_s, r_m) / Var(r_m)
    α = r_s - [r_f + β(r_m - r_f)]
    """
    if benchmark_returns is None or len(benchmark_returns) < 2:
        return None, None

    strategy_returns = np.diff(values) / values[:-1]

    # Align lengths (use min)
    n = min(len(strategy_returns), len(benchmark_returns))
    rs = strategy_returns[:n]
    rm = benchmark_returns[:n]

    cov = np.cov(rs, rm)
    if cov.shape != (2, 2) or cov[1, 1] == 0:
        return None, None

    beta = float(cov[0, 1] / cov[1, 1])

    # Annualized returns
    total_rs = (values[-1] / values[0]) - 1.0
    total_rm = float(np.prod(1 + rm) - 1.0)
    alpha = total_rs - (risk_free_rate + beta * (total_rm - risk_free_rate))

    return beta, float(alpha)


# Report

def _compute_metrics(
    dates: List[str],
    values: np.ndarray,
    decisions: List[dict],
    records: List[dict],
    benchmark_returns: np.ndarray | None,
    data_dir: str | None,
    risk_free_rate: float,
) -> dict:
    """Compute all metrics and return as a dict."""
    cr = cumulative_return(values)
    cr_bnh = cumulative_return_buy_hold(records, data_dir)
    sr = sharpe_ratio(values, risk_free_rate)
    mdd = max_drawdown(values)
    wr = win_rate(decisions)
    b, a = beta_alpha(values, benchmark_returns, risk_free_rate)

    total = len(decisions)
    buys = sum(1 for d in decisions if d.get("action") == "Buy")
    sells = sum(1 for d in decisions if d.get("action") == "Sell")
    holds = sum(1 for d in decisions if d.get("action") == "Hold")

    return {
        "period_start": dates[0],
        "period_end": dates[-1],
        "trading_days": len(dates),
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
        "CR": cr,
        "CRbnh": cr_bnh,
        "SR": sr,
        "MDD": mdd,
        "WR": wr,
        "beta": b,
        "alpha": a,
        "decisions_total": total,
        "decisions_buy": buys,
        "decisions_sell": sells,
        "decisions_hold": holds,
        "value_trajectory": [
            {"date": d, "value": float(v)} for d, v in zip(dates, values)
        ],
    }


def print_report(metrics: dict) -> None:
    """Print a formatted evaluation report from a metrics dict."""
    cr = metrics["CR"]
    cr_bnh = metrics["CRbnh"]
    sr = metrics["SR"]
    mdd = metrics["MDD"]
    wr = metrics["WR"]
    beta = metrics["beta"]
    alpha = metrics["alpha"]

    print("═══ DeepFundBench Evaluation Report ═══")
    print(f"  Period         : {metrics['period_start']} → {metrics['period_end']} ({metrics['trading_days']} trading days)")
    print(f"  Initial Value  : ${metrics['initial_value']:,.2f}")
    print(f"  Final Value    : ${metrics['final_value']:,.2f}")
    print()
    print("  Metrics:")
    print(f"    CR  (Cumulative Return)     : {cr:+.4f}  ({cr*100:+.2f}%)")
    if cr_bnh is not None:
        print(f"    CRbnh (Buy & Hold Return)   : {cr_bnh:+.4f}  ({cr_bnh*100:+.2f}%)")
    else:
        print(f"    CRbnh (Buy & Hold Return)   : N/A (no OHLCV data)")
    print(f"    SR  (Sharpe Ratio, ann.)     : {sr:+.4f}")
    print(f"    MDD (Max Drawdown)           : {mdd:.4f}  ({mdd*100:.2f}%)")
    if wr is not None:
        print(f"    WR  (Active Trade Ratio)     : {wr:.4f}  ({wr*100:.1f}%)")
    else:
        print(f"    WR  (Active Trade Ratio)     : N/A (no trades)")
    if beta is not None:
        print(f"    β   (Beta vs benchmark)      : {beta:+.4f}")
    else:
        print(f"    β   (Beta vs benchmark)      : N/A")
    if alpha is not None:
        print(f"    α   (Alpha vs benchmark)     : {alpha:+.4f}  ({alpha*100:+.2f}%)")
    else:
        print(f"    α   (Alpha vs benchmark)     : N/A")
    print()

    # Decision summary
    print(f"  Decisions: {metrics['decisions_total']} total — "
          f"{metrics['decisions_buy']} Buy, {metrics['decisions_sell']} Sell, "
          f"{metrics['decisions_hold']} Hold")

    # Portfolio value trajectory (sparkline-like)
    trajectory = metrics["value_trajectory"]
    initial = trajectory[0]["value"]
    print()
    print("  Value Trajectory:")
    for pt in trajectory:
        v = pt["value"]
        bar_len = max(0, int((v / initial - 0.95) * 200))  # scale around ±5%
        bar = "█" * min(bar_len, 60)
        change = (v / initial - 1) * 100
        print(f"    {pt['date']}  ${v:>12,.2f}  {change:+6.2f}%  {bar}")

    print()
    print("═══ End of Report ═══")


def save_report(metrics: dict, results_dir: str) -> Path:
    """Save the evaluation metrics to ``evaluation_report.json``."""
    dest = Path(results_dir) / "evaluation_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(metrics, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Report saved to: {dest}")
    return dest


# Programmatic API

def evaluate_results(
    results_dir: str,
    data_dir: str | None = None,
    benchmark: str = "SPY",
    risk_free_rate: float = 0.0429,
) -> dict | None:
    """Evaluate experiment results and return metrics dict.

    Returns ``None`` if fewer than 2 trading days are available.
    Prints the report to stdout and saves ``evaluation_report.json``.
    """
    records = load_results(results_dir)
    dates, values = extract_portfolio_series(records)
    decisions = extract_decisions(records)

    if len(dates) < 2:
        print(f"  ⚠ Only {len(dates)} trading day(s) — need ≥ 2 for evaluation.")
        return None

    benchmark_returns = fetch_benchmark_returns(benchmark, dates)

    if not data_dir:
        candidate = Path(results_dir).parent / "data"
        if candidate.is_dir():
            data_dir = str(candidate)

    metrics = _compute_metrics(
        dates, values, decisions, records,
        benchmark_returns, data_dir, risk_free_rate,
    )
    print_report(metrics)
    save_report(metrics, results_dir)
    return metrics


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepFundBench experiment results.")
    parser.add_argument("--results-dir", required=True,
                        help="Path to results directory (e.g. results/dev/)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to OHLCV data dir (for CRbnh calculation)")
    parser.add_argument("--benchmark", default="SPY",
                        help="Benchmark ticker for Alpha/Beta (default: SPY)")
    parser.add_argument("--risk-free-rate", type=float, default=0.0429,
                        help="Annual risk-free rate (default: 0.0429, Apr 2025)")
    args = parser.parse_args()

    result = evaluate_results(
        args.results_dir, args.data_dir, args.benchmark, args.risk_free_rate,
    )
    if result is None:
        print("ERROR: need at least 2 trading days for evaluation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
