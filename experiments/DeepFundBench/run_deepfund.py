#!/usr/bin/env python
"""Run DeepFundBench experiments on MAS-Sys.

Runs a multi-day trading simulation from ``start_date`` to ``end_date``
defined in the YAML config, iterating over US business days.  Each day
goes through the full PlanAgent → FlowFactory → AgentFlow pipeline.
Portfolio state and decision memory carry forward in-memory across days.

Results are saved to ``{results_dir}/{exp_name}_{YYYYMMDD_HHMMSS}/``.
Evaluation runs once after all days complete.

Usage:

    # 1. Run experiment (iterates start_date → end_date from YAML)
    python -m experiments.DeepFundBench.run_deepfund \
        --config experiments/DeepFundBench/configs/dev.yaml

    # 2. Custom data / results directory
    python -m experiments.DeepFundBench.run_deepfund \
        --config experiments/DeepFundBench/configs/buffett.yaml \
        --data-dir path/to/data --results-dir path/to/results

    # 3. Skip auto-evaluation
    python -m experiments.DeepFundBench.run_deepfund \
        --config experiments/DeepFundBench/configs/dev.yaml --no-eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from app.agent.fund.analyst import _load_current_price
from experiments.DeepFundBench.task_prompt import (
    TASK_PROMPT_TEMPLATE,
)

_THIS_DIR = Path(__file__).resolve().parent
_MAS_SYS_ROOT = _THIS_DIR.parent.parent
if str(_MAS_SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAS_SYS_ROOT))


# YAML config loader


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Trading day generation


def _generate_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate US business days between *start_date* and *end_date* (inclusive)."""
    days = pd.bdate_range(start=start_date, end=end_date)
    return [d.strftime("%Y-%m-%d") for d in days]


def build_task(
    cfg: dict[str, object],
    ticker: str,
    trading_date: str,
    portfolio: dict[str, object],
    decision_memory: list[dict[str, object]],
) -> str:
    cashflow = float(portfolio.get("cashflow", cfg.get("cashflow", 100_000.0)))
    recent_memory = [d for d in decision_memory if d.get("ticker") == ticker][-5:]
    memory_text = json.dumps(recent_memory, ensure_ascii=False, indent=2) if recent_memory else "[]"
    portfolio_text = json.dumps(portfolio, ensure_ascii=False, indent=2)
    return TASK_PROMPT_TEMPLATE.format(
        ticker=ticker,
        trading_date=trading_date,
        cashflow=cashflow,
        portfolio_text=portfolio_text,
        memory_text=memory_text,
    )


# Results persistence


def _save_results(out_dir: Path, results: dict) -> Path:
    """Save structured results for one trading day."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trading_date = results.get("trading_date", "unknown")
    dest = out_dir / f"{trading_date}.json"
    dest.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return dest


def _extract_signal_from_output(raw: str) -> dict | None:
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and "signal" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        data = json.loads(raw)
        if "signal" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_decision_from_output(raw: str) -> dict | None:
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and "action" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        data = json.loads(raw)
        if "action" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _find_manager_id(mas_config) -> str:
    sources = {r.source for r in mas_config.routes}
    targets = {r.target for r in mas_config.routes}
    manager_only = targets - sources
    if len(manager_only) == 1:
        return manager_only.pop()
    for node in mas_config.agents:
        if "manager" in node.agent_key:
            return node.instance_id
    return mas_config.agents[-1].instance_id


def _apply_decision(portfolio: dict, ticker: str, decision: dict) -> None:
    action = decision.get("action", "Hold")
    shares = int(decision.get("shares", 0))
    price = float(decision.get("price", 0.0))
    if ticker not in portfolio["positions"]:
        portfolio["positions"][ticker] = {"shares": 0, "value": 0.0}
    position = portfolio["positions"][ticker]
    if action == "Buy" and shares > 0:
        cost = shares * price
        if cost <= portfolio["cashflow"]:
            position["shares"] += shares
            position["value"] = position["shares"] * price
            portfolio["cashflow"] -= cost
    elif action == "Sell" and shares > 0:
        sell_shares = min(shares, position["shares"])
        proceeds = sell_shares * price
        position["shares"] -= sell_shares
        position["value"] = position["shares"] * price
        portfolio["cashflow"] += proceeds


def _mark_to_market_portfolio(portfolio: dict, trading_date: str) -> None:
    positions = portfolio.get("positions", {}) or {}
    for ticker, position in positions.items():
        shares = int((position or {}).get("shares", 0) or 0)
        price = _load_current_price(str(ticker), trading_date)
        position["value"] = shares * price if price > 0 else 0.0


# Experiment runner


async def run_experiment(
    cfg: dict,
    data_dir: str,
    results_dir: str,
    timeout: int,
    runtime: str = "legacy",
) -> dict:
    """Execute a multi-day DeepFundBench experiment.

    For each trading day in [start_date, end_date]:
      PlanAgent → FlowFactory → AgentFlow → save results.
    Portfolio and decision memory carry forward in-memory.
    """
    from app.flow.flow_factory import FlowFactory
    from app.logger import logger
    from app.mas.plan_agent import PlanAgent
    from app.tracer import tracer

    exp_name = cfg.get("exp_name", "unnamed")
    start_date = cfg["start_date"]
    end_date = cfg["end_date"]
    initial_cashflow = cfg.get("cashflow", 100_000.0)

    trading_days = _generate_trading_days(start_date, end_date)
    if not trading_days:
        return {
            "exp_name": exp_name,
            "error": f"No business days between {start_date} and {end_date}",
            "elapsed_seconds": 0.0,
            "days_completed": 0,
            "days_total": 0,
        }

    # Output directory: {results_dir}/{exp_name}_{YYYYMMDD_HHMMSS}
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_dir) / f"{exp_name}_{run_tag}"

    total_start = time.time()
    portfolio: dict = {"cashflow": initial_cashflow, "positions": {}}
    decision_memory: list[dict] = []
    days_completed = 0
    day_errors: list[dict] = []

    print(
        f"\n  Trading days: {len(trading_days)} "
        f"({trading_days[0]} → {trading_days[-1]})\n"
    )

    for i, trading_date in enumerate(trading_days, 1):
        day_start = time.time()
        print(f"  [{i}/{len(trading_days)}] {trading_date} ", end="", flush=True)

        try:
            tracer.reset()
            tracer.start_e2e()

            day_signals: list[dict] = []
            day_decisions: list[dict] = []

            for ticker in cfg["tickers"]:
                task_prompt = build_task(
                    cfg=cfg,
                    ticker=ticker,
                    trading_date=trading_date,
                    portfolio=portfolio,
                    decision_memory=decision_memory,
                )
                plan_agent = PlanAgent(runtime=runtime)
                mas_config = await plan_agent.plan(task_prompt, runtime=runtime)

                flow = FlowFactory.from_config(
                    mas_config,
                    plan_agent=plan_agent,
                    runtime=runtime,
                )
                manager_output = await asyncio.wait_for(flow.execute(task_prompt), timeout=timeout)
                manager_id = _find_manager_id(mas_config)

                for node in mas_config.agents:
                    agent = flow.agents.get(node.instance_id)
                    if not agent or not agent.memory.messages:
                        continue
                    agent_output = agent.memory.messages[-1].content or ""
                    if node.instance_id == manager_id:
                        continue
                    parsed_signal = _extract_signal_from_output(agent_output)
                    if parsed_signal:
                        parsed_signal["ticker"] = ticker
                        parsed_signal["analyst"] = node.instance_id
                        day_signals.append(parsed_signal)

                decision = _extract_decision_from_output(manager_output)
                if decision:
                    decision["ticker"] = ticker
                    decision["trading_date"] = trading_date
                    day_decisions.append(decision)
                    decision_memory.append(decision)
                    _apply_decision(portfolio, ticker, decision)

            _mark_to_market_portfolio(portfolio, trading_date)
            total_value = portfolio["cashflow"] + sum(
                position.get("value", 0) for position in portfolio["positions"].values()
            )
            day_results = {
                "trading_date": trading_date,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "tickers": cfg["tickers"],
                    "initial_cashflow": initial_cashflow,
                },
                "portfolio": {
                    "cashflow": portfolio["cashflow"],
                    "total_value": total_value,
                    "positions": portfolio["positions"],
                },
                "decisions": day_decisions,
                "signals": day_signals,
            }
            dest = _save_results(out_dir, day_results)
            elapsed = time.time() - day_start
            print(f"✓ {elapsed:.0f}s → {dest.name}")
            days_completed += 1
            tracer.end_e2e()
            tracer.save(out_dir, prefix=f"{trading_date}_")

        except asyncio.TimeoutError:
            elapsed = time.time() - day_start
            print(f"✗ timeout {elapsed:.0f}s")
            day_errors.append({"date": trading_date, "error": "timeout"})
            logger.error(f"[DeepFundBench] {trading_date} timed out")
            tracer.end_e2e()
            tracer.save(out_dir, prefix=f"{trading_date}_")

        except Exception as e:
            elapsed = time.time() - day_start
            print(f"✗ {e} {elapsed:.0f}s")
            day_errors.append({"date": trading_date, "error": str(e)})
            logger.error(f"[DeepFundBench] {trading_date} failed: {e}")
            tracer.end_e2e()
            tracer.save(out_dir, prefix=f"{trading_date}_")

    total_elapsed = round(time.time() - total_start, 2)
    return {
        "exp_name": exp_name,
        "out_dir": str(out_dir),
        "error": None if not day_errors else f"{len(day_errors)} day(s) failed",
        "day_errors": day_errors,
        "elapsed_seconds": total_elapsed,
        "days_completed": days_completed,
        "days_total": len(trading_days),
    }


# ── Auto-evaluation ──


def _auto_evaluate(out_dir: str, data_dir: str) -> None:
    """Run evaluation on the completed run output directory."""
    out_path = Path(out_dir)
    json_files = sorted(out_path.glob("*.json")) if out_path.is_dir() else []
    json_files = [f for f in json_files if f.name != "evaluation_report.json"]
    if len(json_files) < 2:
        print(
            f"\n  ⚠ Only {len(json_files)} result file(s) — "
            f"need ≥ 2 for evaluation. Skipping."
        )
        return
    print(f"\n═══ Auto-Evaluating ({len(json_files)} trading days) ═══\n")
    from experiments.DeepFundBench.evaluate import evaluate_results

    evaluate_results(out_dir, data_dir=data_dir)


# ── CLI ──


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a DeepFundBench experiment on MAS-Sys."
    )
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config"
    )
    _default_data = str(_THIS_DIR / "data")
    parser.add_argument(
        "--data-dir",
        default=_default_data,
        help=f"Financial data directory (default: {_default_data})",
    )
    _default_results = str(_THIS_DIR / "results")
    parser.add_argument(
        "--results-dir",
        default=_default_results,
        help=f"Results output directory (default: {_default_results})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-day execution timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after the run",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["legacy", "orla", "orla_dry_run"],
        default="legacy",
        help="Execution runtime for benchmark generation; orla_dry_run only validates ORLA lowering",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_yaml(args.config)
    exp_name = cfg.get("exp_name", "unnamed")
    start_date = cfg.get("start_date", "N/A")
    end_date = cfg.get("end_date", "N/A")

    trading_days = _generate_trading_days(start_date, end_date) if start_date != "N/A" and end_date != "N/A" else []

    print(f"═══ DeepFundBench: {exp_name} ═══")
    print(f"  Config      : {args.config}")
    print(f"  Tickers     : {cfg.get('tickers', [])}")
    print(f"  Date Range  : {start_date} → {end_date} ({len(trading_days)} business days)")
    print(f"  Cashflow    : ${cfg.get('cashflow', 100000):,.0f}")
    print(f"  Data Dir    : {args.data_dir}")
    print(f"  Results Dir : {args.results_dir}")
    print(f"  Timeout     : {args.timeout}s per day")
    print(f"  Runtime     : {args.runtime}")

    result = asyncio.run(
        run_experiment(
            cfg,
            args.data_dir,
            args.results_dir,
            args.timeout,
            runtime=args.runtime,
        )
    )

    print(f"\n═══ {result['days_completed']}/{result['days_total']} days completed "
          f"in {result['elapsed_seconds']:.1f}s ═══")

    if result.get("day_errors"):
        for de in result["day_errors"]:
            print(f"  ✗ {de['date']}: {de['error']}")

    out_dir = result.get("out_dir", "")
    if out_dir:
        print(f"  Output: {out_dir}")

    if not args.no_eval and result["days_completed"] >= 2 and out_dir:
        _auto_evaluate(out_dir, args.data_dir)


if __name__ == "__main__":
    main()
