"""Fetch real financial data for DeepFundBench experiments.

Data sources are auto-detected based on API keys in ``config/config.toml [fund]`` and applied in priority order:

  1. **yfinance** (always, no API key): OHLCV, fundamentals, insider trades
  2. **FRED** (if ``fred_api_key`` configured): macroeconomic indicators
  3. **Alpha Vantage** (if ``alpha_vantage_api_key`` configured): OHLCV, fundamentals, insider trades, news, policy news, macro indicators

All data is **merged** with existing files — nothing is deleted or overwritten.
CSV rows are merged by date (later source wins for the same date).
JSON dicts are shallow-merged (later keys overwrite). JSON lists are appended with deduplication.

Each record includes a ``_source`` tag indicating which API produced it (for human reference only; agent tools ignore this field).

A ``meta.json`` in the output directory tracks tickers and date ranges.
Subsequent runs take the *union* of tickers and date ranges.

Usage:

    python -m experiments.DeepFundBench.fetch_data
    python -m experiments.DeepFundBench.fetch_data --tickers TSLA NVDA --start 2024-01-01 --end 2025-01-01
    python -m experiments.DeepFundBench.fetch_data -o custom/data/path
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Defaults

_DEFAULT_TICKERS = [
    "AAPL", "AXP", "BAC", "KO", "CVX", "TSLA", "NVDA", "GLD", "NEM", "GOLD",
]
_DEFAULT_START = "2021-04-01"
_DEFAULT_END = "2026-04-01"

# Config helpers 


def _load_fund_config() -> dict:
    """Read [fund] section from config/config.toml if it exists."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "config" / "config.toml",
    ]
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                import tomllib
            except ModuleNotFoundError:
                try:
                    import tomli as tomllib  # Python < 3.11
                except ImportError:
                    return {}
            with open(cfg_path, "rb") as f:
                data = tomllib.load(f)
            return data.get("fund", {})
    return {}


def _get_api_key(config_key: str) -> str:
    return _load_fund_config().get(config_key, "")


# ── Merge helpers ──


def _merge_json_dict(dest: Path, new_data: dict) -> dict:
    """Shallow-merge *new_data* into existing JSON dict at *dest*."""
    existing: dict = {}
    if dest.exists():
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(new_data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(existing, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return existing


def _merge_json_list(
    dest: Path, new_items: list, dedup_key: str | None = None
) -> list:
    """Append *new_items* to existing JSON list at *dest*, with dedup."""
    existing: list = []
    if dest.exists():
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if dedup_key:
        seen = {item.get(dedup_key) for item in existing if item.get(dedup_key)}
        for item in new_items:
            if item.get(dedup_key) not in seen:
                existing.append(item)
                seen.add(item.get(dedup_key))
    else:
        existing_strs = {json.dumps(r, sort_keys=True) for r in existing}
        for item in new_items:
            s = json.dumps(item, sort_keys=True)
            if s not in existing_strs:
                existing.append(item)
                existing_strs.add(s)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return existing


def _merge_csv(dest: Path, new_df) -> None:
    """Merge *new_df* into existing CSV at *dest* (by date index, newer wins)."""
    import pandas as pd

    # Normalise index to string for consistent merging
    new_df.index = new_df.index.astype(str)
    if dest.exists():
        old = pd.read_csv(dest, index_col="date", dtype={"date": str})
        old.index = old.index.astype(str)
        new_df = pd.concat([old, new_df])
        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()
    dest.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(dest)


# ── Retry helper ──


def _call_with_retry(fn, *args, max_retries: int = 6, sleep_sec: int = 10, **kwargs):
    """Call *fn* with retry on rate-limit / transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            retryable = any(
                kw in err
                for kw in (
                    "rate limit", "too many", "429", "throttl",
                    "timeout", "connection", "temporary", "timed out",
                )
            )
            if retryable and attempt < max_retries:
                print(f"    ⏳ Error (attempt {attempt}/{max_retries}): {e}")
                print(f"       Sleeping {sleep_sec}s...")
                _time.sleep(sleep_sec)
                continue
            if attempt >= max_retries:
                print(f"    WARNING: Failed after {max_retries} attempts: {e}")
                return None
            raise
    return None


# ── Lazy imports ──


def _import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        print("ERROR: yfinance is required. Install via: pip install yfinance")
        sys.exit(1)


def _import_requests():
    import requests
    return requests


def _import_fredapi(api_key: str):
    try:
        from fredapi import Fred
        return Fred(api_key=api_key)
    except ImportError:
        print("WARNING: fredapi not installed. Install via: pip install fredapi")
        return None


# ── yfinance fetcher ──


class YFinanceFetcher:
    """Download stock data using yfinance (free, no API key)."""

    SOURCE = "yfinance"

    def __init__(self):
        self.yf = _import_yfinance()

    def fetch_daily(self, ticker: str, start_date: str, end_date: str, out_dir: Path):
        """Download daily OHLCV for [start_date, end_date]."""
        t = self.yf.Ticker(ticker)
        # yfinance end is exclusive, so +1 day
        end_excl = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        df = _call_with_retry(t.history, start=start_date, end=end_excl)
        if df is None or df.empty:
            print(f"  WARNING: no daily data for {ticker}")
            return
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df["_source"] = self.SOURCE
        dest = out_dir / "daily" / f"{ticker}.csv"
        _merge_csv(dest, df)
        print(f"  ✓ daily/{ticker}.csv  ({len(df)} rows, {self.SOURCE})")

    def fetch_fundamentals(self, ticker: str, out_dir: Path):
        t = self.yf.Ticker(ticker)
        info = _call_with_retry(lambda: t.info)
        if not info:
            print(f"  WARNING: no fundamentals for {ticker}")
            return
        info["_source"] = self.SOURCE
        info["_fetched_as_of"] = datetime.now().strftime("%Y-%m-%d")
        dest = out_dir / "fundamentals" / f"{ticker}.json"
        merged = _merge_json_dict(dest, info)
        print(f"  ✓ fundamentals/{ticker}.json  ({len(merged)} keys, {self.SOURCE})")

    def fetch_insider(self, ticker: str, start_date: str, end_date: str, out_dir: Path):
        t = self.yf.Ticker(ticker)
        df = _call_with_retry(lambda: t.insider_transactions)
        if df is None or df.empty:
            print(f"  WARNING: no insider data for {ticker}")
            return
        import pandas as pd
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if "Start Date" in df.columns:
            df["Start Date"] = pd.to_datetime(
                df["Start Date"], errors="coerce", utc=True
            )
            ts = df["Start Date"].dt.tz_localize(None)
            df = df[(ts >= start_dt) & (ts < end_dt)]
        if df.empty:
            print(f"  WARNING: no insider data for {ticker} in [{start_date}, {end_date})")
            return
        records = json.loads(df.to_json(orient="records", date_format="iso"))
        for r in records:
            r["_source"] = self.SOURCE
        dest = out_dir / "insider" / f"{ticker}.json"
        merged = _merge_json_list(dest, records)
        print(f"  ✓ insider/{ticker}.json  ({len(merged)} trades, {self.SOURCE})")


# ── Alpha Vantage fetcher ──


class AlphaVantageFetcher:
    """Download stock data using Alpha Vantage (requires API key)."""

    BASE = "https://www.alphavantage.co/query"
    SOURCE = "alpha_vantage"

    def __init__(self, api_key: str):
        if not api_key:
            print("ERROR: alpha_vantage_api_key not set.")
            sys.exit(1)
        self.api_key = api_key
        self.requests = _import_requests()
        self._call_count = 0

    def _get(self, params: dict) -> dict:
        """Make an AV API call with rate-limit retry (up to 6 attempts)."""
        params["apikey"] = self.api_key
        for attempt in range(1, 7):
            self._call_count += 1
            try:
                resp = self.requests.get(self.BASE, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                if attempt < 6:
                    print(f"    ⏳ HTTP error (attempt {attempt}/6): {e}, sleeping 10s...")
                    _time.sleep(10)
                    continue
                print(f"    WARNING: HTTP failed after 6 attempts: {e}")
                return {}
            data = resp.json()
            note = data.get("Note", "") or data.get("Information", "")
            if note and any(
                kw in note.lower()
                for kw in ("limit", "premium", "rate", "frequency")
            ):
                if attempt < 6:
                    print(f"    ⏳ Rate limited (attempt {attempt}/6), sleeping 10s...")
                    _time.sleep(10)
                    continue
                print(f"    WARNING: Rate limit persists after 6 attempts, skipping")
                return {}
            if "Error Message" in data:
                print(f"  WARNING: Alpha Vantage: {data['Error Message'][:120]}")
            return data
        return {}

    def fetch_daily(self, ticker: str, start_date: str, end_date: str, out_dir: Path):
        data = self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
        })
        series = data.get("Time Series (Daily)", {})
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        rows = []
        for date_str, vals in sorted(series.items()):
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt < start_dt or dt > end_dt:
                continue
            rows.append({
                "date": date_str,
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": int(vals["5. volume"]),
                "_source": self.SOURCE,
            })
        if not rows:
            print(f"  WARNING: no daily data for {ticker}")
            return
        import pandas as pd
        df = pd.DataFrame(rows).set_index("date").sort_index()
        dest = out_dir / "daily" / f"{ticker}.csv"
        _merge_csv(dest, df)
        print(f"  ✓ daily/{ticker}.csv  ({len(df)} rows, {self.SOURCE})")

    def fetch_fundamentals(self, ticker: str, out_dir: Path):
        data = self._get({"function": "OVERVIEW", "symbol": ticker})
        if not data or "Symbol" not in data:
            print(f"  WARNING: no fundamentals for {ticker}")
            return
        data["_source"] = self.SOURCE
        dest = out_dir / "fundamentals" / f"{ticker}.json"
        merged = _merge_json_dict(dest, data)
        print(f"  ✓ fundamentals/{ticker}.json  ({len(merged)} keys, {self.SOURCE})")

    def fetch_insider(self, ticker: str, start_date: str, end_date: str, out_dir: Path):
        data = self._get({"function": "INSIDER_TRANSACTIONS", "symbol": ticker})
        trades = data.get("data", [])
        if trades:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            filtered = []
            for t in trades:
                td = t.get("transaction_date") or ""
                if not td:
                    continue
                try:
                    dt = datetime.strptime(td, "%Y-%m-%d")
                except ValueError:
                    continue
                if start_dt <= dt < end_dt:
                    filtered.append(t)
            trades = filtered
        for t in trades:
            t["_source"] = self.SOURCE
        if not trades:
            print(f"  WARNING: no insider data for {ticker} in [{start_date}, {end_date})")
            return
        dest = out_dir / "insider" / f"{ticker}.json"
        merged = _merge_json_list(dest, trades)
        print(f"  ✓ insider/{ticker}.json  ({len(merged)} trades, {self.SOURCE})")

    def fetch_news(self, ticker: str, start_date: str, end_date: str, out_dir: Path):
        params: Dict[str, str] = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": datetime.strptime(start_date, "%Y-%m-%d").strftime(
                "%Y%m%dT0000"
            ),
            "time_to": datetime.strptime(end_date, "%Y-%m-%d").strftime(
                "%Y%m%dT2359"
            ),
        }
        data = self._get(params)
        feed = data.get("feed", [])
        articles = []
        for item in feed[:50]:
            articles.append({
                "title": item.get("title", ""),
                "publish_time": item.get("time_published", ""),
                "publisher": item.get("source", ""),
                "summary": item.get("summary", ""),
                "_source": self.SOURCE,
            })
        if not articles:
            print(f"  WARNING: no news for {ticker}")
            return
        dest = out_dir / "news" / f"{ticker}.json"
        merged = _merge_json_list(dest, articles, dedup_key="title")
        print(f"  ✓ news/{ticker}.json  ({len(merged)} articles, {self.SOURCE})")

    def fetch_policy_news(self, start_date: str, end_date: str, out_dir: Path):
        policy_dir = out_dir / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        time_from = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%dT0000")
        time_to = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%dT2359")
        for topic, filename in [
            ("economy_fiscal", "fiscal.json"),
            ("economy_monetary", "monetary.json"),
        ]:
            params: Dict[str, str] = {
                "function": "NEWS_SENTIMENT",
                "topics": topic,
                "time_from": time_from,
                "time_to": time_to,
            }
            data = self._get(params)
            feed = data.get("feed", [])
            articles = []
            for item in feed[:30]:
                articles.append({
                    "title": item.get("title", ""),
                    "publish_time": item.get("time_published", ""),
                    "publisher": item.get("source", ""),
                    "summary": item.get("summary", ""),
                    "_source": self.SOURCE,
                })
            if articles:
                merged = _merge_json_list(
                    policy_dir / filename, articles, dedup_key="title"
                )
                print(f"  ✓ policy/{filename}  ({len(merged)} articles, {self.SOURCE})")
            else:
                print(f"  WARNING: no {topic} policy news")

    def fetch_macro(self, start_date: str, end_date: str, out_dir: Path):
        """Fetch macro indicators in [*start_date*, *end_date*] range."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        indicators: Dict[str, Any] = {}
        for func_name, key in [
            ("REAL_GDP", "real_gdp"),
            ("CPI", "cpi"),
            ("TREASURY_YIELD", "treasury_yield"),
            ("FEDERAL_FUNDS_RATE", "federal_funds_rate"),
            ("UNEMPLOYMENT", "unemployment"),
            ("NONFARM_PAYROLL", "nonfarm_payrolls"),
        ]:
            try:
                data = self._get({"function": func_name})
                points = data.get("data", [])
                valid = [
                    p
                    for p in points
                    if start_dt
                    <= datetime.strptime(
                        p.get("date", "9999-12-31"), "%Y-%m-%d"
                    )
                    <= end_dt
                ]
                for v in valid:
                    v["_source"] = self.SOURCE
                indicators[key] = valid
            except Exception as e:
                print(f"  WARNING: failed to fetch {func_name}: {e}")
                indicators[key] = []
        dest = out_dir / "macro" / "indicators.json"
        _merge_json_dict(dest, indicators)
        print(f"  ✓ macro/indicators.json  ({self.SOURCE}, [{start_date}, {end_date}])")


# ── FRED macro fetcher ──


def fetch_fred_macro(api_key: str, start_date: str, end_date: str, out_dir: Path):
    """Download macro indicators from FRED in [*start_date*, *end_date*] range."""
    fred = _import_fredapi(api_key)
    if fred is None:
        return
    import pandas as pd

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    indicators: Dict[str, Any] = {}
    series_map = {
        "real_gdp": "GDP",
        "cpi": "CPIAUCSL",
        "treasury_yield": "DGS10",
        "federal_funds_rate": "FEDFUNDS",
        "unemployment": "UNRATE",
        "nonfarm_payrolls": "PAYEMS",
    }
    for key, series_id in series_map.items():
        s = _call_with_retry(fred.get_series, series_id)
        if s is None:
            indicators[key] = []
            continue
        try:
            s = s[(s.index >= start_ts) & (s.index <= end_ts)].dropna()
            points = []
            for idx, val in s.items():
                points.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "value": str(round(float(val), 4)),
                    "_source": "fred",
                })
            indicators[key] = points
        except Exception as e:
            print(f"  WARNING: FRED {series_id}: {e}")
            indicators[key] = []
    dest = out_dir / "macro" / "indicators.json"
    _merge_json_dict(dest, indicators)
    print(f"  ✓ macro/indicators.json  (FRED, [{start_date}, {end_date}])")


# ── Meta file ──


def _update_meta(
    out_dir: Path,
    tickers: List[str],
    start_date: str,
    end_date: str,
    sources: List[str],
) -> None:
    """Create or update meta.json with the union of tickers and date ranges."""
    meta_path = out_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    old_tickers = set(meta.get("tickers", []))
    old_sources = set(meta.get("sources", []))
    old_start = meta.get("start_date", start_date)
    old_end = meta.get("end_date", end_date)
    meta["tickers"] = sorted(old_tickers | set(tickers))
    meta["start_date"] = min(old_start, start_date)
    meta["end_date"] = max(old_end, end_date)
    meta["sources"] = sorted(old_sources | set(sources))
    meta["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Programmatic API ──


def fetch_all(
    tickers: List[str],
    start_date: str,
    end_date: str,
    out_dir: str | Path,
) -> None:
    """Fetch data for *tickers* in [*start_date*, *end_date*] from all sources.

    Auto-detects API keys from ``config/config.toml [fund]``.
    """
    out_dir = Path(out_dir)
    fred_key = _get_api_key("fred_api_key")
    av_key = _get_api_key("alpha_vantage_api_key")

    sources = ["yfinance"]
    if fred_key:
        sources.append("FRED")
    if av_key:
        sources.append("Alpha Vantage")

    print("═══ DeepFundBench Data Fetch ═══")
    print(f"  Tickers    : {tickers}")
    print(f"  Date Range : {start_date} → {end_date}")
    print(f"  Output Dir : {out_dir}")
    print(f"  Sources    : {' → '.join(sources)}")
    print()

    # ── Layer 1: yfinance (always) ──
    print("── Layer 1: yfinance (free) ──")
    yf_fetcher = YFinanceFetcher()
    for ticker in tickers:
        print(f"[{ticker}]")
        yf_fetcher.fetch_daily(ticker, start_date, end_date, out_dir)
        yf_fetcher.fetch_fundamentals(ticker, out_dir)
        yf_fetcher.fetch_insider(ticker, start_date, end_date, out_dir)

    # ── Layer 2: FRED ──
    if fred_key:
        print()
        print("── Layer 2: FRED ──")
        print("[Macro]")
        fetch_fred_macro(fred_key, start_date, end_date, out_dir)
    else:
        print()
        print("  ⚠ fred_api_key not set — skipping FRED")

    # ── Layer 3: Alpha Vantage ──
    if av_key:
        print()
        print("── Layer 3: Alpha Vantage ──")
        av_fetcher = AlphaVantageFetcher(av_key)
        for ticker in tickers:
            print(f"[{ticker}]")
            av_fetcher.fetch_daily(ticker, start_date, end_date, out_dir)
            av_fetcher.fetch_fundamentals(ticker, out_dir)
            av_fetcher.fetch_insider(ticker, start_date, end_date, out_dir)
            av_fetcher.fetch_news(ticker, start_date, end_date, out_dir)
        print()
        print("[Macro & Policy]")
        av_fetcher.fetch_macro(start_date, end_date, out_dir)
        av_fetcher.fetch_policy_news(start_date, end_date, out_dir)
        print(f"\n  Total Alpha Vantage API calls: {av_fetcher._call_count}")
    else:
        print()
        print("  ⚠ alpha_vantage_api_key not set — skipping Alpha Vantage")

    _update_meta(out_dir, tickers, start_date, end_date, sources)
    print(f"\n═══ Done. Data saved to {out_dir} ═══")


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Fetch financial data for DeepFundBench experiments."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=_DEFAULT_TICKERS,
        help=f"Ticker symbols (default: {' '.join(_DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--start",
        default=_DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {_DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=_DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {_DEFAULT_END})",
    )
    _default_out = str(Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=_default_out,
        help=f"Output directory (default: {_default_out})",
    )
    args = parser.parse_args()
    fetch_all(args.tickers, args.start, args.end, args.output_dir)


if __name__ == "__main__":
    main()
