"""
tools/finnhub_engine.py — Finnhub API wrapper.

Provides analyst ratings, price targets, earnings surprises,
insider transactions, and company news — data not available in yfinance.

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数:
  FINNHUB_API_KEY 已在 .env 配置。
  Finnhub 免费版限速: 60次/分钟，约 60万次/月。
  如需升级，访问 https://finnhub.io/pricing
════════════════════════════════════════════════════════════════════
"""

import os
import time
from datetime import datetime, timedelta

import requests
from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state

_BASE = "https://finnhub.io/api/v1"

# Finnhub free tier only supports US-listed securities.
# International tickers have exchange suffixes after a dot (e.g. HSBA.L, 8306.T, 0700.HK).
# Exception: BRK.A / BRK.B / BRK.C and similar US share-class suffixes are valid.
_US_CLASS_SUFFIXES = {"A", "B", "C", "D", "E", "F", "PR", "WS", "U", "W"}


def _is_us_ticker(ticker: str) -> bool:
    """
    Returns True if the ticker is US-listed and safe to query on Finnhub.
    Non-US tickers carry an exchange code after a dot (e.g. .L, .T, .HK, .PA).
    US share-class suffixes (.A, .B …) are treated as US.
    """
    if "." not in ticker:
        return True
    suffix = ticker.rsplit(".", 1)[-1].upper()
    return suffix in _US_CLASS_SUFFIXES


def _non_us_message(ticker: str, method: str) -> str:
    return (
        f"[Finnhub] Skipped {method} for {ticker.upper()} — "
        f"Finnhub free tier only covers US-listed stocks. "
        f"Use TavilyEngine.finance_search() or NewsEngine for international tickers."
    )


class FinnhubEngine(BaseTool):
    tool_name = "finnhub_engine"
    tool_description = (
        "Fetches analyst ratings, price targets, earnings surprises, "
        "insider activity, and company news from Finnhub API."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        # ════════════════════════════════════════════════════════════════
        if not self.api_key:
            logger.warning("[finnhub_engine] FINNHUB_API_KEY not set in .env")

        self.register(self.get_analyst_ratings)
        # get_price_targets is a Finnhub premium endpoint (403 on free tier) — not registered
        self.register(self.get_earnings_surprise)
        self.register(self.get_company_news)
        self.register(self.get_insider_transactions)

    def _get(self, endpoint: str, params: dict) -> dict | None:
        """
        Internal HTTP GET. Returns None on any failure so callers
        always fall back to their 'no data' message gracefully.
        Non-US tickers are short-circuited before hitting the network.
        """
        if not self.api_key:
            return None
        # Guard: skip non-US tickers silently — 403 guaranteed on free tier
        ticker = params.get("symbol", "")
        if ticker and not _is_us_ticker(ticker):
            logger.debug(f"[finnhub_engine] Skipping non-US ticker '{ticker}' on {endpoint}")
            return None
        try:
            params["token"] = self.api_key
            resp = requests.get(f"{_BASE}{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            # Use string check as a safety net in case e.response is None
            if "403" in str(e):
                logger.debug(f"[finnhub_engine] 403 on {endpoint} for '{ticker}' — free-tier limit, skipping")
            else:
                logger.error(f"[finnhub_engine] {endpoint}: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"[finnhub_engine] {endpoint}: {e}")
            return None

    # ── Analyst Ratings ───────────────────────────────────────────────────────

    def get_analyst_ratings(self, ticker: str) -> str:
        """
        Returns the latest analyst consensus ratings (Buy/Hold/Sell counts)
        and recommendation trend over the past 4 months.

        Args:
            ticker: Stock ticker, e.g. 'TSLA'.
        """
        if not _is_us_ticker(ticker):
            return _non_us_message(ticker, "get_analyst_ratings")
        data = self._get("/stock/recommendation", {"symbol": ticker})
        if not data:
            return f"[Finnhub] No analyst rating data for {ticker}."

        lines = [f"Analyst Consensus — {ticker.upper()} (last 4 months):"]
        for row in data[:4]:
            period = row.get("period", "")
            buy    = row.get("buy", 0) + row.get("strongBuy", 0)
            hold   = row.get("hold", 0)
            sell   = row.get("sell", 0) + row.get("strongSell", 0)
            total  = buy + hold + sell or 1
            verdict = "Bullish" if buy / total > 0.5 else ("Bearish" if sell / total > 0.3 else "Neutral")
            lines.append(
                f"  {period}  Buy:{buy}  Hold:{hold}  Sell:{sell}  "
                f"=> {verdict} ({round(buy/total*100)}% bullish)"
            )
        return "\n".join(lines)

    # ── Price Targets ─────────────────────────────────────────────────────────

    def get_price_targets(self, ticker: str) -> str:
        """
        Returns analyst consensus price targets (high, low, mean, median)
        and the last update date.

        Args:
            ticker: Stock ticker.
        """
        if not _is_us_ticker(ticker):
            return _non_us_message(ticker, "get_price_targets")
        data = self._get("/stock/price-target", {"symbol": ticker})
        if not data or not data.get("targetMean"):
            return f"[Finnhub] No price target data for {ticker}."

        current  = data.get("lastUpdated", "N/A")[:10]
        mean_t   = data.get("targetMean", "N/A")
        median_t = data.get("targetMedian", "N/A")
        high_t   = data.get("targetHigh", "N/A")
        low_t    = data.get("targetLow", "N/A")

        lines = [
            f"Analyst Price Targets — {ticker.upper()} (as of {current}):",
            f"  Mean Target   : ${mean_t}",
            f"  Median Target : ${median_t}",
            f"  High Target   : ${high_t}",
            f"  Low Target    : ${low_t}",
        ]
        return "\n".join(lines)

    # ── Earnings Surprises ────────────────────────────────────────────────────

    def get_earnings_surprise(self, ticker: str, quarters: int = 6) -> str:
        """
        Returns the last N quarters of EPS actual vs. estimate and
        the surprise percentage. Identifies consistent beat/miss patterns.

        Args:
            ticker:   Stock ticker.
            quarters: Number of quarters to return (default 6).
        """
        if not _is_us_ticker(ticker):
            return _non_us_message(ticker, "get_earnings_surprise")
        data = self._get("/stock/earnings", {"symbol": ticker})
        if not data:
            return f"[Finnhub] No earnings data for {ticker}."

        subset = data[:quarters]
        lines = [f"EPS Surprises — {ticker.upper()} (last {quarters} quarters):"]
        beat_count = 0
        for q in subset:
            period   = q.get("period", "N/A")
            actual   = q.get("actual", "N/A")
            estimate = q.get("estimate", "N/A")
            surprise = q.get("surprisePercent", 0)
            emoji = "✅" if surprise and surprise > 0 else ("❌" if surprise and surprise < 0 else "—")
            if surprise and surprise > 0:
                beat_count += 1
            surprise_str = f"{surprise:+.1f}%" if isinstance(surprise, (int, float)) else "N/A"
            lines.append(
                f"  {emoji} {period}  Actual:{actual}  Est:{estimate}  "
                f"Surprise:{surprise_str}"
            )
        if subset:
            lines.append(f"  => Beat rate: {beat_count}/{len(subset)} quarters")
        return "\n".join(lines)

    # ── Company News ──────────────────────────────────────────────────────────

    def get_company_news(self, ticker: str, days_back: int = 14) -> str:
        """
        Returns recent company-specific news articles from Finnhub.
        NOTE: Only covers US-listed stocks on the free tier.
        For international tickers, use TavilyEngine.news_search() instead.

        Args:
            ticker:    Stock ticker (US only on free tier).
            days_back: How many days of news to retrieve (default 14).
        """
        if not _is_us_ticker(ticker):
            return _non_us_message(ticker, "get_company_news")
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        data = self._get("/company-news", {
            "symbol": ticker,
            "from": start,
            "to": end,
        })
        if not data:
            return f"[Finnhub] No news for {ticker} in last {days_back} days."

        top = data[:6]
        lines = [f"Finnhub News — {ticker.upper()} (last {days_back} days):"]
        for a in top:
            date    = datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
            headline = a.get("headline", "No headline")[:120]
            source   = a.get("source", "N/A")
            lines.append(f"  [{date}] {headline}")
            lines.append(f"    Source: {source}")

        # Write headlines to SharedState
        state = get_state()
        existing = state.news_headlines or []
        new_heads = [
            f"[{datetime.fromtimestamp(a.get('datetime', 0)).strftime('%Y-%m-%d')}] "
            f"{a.get('headline', '')[:100]} (Finnhub)"
            for a in top
        ]
        state.news_headlines = (existing + new_heads)[:10]

        return "\n".join(lines)

    # ── Insider Transactions ──────────────────────────────────────────────────

    def get_insider_transactions(self, ticker: str) -> str:
        """
        Returns the most recent insider buy/sell transactions.
        Heavy insider buying is a bullish signal; sustained selling is bearish.

        Args:
            ticker: Stock ticker.
        """
        if not _is_us_ticker(ticker):
            return _non_us_message(ticker, "get_insider_transactions")
        data = self._get("/stock/insider-transactions", {"symbol": ticker})
        if not data or not data.get("data"):
            return f"[Finnhub] No insider transaction data for {ticker}."

        transactions = data["data"][:8]
        lines = [f"Insider Transactions — {ticker.upper()} (recent):"]
        buy_val = sell_val = 0
        for t in transactions:
            name    = t.get("name", "N/A")
            txn     = t.get("transactionCode", "N/A")
            shares  = t.get("share", 0)
            price   = t.get("price", 0)
            date    = t.get("transactionDate", "N/A")
            value   = (shares or 0) * (price or 0)
            direction = "BUY" if txn in ("P", "A") else "SELL"
            emoji   = "📈" if direction == "BUY" else "📉"
            if direction == "BUY":
                buy_val += value
            else:
                sell_val += value
            lines.append(
                f"  {emoji} {date}  {name[:25]}  {direction}  "
                f"{shares:,} shares @ ${price}  (${value:,.0f})"
            )
        net = buy_val - sell_val
        signal = "Net Buying (+)" if net > 0 else "Net Selling (-)"
        lines.append(f"  => Insider signal: {signal}  Net: ${abs(net):,.0f}")
        return "\n".join(lines)
