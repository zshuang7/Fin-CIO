"""
tools/news_engine.py — DuckDuckGo news search wrapped as BaseTool.

Fixes applied vs v1:
  1. Resolves company name from yfinance before searching, so non-US
     tickers like '8306.T' (Mitsubishi UFJ) search correctly.
  2. Graceful fallback: if DDGS returns empty, logs a warning instead
     of crashing.  FinnhubEngine handles non-US news as the primary
     source; DDGS is a complementary fallback.
  3. "Impersonate chrome_119" warning suppressed — DDGS uses 'random'
     automatically when the specific version isn't available.

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数（例如 NewsAPI / Benzinga）:
  NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
════════════════════════════════════════════════════════════════════
"""

import os
import warnings
import logging
from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state

# ── Suppress harmless DDGS / curl_cffi impersonate warnings ──────────────────
# curl_cffi rotates browser fingerprint versions; DDGS falls back to 'random'
# automatically — this is safe and does not affect functionality.
warnings.filterwarnings("ignore", message=".*[Ii]mpersonate.*does not exist.*")
warnings.filterwarnings("ignore", message=".*[Ii]mpersonate.*")
warnings.filterwarnings("ignore", category=UserWarning, module="curl_cffi")
# Also suppress at the logging level in case it comes through as a log record
logging.getLogger("curl_cffi").setLevel(logging.ERROR)
logging.getLogger("ddgs").setLevel(logging.ERROR)


class NewsEngine(BaseTool):
    tool_name = "news_engine"
    tool_description = (
        "Searches DuckDuckGo for financial news and market sentiment. "
        "Resolves tickers to company names before searching to handle "
        "non-US securities (e.g. 8306.T, 0700.HK) correctly."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key（新闻数据源）
        # self.news_api_key  = os.getenv("NEWS_API_KEY", "")
        # self.benzinga_key  = os.getenv("BENZINGA_API_KEY", "")
        # ════════════════════════════════════════════════════════════════
        self._name_cache: dict[str, str] = {}

        self.register(self.get_stock_news)
        self.register(self.get_market_sentiment)
        self.register(self.get_earnings_news)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_name(self, ticker: str) -> str:
        """
        Look up the company's long name via yfinance to improve DDGS search
        quality for non-US tickers (e.g. '8306.T' → 'Mitsubishi UFJ Financial').
        Result is cached per session.
        """
        if ticker in self._name_cache:
            return self._name_cache[ticker]
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ticker
            self._name_cache[ticker] = name
            return name
        except Exception:
            return ticker

    def _ddgs_news(self, query: str, max_results: int) -> list[dict]:
        """Run DDGS news search and return raw result list. Empty on failure."""
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
            return results
        except Exception as e:
            logger.warning(f"[news_engine] DDGS search failed: {e}")
            return []

    def _format_articles(self, articles: list[dict], header: str) -> str:
        if not articles:
            return f"{header}\n  (No results — try FinnhubEngine.get_company_news() for non-US stocks)"
        lines = [header, ""]
        for i, r in enumerate(articles, 1):
            title  = r.get("title", "No title")
            source = r.get("source", "Unknown")
            date   = (r.get("date") or "")[:10]
            body   = (r.get("body") or "")[:100]
            lines.append(f"  {i}. [{date}] {title}")
            lines.append(f"     Source: {source}")
            if body:
                lines.append(f"     {body}...")
            lines.append("")
        return "\n".join(lines)

    # ── Public tool functions ────────────────────────────────────────────────

    def get_stock_news(self, ticker: str, max_results: int = 6) -> str:
        """
        Fetches recent news for a stock. Automatically resolves the ticker
        to a company name for better search accuracy.

        Args:
            ticker:      Stock ticker (US or international, e.g. '8306.T').
            max_results: Maximum headlines to return (default 6).
        """
        company = self._resolve_name(ticker)
        # Use company name for better international coverage
        query = f"{company} stock news 2025"
        articles = self._ddgs_news(query, max_results)

        if not articles and company != ticker:
            # Second attempt: just name without "stock"
            articles = self._ddgs_news(f"{company} financial news", max_results)

        if not articles:
            logger.warning(f"[news_engine] No DDGS results for {ticker} ({company}). "
                           f"Use FinnhubEngine.get_company_news('{ticker}') for reliable results.")
            return (
                f"[NewsEngine] No DDGS results for {ticker} ({company}).\n"
                f"Recommendation: use FinnhubEngine.get_company_news('{ticker}') instead."
            )

        # Write to SharedState
        state = get_state()
        state.ticker = ticker.upper()
        new_heads = [
            f"[{(r.get('date') or '')[:10]}] {r.get('title','')[:100]} ({r.get('source','')})"
            for r in articles
        ]
        state.news_headlines = ((state.news_headlines or []) + new_heads)[:10]

        return self._format_articles(articles, f"News — {company} ({ticker.upper()})")

    def get_market_sentiment(self, ticker: str) -> str:
        """
        Searches for analyst ratings, upgrades, and price target changes.

        Args:
            ticker: Stock ticker.
        """
        company = self._resolve_name(ticker)
        query = f"{company} analyst rating upgrade downgrade price target 2025"
        articles = self._ddgs_news(query, max_results=5)

        if not articles:
            return (
                f"[NewsEngine] No sentiment results for {ticker} ({company}).\n"
                f"Try FinnhubEngine.get_analyst_ratings('{ticker}') for structured data."
            )

        all_text = " ".join(r.get("title", "") for r in articles).lower()
        bullish_kw = ["upgrade", "buy", "overweight", "outperform", "bullish",
                      "strong", "raise", "increase", "beat", "positive"]
        bearish_kw = ["downgrade", "sell", "underweight", "underperform",
                      "bearish", "weak", "cut", "miss", "lower", "concern"]
        bull = sum(all_text.count(w) for w in bullish_kw)
        bear = sum(all_text.count(w) for w in bearish_kw)

        if bull > bear + 1:
            sentiment = "Bullish"
        elif bear > bull + 1:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        get_state().news_sentiment = sentiment

        lines = [f"Market Sentiment — {company} ({ticker.upper()}):"]
        for r in articles:
            title  = r.get("title", "")
            source = r.get("source", "")
            date   = (r.get("date") or "")[:10]
            lines.append(f"  [{date}] {title} ({source})")
        lines.append(f"\n  => Sentiment: {sentiment}")
        return "\n".join(lines)

    def get_earnings_news(self, ticker: str) -> str:
        """
        Searches for recent earnings results, guidance, and analyst reactions.

        Args:
            ticker: Stock ticker.
        """
        company = self._resolve_name(ticker)
        query = f"{company} earnings results guidance 2025"
        articles = self._ddgs_news(query, max_results=4)

        if not articles:
            return (
                f"[NewsEngine] No earnings news for {ticker} ({company}).\n"
                f"Try FinnhubEngine.get_earnings_surprise('{ticker}') for structured data."
            )

        return self._format_articles(
            articles, f"Earnings & Guidance — {company} ({ticker.upper()})"
        )
