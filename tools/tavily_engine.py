"""
tools/tavily_engine.py — AI-powered web search via Tavily.

Used by QueryAnalyst to understand ambiguous user queries BEFORE any
analysis begins.  Also used by MacroAgent and NewsAgent to ground
responses in real web search results (especially for historical periods
where yfinance has no data).

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数:
  TAVILY_API_KEY 已在 .env 中配置。
  如需接入其他搜索引擎（如 Bing, Exa, Perplexity），在此添加：
    BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY", "")
    EXA_API_KEY     = os.getenv("EXA_API_KEY", "")
════════════════════════════════════════════════════════════════════
"""

import os
import json
from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state


class TavilyEngine(BaseTool):
    tool_name = "tavily_engine"
    tool_description = (
        "AI-powered web search that understands user intent and retrieves "
        "grounded, up-to-date information for any query — stocks, markets, "
        "historical events, financial concepts, or comparisons."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        # self.bing_key = os.getenv("BING_SEARCH_KEY", "")
        # self.exa_key  = os.getenv("EXA_API_KEY", "")
        # ════════════════════════════════════════════════════════════════

        self._client = None  # lazy-loaded
        self.register(self.understand_query)
        self.register(self.web_search)
        self.register(self.finance_search)
        self.register(self.news_search)

    def _get_client(self):
        """Lazy-load Tavily client."""
        if self._client is None:
            from tavily import TavilyClient
            if not self.api_key:
                raise ValueError("TAVILY_API_KEY not set in .env")
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    # ── Primary: intent understanding ────────────────────────────────────

    def understand_query(self, user_input: str) -> str:
        """
        THE MOST IMPORTANT METHOD.
        Analyses the user's raw input and returns a structured breakdown:
          - What the user is actually asking about
          - Query type classification
          - Extracted tickers / markets / time periods
          - AI-generated context from the web
          - Recommended analysis approach

        Args:
            user_input: The raw, unprocessed user message.
        """
        try:
            client = self._get_client()

            _TAVILY_MAX = 400
            _PREFIX = "financial analysis: "

            def _build_query(text: str) -> str:
                """Trim text so the full query stays within Tavily's 400-char limit."""
                budget = _TAVILY_MAX - len(_PREFIX)
                return _PREFIX + (text if len(text) <= budget else text[:budget - 3] + "...")

            # Attempt 1 – full query
            search_q = _build_query(user_input)

            def _do_search(q: str):
                return client.search(
                    q,
                    search_depth="advanced",
                    include_answer=True,
                    max_results=6,
                )

            try:
                resp = _do_search(search_q)
            except Exception as _e1:
                if "too long" in str(_e1).lower() or "400" in str(_e1):
                    # Retry with an even shorter slice (first 120 chars of user_input)
                    fallback_q = _build_query(user_input[:120])
                    logger.warning(
                        f"[tavily_engine] Query too long, retrying with truncated version: {fallback_q!r}"
                    )
                    resp = _do_search(fallback_q)
                else:
                    raise

            answer = resp.get("answer") or ""
            results = resp.get("results") or []

            # ── Classify query type from content ──────────────────────
            lower = user_input.lower()
            combined_text = user_input + " " + answer + " " + " ".join(
                (r.get("title") or "") + " " + (r.get("content") or "")[:200]
                for r in results[:3]
            )

            query_type = _classify_intent(user_input, combined_text)
            tickers    = _extract_tickers(combined_text)
            time_period = _extract_time_period(user_input, combined_text)
            market      = _extract_market(user_input, combined_text)

            # ── Format the structured understanding ───────────────────
            lines = [
                "═" * 58,
                "  QUERY UNDERSTANDING (Tavily AI Search)",
                "═" * 58,
                f"  Raw Input    : {user_input}",
                f"  Query Type   : {query_type}",
                f"  Tickers Found: {', '.join(tickers) if tickers else 'N/A'}",
                f"  Market/Index : {market or 'N/A'}",
                f"  Time Period  : {time_period or 'Current / Present'}",
                "",
                "  AI Context from Web:",
            ]
            if answer:
                for line in answer.split("\n"):
                    if line.strip():
                        lines.append(f"    {line.strip()}")
            else:
                lines.append("    (no AI summary available)")

            lines += [
                "",
                "  Top Sources:",
            ]
            for r in results[:3]:
                title   = r.get("title", "")[:80]
                url     = r.get("url", "")
                snippet = r.get("content", "")[:120].replace("\n", " ")
                lines.append(f"    - {title}")
                lines.append(f"      {snippet}...")
                lines.append(f"      Source: {url}")

            lines += [
                "",
                "  Recommended Analysis Approach:",
            ]
            approach = _recommend_approach(query_type, tickers, time_period)
            for step in approach:
                lines.append(f"    → {step}")

            lines.append("═" * 58)

            understanding = "\n".join(lines)

            # Write key facts to SharedState
            state = get_state()
            if tickers:
                state.ticker = tickers[0]
            state.query = user_input

            return understanding

        except Exception as e:
            logger.error(f"[tavily_engine] understand_query error: {e}")
            return (
                f"[QueryUnderstanding] Could not perform Tavily search: {e}\n"
                f"Raw query: {user_input}\n"
                f"Proceeding with best-effort analysis based on query text."
            )

    # ── General web search ────────────────────────────────────────────────

    def web_search(self, query: str, max_results: int = 5) -> str:
        """
        General-purpose AI web search. Returns titles, snippets, and
        an AI-synthesized answer for the query.

        Args:
            query:       Search query string.
            max_results: Number of results to return (default 5).
        """
        try:
            client = self._get_client()
            resp = client.search(
                _trim_query(query),
                search_depth="advanced",
                include_answer=True,
                max_results=max_results,
            )
            return _format_results(query, resp)
        except Exception as e:
            logger.error(f"[tavily_engine] web_search error: {e}")
            return f"Search error for '{query}': {e}"

    # ── Finance-focused search ────────────────────────────────────────────

    def finance_search(self, query: str, max_results: int = 6) -> str:
        """
        Finance-specific search — best for stock analysis, earnings,
        analyst ratings, valuations, and financial news.

        Args:
            query:       Finance-related search query.
            max_results: Number of results (default 6).
        """
        try:
            client = self._get_client()
            resp = client.search(
                _trim_query(f"{query} financial analysis investment"),
                search_depth="advanced",
                include_answer=True,
                max_results=max_results,
            )
            return _format_results(query, resp)
        except Exception as e:
            logger.error(f"[tavily_engine] finance_search error: {e}")
            return f"Finance search error: {e}"

    # ── News search ───────────────────────────────────────────────────────

    def news_search(self, query: str, days_back: int = 30, max_results: int = 6) -> str:
        """
        Recent news search — returns the latest headlines and events
        related to the query within the specified lookback period.

        Args:
            query:       News search query.
            days_back:   How many days back to search (default 30).
            max_results: Number of results (default 6).
        """
        try:
            client = self._get_client()
            resp = client.search(
                _trim_query(query),
                topic="news",
                days=days_back,
                max_results=max_results,
                include_answer=True,
            )
            return _format_results(query, resp, is_news=True)
        except Exception as e:
            logger.error(f"[tavily_engine] news_search error: {e}")
            return f"News search error: {e}"


# ── Private helpers ────────────────────────────────────────────────────────────

_TAVILY_QUERY_MAX = 400


def _trim_query(query: str, max_len: int = _TAVILY_QUERY_MAX) -> str:
    """Ensure a Tavily query never exceeds the 400-character API limit."""
    if len(query) <= max_len:
        return query
    return query[: max_len - 3] + "..."


def _format_results(query: str, resp: dict, is_news: bool = False) -> str:
    answer  = resp.get("answer", "")
    results = resp.get("results", [])
    label   = "📰 News Results" if is_news else "🔍 Search Results"

    lines = [f"{label} for: {query}", ""]
    if answer:
        lines += ["  AI Answer:", f"  {answer}", ""]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "No title")
        url     = r.get("url", "")
        content = r.get("content", "")[:200].replace("\n", " ")
        date    = r.get("published_date", "")
        date_s  = f" [{date[:10]}]" if date else ""
        lines += [
            f"  {i}.{date_s} {title}",
            f"     {content}...",
            f"     Source: {url}",
            "",
        ]
    return "\n".join(lines)


def _classify_intent(raw: str, context: str) -> str:
    """Rule-based intent classification from query text + search context."""
    r = raw.lower()
    c = context.lower()

    # Historical period keywords
    hist_years = [str(y) for y in range(2000, 2024)]
    if any(yr in r for yr in hist_years):
        return "HISTORICAL_ANALYSIS"

    # Comparison keywords
    if any(w in r for w in ["vs", "versus", "compare", "比较", "对比", "差距"]):
        return "COMPARISON"

    # Concept / educational
    concept_kw = ["what is", "explain", "how does", "define", "什么是", "如何",
                  "why", "when", "dcf", "p/e", "eps", "wacc", "roi", "capm"]
    if any(w in r for w in concept_kw):
        return "CONCEPT_EXPLANATION"

    # Macro / market / index
    macro_kw = ["market", "index", "economy", "gdp", "inflation", "rate", "fed",
                "hsi", "s&p", "nasdaq", "dow", "nikkei", "ftse", "sector",
                "港股", "美股", "a股", "大盘", "行业", "板块", "宏观", "利率"]
    if any(w in r for w in macro_kw):
        # Also check if a specific ticker is mentioned
        tickers = _extract_tickers(raw)
        if not tickers:
            return "MARKET_ANALYSIS"

    # Crypto
    crypto_kw = ["bitcoin", "btc", "eth", "ethereum", "crypto", "defi", "web3",
                 "加密", "比特币", "以太坊"]
    if any(w in r for w in crypto_kw):
        return "CRYPTO_ANALYSIS"

    # IPO
    if any(w in r for w in ["ipo", "上市", "新股"]):
        return "IPO_ANALYSIS"

    # Default: stock-level analysis
    return "STOCK_ANALYSIS"


def _extract_tickers(text: str) -> list[str]:
    """Extract likely stock ticker symbols from text."""
    import re
    # Common explicit tickers (uppercase 1-5 letters, standalone)
    explicit = re.findall(r'\b([A-Z]{1,5})\b', text)
    # Known name→ticker map for common queries
    name_map = {
        "tesla": "TSLA", "apple": "AAPL", "nvidia": "NVDA",
        "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
        "amazon": "AMZN", "meta": "META", "netflix": "NFLX",
        "berkshire": "BRK-B", "jpmorgan": "JPM", "goldman": "GS",
        "特斯拉": "TSLA", "苹果": "AAPL", "微软": "MSFT",
        "谷歌": "GOOGL", "亚马逊": "AMZN", "英伟达": "NVDA",
        "tencent": "0700.HK", "腾讯": "0700.HK",
        "alibaba": "BABA", "阿里": "BABA", "阿里巴巴": "BABA",
        "meituan": "3690.HK", "美团": "3690.HK",
    }
    found = []
    lower = text.lower()
    for name, ticker in name_map.items():
        if name in lower and ticker not in found:
            found.append(ticker)

    # Filter explicit tickers (exclude common English words)
    stop = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
            "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET",
            "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW",
            "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS",
            "LET", "PUT", "SAY", "SHE", "TOO", "USE", "ETF", "CEO",
            "IPO", "GDP", "FED", "USD", "HKD", "CNY", "EUR", "AI",
            "USA", "HK", "US", "UK", "CN", "PE", "PB", "EV"}
    for t in explicit:
        if t not in stop and len(t) >= 2 and t not in found:
            found.append(t)
    return found[:5]


def _extract_time_period(raw: str, context: str) -> str:
    """Extract mentioned time period from query."""
    import re
    # Explicit years
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', raw)
    if years:
        return f"{min(years)}–{max(years)}" if len(set(years)) > 1 else years[0]
    # Quarter patterns
    quarters = re.findall(r'\b(Q[1-4]\s*\d{4}|\d{4}\s*Q[1-4])\b', raw, re.I)
    if quarters:
        return quarters[0]
    # Relative terms
    if any(w in raw.lower() for w in ["this year", "今年"]):
        from datetime import datetime
        return str(datetime.now().year)
    if any(w in raw.lower() for w in ["last year", "去年"]):
        from datetime import datetime
        return str(datetime.now().year - 1)
    return ""


def _extract_market(raw: str, context: str) -> str:
    """Extract the market or index being discussed."""
    r = raw.lower()
    if any(w in r for w in ["hk", "hong kong", "港股", "恒指", "hang seng", "hsi"]):
        return "Hong Kong / Hang Seng Index (HSI)"
    if any(w in r for w in ["s&p", "sp500", "us market", "us stock", "美股"]):
        return "US Market / S&P 500"
    if any(w in r for w in ["nasdaq", "tech", "科技股"]):
        return "NASDAQ / US Tech"
    if any(w in r for w in ["a股", "a share", "沪深", "上证", "深证", "sse", "szse"]):
        return "China A-Shares"
    if any(w in r for w in ["nikkei", "japan", "日经", "日股"]):
        return "Japan / Nikkei 225"
    if any(w in r for w in ["europe", "ftse", "dax", "欧股"]):
        return "European Markets"
    return ""


def _recommend_approach(query_type: str, tickers: list, time_period: str) -> list[str]:
    """Return a list of recommended analysis steps based on query type."""
    is_historical = bool(time_period and any(
        c.isdigit() for c in time_period
    ) and int(time_period[:4]) < 2024 if time_period else False)

    steps = {
        "STOCK_ANALYSIS": [
            f"CompanyAgent: fetch real-time fundamentals via yfinance for {tickers[0] if tickers else 'ticker'}",
            "MacroAgent: analyse macro + sector context",
            "NewsAgent: fetch recent headlines and analyst ratings",
            "CIO: synthesise BUY / HOLD / SELL recommendation",
            "ReportManager: generate Excel + PDF report",
        ],
        "MARKET_ANALYSIS": [
            "MacroAgent: broad market + sector analysis using Tavily search",
            "NewsAgent: latest market headlines and sentiment",
            "CIO: synthesise market outlook (NO yfinance needed)",
        ],
        "HISTORICAL_ANALYSIS": [
            "Use Tavily web_search for historical data (yfinance not needed)",
            "MacroAgent: search historical macro context via Tavily",
            "NewsAgent: search historical events and sentiment via Tavily",
            "CIO: synthesise historical analysis with proper time-period framing",
        ],
        "COMPARISON": [
            f"Run STOCK_ANALYSIS for each of: {', '.join(tickers[:3]) or 'entities'}",
            "CIO: synthesise side-by-side comparison",
        ],
        "CONCEPT_EXPLANATION": [
            "CIO: answer directly from knowledge — no sub-agents needed",
        ],
        "CRYPTO_ANALYSIS": [
            "MacroAgent: crypto market context via Tavily",
            "NewsAgent: crypto news and sentiment via Tavily",
            "CIO: synthesise crypto outlook (yfinance crypto data may be limited)",
        ],
        "IPO_ANALYSIS": [
            "MacroAgent + NewsAgent: IPO details via Tavily search",
            "CIO: synthesise IPO investment thesis",
        ],
    }
    return steps.get(query_type, [
        "Use Tavily to gather relevant context",
        "CIO: synthesise based on available information",
    ])
