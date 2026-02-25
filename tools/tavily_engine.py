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
        self.register(self.wall_street_search)

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

            # ── Step 0: Pre-detect numeric exchange tickers BEFORE Tavily ─
            # e.g. "2015 HK" = 2015.HK (Li Auto HKEX), not "HK market in 2015"
            #      "1211 HK" = 1211.HK (BYD), "8306 T" = 8306.T (Mitsubishi UFJ)
            # This MUST run before Tavily and before any year/market heuristics.
            pre_numeric_tickers = _extract_numeric_tickers(user_input)

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
            combined_text = user_input + " " + answer + " " + " ".join(
                (r.get("title") or "") + " " + (r.get("content") or "")[:200]
                for r in results[:3]
            )

            query_type  = _classify_intent(user_input, combined_text, pre_numeric_tickers)
            tickers     = _extract_tickers(combined_text, pre_numeric_tickers)
            time_period = _extract_time_period(user_input, combined_text, pre_numeric_tickers)
            market      = _extract_market(user_input, combined_text, pre_numeric_tickers)

            # ── Format the structured understanding ───────────────────
            # Prepend a disambiguation banner when numeric tickers were detected.
            # This prevents the CIO from treating "2015 HK" as "year 2015" etc.
            ticker_alert = []
            if pre_numeric_tickers:
                ticker_alert = [
                    "  ⚠  TICKER DISAMBIGUATION:",
                    f"     Numeric exchange codes detected: {', '.join(pre_numeric_tickers)}",
                    "     These are STOCK TICKERS, not years or market names.",
                    "     e.g. '2015 HK' → 2015.HK (Li Auto, HKEX)",
                    "          '1211 HK' → 1211.HK (BYD, HKEX)",
                    "          '8306 T'  → 8306.T  (Mitsubishi UFJ, TSE)",
                    "",
                ]

            lines = [
                "═" * 58,
                "  QUERY UNDERSTANDING (Tavily AI Search)",
                "═" * 58,
                *ticker_alert,
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

    # ── Wall Street deep analyst search ──────────────────────────────────

    def wall_street_search(self, ticker: str, context: str = "") -> str:
        """
        Deep-dive search for Wall Street analyst reports, price target changes,
        and broker research notes for a specific ticker.
        Uses search_depth='advanced' with targeted financial queries.

        Args:
            ticker:  Stock ticker (e.g. TSLA, NVDA, 0700.HK).
            context: Optional extra context to refine the search
                     (e.g. 'Goldman Sachs bull case' or 'EPS revision Q1 2025').
        """
        try:
            client = self._get_client()
            ticker = ticker.upper().strip()
            from datetime import datetime
            year = datetime.now().year

            # Build a targeted analyst-report query
            base = f"{ticker} analyst rating price target {year}"
            if context:
                base = f"{ticker} {context} analyst note {year}"

            queries = [
                base,
                f"Goldman Sachs Morgan Stanley JPMorgan {ticker} research note {year}",
                f"{ticker} upgrade downgrade overweight underweight {year}",
            ]

            all_results = []
            seen_urls: set[str] = set()

            for q in queries[:2]:   # 2 calls max to conserve quota
                try:
                    resp = client.search(
                        _trim_query(q),
                        search_depth="advanced",
                        include_answer=True,
                        max_results=5,
                        topic="finance",
                    )
                    for r in resp.get("results") or []:
                        url = r.get("url", "")
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                    # Use the AI answer from the first query
                    if not hasattr(self, "_ws_answer"):
                        self._ws_answer = resp.get("answer") or ""
                except Exception:
                    continue

            answer = getattr(self, "_ws_answer", "")
            if hasattr(self, "_ws_answer"):
                del self._ws_answer   # reset for next call

            lines = [
                f"## Wall Street Deep Search — {ticker} (Tavily Advanced)\n",
            ]
            if answer:
                lines += ["### AI Analyst Summary", f"  {answer}", ""]

            # Classify results by broker mentions
            broker_kw = ["Goldman", "Morgan", "JPMorgan", "Citi", "UBS",
                         "Barclays", "BofA", "Wells Fargo", "Deutsche",
                         "Jefferies", "Needham", "Wedbush", "RBC", "Cowen",
                         "Piper", "Baird", "Bernstein", "Evercore"]

            research, general = [], []
            for r in all_results[:10]:
                title = r.get("title", "")
                is_research = any(b.lower() in title.lower() for b in broker_kw) or \
                              any(kw in title.lower() for kw in
                                  ["analyst", "rating", "price target", "upgrade",
                                   "downgrade", "overweight", "pt ", "pt$"])
                (research if is_research else general).append(r)

            if research:
                lines.append("### Broker Research Notes")
                for r in research[:5]:
                    date = (r.get("published_date") or "")[:10]
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = (r.get("content") or "")[:180].replace("\n", " ")
                    lines.append(f"  🏦 [{date}] {title}")
                    lines.append(f"     › {snippet}...")
                    lines.append(f"     Source: {url}")
                lines.append("")

            if general:
                lines.append("### Related Market Coverage")
                for r in general[:3]:
                    date = (r.get("published_date") or "")[:10]
                    title = r.get("title", "")
                    url = r.get("url", "")
                    lines.append(f"  📰 [{date}] {title}  —  {url}")

            lines.append(
                "\n  Confidence: MEDIUM (web-sourced; cross-verify price targets "
                "with EODHD structured data)"
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[tavily_engine] wall_street_search error: {e}")
            return f"Wall Street search error for {ticker}: {e}"

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


def _extract_numeric_tickers(raw: str) -> list[str]:
    """
    Detect Asian and London market numeric tickers BEFORE any year/market
    heuristics fire.  This prevents false positives like "2015 HK" being
    read as "the HK market in year 2015".

    Supported formats:
      • Explicit dotted  : 2015.HK, 8306.T, HSBA.L, 1211.HK
      • HKEX spaced      : 2015 HK, 1211 HK, 0700 HK
      • TSE/Japan spaced : 8306 T, 7203 T, 6758 JP
      • Shanghai/Shenzhen: 600519 SS, 000858 SZ
      • Korean Exchange  : 005930 KS
      • Taiwan Exchange  : 2330 TW
    """
    import re
    found: list[str] = []

    # Format 1: explicit dotted (highest confidence)
    dotted = re.findall(
        r'\b(\d{1,6})\.(HK|T|L|SS|SZ|KS|TW|PA|DE|AS|BR|AX|NS|BO)\b',
        raw, re.I,
    )
    for num, exch in dotted:
        t = f"{num}.{exch.upper()}"
        if t not in found:
            found.append(t)

    # Format 2: "NNNN HK" — most common ambiguity (HKEX, 4–5 digits + space + HK)
    # Requires a word boundary before the number and after HK to avoid matching
    # substrings like "in 2015 HKD" (HKD would be matched by HK if we're not careful)
    hk_spaced = re.findall(r'(?<!\w)(\d{1,5})\s+HK(?!\w)', raw, re.I)
    for num in hk_spaced:
        t = f"{num}.HK"
        if t not in found:
            found.append(t)

    # Format 3: "NNNN T" or "NNNN JP" — Tokyo/Japan (4–5 digits)
    jp_spaced = re.findall(r'(?<!\w)(\d{4,5})\s+(?:T|JP)(?!\w)', raw, re.I)
    for num in jp_spaced:
        t = f"{num}.T"
        if t not in found:
            found.append(t)

    # Format 4: Other common spaced exchange patterns
    other_patterns = [
        (r'(?<!\w)(\d{1,6})\s+SS(?!\w)', "SS"),   # Shanghai
        (r'(?<!\w)(\d{1,6})\s+SZ(?!\w)', "SZ"),   # Shenzhen
        (r'(?<!\w)(\d{6})\s+KS(?!\w)',   "KS"),   # Korea
        (r'(?<!\w)(\d{4})\s+TW(?!\w)',   "TW"),   # Taiwan
    ]
    for pattern, exch in other_patterns:
        for num in re.findall(pattern, raw, re.I):
            t = f"{num}.{exch}"
            if t not in found:
                found.append(t)

    return found


def _classify_intent(raw: str, context: str,
                     pre_numeric: list[str] | None = None) -> str:
    """
    Rule-based intent classification from query text + search context.

    IMPORTANT: numeric exchange tickers (pre_numeric) are checked FIRST so
    that '2015 HK' is never mis-classified as HISTORICAL_ANALYSIS just
    because '2015' is a year-like number.
    """
    r = raw.lower()

    # ── Priority 0: numeric tickers win over year detection ────────────
    if pre_numeric:
        # If comparison keywords also present → COMPARISON
        if any(w in r for w in ["vs", "versus", "compare", "比较", "对比", "差距",
                                 "with", "和", "与"]):
            return "COMPARISON"
        return "STOCK_ANALYSIS"

    # ── Historical period keywords ──────────────────────────────────────
    # Only check AFTER confirming there are no numeric tickers.
    import re as _re
    hist_match = _re.search(r'\b(19\d{2}|20(?:0[0-9]|1[0-9]|2[0-3]))\b', raw)
    if hist_match:
        # Double-check: is that number immediately followed by an exchange suffix?
        after = raw[hist_match.end():hist_match.end() + 5].strip().upper()
        if not (after.startswith("HK") or after.startswith(".HK") or
                after.startswith("T") or after.startswith(".T")):
            return "HISTORICAL_ANALYSIS"

    # ── Comparison keywords ─────────────────────────────────────────────
    if any(w in r for w in ["vs", "versus", "compare", "比较", "对比", "差距"]):
        return "COMPARISON"

    # ── Concept / educational ───────────────────────────────────────────
    concept_kw = ["what is", "explain", "how does", "define", "什么是", "如何",
                  "why", "when", "dcf", "p/e", "eps", "wacc", "roi", "capm"]
    if any(w in r for w in concept_kw):
        return "CONCEPT_EXPLANATION"

    # ── Macro / market / index ──────────────────────────────────────────
    macro_kw = ["market", "index", "economy", "gdp", "inflation", "rate", "fed",
                "hsi", "s&p", "nasdaq", "dow", "nikkei", "ftse", "sector",
                "港股", "美股", "a股", "大盘", "行业", "板块", "宏观", "利率"]
    if any(w in r for w in macro_kw):
        if not _extract_tickers(raw):
            return "MARKET_ANALYSIS"

    # ── Crypto ──────────────────────────────────────────────────────────
    crypto_kw = ["bitcoin", "btc", "eth", "ethereum", "crypto", "defi", "web3",
                 "加密", "比特币", "以太坊"]
    if any(w in r for w in crypto_kw):
        return "CRYPTO_ANALYSIS"

    # ── IPO ─────────────────────────────────────────────────────────────
    if any(w in r for w in ["ipo", "上市", "新股"]):
        return "IPO_ANALYSIS"

    return "STOCK_ANALYSIS"


def _extract_tickers(text: str,
                     pre_numeric: list[str] | None = None) -> list[str]:
    """
    Extract likely stock ticker symbols from text.
    pre_numeric (from _extract_numeric_tickers) is merged in with highest priority
    so that numeric HKEX / TSE codes are always present regardless of letter case.
    """
    import re
    found: list[str] = list(pre_numeric) if pre_numeric else []

    # Known name→ticker map
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
        "byd": "1211.HK", "比亚迪": "1211.HK",
        "li auto": "2015.HK", "理想汽车": "2015.HK",
        "xpeng": "9868.HK", "小鹏": "9868.HK",
        "nio": "9866.HK", "蔚来": "9866.HK",
        "ping an": "2318.HK", "平安": "2318.HK",
    }
    lower = text.lower()
    for name, ticker in name_map.items():
        if name in lower and ticker not in found:
            found.append(ticker)

    # Common explicit uppercase tickers (1-5 letters, standalone word)
    explicit = re.findall(r'\b([A-Z]{1,5})\b', text)
    stop = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
            "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET",
            "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW",
            "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS",
            "LET", "PUT", "SAY", "SHE", "TOO", "USE", "ETF", "CEO",
            "IPO", "GDP", "FED", "USD", "HKD", "CNY", "EUR", "AI",
            "USA", "HK", "US", "UK", "CN", "PE", "PB", "EV", "TTM",
            "YOY", "QOQ", "FCF", "EPS", "DCF"}
    for t in explicit:
        if t not in stop and len(t) >= 2 and t not in found:
            found.append(t)

    return found[:6]


def _extract_time_period(raw: str, context: str,
                         pre_numeric: list[str] | None = None) -> str:
    """
    Extract mentioned time period from query.
    When numeric tickers are present, year-like numbers (e.g. 2015 in '2015 HK')
    are excluded from the time period result to avoid false positives.
    """
    import re

    # Collect the numeric parts of known tickers so we can exclude them
    ticker_numbers: set[str] = set()
    if pre_numeric:
        for t in pre_numeric:
            m = re.match(r'^(\d+)', t)
            if m:
                ticker_numbers.add(m.group(1))

    years = re.findall(r'\b(19\d{2}|20\d{2})\b', raw)
    # Filter out years that are actually ticker codes
    years = [y for y in years if y not in ticker_numbers]

    if years:
        return f"{min(years)}\u2013{max(years)}" if len(set(years)) > 1 else years[0]

    quarters = re.findall(r'\b(Q[1-4]\s*\d{4}|\d{4}\s*Q[1-4])\b', raw, re.I)
    if quarters:
        return quarters[0]

    if any(w in raw.lower() for w in ["this year", "今年"]):
        from datetime import datetime
        return str(datetime.now().year)
    if any(w in raw.lower() for w in ["last year", "去年"]):
        from datetime import datetime
        return str(datetime.now().year - 1)
    return ""


def _extract_market(raw: str, context: str,
                    pre_numeric: list[str] | None = None) -> str:
    """
    Extract the market or index being discussed.
    When numeric HK tickers are detected, 'HK' in the query is treated as
    an exchange suffix (e.g. 2015.HK), NOT as 'Hong Kong market'.
    """
    r = raw.lower()

    # If the only 'hk' present is as an exchange suffix for numeric tickers,
    # don't claim the whole query is about the HK market index.
    hk_is_exchange_only = bool(pre_numeric) and all(
        t.endswith(".HK") for t in (pre_numeric or [])
    ) and "hong kong" not in r and "hang seng" not in r and "hsi" not in r

    if not hk_is_exchange_only:
        if any(w in r for w in ["hk market", "hong kong", "港股", "恒指",
                                 "hang seng", "hsi", "hk stock market"]):
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
