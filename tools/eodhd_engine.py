"""
tools/eodhd_engine.py — EODHD Financial Data Engine

Provides Wall Street analyst intelligence:
  • Analyst ratings, consensus, and price targets  (EODHD Fundamentals API)
  • Financial news filtered for analyst mentions   (EODHD News API)
  • Full "Mosaic" Wall Street breakdown            (combined)

Design decisions:
  ─ Rate limiting  : 1.1 s gap between calls (EODHD free tier)
  ─ Caching        : JSON files in OS temp dir, 1-hour TTL
                     prevents re-burning daily API quota on repeat queries
  ─ Ticker normalization: Yahoo Finance format (TSLA, 8306.T) → EODHD format
  ─ Graceful degradation: returns a human-readable message if the free tier
                          blocks an endpoint (402/403)

════════════════════════════════════════════════════════════════════
Adding new EODHD endpoints:
  1. Add a method below that calls self._get(endpoint, params, ticker)
  2. Register it in __init__ with  self.register(self.new_method)
════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import requests

from tools.base_tool import BaseTool

logger = logging.getLogger("eodhd_engine")

# ── Cache config ──────────────────────────────────────────────────────────────
_CACHE_DIR = Path(tempfile.gettempdir()) / "finagent_eodhd_cache"
_CACHE_TTL_HOURS = 1     # re-use cached data for up to 1 hour

# ── Exchange suffix mapping (Yahoo Finance → EODHD) ───────────────────────────
_EXCHANGE_MAP = {
    ".T":   ".TSE",    # Tokyo
    ".L":   ".LSE",    # London
    ".HK":  ".HK",     # Hong Kong (same)
    ".PA":  ".PA",     # Paris
    ".DE":  ".XETRA",  # Frankfurt / XETRA
    ".AX":  ".AU",     # Australia
    ".TO":  ".TO",     # Toronto
    ".SS":  ".SHG",    # Shanghai
    ".SZ":  ".SHE",    # Shenzhen
}

# Major broker / analyst names used to identify research mentions
_ANALYST_KEYWORDS = [
    "analyst", "rating", "price target", "upgrade", "downgrade",
    "overweight", "underweight", "outperform", "underperform",
    "buy", "sell", "neutral", "hold",
    "Goldman", "Morgan Stanley", "JPMorgan", "Citi", "UBS",
    "Barclays", "Bank of America", "BofA", "Wells Fargo",
    "Deutsche Bank", "Jefferies", "Needham", "Wedbush", "RBC",
    "TD Cowen", "Piper Sandler", "Baird", "Bernstein", "HSBC",
    "Evercore", "Oppenheimer", "Raymond James", "Truist",
    "PT ", "target price", "EPS estimate", "coverage", "initiates",
    "reiterates", "raises", "lowers", "maintains",
]

# Banks / brokers for vote extraction (title-based heuristic)
_BANK_NAMES = [
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "JP Morgan", "J.P. Morgan",
    "Citi", "Citigroup", "UBS", "Barclays", "Bank of America", "BofA",
    "Wells Fargo", "Deutsche Bank", "Jefferies", "Needham", "Wedbush", "RBC",
    "TD Cowen", "Piper Sandler", "Baird", "Bernstein", "HSBC", "Evercore",
    "Oppenheimer", "Raymond James", "Truist", "DA Davidson", "D.A. Davidson",
    "Mizuho", "Nomura", "Macquarie", "Credit Suisse", "BNP Paribas",
]


class EODHDEngine(BaseTool):
    """
    Wall Street intelligence engine backed by EODHD.
    All public methods are registered as Agno tools.
    """
    tool_name = "EODHDEngine"
    tool_description = (
        "Fetches Wall Street analyst ratings, price targets, EPS estimates, "
        "and broker research mentions from EODHD. Best for institutional "
        "consensus and analyst upgrade/downgrade signals."
    )

    # class-level rate-limit state shared across all instances
    _last_call_ts: float = 0.0

    def __init__(self):
        super().__init__(name=self.tool_name)
        self._api_key = os.getenv("EODHD_API_KEY", "699efdd4dfa1a0.46374025")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.register(self.get_analyst_ratings)
        self.register(self.get_analyst_news)
        self.register(self.get_wall_street_breakdown)
        self.register(self.get_wall_street_signals)

    # ── Ticker normalisation ──────────────────────────────────────────────────

    def _eodhd_ticker(self, ticker: str) -> str:
        """
        Convert Yahoo Finance ticker format to EODHD format.
        Plain US tickers (TSLA, NVDA) get the '.US' suffix required by
        the fundamentals endpoint.
        """
        t = ticker.upper().strip()
        # Already has an exchange suffix
        for yf_sfx, eodhd_sfx in _EXCHANGE_MAP.items():
            if t.endswith(yf_sfx):
                return t[: -len(yf_sfx)] + eodhd_sfx
        # Plain US ticker (no dot)
        if "." not in t:
            return f"{t}.US"
        return t

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(".", "_")
        return _CACHE_DIR / f"{safe}.json"

    def _read_cache(self, key: str):
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(raw.get("_at", "1970-01-01"))
            if datetime.now() - cached_at < timedelta(hours=_CACHE_TTL_HOURS):
                logger.debug("[eodhd_engine] cache hit: %s", key)
                return raw.get("data")
        except Exception:
            pass
        return None

    def _write_cache(self, key: str, data) -> None:
        try:
            path = self._cache_path(key)
            path.write_text(
                json.dumps({"_at": datetime.now().isoformat(), "data": data},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── Rate limiter ──────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Enforce ≥ 1.1 s between EODHD API calls (free tier requirement)."""
        elapsed = time.time() - EODHDEngine._last_call_ts
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        EODHDEngine._last_call_ts = time.time()

    # ── Core HTTP request ─────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict | None = None,
             cache_key: str = "") -> dict | list | None:
        """
        GET request to EODHD with cache + rate limiting.
        Returns parsed JSON or None on error.
        """
        key = cache_key or endpoint
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        self._throttle()
        p = dict(params or {})
        p["api_token"] = self._api_key
        p["fmt"] = "json"

        url = f"https://eodhd.com/api/{endpoint}"
        try:
            resp = requests.get(url, params=p, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            self._write_cache(key, data)
            return data
        except requests.exceptions.HTTPError as e:
            code = str(e)
            if "429" in code:
                logger.warning("[eodhd_engine] Rate limit (429) hit for %s", endpoint)
            elif "402" in code or "403" in code:
                logger.debug("[eodhd_engine] Endpoint %s not available on free tier.", endpoint)
            else:
                logger.error("[eodhd_engine] HTTP error on %s: %s", endpoint, e)
            return None
        except Exception as e:
            logger.error("[eodhd_engine] %s: %s", endpoint, e)
            return None

    # ── Public tools ──────────────────────────────────────────────────────────

    def get_analyst_ratings(self, ticker: str) -> str:
        """
        Fetch Wall Street analyst consensus, average price target, and
        Buy/Hold/Sell distribution for a ticker from EODHD.

        Args:
            ticker: Stock ticker (e.g. TSLA, NVDA, 0700.HK).
        """
        eodhd_t = self._eodhd_ticker(ticker)
        data = self._get(
            f"fundamentals/{eodhd_t}",
            {"filter": "Highlights,AnalystRatings,Valuation,Earnings"},
            cache_key=f"fundamentals_{eodhd_t}",
        )

        if not data or not isinstance(data, dict):
            return (
                f"Analyst ratings for {ticker} not available via EODHD "
                f"(may require paid tier or ticker not found)."
            )

        lines = [f"## Wall Street Analyst Ratings — {ticker} (EODHD)\n"]

        # ── Analyst consensus ──────────────────────────────────────────────
        ar = data.get("AnalystRatings") or {}
        if ar:
            rating = ar.get("Rating", "N/A")
            target = ar.get("TargetPrice", "N/A")
            sb  = ar.get("StrongBuy", 0)
            b   = ar.get("Buy", 0)
            h   = ar.get("Hold", 0)
            s   = ar.get("Sell", 0)
            ss  = ar.get("StrongSell", 0)
            total = (sb or 0) + (b or 0) + (h or 0) + (s or 0) + (ss or 0)
            lines += [
                "### Consensus",
                f"  Avg Rating   : {rating}  (1=Strong Buy … 5=Strong Sell)",
                f"  Avg Target   : ${target}",
                f"  # Analysts   : {total}",
                f"  Distribution : StrBuy={sb}  Buy={b}  Hold={h}  Sell={s}  StrSell={ss}",
            ]

        # ── Key highlights ────────────────────────────────────────────────
        h_data = data.get("Highlights") or {}
        if h_data:
            lines.append("\n### Key Highlights")
            highlight_fields = [
                ("MarketCapitalization", "Market Cap"),
                ("EBITDA", "EBITDA"),
                ("PERatio", "P/E Ratio"),
                ("PEGRatio", "PEG Ratio"),
                ("EPS", "EPS (TTM)"),
                ("EPSEstimateCurrentYear", "EPS Est. (CY)"),
                ("EPSEstimateNextYear", "EPS Est. (NY)"),
                ("RevenuePerShareTTM", "Revenue/Share TTM"),
                ("ProfitMargin", "Profit Margin"),
                ("ReturnOnEquityTTM", "ROE TTM"),
                ("52WeekHigh", "52-Week High"),
                ("52WeekLow", "52-Week Low"),
                ("50DayMA", "50-Day MA"),
                ("200DayMA", "200-Day MA"),
            ]
            for key, label in highlight_fields:
                val = h_data.get(key)
                if val not in (None, "None", "", 0):
                    lines.append(f"  {label:<26}: {val}")

        # ── Valuation multiples ───────────────────────────────────────────
        v_data = data.get("Valuation") or {}
        if v_data:
            lines.append("\n### Valuation Multiples")
            for k, v in v_data.items():
                if v not in (None, "None", "", 0):
                    lines.append(f"  {k:<26}: {v}")

        lines.append(f"\n  Source: EODHD   |   Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("  Confidence: HIGH (structured institutional data)")
        return "\n".join(lines)

    def get_analyst_news(self, ticker: str, limit: int = 12) -> str:
        """
        Fetch the latest financial news for a ticker, highlighting
        analyst-related articles (upgrades, downgrades, price target changes).

        Args:
            ticker: Stock ticker.
            limit:  Max number of articles to return (default 12).
        """
        eodhd_t = self._eodhd_ticker(ticker)
        from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        data = self._get(
            "news",
            {"s": eodhd_t, "limit": limit, "from": from_date},
            cache_key=f"news_{eodhd_t}_{from_date}",
        )

        if not data or not isinstance(data, list):
            return f"No recent news found for {ticker} via EODHD."

        analyst_hits, general_news = [], []
        for item in data[:limit]:
            title  = item.get("title", "N/A")
            date   = (item.get("date") or "")[:10]
            source = item.get("source", "Unknown")
            url    = item.get("url", "")
            snippet = (item.get("content") or "")[:200].replace("\n", " ")

            is_analyst = any(kw.lower() in title.lower() for kw in _ANALYST_KEYWORDS)
            entry = {
                "title": title, "date": date,
                "source": source, "url": url,
                "snippet": snippet, "is_analyst": is_analyst,
            }
            if is_analyst:
                analyst_hits.append(entry)
            else:
                general_news.append(entry)

        lines = [f"## Financial News — {ticker} (EODHD, last 90 days)\n"]

        if analyst_hits:
            lines.append("### Analyst & Research Mentions")
            for e in analyst_hits[:6]:
                lines.append(f"  🏦 [{e['date']}] {e['source']} — {e['title']}")
                if e["snippet"]:
                    lines.append(f"     › {e['snippet']}...")
                if e["url"]:
                    lines.append(f"     Source: {e['url']}")
            lines.append("")

        if general_news:
            lines.append("### General Market News")
            for e in general_news[:4]:
                lines.append(f"  📰 [{e['date']}] {e['source']} — {e['title']}")

        return "\n".join(lines)

    def get_wall_street_breakdown(self, ticker: str) -> str:
        """
        Full Wall Street intelligence package for a ticker:
          Phase 1 — EODHD structured analyst ratings + price targets
          Phase 2 — EODHD news filtered for broker research mentions
          Phase 3 — Cross-verification note on data consistency

        Returns a structured 'Mosaic' breakdown ready for CIO synthesis.

        Args:
            ticker: Stock ticker (e.g. TSLA, NVDA, 0700.HK).
        """
        ticker = ticker.upper().strip()

        # Phase 1: Structured data
        ratings_text = self.get_analyst_ratings(ticker)

        # Phase 2: News context
        news_text = self.get_analyst_news(ticker, limit=10)

        # Phase 3: Cross-verification + data freshness note
        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        xv_note = (
            "\n## Cross-Verification & Data Freshness\n"
            f"  EODHD data retrieved: {retrieved_at}  (cache TTL = {_CACHE_TTL_HOURS}h)\n"
            "  ⚠  CIO must compare this retrieval date against:\n"
            "     • yfinance financial data date (from get_financial_summary)\n"
            "     • Most recent news date (from NewsAgent)\n"
            "  If financial data is > 6 months older than latest news → flag DATA GAP.\n"
            "  Flag any price-target discrepancy > 15% as a HIGH-PRIORITY data conflict."
        )

        return "\n\n".join([
            f"# Wall Street Intelligence — {ticker}",
            ratings_text,
            news_text,
            xv_note,
        ])

    # ── NEW: Structured "Conviction Board" signals ────────────────────────────

    @staticmethod
    def _guess_stance(text: str) -> str:
        """
        Heuristic stance classifier from a headline/title/snippet.
        Returns one of: 'bullish' | 'neutral' | 'bearish'
        """
        t = (text or "").lower()
        bullish = ("buy", "strong buy", "overweight", "outperform", "upgrade",
                   "raises target", "raise target", "initiates coverage")
        bearish = ("sell", "strong sell", "underweight", "underperform", "downgrade",
                   "cuts target", "cut target")
        neutral = ("hold", "neutral", "market perform", "equal weight", "maintains")

        if any(k in t for k in bearish):
            return "bearish"
        if any(k in t for k in bullish):
            return "bullish"
        if any(k in t for k in neutral):
            return "neutral"
        return "neutral"

    @staticmethod
    def _extract_bank(title: str) -> str | None:
        if not title:
            return None
        tl = title.lower()
        for b in _BANK_NAMES:
            if b.lower() in tl:
                # normalise a few aliases
                if b in ("JP Morgan", "J.P. Morgan"):
                    return "JPMorgan"
                if b in ("DA Davidson", "D.A. Davidson"):
                    return "D.A. Davidson"
                if b == "BofA":
                    return "Bank of America"
                return b
        return None

    def get_wall_street_signals(self, ticker: str, limit: int = 25) -> str:
        """
        Return a JSON string used to build the "🗳️ Wall Street Conviction Board".

        It combines:
          - EODHD structured AnalystRatings (consensus + buy/hold/sell counts)
          - EODHD News (last 180d) filtered for analyst/broker mentions

        Output schema (JSON):
          {
            "ticker": "...",
            "retrieved_at": "YYYY-MM-DD HH:MM",
            "consensus": { "rating": ..., "target_price": ..., "distribution": {...} },
            "bank_votes": [ { "bank": ..., "stance": ..., "date": ..., "source": ..., "title": ..., "url": ... }, ... ],
            "vote_tally": { "bullish": N, "neutral": N, "bearish": N },
            "news_max_date": "YYYY-MM-DD" | null
          }
        """
        ticker = ticker.upper().strip()
        eodhd_t = self._eodhd_ticker(ticker)

        # Structured fundamentals for distribution counts
        fundamentals = self._get(
            f"fundamentals/{eodhd_t}",
            {"filter": "AnalystRatings"},
            cache_key=f"fundamentals_ratings_{eodhd_t}",
        )
        ar = (fundamentals or {}).get("AnalystRatings") if isinstance(fundamentals, dict) else {}
        dist = {
            "strong_buy": int(ar.get("StrongBuy") or 0),
            "buy": int(ar.get("Buy") or 0),
            "hold": int(ar.get("Hold") or 0),
            "sell": int(ar.get("Sell") or 0),
            "strong_sell": int(ar.get("StrongSell") or 0),
        }
        consensus = {
            "rating": ar.get("Rating", None),
            "target_price": ar.get("TargetPrice", None),
            "distribution": dist,
            "analyst_count": sum(dist.values()),
        }

        # News signals (use longer horizon to catch broker notes)
        from_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        news = self._get(
            "news",
            {"s": eodhd_t, "limit": int(limit), "from": from_date},
            cache_key=f"news_ws_{eodhd_t}_{from_date}_{limit}",
        )

        bank_latest: dict[str, dict] = {}
        news_max_date: str | None = None
        if isinstance(news, list):
            for item in news[: int(limit)]:
                title = item.get("title", "") or ""
                bank = self._extract_bank(title)
                if not bank:
                    continue
                date = (item.get("date") or "")[:10] or None
                if date and (news_max_date is None or date > news_max_date):
                    news_max_date = date
                stance = self._guess_stance(title)
                entry = {
                    "bank": bank,
                    "stance": stance,
                    "date": date,
                    "source": item.get("source", None),
                    "title": title[:220],
                    "url": item.get("url", None),
                }
                # Keep the most recent mention per bank
                prev = bank_latest.get(bank)
                if prev is None or (entry["date"] and prev.get("date") and entry["date"] > prev.get("date")):
                    bank_latest[bank] = entry

        votes = list(bank_latest.values())
        tally = {"bullish": 0, "neutral": 0, "bearish": 0}
        for v in votes:
            s = v.get("stance", "neutral")
            if s in tally:
                tally[s] += 1

        payload = {
            "ticker": ticker,
            "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "consensus": consensus,
            "bank_votes": sorted(votes, key=lambda x: (x.get("date") or ""), reverse=True)[:12],
            "vote_tally": tally,
            "news_max_date": news_max_date,
            "confidence": "MEDIUM (headline-based bank extraction; verify primary notes for details)",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
