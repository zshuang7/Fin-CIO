"""
tools/fmp_engine.py — Financial Modeling Prep (FMP) consensus engine.

Primary purpose:
  - Provide a quantitative "Wall Street & Expert Consensus" vote for any symbol.
  - Equity: use analyst recommendation / grades-consensus counts.
  - Non-equity (crypto/forex): fallback to social sentiment (if available).

Notes:
  - Uses simple JSON caching (1 hour) to reduce API usage.
  - Avoids hardcoding API keys (public repo safety). Set FMP_API_KEY in .env / Streamlit secrets.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from tools.base_tool import BaseTool

logger = logging.getLogger("fmp_engine")

_CACHE_DIR = Path(tempfile.gettempdir()) / "finagent_fmp_cache"
_CACHE_TTL_HOURS = 1


class FmpEngine(BaseTool):
    tool_name = "FmpEngine"
    tool_description = (
        "Fetches analyst consensus (Bullish/Neutral/Bearish vote) from Financial Modeling Prep (FMP). "
        "For equities uses grades-consensus; for non-equities falls back to social sentiment where available."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        self._api_key = os.getenv("FMP_API_KEY", "").strip()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.register(self.get_consensus_data)
        self.register(self.format_consensus_vote)

    # ── Cache helpers ─────────────────────────────────────────────────────────
    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace(".", "_")
        return _CACHE_DIR / f"{safe}.json"

    def _read_cache(self, key: str):
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(raw.get("_at", "1970-01-01"))
            if datetime.now() - cached_at < timedelta(hours=_CACHE_TTL_HOURS):
                return raw.get("data")
        except Exception:
            return None
        return None

    def _write_cache(self, key: str, data) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps({"_at": datetime.now().isoformat(), "data": data}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── HTTP helpers ─────────────────────────────────────────────────────────
    def _get(self, url: str, params: dict) -> object | None:
        if not self._api_key:
            return None
        cache_key = f"{url}?{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        try:
            p = dict(params)
            p["apikey"] = self._api_key
            resp = requests.get(url, params=p, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            self._write_cache(cache_key, data)
            return data
        except Exception as e:
            logger.debug("[fmp_engine] GET failed %s: %s", url, e)
            return None

    # ── Parsing helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _coerce_int(v) -> int:
        try:
            if v is None or v == "":
                return 0
            return int(float(v))
        except Exception:
            return 0

    @staticmethod
    def _looks_like_crypto_or_fx(symbol: str) -> bool:
        s = (symbol or "").upper().strip()
        # Common FMP formats: BTCUSD, ETHUSD, EURUSD, USDJPY, XAUUSD, etc.
        if "/" in s:
            return True
        if len(s) == 6 and s.isalpha():  # e.g. EURUSD
            return True
        if s.endswith("USD") and len(s) in (6, 7, 8, 9):  # BTCUSD, DOGEUSD
            return True
        return False

    @staticmethod
    def _extract_counts(obj: object) -> dict:
        """
        Try hard to extract recommendation counts from unknown FMP schema.
        Expected keys (any casing):
          strongBuy, buy, hold, sell, strongSell
        Returns normalised dict with those five.
        """
        rec = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
        if obj is None:
            return rec
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            return rec

        # Normalise casing
        lower_map = {str(k).lower(): v for k, v in obj.items()}

        def get_any(*names: str) -> int:
            for n in names:
                if n in lower_map:
                    return FmpEngine._coerce_int(lower_map[n])
            return 0

        rec["strongBuy"] = get_any("strongbuy", "strong_buy", "strong buy")
        rec["buy"] = get_any("buy")
        rec["hold"] = get_any("hold", "neutral")
        rec["sell"] = get_any("sell")
        rec["strongSell"] = get_any("strongsell", "strong_sell", "strong sell")
        return rec

    @staticmethod
    def _vote_from_counts(counts: dict) -> dict:
        sb = int(counts.get("strongBuy") or 0)
        b = int(counts.get("buy") or 0)
        h = int(counts.get("hold") or 0)
        s = int(counts.get("sell") or 0)
        ss = int(counts.get("strongSell") or 0)
        bull = sb + b
        neu = h
        bear = s + ss
        total = bull + neu + bear
        if total <= 0:
            return {
                "bullish_pct": None,
                "neutral_pct": None,
                "bearish_pct": None,
                "total_votes": 0,
                "bucket_counts": {"bullish": bull, "neutral": neu, "bearish": bear},
            }
        return {
            "bullish_pct": round(100 * bull / total, 1),
            "neutral_pct": round(100 * neu / total, 1),
            "bearish_pct": round(100 * bear / total, 1),
            "total_votes": total,
            "bucket_counts": {"bullish": bull, "neutral": neu, "bearish": bear},
        }

    @staticmethod
    def _market_sentiment_label(vote: dict) -> str:
        b = vote.get("bullish_pct")
        n = vote.get("neutral_pct")
        r = vote.get("bearish_pct")
        if b is None or n is None or r is None:
            return "N/A"
        if b >= 65:
            return "Strong Buy / Overweight"
        if b >= 50:
            return "Overweight / Constructive"
        if r >= 50:
            return "Underweight / Cautious"
        if n >= 45:
            return "Neutral / Wait-and-see"
        return "Mixed / Split"

    # ── Public tool ───────────────────────────────────────────────────────────
    def get_consensus_data(self, symbol: str) -> str:
        """
        Fetches consensus recommendation counts and maps them into a 3-tier vote:
          Bullish = Strong Buy + Buy
          Neutral = Hold
          Bearish = Sell + Strong Sell

        For non-equity (crypto/forex), attempts to derive the vote from social sentiment.
        Returns JSON string (for WallStreetAgent to render cleanly).
        """
        sym = (symbol or "").upper().strip()
        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        if not self._api_key:
            payload = {
                "symbol": sym,
                "retrieved_at": retrieved_at,
                "source": "FMP",
                "error": "FMP_API_KEY is not set. Add it to .env or Streamlit secrets as FMP_API_KEY.",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        # ── Equity path ────────────────────────────────────────────────────
        is_non_equity = self._looks_like_crypto_or_fx(sym)
        if not is_non_equity:
            # Preferred stable endpoint
            data = self._get(
                "https://financialmodelingprep.com/stable/grades-consensus",
                {"symbol": sym},
            )
            counts = self._extract_counts(data)

            # Fallback legacy endpoint (if stable schema changes or returns empty)
            if sum(counts.values()) == 0:
                data_legacy = self._get(
                    f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{sym}",
                    {},
                )
                counts = self._extract_counts(data_legacy)
                data = data_legacy or data

            vote = self._vote_from_counts(counts)
            payload = {
                "symbol": sym,
                "asset_type": "equity",
                "retrieved_at": retrieved_at,
                "source": "FMP grades-consensus / analyst-stock-recommendations",
                "raw_counts": counts,
                "vote": vote,
                "market_sentiment": self._market_sentiment_label(vote),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        # ── Non-equity path (crypto/forex) ─────────────────────────────────
        ss = self._get(
            "https://financialmodelingprep.com/api/v4/historical/social-sentiment",
            {"symbol": sym, "page": 0},
        )
        # Try to compute vote from latest record if present
        latest = None
        if isinstance(ss, list) and ss:
            latest = ss[0]
        elif isinstance(ss, dict):
            latest = ss

        # Extract any usable positive/neutral/negative fields
        pos = neu = neg = None
        if isinstance(latest, dict):
            lower_map = {str(k).lower(): v for k, v in latest.items()}
            for key in ("positive", "positivescore", "positive_score", "bullish"):
                if key in lower_map:
                    pos = float(lower_map[key] or 0)
                    break
            for key in ("neutral", "neutralscore", "neutral_score"):
                if key in lower_map:
                    neu = float(lower_map[key] or 0)
                    break
            for key in ("negative", "negativescore", "negative_score", "bearish"):
                if key in lower_map:
                    neg = float(lower_map[key] or 0)
                    break

        payload = {
            "symbol": sym,
            "asset_type": "non_equity",
            "retrieved_at": retrieved_at,
            "source": "FMP social sentiment (fallback)",
            "raw_latest": latest,
        }
        if pos is None and neu is None and neg is None:
            payload.update(
                {
                    "vote": {
                        "bullish_pct": None,
                        "neutral_pct": None,
                        "bearish_pct": None,
                        "total_votes": 0,
                        "bucket_counts": {"bullish": 0, "neutral": 0, "bearish": 0},
                    },
                    "market_sentiment": "N/A",
                    "warning": "No usable social-sentiment fields returned by FMP for this symbol.",
                }
            )
            return json.dumps(payload, ensure_ascii=False, indent=2)

        total = float(pos or 0) + float(neu or 0) + float(neg or 0)
        if total <= 0:
            total = 1.0
        vote = {
            "bullish_pct": round(100 * float(pos or 0) / total, 1),
            "neutral_pct": round(100 * float(neu or 0) / total, 1),
            "bearish_pct": round(100 * float(neg or 0) / total, 1),
            "total_votes": None,
            "bucket_counts": {"bullish": pos, "neutral": neu, "bearish": neg},
        }
        payload.update({"vote": vote, "market_sentiment": self._market_sentiment_label(vote)})
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ── Rendering helper (deterministic, prevents LLM hallucinations) ──────────
    @staticmethod
    def _bar(pct: float | None, width: int = 12) -> str:
        if pct is None:
            return "░" * width
        filled = int(round(width * (pct / 100.0)))
        filled = max(0, min(width, filled))
        return ("█" * filled) + ("░" * (width - filled))

    # ── Analytical fallback: derive consensus from yfinance + EODHD when FMP is empty ──

    def _derive_consensus_from_yfinance(self, symbol: str) -> dict | None:
        """Try yfinance recommendationKey + targetMeanPrice as a rough consensus proxy."""
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info or {}
            rec = (info.get("recommendationKey") or "").lower()
            n_analysts = info.get("numberOfAnalystOpinions") or 0
            if not rec or rec == "none":
                return None
            mapping = {
                "strong_buy": {"bullish_pct": 90, "neutral_pct": 8, "bearish_pct": 2},
                "buy":        {"bullish_pct": 75, "neutral_pct": 20, "bearish_pct": 5},
                "hold":       {"bullish_pct": 25, "neutral_pct": 55, "bearish_pct": 20},
                "sell":       {"bullish_pct": 5, "neutral_pct": 20, "bearish_pct": 75},
                "strong_sell":{"bullish_pct": 2, "neutral_pct": 8, "bearish_pct": 90},
            }
            vote = mapping.get(rec, {"bullish_pct": 33, "neutral_pct": 34, "bearish_pct": 33})
            vote["total_votes"] = n_analysts or None
            vote["bucket_counts"] = None
            return {
                "vote": vote,
                "market_sentiment": self._market_sentiment_label(vote),
                "source": f"yfinance (recommendation: {rec.upper()}, ~{n_analysts} analysts)",
                "approximate": True,
            }
        except Exception as exc:
            logger.debug("[fmp_engine] yfinance fallback failed for %s: %s", symbol, exc)
            return None

    def _derive_consensus_from_eodhd(self, symbol: str) -> dict | None:
        """Try EODHD AnalystRatings distribution as a consensus proxy."""
        try:
            from tools.eodhd_engine import EODHDEngine
            eng = EODHDEngine()
            raw = eng.get_wall_street_signals(symbol)
            data = json.loads(raw) if raw else {}
            consensus = data.get("consensus") or {}
            dist = consensus.get("distribution") or {}
            total = consensus.get("analyst_count") or 0
            if total <= 0:
                return None
            bull = (dist.get("strong_buy") or 0) + (dist.get("buy") or 0)
            neu = dist.get("hold") or 0
            bear = (dist.get("sell") or 0) + (dist.get("strong_sell") or 0)
            vote = {
                "bullish_pct": round(100 * bull / total, 1),
                "neutral_pct": round(100 * neu / total, 1),
                "bearish_pct": round(100 * bear / total, 1),
                "total_votes": total,
                "bucket_counts": {"bullish": bull, "neutral": neu, "bearish": bear},
            }
            return {
                "vote": vote,
                "market_sentiment": self._market_sentiment_label(vote),
                "source": f"EODHD AnalystRatings ({total} analysts)",
                "approximate": False,
            }
        except Exception as exc:
            logger.debug("[fmp_engine] EODHD fallback failed for %s: %s", symbol, exc)
            return None

    def _derive_consensus_from_news_sentiment(self, symbol: str) -> dict | None:
        """
        Last-resort analytical method: fetch recent headlines from multiple
        news APIs and classify sentiment to build an approximate consensus.
        """
        try:
            positive = neutral = negative = 0
            sources_used: list[str] = []

            # Try Polygon
            try:
                from tools.polygon_engine import PolygonEngine
                poly = PolygonEngine()
                raw = poly.get_media_news(symbol, limit=15, days=30)
                items = json.loads(raw).get("items") or []
                if items:
                    sources_used.append("Polygon")
                    for it in items:
                        t = ((it.get("title") or "") + " " + (it.get("summary") or "")).lower()
                        if any(w in t for w in ("upgrade", "buy", "outperform", "bullish", "beat", "strong", "surge", "rally")):
                            positive += 1
                        elif any(w in t for w in ("downgrade", "sell", "underperform", "bearish", "miss", "decline", "crash", "cut")):
                            negative += 1
                        else:
                            neutral += 1
            except Exception:
                pass

            # Try NewsData.io
            try:
                from tools.newsdata_engine import NewsDataEngine
                nd = NewsDataEngine()
                raw = nd.get_latest_news(f"{symbol} stock", size=15, days=30)
                items = json.loads(raw).get("items") or []
                if items:
                    sources_used.append("NewsData.io")
                    for it in items:
                        t = ((it.get("title") or "") + " " + (it.get("summary") or "")).lower()
                        if any(w in t for w in ("upgrade", "buy", "outperform", "bullish", "beat", "strong", "surge", "rally")):
                            positive += 1
                        elif any(w in t for w in ("downgrade", "sell", "underperform", "bearish", "miss", "decline", "crash", "cut")):
                            negative += 1
                        else:
                            neutral += 1
            except Exception:
                pass

            total = positive + neutral + negative
            if total < 3:
                return None
            vote = {
                "bullish_pct": round(100 * positive / total, 1),
                "neutral_pct": round(100 * neutral / total, 1),
                "bearish_pct": round(100 * negative / total, 1),
                "total_votes": total,
                "bucket_counts": {"bullish": positive, "neutral": neutral, "bearish": negative},
            }
            return {
                "vote": vote,
                "market_sentiment": self._market_sentiment_label(vote),
                "source": f"News sentiment analysis ({total} headlines from {', '.join(sources_used)})",
                "approximate": True,
            }
        except Exception as exc:
            logger.debug("[fmp_engine] news-sentiment fallback failed for %s: %s", symbol, exc)
            return None

    def format_consensus_vote(self, symbol: str) -> str:
        """
        Returns a pre-formatted, in-context vote tally (text bars) for the
        Institutional & Expert Consensus module.

        CASCADE LOGIC (tries each source until one returns data):
          1. FMP grades-consensus (direct analyst vote)
          2. EODHD AnalystRatings (structured distribution)
          3. yfinance recommendationKey (approximate mapping)
          4. News headline sentiment analysis (analytical semi-consensus)
        """
        sym = (symbol or "").upper().strip()
        raw = self.get_consensus_data(sym)
        try:
            data = json.loads(raw)
        except Exception:
            data = {"symbol": sym, "error": "Invalid JSON from get_consensus_data"}

        vote = (data.get("vote") or {}) if isinstance(data, dict) else {}
        b = vote.get("bullish_pct")
        n = vote.get("neutral_pct")
        r = vote.get("bearish_pct")
        total_votes = vote.get("total_votes")
        sentiment = data.get("market_sentiment", "N/A") if isinstance(data, dict) else "N/A"
        retrieved_at = data.get("retrieved_at") if isinstance(data, dict) else None
        src = data.get("source", "FMP") if isinstance(data, dict) else "FMP"
        is_approximate = False

        # ── CASCADE: if FMP returned empty vote, try alternatives ──────────
        if b is None or (total_votes is not None and total_votes == 0):
            for fallback_name, fallback_fn in [
                ("EODHD", self._derive_consensus_from_eodhd),
                ("yfinance", self._derive_consensus_from_yfinance),
                ("News Sentiment", self._derive_consensus_from_news_sentiment),
            ]:
                logger.debug("[fmp_engine] FMP empty for %s, trying %s fallback", sym, fallback_name)
                result = fallback_fn(sym)
                if result and result.get("vote"):
                    fb_vote = result["vote"]
                    if fb_vote.get("bullish_pct") is not None:
                        b = fb_vote["bullish_pct"]
                        n = fb_vote["neutral_pct"]
                        r = fb_vote["bearish_pct"]
                        total_votes = fb_vote.get("total_votes")
                        sentiment = result.get("market_sentiment", "N/A")
                        src = result["source"]
                        is_approximate = result.get("approximate", False)
                        break

        def fmt(x):
            return "N/A" if x is None else f"{float(x):.1f}%"

        lines = [
            f"🏛️ Institutional & Expert Consensus: {sym}",
            "[The Consensus Vote]",
            f"🟢 Bullish: {fmt(b)} | 🟡 Neutral: {fmt(n)} | 🔴 Bearish: {fmt(r)}",
        ]
        if b is not None and n is not None and r is not None:
            lines += [
                f"🐂 Bullish  {self._bar(float(b))}  {fmt(b)}",
                f"😐 Neutral  {self._bar(float(n))}  {fmt(n)}",
                f"🐻 Bearish  {self._bar(float(r))}  {fmt(r)}",
            ]
        else:
            lines.append("Vote Tally: N/A (no data from any source)")

        if isinstance(total_votes, int) and total_votes > 0:
            lines.append(f"Total analyst votes: {total_votes}")

        if is_approximate:
            lines.append("⚠️ Note: This is an approximate consensus derived from alternative data sources.")

        lines.append(f"Market Sentiment: {sentiment}")
        if retrieved_at:
            lines.append(f"Source: {src} | Retrieved: {retrieved_at}")
        else:
            lines.append(f"Source: {src}")

        return "\n".join(lines)

