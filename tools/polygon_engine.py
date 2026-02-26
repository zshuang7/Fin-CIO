"""
tools/polygon_engine.py — Polygon.io market data + news.

Used primarily for:
  - High-quality, up-to-date media/news coverage for tickers (US-focused).

API key:
  - POLYGON_API_KEY (set in .env or Streamlit secrets)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import requests

from tools.base_tool import BaseTool

logger = logging.getLogger("polygon_engine")

_CACHE_DIR = Path(tempfile.gettempdir()) / "finagent_polygon_cache"
_CACHE_TTL_MIN = 15


class PolygonEngine(BaseTool):
    tool_name = "polygon_engine"
    tool_description = (
        "Fetches up-to-date market news from Polygon.io. Best for US tickers; "
        "returns structured JSON for downstream synthesis."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        self.api_key = os.getenv("POLYGON_API_KEY", "").strip()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.register(self.get_media_news)
        self.register(self.get_tier1_media_news)

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace(".", "_")
        return _CACHE_DIR / f"{safe}.json"

    def _read_cache(self, key: str):
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            at = datetime.fromisoformat(raw.get("_at", "1970-01-01"))
            if datetime.now() - at < timedelta(minutes=_CACHE_TTL_MIN):
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

    def _get(self, url: str, params: dict) -> dict | None:
        if not self.api_key:
            return None
        cache_key = f"{url}?{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        try:
            p = dict(params)
            p["apiKey"] = self.api_key
            resp = requests.get(url, params=p, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            self._write_cache(cache_key, data)
            return data
        except Exception as e:
            logger.debug("[polygon_engine] request failed: %s", e)
            return None

    @staticmethod
    def _norm_source(publisher: dict | None) -> str:
        if not publisher:
            return "Unknown"
        return publisher.get("name") or publisher.get("homepage_url") or "Unknown"

    @staticmethod
    def _is_tier1(source: str, url: str) -> bool:
        s = (source or "").lower()
        u = (url or "").lower()
        tier1 = (
            "reuters", "bloomberg", "wall street journal", "wsj",
            "cnbc", "yahoo", "financial times", "ft.com",
        )
        return any(t in s for t in tier1) or any(t in u for t in ("reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "finance.yahoo.com", "ft.com"))

    def get_media_news(self, ticker: str, limit: int = 10, days: int = 30) -> str:
        """
        Fetch recent news for a ticker from Polygon.

        Returns JSON string:
          { "ticker": "...", "retrieved_at": "...", "items": [ {date, source, title, url, summary} ... ] }
        """
        t = (ticker or "").upper().strip()
        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        if not self.api_key:
            return json.dumps(
                {
                    "ticker": t,
                    "retrieved_at": retrieved_at,
                    "source": "Polygon",
                    "error": "POLYGON_API_KEY is not set.",
                    "items": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        from_date = (datetime.utcnow() - timedelta(days=int(days))).strftime("%Y-%m-%d")
        data = self._get(
            "https://api.polygon.io/v2/reference/news",
            {
                "ticker": t,
                "limit": int(limit),
                "published_utc.gte": from_date,
                "order": "desc",
                "sort": "published_utc",
            },
        )

        items: list[dict] = []
        if isinstance(data, dict):
            for r in (data.get("results") or [])[: int(limit)]:
                date = (r.get("published_utc") or "")[:10]
                items.append(
                    {
                        "date": date,
                        "source": self._norm_source(r.get("publisher")),
                        "title": (r.get("title") or "")[:220],
                        "url": r.get("article_url") or "",
                        "summary": (r.get("description") or "")[:240],
                    }
                )

        payload = {
            "ticker": t,
            "retrieved_at": retrieved_at,
            "source": "Polygon",
            "window_days": int(days),
            "items": items,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_tier1_media_news(self, ticker: str, limit: int = 8, days: int = 30) -> str:
        """
        Same as get_media_news but filters to Tier-1 outlets when possible.
        Returns JSON. If no Tier-1 items found, returns best-effort unfiltered list with a warning.
        """
        raw = self.get_media_news(ticker, limit=max(int(limit), 10), days=days)
        try:
            data = json.loads(raw)
        except Exception:
            return raw
        items = data.get("items") or []
        tier1_items = [i for i in items if self._is_tier1(i.get("source", ""), i.get("url", ""))]
        if tier1_items:
            data["items"] = tier1_items[: int(limit)]
            data["tier1_only"] = True
            return json.dumps(data, ensure_ascii=False, indent=2)
        data["tier1_only"] = False
        data["warning"] = "No Tier-1 sources found in Polygon results for this query window."
        data["items"] = items[: int(limit)]
        return json.dumps(data, ensure_ascii=False, indent=2)

