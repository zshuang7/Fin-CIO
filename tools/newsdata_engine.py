"""
tools/newsdata_engine.py — NewsData.io media/news coverage tool.

Used for:
  - Broad global media coverage (WSJ/Bloomberg/Reuters/CNBC/Yahoo where available)
  - Helpful fallback for non-US tickers where Finnhub is restricted.

API key:
  - NEWSDATA_API_KEY (set in .env or Streamlit secrets)
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

logger = logging.getLogger("newsdata_engine")

_CACHE_DIR = Path(tempfile.gettempdir()) / "finagent_newsdata_cache"
_CACHE_TTL_MIN = 15


class NewsDataEngine(BaseTool):
    tool_name = "newsdata_engine"
    tool_description = (
        "Fetches recent business/market news from NewsData.io and returns structured JSON."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        self.api_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.register(self.get_latest_news)

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
            p["apikey"] = self.api_key
            resp = requests.get(url, params=p, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            self._write_cache(cache_key, data)
            return data
        except Exception as e:
            logger.debug("[newsdata_engine] request failed: %s", e)
            return None

    def get_latest_news(self, query: str, size: int = 10, days: int = 30) -> str:
        """
        Search latest business news for a query string.
        Returns JSON string: { query, retrieved_at, items:[{date, source, title, url, summary}] }
        """
        q = (query or "").strip()
        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        if not self.api_key:
            return json.dumps(
                {
                    "query": q,
                    "retrieved_at": retrieved_at,
                    "source": "NewsData.io",
                    "error": "NEWSDATA_API_KEY is not set.",
                    "items": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        # NewsData supports time filter via 'from_date' (YYYY-MM-DD) in some plans;
        # if unsupported, the API will ignore it — still ok.
        from_date = (datetime.utcnow() - timedelta(days=int(days))).strftime("%Y-%m-%d")
        data = self._get(
            "https://newsdata.io/api/1/news",
            {
                "q": q,
                "language": "en",
                "category": "business",
                "size": int(size),
                "from_date": from_date,
            },
        )

        items: list[dict] = []
        if isinstance(data, dict):
            for r in (data.get("results") or [])[: int(size)]:
                date = (r.get("pubDate") or r.get("publishedAt") or "")[:10]
                items.append(
                    {
                        "date": date,
                        "source": r.get("source_id") or r.get("source") or "Unknown",
                        "title": (r.get("title") or "")[:220],
                        "url": r.get("link") or r.get("url") or "",
                        "summary": (r.get("description") or r.get("content") or "")[:240],
                    }
                )

        payload = {
            "query": q,
            "retrieved_at": retrieved_at,
            "source": "NewsData.io",
            "window_days": int(days),
            "items": items,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

