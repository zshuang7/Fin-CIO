"""
dspy_router.py — Deterministic query router and parallel agent data gatherer.

Replaces the Agno Team coordinator's LLM-based routing with a fast Python
function, then runs selected Agno agents in parallel to collect data for
the DSPy CIO synthesis layer.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ── Query type classification ────────────────────────────────────────────────

_COMPARE_KW = re.compile(r"\b(compare|vs\.?|versus)\b", re.I)
_CONCEPT_STARTS = (
    "what is", "what are", "what's a", "explain ", "define ",
    "how does", "how do ", "what does",
)


def classify_query(query: str, has_ticker: bool) -> str:
    """Classify the query into a routing category.

    Returns one of: STOCK_ANALYSIS, COMPARISON, MARKET_ANALYSIS,
                    CONCEPT, REPORT_REQUEST
    """
    q = query.lower().strip()

    if any(kw in q for kw in ("generate report", "save report", "export", "pdf", "excel")):
        return "REPORT_REQUEST"

    if _COMPARE_KW.search(q) and has_ticker:
        return "COMPARISON"

    if has_ticker:
        return "STOCK_ANALYSIS"

    if any(q.startswith(s) for s in _CONCEPT_STARTS) or len(q.split()) <= 5:
        return "CONCEPT"

    return "MARKET_ANALYSIS"


# ── Route: map (level, query_type) → agent names to call ────────────────────

def route(level: str, query_type: str, has_ticker: bool) -> list[str]:
    """Return an ordered list of agent names to call.

    The names match keys in the AGENTS dict exported by team_config.py.
    """
    if query_type == "CONCEPT":
        return []

    if query_type == "REPORT_REQUEST":
        return []

    if level == "fast":
        return ["CompanyAgent"] if has_ticker else []

    if query_type == "COMPARISON":
        return ["CompanyAgent", "WallStreetAgent", "NewsAgent"]

    if level == "standard":
        return ["CompanyAgent", "WallStreetAgent", "NewsAgent"]

    # master / deep_dive
    return ["MacroAgent", "CompanyAgent", "WallStreetAgent", "NewsAgent"]


# ── Parallel agent data gathering ────────────────────────────────────────────

def _run_agent(agent, query: str) -> str:
    """Run a single Agno agent and extract its text response."""
    try:
        resp = agent.run(query)
        if hasattr(resp, "content") and resp.content:
            return str(resp.content)
        return str(resp)
    except Exception as e:
        logger.error("Agent %s failed: %s", getattr(agent, "name", "?"), e)
        return ""


def gather_agent_data(
    query: str,
    agent_names: list[str],
    agents_dict: dict,
    on_agent_start=None,
    on_agent_done=None,
    is_comparison: bool = False,
    tickers: list[str] | None = None,
) -> dict[str, str]:
    """Run selected Agno agents in parallel and collect their text outputs.

    Args:
        query: The user's question.
        agent_names: List of agent name strings (e.g. ["CompanyAgent", "NewsAgent"]).
        agents_dict: The AGENTS dict from team_config.py mapping name -> Agent.
        on_agent_start: Optional callback(name: str) when an agent starts.
        on_agent_done: Optional callback(name: str) when an agent finishes.
        is_comparison: If True, runs CompanyAgent/WallStreetAgent/NewsAgent
                       separately for each ticker.
        tickers: List of ticker symbols for comparison queries.

    Returns:
        Dict with keys like 'company_data', 'wall_street_data', 'news_data',
        'macro_data', or for comparisons: 'ticker_a_data', 'ticker_b_data', etc.
    """
    results: dict[str, str] = {}

    if is_comparison and tickers and len(tickers) >= 2:
        return _gather_comparison(query, agent_names, agents_dict, tickers,
                                  on_agent_start, on_agent_done)

    agent_key_map = {
        "MacroAgent": "macro_data",
        "CompanyAgent": "company_data",
        "WallStreetAgent": "wall_street_data",
        "NewsAgent": "news_data",
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in agent_names:
            agent = agents_dict.get(name)
            if agent is None:
                logger.warning("Agent %s not found in agents_dict", name)
                continue
            if on_agent_start:
                on_agent_start(name)
            fut = pool.submit(_run_agent, agent, query)
            futures[fut] = name

        for fut in as_completed(futures):
            name = futures[fut]
            key = agent_key_map.get(name, name.lower())
            try:
                results[key] = fut.result()
            except Exception as e:
                logger.error("Agent %s raised: %s", name, e)
                results[key] = ""
            if on_agent_done:
                on_agent_done(name)

    return results


def _gather_comparison(
    query: str,
    agent_names: list[str],
    agents_dict: dict,
    tickers: list[str],
    on_agent_start=None,
    on_agent_done=None,
) -> dict[str, str]:
    """Run agents separately for each ticker in a comparison query."""
    ta, tb = tickers[0], tickers[1]
    results: dict[str, str] = {}

    per_ticker_agents = ["CompanyAgent", "WallStreetAgent", "NewsAgent"]
    shared_agents = [n for n in agent_names if n not in per_ticker_agents]

    futures = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name in per_ticker_agents:
            if name not in agent_names:
                continue
            agent = agents_dict.get(name)
            if agent is None:
                continue
            for ticker, suffix in [(ta, "a"), (tb, "b")]:
                label = f"{name}_{suffix}"
                ticker_query = f"{query}\n\nFocus specifically on: {ticker}"
                if on_agent_start:
                    on_agent_start(f"{name} ({ticker})")
                fut = pool.submit(_run_agent, agent, ticker_query)
                futures[fut] = (name, suffix)

        for name in shared_agents:
            agent = agents_dict.get(name)
            if agent is None:
                continue
            if on_agent_start:
                on_agent_start(name)
            fut = pool.submit(_run_agent, agent, query)
            futures[fut] = (name, "shared")

        for fut in as_completed(futures):
            name, suffix = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                logger.error("Agent %s (%s) raised: %s", name, suffix, e)
                text = ""

            key_map = {
                "CompanyAgent": "ticker_{}_data",
                "WallStreetAgent": "ws_{}",
                "NewsAgent": "news_{}",
                "MacroAgent": "macro_data",
            }
            if suffix == "shared":
                results[key_map.get(name, name.lower()).format("")] = text
            else:
                results[key_map.get(name, name.lower()).format(suffix)] = text

            if on_agent_done:
                on_agent_done(f"{name}" if suffix == "shared" else f"{name} ({suffix})")

    return results
