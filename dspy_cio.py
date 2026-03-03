"""
dspy_cio.py — DSPy-powered CIO synthesis layer.

Sits on top of the existing Agno agent pipeline: Agno agents gather data
via their tools, then this module synthesizes the final CIO response using
DSPy signatures that are compilable and optimizable with gold examples.
"""

import os
import json
import logging
from pathlib import Path

import dspy

logger = logging.getLogger(__name__)

# ── LM configuration ────────────────────────────────────────────────────────

_lm_instance: dspy.LM | None = None


def _get_lm() -> dspy.LM:
    global _lm_instance
    if _lm_instance is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_id = os.getenv("MODEL_ID", "deepseek/deepseek-chat")
        _lm_instance = dspy.LM(model_id, api_key=api_key, max_tokens=4096)
        dspy.configure(lm=_lm_instance)
    return _lm_instance


# ── DSPy Signatures ──────────────────────────────────────────────────────────
# Each level has a different input/output schema so the LLM gets the right
# amount of context and produces the right depth of analysis.

class CIOFast(dspy.Signature):
    """You are a CIO giving a quick, high-level investment take.
    Speak like a smart human investor: clear, concise, opinionated but humble.
    3-6 short paragraphs, minimal metrics, focus on big picture and key drivers.
    End with a simple 'how to think about it' frame.
    NO section headers. NO tables. NO AI chatter."""

    query: str = dspy.InputField(desc="The user's question")
    company_data: str = dspy.InputField(desc="Live fundamentals from CompanyAgent (may be empty)")

    opening_take: str = dspy.OutputField(desc="1-2 sentence big-picture view of what matters")
    analysis: str = dspy.OutputField(desc="2-4 short paragraphs with key drivers, narrative style")
    decision_frame: str = dspy.OutputField(desc="1-2 sentences: who this fits, key trade-off, how to think about it")


class CIOStandard(dspy.Signature):
    """You are a CIO providing a balanced investment analysis (Standard level).
    Narrative-driven: 3-5 sections with headings, 1 small table only if comparing.
    Key metrics woven into narrative (not metric dumps).
    MUST separate Wall Street bank research from media news.
    End with a decision frame, not just a price target.
    Style: conversational, opinionated, evidence-backed. No AI chatter."""

    query: str = dspy.InputField(desc="The user's question")
    company_data: str = dspy.InputField(desc="Fundamentals from CompanyAgent")
    wall_street_data: str = dspy.InputField(desc="Analyst consensus + bank research from WallStreetAgent")
    news_data: str = dspy.InputField(desc="Tier-1 media headlines + sentiment from NewsAgent")

    opening_take: str = dspy.OutputField(desc="1-2 sentence big-picture insight")
    business_and_numbers: str = dspy.OutputField(
        desc="Narrative on business position + key metrics (weave numbers into prose, "
             "small table OK if genuinely helpful). Flag data gaps if financials > 9 months old."
    )
    consensus_section: str = dspy.OutputField(
        desc="## 🏛️ Institutional & Expert Consensus section. "
             "Wall Street banks ONLY (Goldman, JPM, etc.). "
             "Include: consensus vote (🟢/🟡/🔴 percentages), "
             "bank-by-bank research (Tier 1 first), your CIO take on whether you agree."
    )
    media_section: str = dspy.OutputField(
        desc="## 📰 Latest Media News & Sentiment section. "
             "Tier-1 media ONLY (Bloomberg, WSJ, Reuters, CNBC, FT). "
             "Compress headlines into narrative. Sentiment verdict with confidence."
    )
    decision_frame: str = dspy.OutputField(
        desc="How to think about owning this: investor type, key risks woven in, "
             "catalysts, and a clear stance (not just BUY/HOLD/SELL but a reasoning frame)."
    )


class CIOMaster(dspy.Signature):
    """You are a CIO presenting to an investment committee (Master level).
    As Standard PLUS: explicit scenario thinking (base/bull/bear with probabilities),
    analogy to past cycles or peer companies, short decision framework.
    Still narrative-driven — avoid metric spam. Frame as IC presentation.
    MUST separate Wall Street research from media news."""

    query: str = dspy.InputField(desc="The user's question")
    company_data: str = dspy.InputField(desc="Fundamentals from CompanyAgent")
    wall_street_data: str = dspy.InputField(desc="Analyst consensus from WallStreetAgent")
    news_data: str = dspy.InputField(desc="Media coverage from NewsAgent")
    macro_data: str = dspy.InputField(desc="Macro/sector context from MacroAgent")

    opening_take: str = dspy.OutputField(desc="1-2 sentence big-picture framing for the IC")
    business_and_numbers: str = dspy.OutputField(
        desc="Business position, moat, key metrics woven into narrative. "
             "Include peer/cycle analogy if it genuinely clarifies risk/reward."
    )
    consensus_section: str = dspy.OutputField(
        desc="## 🏛️ Institutional & Expert Consensus. Wall Street banks only, "
             "vote + bank-by-bank + your CIO agreement/disagreement."
    )
    media_section: str = dspy.OutputField(
        desc="## 📰 Latest Media News & Sentiment. Media only, narrative, sentiment verdict."
    )
    macro_context: str = dspy.OutputField(
        desc="## 🌐 Macro & Sector Context. Where we are in the cycle, "
             "how it affects this name, regime awareness."
    )
    scenario_analysis: str = dspy.OutputField(
        desc="Scenario thinking: Bull case (probability + thesis), "
             "Base case (probability + thesis), Bear case (probability + thesis)."
    )
    decision_frame: str = dspy.OutputField(
        desc="IC-level decision framework: investor type, position sizing logic, "
             "key risks, catalysts, time horizon, conviction level."
    )


class CIODeepDive(dspy.Signature):
    """You are a CIO writing a mini research note (Deep Dive level).
    Most comprehensive level: more metrics, capital structure, unit economics.
    But STILL narrative-driven with a clear 'so what' at the end.
    MUST separate Wall Street research from media news."""

    query: str = dspy.InputField(desc="The user's question")
    company_data: str = dspy.InputField(desc="Full fundamentals from CompanyAgent")
    wall_street_data: str = dspy.InputField(desc="Full analyst consensus from WallStreetAgent")
    news_data: str = dspy.InputField(desc="Full media coverage from NewsAgent")
    macro_data: str = dspy.InputField(desc="Full macro/sector context from MacroAgent")

    opening_take: str = dspy.OutputField(desc="1-2 sentence thesis statement")
    business_deep_dive: str = dspy.OutputField(
        desc="Deep business analysis: competitive position, unit economics, "
             "capital structure, margin trajectory. Use numbers but keep narrative."
    )
    consensus_section: str = dspy.OutputField(
        desc="## 🏛️ Institutional & Expert Consensus. Comprehensive: vote + "
             "all available bank research + price target range + your CIO view."
    )
    media_section: str = dspy.OutputField(
        desc="## 📰 Latest Media News & Sentiment. Comprehensive media analysis."
    )
    macro_context: str = dspy.OutputField(
        desc="## 🌐 Macro & Sector Context. Cycle position, sector dynamics, regulatory."
    )
    scenario_analysis: str = dspy.OutputField(
        desc="## Scenario Analysis table: Bull/Base/Bear with probabilities, "
             "key assumptions, target prices, and time horizons."
    )
    decision_frame: str = dspy.OutputField(
        desc="Research-note-quality conclusion: full risk/reward assessment, "
             "position sizing, entry/exit framework, key monitoring metrics."
    )


class CIOComparison(dspy.Signature):
    """You are a CIO comparing two investment opportunities.
    Start with a framing sentence, then a comparison table with REAL numbers,
    then per-ticker consensus, then a clear CIO verdict.
    MUST separate Wall Street research from media news for EACH ticker."""

    query: str = dspy.InputField(desc="The user's comparison question")
    ticker_a_data: str = dspy.InputField(desc="Fundamentals for Ticker A")
    ticker_b_data: str = dspy.InputField(desc="Fundamentals for Ticker B")
    ws_a: str = dspy.InputField(desc="Wall Street consensus for Ticker A")
    ws_b: str = dspy.InputField(desc="Wall Street consensus for Ticker B")
    news_a: str = dspy.InputField(desc="News/media for Ticker A")
    news_b: str = dspy.InputField(desc="News/media for Ticker B")

    opening_take: str = dspy.OutputField(desc="1-2 sentence comparative framing")
    comparison_table: str = dspy.OutputField(
        desc="Side-by-side markdown table: 6-8 key metrics with real numbers, "
             "trends (→ arrows), and an Edge column. Keep tight."
    )
    consensus_per_ticker: str = dspy.OutputField(
        desc="## 🏛️ Institutional & Expert Consensus for EACH ticker. "
             "Per-ticker: vote + bank research. Wall Street only."
    )
    media_per_ticker: str = dspy.OutputField(
        desc="## 📰 Latest Media News for EACH ticker. Media only, brief."
    )
    verdict: str = dspy.OutputField(
        desc="CIO verdict: which gets the nod and why. Decision frame: "
             "'If you care about X, A is the core; B is the satellite.'"
    )


# ── CIOSynthesizer Module ───────────────────────────────────────────────────

class CIOSynthesizer(dspy.Module):
    """Main DSPy module that selects the right signature based on level
    and produces the structured CIO response."""

    def __init__(self):
        super().__init__()
        self.fast = dspy.ChainOfThought(CIOFast)
        self.standard = dspy.ChainOfThought(CIOStandard)
        self.master = dspy.ChainOfThought(CIOMaster)
        self.deep_dive = dspy.ChainOfThought(CIODeepDive)
        self.comparison = dspy.ChainOfThought(CIOComparison)

    def forward(self, query: str, level: str, agent_outputs: dict) -> dspy.Prediction:
        """Run the appropriate signature based on level.

        Args:
            query: The user's original question.
            level: One of 'fast', 'standard', 'master', 'deep_dive', 'comparison'.
            agent_outputs: Dict with keys like 'company_data', 'wall_street_data',
                          'news_data', 'macro_data'. Missing keys default to ''.
        """
        _get_lm()

        co = agent_outputs.get("company_data", "") or "(No company data available)"
        ws = agent_outputs.get("wall_street_data", "") or "(No Wall Street data available)"
        nw = agent_outputs.get("news_data", "") or "(No news data available)"
        ma = agent_outputs.get("macro_data", "") or "(No macro data available)"

        if level == "fast":
            return self.fast(query=query, company_data=co)

        if level == "comparison":
            return self.comparison(
                query=query,
                ticker_a_data=agent_outputs.get("ticker_a_data", co),
                ticker_b_data=agent_outputs.get("ticker_b_data", ""),
                ws_a=agent_outputs.get("ws_a", ws),
                ws_b=agent_outputs.get("ws_b", ""),
                news_a=agent_outputs.get("news_a", nw),
                news_b=agent_outputs.get("news_b", ""),
            )

        if level == "master":
            return self.master(
                query=query, company_data=co,
                wall_street_data=ws, news_data=nw, macro_data=ma,
            )

        if level == "deep_dive":
            return self.deep_dive(
                query=query, company_data=co,
                wall_street_data=ws, news_data=nw, macro_data=ma,
            )

        # Default: standard
        return self.standard(
            query=query, company_data=co,
            wall_street_data=ws, news_data=nw,
        )


# ── Markdown Renderer ────────────────────────────────────────────────────────

def render_markdown(prediction: dspy.Prediction, level: str) -> str:
    """Convert a DSPy Prediction into the final markdown string for display."""
    parts: list[str] = []

    opening = getattr(prediction, "opening_take", "")
    if opening:
        parts.append(f"**{opening.strip()}**\n")

    biz = getattr(prediction, "business_and_numbers", "") or getattr(prediction, "business_deep_dive", "")
    analysis = getattr(prediction, "analysis", "")
    if biz:
        parts.append(biz.strip())
    elif analysis:
        parts.append(analysis.strip())

    consensus = getattr(prediction, "consensus_section", "") or getattr(prediction, "consensus_per_ticker", "")
    if consensus:
        text = consensus.strip()
        if not text.startswith("##"):
            text = f"## 🏛️ Institutional & Expert Consensus\n\n{text}"
        parts.append(text)

    media = getattr(prediction, "media_section", "") or getattr(prediction, "media_per_ticker", "")
    if media:
        text = media.strip()
        if not text.startswith("##"):
            text = f"## 📰 Latest Media News & Sentiment\n\n{text}"
        parts.append(text)

    macro = getattr(prediction, "macro_context", "")
    if macro:
        text = macro.strip()
        if not text.startswith("##"):
            text = f"## 🌐 Macro & Sector Context\n\n{text}"
        parts.append(text)

    comp_table = getattr(prediction, "comparison_table", "")
    if comp_table:
        parts.insert(1, comp_table.strip())

    scenario = getattr(prediction, "scenario_analysis", "")
    if scenario:
        text = scenario.strip()
        if not text.startswith("##"):
            text = f"## Scenario Analysis\n\n{text}"
        parts.append(text)

    decision = getattr(prediction, "decision_frame", "") or getattr(prediction, "verdict", "")
    if decision:
        parts.append(f"---\n\n{decision.strip()}")

    parts.append("\n> ⚠️ Disclaimer: AI-generated analysis for educational purposes. Not financial advice.")

    return "\n\n".join(parts)


# ── Load optimized module if available ───────────────────────────────────────

_OPTIMIZED_PATH = Path(__file__).parent / "optimized_cio.json"
_synthesizer: CIOSynthesizer | None = None


def get_synthesizer() -> CIOSynthesizer:
    """Return a singleton CIOSynthesizer, loading optimized weights if available."""
    global _synthesizer
    if _synthesizer is None:
        _get_lm()
        _synthesizer = CIOSynthesizer()
        if _OPTIMIZED_PATH.exists():
            try:
                _synthesizer.load(_OPTIMIZED_PATH)
                logger.info("Loaded optimized CIO module from %s", _OPTIMIZED_PATH)
            except Exception as e:
                logger.warning("Failed to load optimized module: %s", e)
    return _synthesizer
