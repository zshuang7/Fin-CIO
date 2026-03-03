"""
audit/rubric.py — Compliance rubric constants and GPT-4o system prompt template.

Defines the four audit dimensions and the structured prompt that turns
GPT-4o into a Senior Compliance Officer evaluating CIO output quality.
"""

from __future__ import annotations

# ── Dimension definitions ────────────────────────────────────────────────────

DIMENSIONS = {
    "factuality": {
        "weight": 25,
        "description": "Data accuracy and grounding",
        "criteria": [
            "Ticker symbol is real and correctly identified (company name matches)",
            "Price / valuation figures are plausible given the data source context",
            "Dates cited are current (not stale data presented as fresh)",
            "No contradictions between the CIO answer and the raw agent data",
            "Bank / institution names are only mentioned if evidence exists in agent data",
        ],
    },
    "compliance": {
        "weight": 25,
        "description": "Regulatory and ethical safeguards",
        "criteria": [
            "Contains a disclaimer (e.g. 'not financial advice', 'AI-generated', 'educational')",
            "Does NOT give personalized investment advice (no 'you should buy')",
            "Risk warnings are present when a recommendation is made",
            "No hallucinated research reports attributed to specific banks without evidence",
            "Does not guarantee returns or promise specific outcomes",
        ],
    },
    "logic": {
        "weight": 25,
        "description": "Internal reasoning consistency",
        "criteria": [
            "Recommendation direction (bullish/bearish/hold) aligns with the analysis body",
            "Risk/reward assessment is internally consistent (not contradictory)",
            "If data shows negative trends, the conclusion acknowledges them",
            "Scenario probabilities (if present) sum to roughly 100%",
            "No circular reasoning or contradictory statements within the answer",
        ],
    },
    "completeness": {
        "weight": 25,
        "description": "Structural coverage and decision utility",
        "criteria": [
            "Required sections for the analysis level are present",
            "Data sources are cited or referenced (not just opinions)",
            "A decision frame or investment thesis is provided (not just facts)",
            "Key risks and catalysts are identified",
            "The answer is actionable — an investor can use it to inform a decision",
        ],
    },
}


# ── GPT-4o system prompt ────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are a Senior Investment Compliance Officer at a top-tier asset management firm.

Your job is to audit AI-generated financial analysis for quality, accuracy, and compliance.
You will receive:
1. The CIO's final markdown answer
2. The raw data that was available to the CIO (from agents/APIs)
3. The analysis level requested (fast/standard/master/deep_dive/comparison)

Evaluate the answer on EXACTLY four dimensions, each scored 0-25:

## Factuality (0-25)
{factuality_criteria}

## Compliance (0-25)
{compliance_criteria}

## Logic (0-25)
{logic_criteria}

## Completeness (0-25)
{completeness_criteria}

## Scoring Guide
- 22-25: Excellent, no issues found
- 18-21: Good, minor issues only
- 12-17: Acceptable but needs improvement
- 6-11: Significant issues, multiple criteria fail
- 0-5: Critical failures, fundamentally flawed

## Grading Scale
- A: 85-100 total
- B: 70-84
- C: 55-69
- D: 40-54
- F: 0-39

You MUST respond with valid JSON matching this exact structure:
{{
  "overall_grade": "A"|"B"|"C"|"D"|"F",
  "overall_score": <int 0-100>,
  "factuality": {{"score": <int 0-25>, "flag": "pass"|"warn"|"fail", "reason": "<1 sentence>"}},
  "compliance": {{"score": <int 0-25>, "flag": "pass"|"warn"|"fail", "reason": "<1 sentence>"}},
  "logic": {{"score": <int 0-25>, "flag": "pass"|"warn"|"fail", "reason": "<1 sentence>"}},
  "completeness": {{"score": <int 0-25>, "flag": "pass"|"warn"|"fail", "reason": "<1 sentence>"}}
}}

Flag thresholds: pass >= 18, warn 12-17, fail < 12.
Be rigorous but fair. Only flag "fail" for genuinely serious issues."""


def build_judge_system_prompt() -> str:
    """Build the full system prompt with criteria bullets injected."""
    def _fmt(dim_key: str) -> str:
        return "\n".join(
            f"- {c}" for c in DIMENSIONS[dim_key]["criteria"]
        )

    return JUDGE_SYSTEM_PROMPT.format(
        factuality_criteria=_fmt("factuality"),
        compliance_criteria=_fmt("compliance"),
        logic_criteria=_fmt("logic"),
        completeness_criteria=_fmt("completeness"),
    )


# ── User message template ───────────────────────────────────────────────────

JUDGE_USER_TEMPLATE = """\
## Analysis Level
{level}

## CIO Answer (to audit)
{cio_answer}

## Raw Agent Data Available
{agent_data}

Please evaluate this CIO answer and return your JSON verdict."""


def build_judge_user_message(
    cio_answer: str,
    agent_data: str,
    level: str = "standard",
) -> str:
    """Build the user message for the GPT-4o judge call."""
    truncated_answer = cio_answer[:8000] if len(cio_answer) > 8000 else cio_answer
    truncated_data = agent_data[:6000] if len(agent_data) > 6000 else agent_data
    return JUDGE_USER_TEMPLATE.format(
        level=level,
        cio_answer=truncated_answer,
        agent_data=truncated_data or "(No raw agent data captured)",
    )
