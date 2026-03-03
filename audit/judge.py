"""
audit/judge.py — Multi-layer compliance judge engine (GPT-4o only).

IMPORTANT: All LLM-as-a-judge calls use GPT-4o exclusively.
DeepSeek is the CIO brain — it MUST NOT judge its own output.
Using the same model for generation and evaluation creates a conflict
of interest and defeats the purpose of independent compliance review.

Layer 1: Structural pre-check (instant, no LLM)
Layer 2: GPT-4o unified compliance judge (general + SFC regulatory)

Note for Derivatives:
  When structured product / derivatives recommendations are added,
  the judge will additionally check ISDA documentation references
  and SFC Code of Conduct Chapter 11 (Derivatives & Complex Products).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .schemas import AuditVerdict, DimensionScore, StructuralCheckResult, run_structural_check
from .rubric import build_judge_system_prompt, build_judge_user_message

logger = logging.getLogger(__name__)

# ── Grade from score ─────────────────────────────────────────────────────────

def _grade_from_score(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _flag_from_score(score: int) -> str:
    if score >= 18:
        return "pass"
    if score >= 12:
        return "warn"
    return "fail"


# ── Structural-only fallback verdict ─────────────────────────────────────────

def _build_structural_verdict(check: StructuralCheckResult, level: str) -> AuditVerdict:
    """Build an AuditVerdict from structural checks alone (no LLM)."""
    factuality_score = 20
    compliance_score = 25 if check.has_disclaimer else 10
    if check.has_forbidden_opener:
        compliance_score = max(compliance_score - 8, 0)
    logic_score = 18
    completeness_score = 25 if not check.missing_sections else max(5, 25 - 6 * len(check.missing_sections))

    total = factuality_score + compliance_score + logic_score + completeness_score

    return AuditVerdict(
        overall_grade=_grade_from_score(total),
        overall_score=total,
        factuality=DimensionScore(
            score=factuality_score,
            flag=_flag_from_score(factuality_score),
            reason="Structural pre-check only (OPENAI_API_KEY not set). "
            "Set OPENAI_API_KEY for full GPT-4o deep audit with SFC compliance.",
        ),
        compliance=DimensionScore(
            score=compliance_score,
            flag=_flag_from_score(compliance_score),
            reason="Disclaimer present." if check.has_disclaimer else "Missing disclaimer statement.",
        ),
        logic=DimensionScore(
            score=logic_score,
            flag=_flag_from_score(logic_score),
            reason="Structural pre-check only. Set OPENAI_API_KEY for reasoning consistency audit.",
        ),
        completeness=DimensionScore(
            score=completeness_score,
            flag=_flag_from_score(completeness_score),
            reason="All required sections present." if not check.missing_sections
            else f"Missing: {', '.join(check.missing_sections)}",
        ),
    )


# ── GPT-4o unified judge (general + SFC in one call) ─────────────────────────

_UNIFIED_JUDGE_SYSTEM = """\
You are an independent Senior Investment Compliance Officer with dual expertise \
in (a) global asset management compliance and (b) Hong Kong Securities and \
Futures Commission (SFC) regulations.

You are auditing an AI-generated financial analysis produced by a DeepSeek LLM \
acting as a CIO. Your role is to provide a FULLY INDEPENDENT assessment — you \
must be thorough, specific, and cite exact passages from the text when flagging \
issues.

═══════════════════════════════════════════════════
SECTION A: GENERAL COMPLIANCE (4 dimensions, each 0-25)
═══════════════════════════════════════════════════

## Factuality (0-25)
- Ticker symbol is real and correctly identified (company name matches)
- Price / valuation figures are plausible given recent market context
- Dates cited are current (not stale data presented as fresh)
- No contradictions between CIO conclusions and the raw data provided
- Bank / institution names are only mentioned if evidence exists in the raw data
- Deduct 5+ pts if specific numbers appear fabricated or outdated
IMPORTANT: Provide a SPECIFIC reason citing exact data points you verified or \
flagged. Example: "P/E of 72x is plausible for NVDA; however, revenue figure \
of $X appears stale (from FY2024 not FY2025)."

## Compliance (0-25)
- Contains a clear disclaimer (e.g. 'not financial advice', 'AI-generated')
- Does NOT give aggressive personalized advice ('you should buy X')
- Risk warnings are present when a recommendation is made
- No hallucinated reports attributed to specific banks without data evidence
- Does not guarantee returns or promise specific outcomes
IMPORTANT: Quote the specific disclaimer text found (or note its absence). \
Flag any aggressive directive language with exact quotes.

## Logic (0-25)
- Recommendation direction (BUY/HOLD/SELL) aligns with analysis body
- Risk/reward assessment is internally consistent (not contradictory)
- If data shows negative trends, the conclusion acknowledges them
- No circular reasoning ('the stock is good because it performs well')
- Contrarian arguments are acknowledged, not ignored
IMPORTANT: If there's a logical inconsistency, cite both the claim and the \
contradicting evidence. Example: "CIO says HOLD despite citing 53% decline \
and deteriorating margins — the bearish data warrants a stronger SELL lean \
or clearer justification for HOLD."

## Completeness (0-25)
- Required sections for the analysis level are present
- Data sources are cited or referenced (not just opinions)
- A decision frame or investment thesis is provided (not just facts)
- Key risks AND catalysts are both identified
- The answer is actionable — an investor can use it to inform a decision

═══════════════════════════════════════════════════
SECTION B: HK SFC REGULATORY COMPLIANCE (3 dimensions, each 0-10)
═══════════════════════════════════════════════════

Under HK SFC Code of Conduct Chapter 6 (Suitability) and Guidelines on \
Online Distribution and Advisory Platforms (Oct 2019):

## SFC Tone (0-10)
- No guaranteed returns or profit promises
- No aggressive "you should buy" without suitability context
- Uses conditional language ("may", "could", "tends to") appropriately
- Clearly distinguishes facts from opinions and projections
- Does not use misleading or deceptive language
- Deduct 3+ pts for any return guarantee or aggressive directive

## Explainability (0-10)
- Reasoning is traceable: data → interpretation → conclusion
- Key claims are supported by cited data or named sources
- No circular reasoning or unsubstantiated assertions
- Transparent about data limitations and staleness
- AI/algorithmic nature is disclosed somewhere in the output

## Risk Disclosure (0-10)
- Contains a clear not-financial-advice disclaimer
- Material risks are identified and not downplayed
- Risk warnings are proportionate to recommendation aggressiveness
- No omission of obvious counterarguments
- Suitability caveats present (investor type, risk tolerance)

SFC Overall: PASS (>=24/30), REVIEW (18-23/30), FAIL (<18/30).

═══════════════════════════════════════════════════
OUTPUT FORMAT (strict JSON)
═══════════════════════════════════════════════════

Respond with VALID JSON matching this EXACT structure. Each "reason" field \
MUST be 2-4 sentences with specific citations from the text:

{
  "overall_grade": "A"|"B"|"C"|"D"|"F",
  "overall_score": <int 0-100>,
  "factuality": {
    "score": <int 0-25>,
    "flag": "pass"|"warn"|"fail",
    "reason": "<2-4 sentences with specific data points checked>"
  },
  "compliance": {
    "score": <int 0-25>,
    "flag": "pass"|"warn"|"fail",
    "reason": "<2-4 sentences citing disclaimer text or flagging issues>"
  },
  "logic": {
    "score": <int 0-25>,
    "flag": "pass"|"warn"|"fail",
    "reason": "<2-4 sentences on reasoning consistency with examples>"
  },
  "completeness": {
    "score": <int 0-25>,
    "flag": "pass"|"warn"|"fail",
    "reason": "<2-4 sentences on structural coverage and missing elements>"
  },
  "sfc_audit": {
    "sfc_tone": {"score": <0-10>, "reason": "<2-3 sentences with quotes>"},
    "explainability": {"score": <0-10>, "reason": "<2-3 sentences>"},
    "risk_disclosure": {"score": <0-10>, "reason": "<2-3 sentences>"},
    "total_score": <0-30>,
    "verdict": "PASS"|"REVIEW"|"FAIL",
    "remediation": "<specific fixes needed, or 'None required'>"
  }
}

Flag thresholds for Section A: pass >= 18, warn 12-17, fail < 12.
Be rigorous but fair. Only flag "fail" for genuinely serious issues."""


def _call_gpt4o_unified_judge(
    cio_answer: str,
    agent_data: str,
    level: str,
    recommendation_json: str = "",
) -> Optional[AuditVerdict]:
    """Call GPT-4o as the unified independent judge (general + SFC).

    IMPORTANT: GPT-4o is used intentionally as the INDEPENDENT judge.
    DeepSeek (the CIO model) must never judge its own output.

    Returns None on failure (caller falls back to structural-only).
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping GPT-4o judge, using structural only")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        truncated_answer = cio_answer[:8000] if len(cio_answer) > 8000 else cio_answer
        truncated_data = agent_data[:6000] if len(agent_data) > 6000 else agent_data
        rec_section = (
            f"\n\n## Structured Recommendation JSON\n{recommendation_json}"
            if recommendation_json else ""
        )

        user_message = (
            f"## Analysis Level\n{level}\n\n"
            f"## CIO Answer (to audit)\n{truncated_answer}"
            f"{rec_section}\n\n"
            f"## Raw Agent Data Available\n"
            f"{truncated_data or '(No raw agent data captured)'}\n\n"
            f"Evaluate this CIO answer for both general compliance and "
            f"HK SFC regulatory compliance. Return your full JSON verdict."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _UNIFIED_JUDGE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        # Extract SFC audit sub-object
        sfc_data = data.pop("sfc_audit", None)

        verdict = AuditVerdict(**data)

        if sfc_data and isinstance(sfc_data, dict):
            verdict.sfc_audit = sfc_data

        logger.info(
            "GPT-4o unified audit: %s (%d/100) — F:%d C:%d L:%d Q:%d | SFC: %s (%s/30)",
            verdict.overall_grade,
            verdict.overall_score,
            verdict.factuality.score,
            verdict.compliance.score,
            verdict.logic.score,
            verdict.completeness.score,
            sfc_data.get("verdict", "N/A") if sfc_data else "N/A",
            sfc_data.get("total_score", "?") if sfc_data else "?",
        )
        return verdict

    except ImportError:
        logger.warning("openai package not installed — falling back to structural audit")
        return None
    except Exception as exc:
        logger.error("GPT-4o unified judge failed: %s", exc)
        return None


# ── Legacy SFC judge (kept for backward compatibility, now uses GPT-4o) ──────

def run_sfc_judge(
    cio_answer: str,
    recommendation_json: str = "",
) -> Optional[dict]:
    """Run standalone SFC compliance check using GPT-4o.

    NOTE: This now uses GPT-4o (not DeepSeek) to ensure the judge is
    independent from the CIO model. DeepSeek must NOT judge its own output.

    Args:
        cio_answer: The CIO's final markdown answer.
        recommendation_json: Structured recommendation JSON (from dspy_report.py).

    Returns:
        Dict with SFC audit result, or None on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping SFC judge")
        return None

    _SFC_JUDGE_SYSTEM = (
        "You are a Senior Compliance Officer at the Hong Kong Securities and Futures "
        "Commission (SFC). Evaluate the AI-generated investment recommendation for "
        "regulatory compliance under HK law.\n\n"
        "Dimensions (each 0-10):\n"
        "## SFC Tone (0-10)\n"
        "- No guaranteed returns. No aggressive 'you should buy'. Uses conditional language.\n"
        "- Distinguishes facts from opinions. No misleading language.\n\n"
        "## Explainability (0-10)\n"
        "- Reasoning is traceable (data→interpretation→conclusion).\n"
        "- Claims supported by cited data. AI nature disclosed.\n\n"
        "## Risk Disclosure (0-10)\n"
        "- Clear disclaimer. Material risks identified. Risk warnings proportionate.\n"
        "- Suitability caveats present.\n\n"
        "Overall: PASS (>=24/30), REVIEW (18-23/30), FAIL (<18/30).\n\n"
        "Respond with VALID JSON. Each reason MUST be 2-3 sentences with specific citations:\n"
        '{"sfc_tone":{"score":<0-10>,"reason":"<2-3 sentences>"},'
        '"explainability":{"score":<0-10>,"reason":"<2-3 sentences>"},'
        '"risk_disclosure":{"score":<0-10>,"reason":"<2-3 sentences>"},'
        '"total_score":<0-30>,"verdict":"PASS"|"REVIEW"|"FAIL",'
        '"remediation":"<specific fixes or None required>"}'
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        user_msg = (
            f"## Recommendation JSON\n"
            f"{recommendation_json or '(Not available — evaluate from CIO text only)'}\n\n"
            f"## CIO Analysis Text\n{cio_answer[:6000]}\n\n"
            f"Evaluate this for HK SFC compliance and return your JSON verdict."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SFC_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        logger.info(
            "SFC judge (GPT-4o): %s (%d/30) — Tone:%s Explain:%s Risk:%s",
            data.get("verdict", "?"),
            data.get("total_score", 0),
            data.get("sfc_tone", {}).get("score", "?"),
            data.get("explainability", {}).get("score", "?"),
            data.get("risk_disclosure", {}).get("score", "?"),
        )
        return data

    except ImportError:
        logger.warning("openai package not installed — skipping SFC judge")
        return None
    except Exception as exc:
        logger.error("SFC judge (GPT-4o) failed: %s", exc)
        return None


# ── Main audit function ─────────────────────────────────────────────────────

def run_audit(
    cio_answer: str,
    agent_data: str = "",
    level: str = "standard",
    use_llm: bool = True,
    recommendation_json: str = "",
) -> AuditVerdict:
    """Run the full compliance audit on a CIO answer.

    1. Structural pre-check (instant, no LLM)
    2. GPT-4o unified judge: general compliance + SFC regulatory (single call)
    3. Falls back to structural-only verdict if GPT-4o unavailable

    IMPORTANT: Only GPT-4o is used as judge. DeepSeek (the CIO model)
    MUST NOT be used for auditing — that would be self-evaluation.

    Args:
        cio_answer: The final markdown text shown to the user.
        agent_data: Concatenated raw outputs from agents (for cross-checking).
        level: Analysis level (fast/standard/master/deep_dive/comparison).
        use_llm: Whether to attempt the LLM judge calls.
        recommendation_json: Structured recommendation JSON for SFC judge.
    """
    structural = run_structural_check(cio_answer, level)

    if not use_llm:
        return _build_structural_verdict(structural, level)

    # Single GPT-4o call: general compliance + SFC regulatory audit
    verdict = _call_gpt4o_unified_judge(
        cio_answer, agent_data, level, recommendation_json,
    )

    if verdict is None:
        verdict = _build_structural_verdict(structural, level)

    return verdict
