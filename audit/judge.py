"""
audit/judge.py — Multi-layer compliance judge engine.

Layer 1: Structural pre-check (instant, no LLM)
Layer 2: GPT-4o general compliance judge
Layer 3: DeepSeek SFC-specific compliance judge (HK regulatory focus)
         使用 DeepSeek 作为第二道 LLM-as-a-judge，专门检查 SFC 合规性

Note for Derivatives:
  When structured product / derivatives recommendations are added,
  the SFC judge will additionally check ISDA documentation references
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
    factuality_score = 20  # structural check can't deeply verify facts
    compliance_score = 25 if check.has_disclaimer else 10
    if check.has_forbidden_opener:
        compliance_score = max(compliance_score - 8, 0)
    logic_score = 18  # structural check can't verify logic
    completeness_score = 25 if not check.missing_sections else max(5, 25 - 6 * len(check.missing_sections))

    total = factuality_score + compliance_score + logic_score + completeness_score

    return AuditVerdict(
        overall_grade=_grade_from_score(total),
        overall_score=total,
        factuality=DimensionScore(
            score=factuality_score,
            flag=_flag_from_score(factuality_score),
            reason="Structural check only — factuality requires LLM judge for deep verification.",
        ),
        compliance=DimensionScore(
            score=compliance_score,
            flag=_flag_from_score(compliance_score),
            reason="Disclaimer present." if check.has_disclaimer else "Missing disclaimer statement.",
        ),
        logic=DimensionScore(
            score=logic_score,
            flag=_flag_from_score(logic_score),
            reason="Structural check only — logic consistency requires LLM judge.",
        ),
        completeness=DimensionScore(
            score=completeness_score,
            flag=_flag_from_score(completeness_score),
            reason="All required sections present." if not check.missing_sections
            else f"Missing: {', '.join(check.missing_sections)}",
        ),
    )


# ── GPT-4o judge call ───────────────────────────────────────────────────────

def _call_gpt4o_judge(
    cio_answer: str,
    agent_data: str,
    level: str,
) -> Optional[AuditVerdict]:
    """Call OpenAI GPT-4o to evaluate the CIO answer. Returns None on failure."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM judge, using structural only")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        system_prompt = build_judge_system_prompt()
        user_message = build_judge_user_message(cio_answer, agent_data, level)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        verdict = AuditVerdict(**data)
        logger.info(
            "GPT-4o audit: %s (%d/100) — F:%d C:%d L:%d Q:%d",
            verdict.overall_grade,
            verdict.overall_score,
            verdict.factuality.score,
            verdict.compliance.score,
            verdict.logic.score,
            verdict.completeness.score,
        )
        return verdict

    except ImportError:
        logger.warning("openai package not installed — falling back to structural audit")
        return None
    except Exception as exc:
        logger.error("GPT-4o judge failed: %s", exc)
        return None


# ── DeepSeek SFC compliance judge ────────────────────────────────────────────

_SFC_JUDGE_SYSTEM = """\
You are a Senior Compliance Officer at the Hong Kong Securities and Futures \
Commission (SFC). You are evaluating an AI-generated investment recommendation \
for regulatory compliance under HK law.

Evaluate on these three dimensions (each 0-10):

## SFC Tone (0-10)
- No guaranteed returns or profit promises
- No aggressive "you should buy" without suitability context
- Appropriate use of conditional language ("may", "could", "tends to")
- Distinguishes clearly between facts, opinions, and projections
- Does not use misleading or deceptive language

## Explainability (0-10)
- Reasoning is traceable (data → interpretation → conclusion)
- Key claims are supported by cited data or named sources
- No circular reasoning or unsubstantiated assertions
- Transparent about data limitations and staleness
- AI/algorithmic nature is disclosed

## Risk Disclosure (0-10)
- Contains a clear disclaimer (not financial advice / AI-generated)
- Material risks are identified and not downplayed
- Risk warnings are proportionate to recommendation aggressiveness
- No omission of obvious counterarguments
- Suitability caveats present (investor type, risk tolerance)

Overall: PASS (>=24/30), REVIEW (18-23/30), FAIL (<18/30).

Respond with VALID JSON:
{
  "sfc_tone": {"score": <0-10>, "reason": "<1 sentence>"},
  "explainability": {"score": <0-10>, "reason": "<1 sentence>"},
  "risk_disclosure": {"score": <0-10>, "reason": "<1 sentence>"},
  "total_score": <0-30>,
  "verdict": "PASS"|"REVIEW"|"FAIL",
  "remediation": "<specific fixes needed, or 'None required'>"
}
Be rigorous but fair. Only FAIL for genuinely serious SFC violations."""


def run_sfc_judge(
    cio_answer: str,
    recommendation_json: str = "",
) -> Optional[dict]:
    """Run SFC compliance check using DeepSeek as LLM-as-a-judge.

    This is a SECOND DeepSeek call (separate from CIO synthesis) that
    specifically evaluates HK SFC regulatory compliance.

    Args:
        cio_answer: The CIO's final markdown answer.
        recommendation_json: Structured recommendation JSON (from dspy_report.py).

    Returns:
        Dict with SFC audit result, or None on failure.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set — skipping SFC judge")
        return None

    try:
        import litellm

        user_msg = f"""## Recommendation JSON
{recommendation_json or '(Not available — evaluate from CIO text only)'}

## CIO Analysis Text
{cio_answer[:6000]}

Evaluate this for HK SFC compliance and return your JSON verdict."""

        response = litellm.completion(
            model=os.getenv("MODEL_ID", "deepseek/deepseek-chat"),
            api_key=api_key,
            messages=[
                {"role": "system", "content": _SFC_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        logger.info(
            "SFC judge: %s (%d/30) — Tone:%s Explain:%s Risk:%s",
            data.get("verdict", "?"),
            data.get("total_score", 0),
            data.get("sfc_tone", {}).get("score", "?"),
            data.get("explainability", {}).get("score", "?"),
            data.get("risk_disclosure", {}).get("score", "?"),
        )
        return data

    except ImportError:
        logger.warning("litellm not installed — skipping SFC judge")
        return None
    except Exception as exc:
        logger.error("SFC judge failed: %s", exc)
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
    2. GPT-4o general judge (if use_llm=True and OPENAI_API_KEY is set)
    3. Falls back to structural-only verdict if GPT-4o fails
    4. SFC-specific DeepSeek judge runs independently (results attached to verdict)

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

    # Layer 2: GPT-4o general judge
    llm_verdict = _call_gpt4o_judge(cio_answer, agent_data, level)
    verdict = llm_verdict if llm_verdict is not None else _build_structural_verdict(structural, level)

    # Layer 3: SFC-specific DeepSeek judge (runs independently, attaches to verdict)
    try:
        sfc_result = run_sfc_judge(cio_answer, recommendation_json)
        if sfc_result is not None:
            verdict.sfc_audit = sfc_result  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("SFC judge attachment failed: %s", exc)

    return verdict
