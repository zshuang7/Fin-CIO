"""
optimize.py — DSPy compilation workflow for the CIO Synthesis module.

Usage:
    python optimize.py                  # Run optimization with default settings
    python optimize.py --examples 20    # Specify number of examples to use
    python optimize.py --output optimized_cio.json  # Specify output path

Prerequisites:
    1. Fill examples/gold_standard.json with 10-20 curated (query, level, ideal_output) pairs
    2. Ensure DEEPSEEK_API_KEY is set in .env or environment
    3. pip install dspy>=2.6
"""

import json
import os
import re
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import dspy
from dspy_cio import CIOSynthesizer, render_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXAMPLES_PATH = Path(__file__).parent / "examples" / "gold_standard.json"
DEFAULT_OUTPUT = Path(__file__).parent / "optimized_cio.json"

# ── Metric function ──────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = {
    "standard": ["🏛️ Institutional & Expert Consensus", "📰 Latest Media News"],
    "master": ["🏛️ Institutional & Expert Consensus", "📰 Latest Media News", "Macro"],
    "deep_dive": ["🏛️ Institutional & Expert Consensus", "📰 Latest Media News", "Macro", "Scenario"],
    "comparison": ["🏛️ Institutional & Expert Consensus", "📰 Latest Media News"],
}

_AI_CHATTER = [
    "certainly", "here is the report", "let me analyze",
    "i'll provide", "i will provide", "as requested",
    "let me start", "i'd be happy to", "sure,",
]

_WORD_COUNT_RANGES = {
    "fast": (80, 400),
    "standard": (300, 1200),
    "master": (500, 1800),
    "deep_dive": (700, 2500),
    "comparison": (400, 1600),
}


def cio_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Score a CIO synthesis output on multiple quality axes.

    Returns a float in [0, 1] — higher is better.
    """
    level = example.get("level", "standard")
    rendered = render_markdown(prediction, level)
    score = 0.0
    max_score = 0.0

    # 1. Required sections present (40% weight)
    required = _REQUIRED_SECTIONS.get(level, [])
    if required:
        max_score += 40
        found = sum(1 for s in required if s in rendered)
        score += 40 * (found / len(required))
    else:
        max_score += 40
        score += 40

    # 2. No AI chatter (20% weight)
    max_score += 20
    lower = rendered.lower()
    chatter_count = sum(1 for phrase in _AI_CHATTER if phrase in lower[:200])
    if chatter_count == 0:
        score += 20
    elif chatter_count == 1:
        score += 10
    # else: 0

    # 3. Word count within range (20% weight)
    max_score += 20
    wc = len(rendered.split())
    lo, hi = _WORD_COUNT_RANGES.get(level, (200, 1500))
    if lo <= wc <= hi:
        score += 20
    elif wc < lo:
        score += max(0, 20 * (wc / lo))
    else:
        score += max(0, 20 * (hi / wc))

    # 4. Narrative quality — no giant tables (10% weight)
    max_score += 10
    table_rows = len(re.findall(r"^\|.+\|$", rendered, re.M))
    if table_rows <= 12:
        score += 10
    elif table_rows <= 20:
        score += 5
    # else: 0

    # 5. Has a decision frame / verdict (10% weight)
    max_score += 10
    decision_markers = ["how to think about", "suits investors", "decision frame",
                        "fits the", "type of investor", "verdict", "how to own"]
    if any(m in lower for m in decision_markers):
        score += 10

    return score / max_score if max_score > 0 else 0.0


# ── Load examples ────────────────────────────────────────────────────────────

def load_gold_examples(path: Path = EXAMPLES_PATH, max_n: int = 50) -> list[dspy.Example]:
    """Load gold-standard examples from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Gold examples not found at {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw[:max_n]:
        ex = dspy.Example(
            query=item["query"],
            level=item.get("level", "standard"),
            ideal_output=item.get("ideal_output", ""),
        ).with_inputs("query", "level")
        examples.append(ex)

    logger.info("Loaded %d gold examples from %s", len(examples), path)
    return examples


# ── Optimization ─────────────────────────────────────────────────────────────

def run_optimization(
    output_path: Path = DEFAULT_OUTPUT,
    max_examples: int = 50,
    optimizer_type: str = "bootstrap",
):
    """Run DSPy compilation and save the optimized module."""

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "deepseek/deepseek-chat")
    lm = dspy.LM(model_id, api_key=api_key, max_tokens=4096)
    dspy.configure(lm=lm)

    examples = load_gold_examples(max_n=max_examples)
    if len(examples) < 3:
        logger.warning("Only %d examples — recommend at least 10 for good optimization", len(examples))

    train_set = examples[:int(len(examples) * 0.8)] or examples
    val_set = examples[int(len(examples) * 0.8):] or examples[:2]

    module = CIOSynthesizer()

    # Wrap forward to accept dspy.Example format
    class _Evaluable(dspy.Module):
        def __init__(self, synth):
            super().__init__()
            self.synth = synth

        def forward(self, query: str, level: str = "standard"):
            return self.synth.forward(
                query=query,
                level=level,
                agent_outputs={},
            )

    evaluable = _Evaluable(module)

    if optimizer_type == "mipro":
        logger.info("Using MIPROv2 optimizer")
        optimizer = dspy.MIPROv2(
            metric=cio_metric,
            auto="medium",
        )
        optimized = optimizer.compile(
            evaluable,
            trainset=train_set,
            valset=val_set,
        )
    else:
        logger.info("Using BootstrapFewShot optimizer")
        optimizer = dspy.BootstrapFewShot(
            metric=cio_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        optimized = optimizer.compile(
            evaluable,
            trainset=train_set,
        )

    optimized.save(str(output_path))
    logger.info("Optimized module saved to %s", output_path)

    # Quick eval on validation set
    scores = []
    for ex in val_set:
        try:
            pred = optimized(query=ex.query, level=ex.level)
            s = cio_metric(ex, pred)
            scores.append(s)
        except Exception as e:
            logger.warning("Eval failed for '%s': %s", ex.query[:40], e)
            scores.append(0.0)

    if scores:
        avg = sum(scores) / len(scores)
        logger.info("Validation score: %.2f (n=%d)", avg, len(scores))
    else:
        logger.warning("No validation scores computed")

    return optimized


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize CIO DSPy module")
    parser.add_argument("--examples", type=int, default=50, help="Max examples to use")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output path")
    parser.add_argument("--optimizer", choices=["bootstrap", "mipro"], default="bootstrap",
                        help="Optimizer type")
    args = parser.parse_args()

    run_optimization(
        output_path=Path(args.output),
        max_examples=args.examples,
        optimizer_type=args.optimizer,
    )
