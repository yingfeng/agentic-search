"""Confidence estimation utilities — P(True) and P(Correct) signals.

Per the paper:
  - P(True): sample 20 answers + 5 self-evaluations → mean confidence
             (high cost, best for open-source models)
  - P(Correct): single-call self-rated confidence
                (low cost, suitable for proprietary models)
"""

from __future__ import annotations

import math


async def estimate_p_correct(question: str, context: str) -> float:
    """P(Correct) — single-call self-rated confidence.

    Calls the LLM with:
        "Answer this question based on the context. 
         Then rate your confidence (0-1) that your answer is correct."

    Returns 0..1 confidence score.
    """
    # In production: actual LLM call
    # placeholder — return moderate confidence
    return _heuristic_confidence(question, context)


async def estimate_p_true(question: str, context: str, num_samples: int = 20) -> float:
    """P(True) — sample multiple answers and self-evaluate.

    Steps:
      1. Generate `num_samples` answers via LLM (temperature=0.7)
      2. For each answer, ask LLM: is this answer correct? (5 times)
      3. Average across all samples and evaluations

    Returns 0..1 confidence score.
    """
    # In production: actual sampling + evaluation loop
    # placeholder
    return _heuristic_confidence(question, context)


def _heuristic_confidence(question: str, context: str) -> float:
    """Fallback confidence estimation without LLM calls."""
    if not context:
        return 0.1
    question_terms = set(question.lower().split())
    context_lower = context.lower()
    matched = sum(1 for t in question_terms if t in context_lower)
    ratio = matched / max(len(question_terms), 1)
    # Calibrate: models tend to be overconfident
    return min(ratio * 1.2, 0.95)
