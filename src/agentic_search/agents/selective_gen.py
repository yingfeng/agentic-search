"""Selective Generation node — dual-signal decision (confidence + sufficiency).

Per the Google paper: use logistic regression on two signals:
  ① Sufficiency score (from the Sufficient Context Agent)
  ② Self-rated confidence (P(Correct) from the LLM)

This node decides whether to answer or abstain.
"""

from __future__ import annotations

import math

from ..state import (
    AgenticSearchState,
    AnswerStatus,
    GroundedAnswer,
    JudgementVerdict,
)


class SelectiveGenerationNode:
    """Decide whether to generate an answer or abstain, based on confidence + sufficiency."""

    # Pre-trained logistic regression weights (in production, train on your data)
    # These approximate the paper's findings:
    #   - Sufficiency is a strong signal (high weight)
    #   - Confidence alone is unreliable (moderate weight)
    _W_SUFFICIENCY = 2.5
    _W_CONFIDENCE = 1.0
    _BIAS = -1.5

    # Coverage-accuracy trade-off thresholds
    _THRESHOLDS = {
        "strict": 0.70,    # Medical, financial → high precision
        "balanced": 0.50,  # General use
        "lenient": 0.30,   # Creative, exploratory → high coverage
    }

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        assessment = state.assessment
        if assessment is None:
            return _make_abstain(state, "No assessment available.")

        # Signal 1: Sufficiency score (0..1)
        sufficiency = assessment.sufficiency_score

        # Signal 2: Self-rated confidence via P(Correct)
        confidence = await _estimate_confidence(state.question, state.get_all_snippets())

        # Dual-signal fusion (logistic regression)
        hallucination_risk = _predict_hallucination_risk(
            confidence=confidence,
            sufficiency=sufficiency,
        )

        # Decision based on current threshold
        threshold = SelectiveGenerationNode._THRESHOLDS.get(
            state.search_config.get("mode", "balanced"),
        )

        if hallucination_risk < threshold:
            # Low risk → answer
            return _make_answer_decision(
                state, confidence, sufficiency, assessment,
            )
        else:
            # High risk → abstain with explanation
            return _make_abstain(
                state,
                f"Hallucination risk {hallucination_risk:.2f} exceeds threshold {threshold:.2f}. "
                f"Confidence={confidence:.2f}, Sufficiency={sufficiency:.2f}. "
                f"Missing: {', '.join(assessment.missing_facts[:3])}",
            )


async def _estimate_confidence(question: str, snippets: list) -> float:
    """P(Correct) — single-call self-rated confidence.
    
    In production, this calls the LLM with:
        "Answer this question. Then rate your confidence (0-1)
         that your answer is correct based on the provided context."
    
    Returns 0..1 confidence score.
    
    The heuristic here simulates model behavior:
      - With good snippet coverage → moderate-high confidence
      - With no snippets → low confidence
      - With minimal snippets → moderate confidence (simulating overconfidence bias)
    """
    if not snippets:
        return 0.05  # No evidence = very low confidence
    # Check how many question terms appear in snippets
    question_terms = {t.lower() for t in question.split() if len(t) > 3}
    if not question_terms:
        return 0.5
    combined = " ".join(s.text for s in snippets).lower()
    matched = sum(1 for t in question_terms if t in combined)
    ratio = matched / len(question_terms)
    # Simulate the paper's finding: models are overconfident with partial info
    # Calibration: conf = ratio * 0.5 + 0.3 (min 0.05, max 0.85)
    confidence = min(ratio * 0.5 + 0.3, 0.85)
    return max(confidence, 0.05)


def _predict_hallucination_risk(confidence: float, sufficiency: float) -> float:
    """Logistic regression: P(hallucination | confidence, sufficiency).
    
    Key insight from the paper:
      - High confidence + low sufficiency → VERY high risk (66% error)
      - Low confidence + high sufficiency → moderate risk
      - High confidence + high sufficiency → low risk
    """
    logit = (
        SelectiveGenerationNode._BIAS
        - SelectiveGenerationNode._W_CONFIDENCE * confidence
        + SelectiveGenerationNode._W_SUFFICIENCY * (1.0 - sufficiency)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _make_answer_decision(
    state: AgenticSearchState,
    confidence: float,
    sufficiency: float,
    assessment,
) -> dict:
    """Decision to answer — record the intent for the synthesizer."""
    status = AnswerStatus.ANSWERED
    if assessment.status == JudgementVerdict.USEFUL_BUT_INCOMPLETE:
        status = AnswerStatus.PARTIAL

    answer = GroundedAnswer(
        question=state.question,
        answer=None,  # Will be filled by synthesizer
        status=status,
        confidence=confidence,
        citations={},
        iteration_count=state.iteration,
        evidence_path=[assessment.reason],
        missing_info=", ".join(assessment.missing_facts[:3]) if assessment.missing_facts else None,
    )
    return {"answer": answer}


def _make_abstain(state: AgenticSearchState, reason: str) -> dict:
    """Decision to abstain — structured refusal."""
    answer = GroundedAnswer(
        question=state.question,
        answer=None,
        status=AnswerStatus.ABSTAINED,
        confidence=0.0,
        citations={},
        iteration_count=state.iteration,
        evidence_path=[reason],
        missing_info=reason,
    )
    return {"answer": answer}
