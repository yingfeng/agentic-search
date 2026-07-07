"""Abstain node — structured abstention when context is insufficient.

Per the Google paper's finding that "insufficient context leads to 66% error rate",
this node provides a structured abstention instead of letting the LLM guess.
"""

from __future__ import annotations

from ..state import AgenticSearchState, AnswerStatus, GroundedAnswer


class AbstainNode:
    """Generate a structured abstention response."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        question = state.question
        assessment = state.assessment

        # Build a helpful abstention message
        # (inspired by the paper's recommendation to explain what's missing)
        parts = [f"I don't have enough information to answer: '{question}'"]

        if assessment:
            if assessment.missing_facts:
                parts.append(f"Missing information: {', '.join(assessment.missing_facts[:5])}")

            if assessment.evidence_counts:
                covered = sum(1 for c in assessment.evidence_counts.values() if c > 0)
                total = len(assessment.evidence_counts)
                parts.append(f"Found {covered}/{total} required facts.")

            if assessment.feedback_queries:
                suggestions = [fq.query for fq in assessment.feedback_queries[:3]]
                parts.append(f"Try searching for: {'; '.join(suggestions)}")

            parts.append(f"Sufficiency score: {assessment.sufficiency_score:.0%}")

        answer = GroundedAnswer(
            question=question,
            answer=None,
            status=AnswerStatus.ABSTAINED,
            confidence=0.0,
            citations={},
            iteration_count=state.iteration,
            evidence_path=[assessment.reason] if assessment else [],
            missing_info="\n".join(parts),
        )

        return {"answer": answer}
