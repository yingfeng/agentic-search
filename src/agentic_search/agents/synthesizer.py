"""Synthesizer node — generates final grounded answer from context."""

from __future__ import annotations

from ..state import AgenticSearchState, AnswerStatus, GroundedAnswer


class SynthesizerNode:
    """Generate the final answer with full traceability."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        question = state.question
        plan = state.plan
        snippets = state.get_all_snippets()
        assessment = state.assessment
        draft = state.draft
        prior_answer = state.answer

        if prior_answer and prior_answer.status == AnswerStatus.ABSTAINED:
            return {"answer": prior_answer}

        # In production, this calls an LLM to generate a grounded answer.
        # The LLM prompt should include:
        #   - The question
        #   - All snippets (with IDs for citation)
        #   - The assessment (which facts are covered, which are missing)
        #   - Instruction: "Only use information from the provided snippets.
        #     Cite each claim with its snippet ID."

        # Build evidence trail
        evidence_path = _build_evidence_path(plan, snippets, assessment)

        # Generate answer text (placeholder for LLM call)
        if prior_answer and prior_answer.status == AnswerStatus.PARTIAL:
            answer_text = _generate_partial_answer(question, snippets, assessment)
        else:
            answer_text = _generate_full_answer(question, snippets)

        confidence = prior_answer.confidence if prior_answer else 0.5
        status = prior_answer.status if prior_answer else AnswerStatus.ANSWERED

        answer = GroundedAnswer(
            question=question,
            answer=answer_text,
            status=status,
            confidence=confidence,
            citations=_build_citations(snippets, answer_text) if answer_text else {},
            iteration_count=state.iteration,
            evidence_path=evidence_path,
            missing_info=prior_answer.missing_info if prior_answer else None,
        )

        return {"answer": answer}


def _build_evidence_path(plan, snippets: list, assessment) -> list[str]:
    """Build human-readable evidence trail."""
    path = []
    if plan:
        for fact in plan.required_facts:
            count = assessment.evidence_counts.get(fact.fact_id, 0)
            if count > 0:
                path.append(f"✓ Fact '{fact.description}' — {count} snippets found")
            else:
                path.append(f"✗ Fact '{fact.description}' — NOT found")
    path.append(f"Sufficiency score: {assessment.sufficiency_score:.0%}")
    return path


def _generate_full_answer(question: str, snippets: list) -> str:
    """Generate a complete answer from snippets (placeholder for LLM)."""
    # In production, replace with LLM call.
    if not snippets:
        return ""
    # Simple extractive summary
    texts = [sn.text[:200] for sn in snippets[:5]]
    return "Based on the retrieved information:\n\n" + "\n\n".join(texts)


def _generate_partial_answer(question: str, snippets: list, assessment) -> str:
    """Generate a partial answer acknowledging gaps."""
    text = _generate_full_answer(question, snippets)
    missing = assessment.missing_facts[:3]
    if missing:
        text += f"\n\n⚠ Note: Some questions could not be fully answered. Missing information: {', '.join(missing)}."
    return text


def _build_citations(snippets: list, answer: str) -> dict[str, list[str]]:
    """Build claim → snippet ID mappings (placeholder for LLM)."""
    # In production, use LLM to assign citations.
    citations: dict[str, list[str]] = {}
    for i, sn in enumerate(snippets[:5]):
        key = f"Source {i + 1}"
        citations[key] = [sn.snippet_id]
    return citations
