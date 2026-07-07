"""Drafter node — generates intermediate draft for sufficiency check."""

from __future__ import annotations

from ..state import AgenticSearchState, DraftAnswer


class DrafterNode:
    """Generate an intermediate draft from current context.
    The draft is used by the Sufficient Context Agent to verify coverage."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        if not state.plan:
            return {}

        question = state.question
        all_snippets = state.get_all_snippets()

        if not all_snippets:
            return {
                "draft": DraftAnswer(
                    text="No context retrieved yet.",
                    claims=[],
                    citations={},
                )
            }

        # In production, this would call an LLM.
        # Here we use a simple extractive approach.
        text_parts = [sn.text for sn in all_snippets[:10]]
        combined = "\n\n".join(text_parts)

        # Simple claim extraction (placeholder for LLM call)
        claims = _extract_claims(question, combined)
        citations = _build_citations(claims, all_snippets)

        draft = DraftAnswer(
            text=combined[:2000],
            claims=claims,
            citations=citations,
        )
        return {"draft": draft}


def _extract_claims(question: str, context: str) -> list[str]:
    """Extract factual claims from context (placeholder for LLM)."""
    # In production, this would be an LLM call with:
    #   "Given this question and context, break down the answer into
    #    individual factual claims. Return a JSON list of claim strings."
    sentences = context.replace("\n", " ").split(". ")
    claims = [s.strip() + "." for s in sentences if len(s.strip()) > 20]
    return claims[:15]


def _build_citations(claims: list[str], snippets: list) -> dict[str, list[str]]:
    """Map each claim to its source snippets (placeholder for LLM)."""
    # In production, use semantic matching or LLM assignment.
    # Simple: assign each claim to the first snippet that contains it.
    citations: dict[str, list[str]] = {}
    for i, claim in enumerate(claims[:5]):
        claim_key = f"claim_{i}"
        claim_lower = claim.lower()
        for sn in snippets:
            if claim_key in citations:
                break
            if any(w in sn.text.lower() for w in claim_lower.split()[:5]):
                citations[claim_key] = citations.get(claim_key, []) + [sn.snippet_id]
                break
    return citations
