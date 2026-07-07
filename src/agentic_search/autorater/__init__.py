"""Autorater implementations for Sufficient Context classification.

Per the paper:
  - FLAMe: efficient online autorater (88% accuracy, 1600 token window)
  - Gemini: high-accuracy offline autorater (93% accuracy)
  - Fallback: token-match heuristic when no LLM available
"""

from __future__ import annotations

from typing import Protocol


class SufficiencyAutorater(Protocol):
    """Protocol for sufficient context classification."""

    async def is_sufficient(self, question: str, context: str) -> tuple[bool, str]:
        """Return (is_sufficient, reasoning)."""
        ...

    async def classify(self, question: str, context: str) -> tuple[float, str]:
        """Return (sufficiency_score, reasoning)."""
        ...


class FLAMeAutorater:
    """FLAMe-based autorater — efficient, ~88% accuracy.

    Uses 1600-token chunks (matching the paper's approach).
    """

    def __init__(self, model_path: str = "flame-24b"):
        self.model_path = model_path
        self.max_tokens = 1600
        # In production, load the actual FLAMe model

    async def is_sufficient(self, question: str, context: str) -> tuple[bool, str]:
        chunks = self._chunk_context(context)
        for i, chunk in enumerate(chunks):
            is_suff, reason = await self._classify_chunk(question, chunk)
            if is_suff:
                return True, f"Chunk {i} is sufficient: {reason}"
        return False, f"No chunk sufficient across {len(chunks)} chunks"

    async def classify(self, question: str, context: str) -> tuple[float, str]:
        chunks = self._chunk_context(context)
        sufficient_chunks = 0
        reasons = []
        for i, chunk in enumerate(chunks):
            is_suff, reason = await self._classify_chunk(question, chunk)
            if is_suff:
                sufficient_chunks += 1
            reasons.append(f"Chunk {i}: {'SUFFICIENT' if is_suff else 'INSUFFICIENT'} — {reason}")
        score = sufficient_chunks / max(len(chunks), 1)
        return score, " | ".join(reasons)

    def _chunk_context(self, context: str) -> list[str]:
        words = context.split()
        chunks, current = [], []
        for w in words:
            current.append(w)
            if len(current) >= self.max_tokens:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks or [""]

    async def _classify_chunk(self, question: str, chunk: str) -> tuple[bool, str]:
        """In production: forward pass through FLAMe model.
        
        The model classifies whether the chunk contains enough info 
        to answer the question.
        """
        # Placeholder: return a heuristic
        question_terms = set(question.lower().split())
        chunk_lower = chunk.lower()
        matched = sum(1 for t in question_terms if t in chunk_lower)
        is_suff = matched >= max(len(question_terms) * 0.3, 1)
        return is_suff, f"term_match={matched}/{len(question_terms)}"


class GeminiAutorater:
    """Gemini-based autorater — high accuracy ~93%.

    Uses 1-shot prompting with chain-of-thought.
    """

    def __init__(self):
        # In production, initialize Gemini client
        pass

    async def is_sufficient(self, question: str, context: str) -> tuple[bool, str]:
        """Classify with Gemini 1.5 Pro (1-shot + CoT)."""
        # In production: call Gemini API
        prompt = _build_autorater_prompt(question, context)
        # response = await gemini_client.generate(prompt)
        # return parse_verdict(response)
        raise NotImplementedError("Configure GEMINI_API_KEY to use")

    async def classify(self, question: str, context: str) -> tuple[float, str]:
        """Get sufficiency score from Gemini."""
        is_suff, reason = await self.is_sufficient(question, context)
        return (1.0 if is_suff else 0.0, reason)


class FallbackAutorater:
    """Token-match heuristic autorater — no external model needed."""

    async def is_sufficient(self, question: str, context: str) -> tuple[bool, str]:
        score, reason = await self.classify(question, context)
        return score >= 0.5, reason

    async def classify(self, question: str, context: str) -> tuple[float, str]:
        question_terms = set(t.lower() for t in question.split() if len(t) > 3)
        if not question_terms:
            return 0.5, "No significant terms to match"
        context_lower = context.lower()
        matched = sum(1 for t in question_terms if t in context_lower)
        score = matched / len(question_terms)
        return score, f"Fallback: {matched}/{len(question_terms)} terms matched"


def _build_autorater_prompt(question: str, context: str) -> str:
    """Paper's autorater prompt template."""
    return f"""Given a QUESTION and CONTEXT, determine if the CONTEXT contains 
enough information to answer the QUESTION.

A context is SUFFICIENT if it contains all the necessary information 
to provide a definitive answer to the question. It is INSUFFICIENT if 
the information is incomplete, inconclusive, or contradictory.

QUESTION: {question}
CONTEXT: {context}

Use chain-of-thought reasoning before providing your final answer.
SUFFICIENT or INSUFFICIENT?"""
