"""Protocol interfaces for pluggable components.

All core components are Protocols, enabling:
  - Deterministic in-memory implementations for testing
  - LLM-based implementations for production
  - Custom third-party implementations
"""

from __future__ import annotations

from typing import Protocol

from .state import (
    RetrievalPlan,
    SubQuery,
    Snippet,
    DraftAnswer,
    ContextAssessment,
    GroundedAnswer,
)


class Planner(Protocol):
    """Decompose question into search plan."""

    async def plan(
        self,
        question: str,
        corpora: dict[str, str],
        dead_corpora: set[str],
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> RetrievalPlan:
        ...


class QueryRewriter(Protocol):
    """Rewrite plan routes into search queries."""

    async def rewrite(
        self,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        tried_queries: set[str],
    ) -> list[SubQuery]:
        ...


class Retriever(Protocol):
    """Multi-source retrieval backend."""

    def supports_corpus(self, corpus: str) -> bool: ...

    async def search(self, query: str, corpus: str, top_k: int = 10) -> list[Snippet]: ...


class Drafter(Protocol):
    """Generate intermediate draft from context."""

    async def draft(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
    ) -> DraftAnswer:
        ...


class SufficiencyJudge(Protocol):
    """Judge context sufficiency and generate feedback."""

    async def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
        draft: DraftAnswer | None,
    ) -> ContextAssessment:
        ...


class ConfidenceScorer(Protocol):
    """Estimate model confidence in its answer."""

    async def score(
        self,
        question: str,
        snippets: list[Snippet],
    ) -> float:
        ...


class Synthesizer(Protocol):
    """Generate final grounded answer."""

    async def synthesize(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
        assessment: ContextAssessment,
        confidence: float,
        status,
    ) -> GroundedAnswer:
        ...


class StructuredLLM(Protocol):
    """LLM that returns structured JSON output."""

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
        temperature: float = 0.0,
    ) -> dict:
        ...
