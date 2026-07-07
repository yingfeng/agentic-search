"""Tests for the Protocol-driven orchestrator."""

import pytest

from agentic_search import (
    AgenticRAGOrchestrator,
    OrchestratorConfig,
    EvaluationFixture,
    evaluate_run,
    AnswerStatus,
)
from agentic_search.state import (
    RetrievalPlan,
    SearchRoute,
    RequiredFact,
    SubQuery,
    DraftAnswer,
    ContextAssessment,
    FeedbackQuery,
    JudgementVerdict,
    GroundedAnswer,
    Snippet,
)
from uuid import uuid4


def _snip(text: str) -> Snippet:
    return Snippet(snippet_id=uuid4().hex[:8], corpus="docs", document_id="d1", text=text)


def _plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        question=question,
        routes=[SearchRoute(corpus="docs", query=question, rationale="test", required_fact_ids=["f0"])],
        required_facts=[RequiredFact(fact_id="f0", description=question, required_terms=question.split())],
    )


class MockPlanner:
    async def plan(self, question, corpora, dead_corpora, prior, iteration):
        return _plan(question)


class MockRewriter:
    async def rewrite(self, plan, prior, tried):
        routes = plan.routes
        return [SubQuery(query_id="q1", text=r.query, target_corpus=r.corpus, required_fact_ids=r.required_fact_ids) for r in routes]


class MockRetriever:
    def supports_corpus(self, corpus): return corpus in ("docs", "all")

    async def search(self, query, corpus, top_k=10):
        return [_snip(f"Result for: {query}")]


class MockDrafter:
    async def draft(self, question, plan, snippets):
        return DraftAnswer(text="Draft answer.", claims=["Claim 1"], citations={"claim_0": [s.snippet_id for s in snippets[:1]]})


class MockJudge:
    def __init__(self, verdict=JudgementVerdict.SUFFICIENT):
        self.verdict = verdict

    async def assess(self, question, plan, snippets, draft):
        return ContextAssessment(
            status=self.verdict,
            sufficiency_score=1.0,
            evidence_counts={"f0": len(snippets)},
            unsupported_claims=[],
            missing_facts=[],
            feedback_queries=[],
            reason=f"Verdict: {self.verdict.value}",
        )


class MockScorer:
    async def score(self, question, snippets):
        return 0.85 if snippets else 0.0


class MockSynthesizer:
    async def synthesize(self, question, plan, snippets, assessment, confidence, status):
        return GroundedAnswer(
            question=question,
            answer=f"Answer with confidence {confidence:.2f}.",
            status=status,
            confidence=confidence,
            citations={},
            iteration_count=1,
            evidence_path=[assessment.reason],
        )


def test_orchestrator_sufficient_route():
    """SUFFICIENT → synthesize with high confidence."""
    orch = AgenticRAGOrchestrator(
        planner=MockPlanner(),
        rewriter=MockRewriter(),
        retrievers=[MockRetriever()],
        drafter=MockDrafter(),
        judge=MockJudge(JudgementVerdict.SUFFICIENT),
        scorer=MockScorer(),
        synthesizer=MockSynthesizer(),
        config=OrchestratorConfig(max_iterations=3, corpus_descriptions={"docs": "Docs"}),
    )

    answer = import_asyncio_and_run(orch.run, "test question")
    assert answer is not None
    assert answer.status == AnswerStatus.ANSWERED
    assert answer.confidence == 0.85


def test_orchestrator_unanswerable_abstains():
    """UNANSWERABLE → abstain immediately."""
    orch = AgenticRAGOrchestrator(
        planner=MockPlanner(),
        rewriter=MockRewriter(),
        retrievers=[MockRetriever()],
        drafter=MockDrafter(),
        judge=MockJudge(JudgementVerdict.UNANSWERABLE),
        scorer=MockScorer(),
        synthesizer=MockSynthesizer(),
        config=OrchestratorConfig(max_iterations=3),
    )

    answer = import_asyncio_and_run(orch.run, "unknown")
    assert answer is not None
    assert answer.status == AnswerStatus.ABSTAINED


def test_orchestrator_partial_with_high_confidence():
    """PARTIAL + high confidence → should answer."""
    orch = AgenticRAGOrchestrator(
        planner=MockPlanner(),
        rewriter=MockRewriter(),
        retrievers=[MockRetriever()],
        drafter=MockDrafter(),
        judge=MockJudge(JudgementVerdict.USEFUL_BUT_INCOMPLETE),
        scorer=MockScorer(),
        synthesizer=MockSynthesizer(),
        config=OrchestratorConfig(max_iterations=2),
    )

    answer = import_asyncio_and_run(orch.run, "partial question")
    assert answer is not None
    assert answer.status == AnswerStatus.PARTIAL


def test_orchestrator_frames_evaluation():
    """End-to-end: orchestrator output should pass evaluation."""
    orch = AgenticRAGOrchestrator(
        planner=MockPlanner(),
        rewriter=MockRewriter(),
        retrievers=[MockRetriever()],
        drafter=MockDrafter(),
        judge=MockJudge(JudgementVerdict.SUFFICIENT),
        scorer=MockScorer(),
        synthesizer=MockSynthesizer(),
    )

    answer = import_asyncio_and_run(orch.run, "test question")

    # Build a result dict for evaluation
    assessment = ContextAssessment(
        status=JudgementVerdict.SUFFICIENT,
        sufficiency_score=1.0,
        evidence_counts={"f0": 1},
        unsupported_claims=[],
        missing_facts=[],
        feedback_queries=[],
        reason="Verdict: sufficient. All facts covered.",
    )
    result = {
        "answer": answer,
        "snippets": {"s1": _snip("test result")},
        "assessment": assessment,
        "iteration": 1,
    }

    fixture = EvaluationFixture(
        question="test question",
        corpora={"docs": "Docs"},
        expected_facts=["Verdict: sufficient"],
        expected_answer_terms=["confidence"],
        max_acceptable_iterations=3,
    )

    report = evaluate_run(result, fixture)
    assert report.metrics is not None
    assert report.metrics.fact_coverage >= 0.8
    assert report.metrics.reasoning_correctness >= 0.5
    assert report.metrics.passed


def import_asyncio_and_run(coro_fn, *args):
    import asyncio
    return asyncio.run(coro_fn(*args, corpora={"docs": "Docs"}))
