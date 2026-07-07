"""Tests for the FRAMES-style evaluation framework."""

from uuid import uuid4

import pytest

from agentic_search import EvaluationFixture, GroundedAnswer, evaluate_run, compare_runs, AnswerStatus
from agentic_search.state import ContextAssessment, JudgementVerdict, Snippet


def _make_snippet(text: str) -> Snippet:
    return Snippet(
        snippet_id=uuid4().hex[:8],
        corpus="docs",
        document_id="doc_1",
        text=text,
    )


def test_evaluate_passing_run():
    """A fully successful run should pass evaluation."""
    result = {
        "answer": GroundedAnswer(
            question="What is X?",
            answer="X is a system [Source: doc_1]. It uses Y [Source: doc_2].",
            status=AnswerStatus.ANSWERED,
            confidence=0.85,
            citations={"claim_0": ["snip_1"], "claim_1": ["snip_2"]},
            iteration_count=1,
            evidence_path=["All facts covered."],
        ),
        "snippets": {"s1": _make_snippet("X is a system."), "s2": _make_snippet("It uses Y.")},
        "assessment": ContextAssessment(
            status=JudgementVerdict.SUFFICIENT,
            sufficiency_score=1.0,
            evidence_counts={"fact_0": 1},
            unsupported_claims=[],
            missing_facts=[],
            feedback_queries=[],
            reason="All facts covered. Fact: Information about X.",
        ),
        "iteration": 1,
    }

    fixture = EvaluationFixture(
        question="What is X?",
        corpora={"docs": "Documentation"},
        expected_facts=["Information about X"],
        expected_answer_terms=["system", "Y"],
    )

    report = evaluate_run(result, fixture)
    assert report.metrics is not None
    assert report.metrics.fact_coverage >= 0.8
    assert report.metrics.reasoning_correctness >= 0.5
    assert report.metrics.passed


def test_evaluate_failing_run():
    """A failed run should not pass."""
    result = {
        "answer": GroundedAnswer(
            question="What is X?",
            answer=None,
            status=AnswerStatus.ABSTAINED,
            confidence=0.0,
            citations={},
            iteration_count=3,
            evidence_path=["Nothing found."],
        ),
        "snippets": {},
        "assessment": ContextAssessment(
            status=JudgementVerdict.UNANSWERABLE,
            sufficiency_score=0.0,
            evidence_counts={},
            unsupported_claims=[],
            missing_facts=["fact_0"],
            feedback_queries=[],
            reason="No evidence found.",
        ),
        "iteration": 3,
    }

    fixture = EvaluationFixture(
        question="What is X?",
        corpora={"docs": "Documentation"},
        expected_facts=["Information about X"],
        expected_answer_terms=["system"],
    )

    report = evaluate_run(result, fixture)
    assert report.metrics is not None
    assert not report.metrics.passed
    assert report.metrics.iteration_count == 3


def test_compare_runs():
    """Ablation comparison between two runs."""
    fixture = EvaluationFixture(
        question="What is X?",
        corpora={"docs": "Docs"},
        expected_facts=["X"],
        expected_answer_terms=["system"],
    )

    baseline_result = {
        "answer": GroundedAnswer(
            question="What is X?",
            answer="X is a system.",
            status=AnswerStatus.ANSWERED,
            confidence=0.5,
            citations={},
            iteration_count=2,
            evidence_path=[],
        ),
        "snippets": {"s1": _make_snippet("X is a system.")},
        "assessment": ContextAssessment(
            status=JudgementVerdict.SUFFICIENT,
            sufficiency_score=0.5,
            evidence_counts={"fact_0": 1},
            unsupported_claims=[],
            missing_facts=[],
            feedback_queries=[],
            reason="Partial.",
        ),
        "iteration": 2,
    }

    candidate_result = {
        "answer": GroundedAnswer(
            question="What is X?",
            answer="X is a system with full details.",
            status=AnswerStatus.ANSWERED,
            confidence=0.9,
            citations={"claim_0": ["snip_1"]},
            iteration_count=1,
            evidence_path=["All covered."],
        ),
        "snippets": {"s1": _make_snippet("X is a system with full details.")},
        "assessment": ContextAssessment(
            status=JudgementVerdict.SUFFICIENT,
            sufficiency_score=1.0,
            evidence_counts={"fact_0": 1},
            unsupported_claims=[],
            missing_facts=[],
            feedback_queries=[],
            reason="All covered.",
        ),
        "iteration": 1,
    }

    baseline_report = evaluate_run(baseline_result, fixture)
    candidate_report = evaluate_run(candidate_result, fixture)
    delta = compare_runs(baseline_report, candidate_report)

    assert delta["fact_coverage_delta"] >= 0
    assert delta["iteration_delta"] <= 0  # Candidate should have fewer iterations
    assert not delta.get("error")
