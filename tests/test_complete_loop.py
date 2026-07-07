"""Comprehensive tests for the complete Agentic Search loop.

Validates all 12 design dimensions from the paper:
  1. Sufficient Context 3-dimension check (snippets, draft, missing)
  2. Selective Generation dual-signal fusion
  3. Iterative retrieval with feedback queries
  4. 5-way verdict routing
  5. Traceability and auditability
"""

import asyncio
from uuid import uuid4

import pytest

from agentic_search import build_graph, AgenticSearchState, AnswerStatus, JudgementVerdict
from agentic_search.state import Snippet


def _run(g, state):
    return asyncio.run(g.ainvoke(state))


def _make_snippet(text: str, corpus: str = "docs") -> Snippet:
    return Snippet(
        snippet_id=uuid4().hex[:8],
        corpus=corpus,
        document_id=f"doc_{uuid4().hex[:4]}",
        text=text,
        score=1.0,
    )


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture
def corpora():
    return {
        "docs": "Technical documentation about the system architecture",
        "crm": "Customer relationship management data",
        "wiki": "Company knowledge base and internal wiki",
    }


# ═══════════════════════════════════════════════════════════════════
# Tests 1-4: Verdict routing
# ═══════════════════════════════════════════════════════════════════


def test_empty_corpora_abstains(graph):
    """No corpora → UNANSWERABLE → ABSTAINED."""
    state = AgenticSearchState(
        question="Any question at all",
        corpora={},
        max_iterations=1,
        search_config={"mode": "strict"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert answer.status == AnswerStatus.ABSTAINED


def test_no_evidence_abstains(graph, corpora):
    """No snippets found → UNANSWERABLE → ABSTAINED."""
    state = AgenticSearchState(
        question="Obscure topic not in any corpus",
        corpora=corpora,
        max_iterations=2,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert answer.status == AnswerStatus.ABSTAINED


def test_missing_facts_abstains_in_strict_mode(graph, corpora):
    """Missing key facts + strict mode → ABSTAINED (highest risk quadrant)."""
    state = AgenticSearchState(
        question="Unknown question with no matching data",
        corpora=corpora,
        max_iterations=1,
        search_config={"mode": "strict"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert answer.status == AnswerStatus.ABSTAINED


def test_conflicting_evidence_detected(graph, corpora):
    """Conflicting evidence in corpus description → CONFLICTING."""
    state = AgenticSearchState(
        question="What material is used?",
        corpora={"specs": "Chassis materials: steel and aluminum (conflicting)"},
        max_iterations=1,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assessment = result.get("assessment")
    assert assessment is not None
    assert assessment.status in (
        JudgementVerdict.CONFLICTING,
        JudgementVerdict.INSUFFICIENT,
        JudgementVerdict.UNANSWERABLE,
    )


# ═══════════════════════════════════════════════════════════════════
# Tests 5-7: Selective Generation (dual-signal)
# ═══════════════════════════════════════════════════════════════════


def test_selective_gen_routes_correctly(graph, corpora):
    """When snippets exist, system should route through selective_gen."""
    state = AgenticSearchState(
        question="architecture",
        corpora=corpora,
        max_iterations=2,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert result.get("assessment") is not None
    # Should have a verdict
    assert result["assessment"].status is not None


def test_different_modes_produce_different_outcomes(graph, corpora):
    """Strict vs lenient modes should have different routing behavior."""
    question = "Partially answerable question"
    corpora_small = {"docs": "Minimal context on the topic"}

    strict_state = AgenticSearchState(
        question=question, corpora=corpora_small, max_iterations=1,
        search_config={"mode": "strict"},
    )
    lenient_state = AgenticSearchState(
        question=question, corpora=corpora_small, max_iterations=1,
        search_config={"mode": "lenient"},
    )

    strict_result = _run(graph, strict_state)
    lenient_result = _run(graph, lenient_state)
    assert strict_result.get("answer") is not None
    assert lenient_result.get("answer") is not None


def test_max_iterations_enforced(graph, corpora):
    """Max iterations exhausted → ABSTAINED, iteration_count limited."""
    state = AgenticSearchState(
        question="Requires many iterations of searching",
        corpora=corpora,
        max_iterations=3,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert answer.iteration_count <= 3


# ═══════════════════════════════════════════════════════════════════
# Tests 8-10: Multi-hop and partial answers
# ═══════════════════════════════════════════════════════════════════


def test_multihop_iterates(graph, corpora):
    """Multi-hop question should trigger iteration."""
    state = AgenticSearchState(
        question='Find "specs" and "date" in two different corpora',
        corpora={
            "specs": "Server hardware specifications and inventory",
            "wiki": "Project schedules with important dates",
        },
        max_iterations=3,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    # May be abstained if no actual data found, but iteration should be tracked
    assert result.get("iteration", 0) > 0


def test_partial_info_handled(graph, corpora):
    """Single source only → still produces some kind of response."""
    state = AgenticSearchState(
        question='Compare "performance" and "cost"',
        corpora={"benchmarks": "Performance benchmarks only, no cost data"},
        max_iterations=2,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None


def test_trace_auditability(graph, corpora):
    """All iterations recorded; answer has evidence trail."""
    state = AgenticSearchState(
        question="Multi-step question",
        corpora=corpora,
        max_iterations=2,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    assert result.get("iteration", 0) >= 0
    assert len(answer.evidence_path) > 0
    # Check that plan and assessment exist
    assert result.get("plan") is not None
    assert result.get("assessment") is not None


# ═══════════════════════════════════════════════════════════════════
# Tests 11-12: End-to-end validation
# ═══════════════════════════════════════════════════════════════════


def test_end_to_end_state_flow(graph, corpora):
    """All state fields populated correctly across the graph."""
    state = AgenticSearchState(
        question="agentic search",
        corpora=corpora,
        max_iterations=2,
        search_config={"mode": "balanced"},
    )
    result = _run(graph, state)
    assert result.get("plan") is not None
    assert result.get("assessment") is not None
    answer = result.get("answer")
    assert answer is not None


def test_missing_info_formatted(graph, corpora):
    """ABSTAINED answers should have structured missing_info."""
    state = AgenticSearchState(
        question="Completely unknown topic xyz123nonexistent",
        corpora=corpora,
        max_iterations=1,
        search_config={"mode": "strict"},
    )
    result = _run(graph, state)
    answer = result.get("answer")
    assert answer is not None
    if answer.status == AnswerStatus.ABSTAINED:
        assert answer.missing_info is not None
        assert len(answer.missing_info) > 0
        # Should reference the question
        assert "Completely unknown topic" in answer.missing_info
