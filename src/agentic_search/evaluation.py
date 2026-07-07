"""FRAMES-style evaluation framework for Agentic Search.

Measures 5 dimensions:
  1. Fact coverage: Did we find all required facts?
  2. Fetch coverage: Did we retrieve data for all sub-queries?
  3. Reasoning correctness: Is the answer logically sound?
  4. Citation completeness: Are all claims grounded in sources?
  5. Iteration count: How many iterations were needed?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationFixture:
    """One test case with expected outcomes."""
    question: str
    corpora: dict[str, str]
    expected_facts: list[str]              # Required fact descriptions
    expected_answer_terms: list[str]       # Terms the answer must contain
    expected_status: str = "answered"       # Expected answer status
    max_acceptable_iterations: int = 3
    tags: list[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Results for one evaluation run."""
    fact_coverage: float                    # 0..1
    fetch_coverage: float                   # 0..1
    reasoning_correctness: float            # 0..1
    citation_completeness: float            # 0..1
    iteration_count: int
    passed: bool


@dataclass
class EvaluationReport:
    """Full report for one test case."""
    fixture: EvaluationFixture
    metrics: EvaluationMetrics | None
    answer: Any = None
    errors: list[str] = field(default_factory=list)


def evaluate_run(
    result: dict,
    fixture: EvaluationFixture,
) -> EvaluationReport:
    """Evaluate one graph run against a fixture."""
    errors: list[str] = []
    answer = result.get("answer")
    assessment = result.get("assessment")

    if answer is None:
        return EvaluationReport(fixture=fixture, errors=["No answer produced."])

    # 1. Fact coverage: did assessment cover expected facts?
    fact_coverage = 0.0
    if assessment and fixture.expected_facts:
        matched = sum(1 for ef in fixture.expected_facts
                      if any(ef.lower() in r.lower() for r in [assessment.reason]))
        fact_coverage = matched / len(fixture.expected_facts)

    # 2. Fetch coverage: did we retrieve something?
    snippets = result.get("snippets", {})
    fetch_coverage = min(len(snippets) / max(len(fixture.expected_facts), 1), 1.0)

    # 3. Reasoning correctness: does answer contain expected terms?
    reasoning_correctness = 0.0
    if answer.answer and fixture.expected_answer_terms:
        answer_lower = answer.answer.lower()
        matched = sum(1 for t in fixture.expected_answer_terms if t.lower() in answer_lower)
        reasoning_correctness = matched / len(fixture.expected_answer_terms)

    # 4. Citation completeness
    citation_completeness = 0.0
    if answer.citations and answer.answer:
        # Count how many answer sentences have citations
        sentences = [s.strip() for s in answer.answer.replace("\n", " ").split(". ") if s.strip()]
        cited = sum(1 for s in sentences if any(c in s for c in ["[", "]", "(Source", "Source:"]))
        citation_completeness = cited / len(sentences) if sentences else 0.0

    # 5. Iteration count
    iteration_count = result.get("iteration", 0)

    # Composite pass/fail
    passed = (
        fact_coverage >= 0.8
        and reasoning_correctness >= 0.5
        and iteration_count <= fixture.max_acceptable_iterations
    )

    metrics = EvaluationMetrics(
        fact_coverage=fact_coverage,
        fetch_coverage=fetch_coverage,
        reasoning_correctness=reasoning_correctness,
        citation_completeness=citation_completeness,
        iteration_count=iteration_count,
        passed=passed,
    )

    return EvaluationReport(fixture=fixture, metrics=metrics, answer=answer, errors=errors)


def compare_runs(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> dict:
    """Compare two evaluation runs for ablation studies."""
    if not baseline.metrics or not candidate.metrics:
        return {"error": "Both runs must have metrics"}
    b, c = baseline.metrics, candidate.metrics
    return {
        "fact_coverage_delta": c.fact_coverage - b.fact_coverage,
        "fetch_coverage_delta": c.fetch_coverage - b.fetch_coverage,
        "reasoning_delta": c.reasoning_correctness - b.reasoning_correctness,
        "citation_delta": c.citation_completeness - b.citation_completeness,
        "iteration_delta": c.iteration_count - b.iteration_count,
        "passed_baseline": b.passed,
        "passed_candidate": c.passed,
    }
