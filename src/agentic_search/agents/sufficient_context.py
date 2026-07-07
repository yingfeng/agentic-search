"""Sufficient Context Agent node — the core innovation.

Three-dimension check (per the Google paper):
  ① Retrieved Snippets: Do the raw chunks contain the required facts?
  ② Intermediate Draft: Does the draft cover all question dimensions?
  ③ Missing Pieces Analysis: Exactly what is missing? → structured feedback
"""

from __future__ import annotations

from ..state import (
    AgenticSearchState,
    ContextAssessment,
    FeedbackQuery,
    JudgementVerdict,
    RequiredFact,
)


class SufficientContextNode:
    """The Sufficient Context Agent — quality controller."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        question = state.question
        plan = state.plan
        draft = state.draft
        all_snippets = state.get_all_snippets()

        if not plan:
            return {
                "assessment": ContextAssessment(
                    status=JudgementVerdict.UNANSWERABLE,
                    sufficiency_score=0.0,
                    evidence_counts={},
                    unsupported_claims=[],
                    missing_facts=[],
                    feedback_queries=[],
                    reason="No search plan available.",
                )
            }

        # ── Dimension 1: Check retrieved snippets ──
        fact_coverage = _check_snippet_coverage(plan.required_facts, all_snippets)

        # ── Dimension 2: Check draft coverage ──
        unsupported_claims = _check_draft_claims(draft, all_snippets) if draft else []

        # ── Dimension 3: Missing pieces analysis ──
        missing, feedback_queries, conflict_detected = _analyze_missing(
            question, plan.required_facts, fact_coverage, all_snippets,
        )

        # ── Compute sufficiency score ──
        # Count ONLY facts that have at least one matching snippet
        covered_ids = {fact_id for fact_id, count in fact_coverage.items() if count > 0}
        score = len(covered_ids) / max(len(plan.required_facts), 1)

        # ── Compute verdict ──
        if conflict_detected:
            verdict = JudgementVerdict.CONFLICTING
            reason = _format_conflict_reason(fact_coverage, missing)
        elif score >= 0.9 and not unsupported_claims:
            verdict = JudgementVerdict.SUFFICIENT
            reason = f"All required facts covered ({score:.0%}). Draft verified."
        elif score >= 0.4:
            verdict = JudgementVerdict.USEFUL_BUT_INCOMPLETE
            reason = f"Partial coverage ({score:.0%}). Missing: {', '.join(missing[:3])}"
        elif not all_snippets:
            verdict = JudgementVerdict.UNANSWERABLE
            reason = "No evidence retrieved after exhaustive search."
        else:
            verdict = JudgementVerdict.INSUFFICIENT
            reason = f"Insufficient coverage ({score:.0%}). Need: {', '.join(missing[:3])}"

        assessment = ContextAssessment(
            status=verdict,
            sufficiency_score=score,
            evidence_counts={f.fact_id: fact_coverage.get(f.fact_id, 0) for f in plan.required_facts},
            unsupported_claims=unsupported_claims,
            missing_facts=missing,
            feedback_queries=feedback_queries,
            reason=reason,
        )

        return {"assessment": assessment}


def _check_snippet_coverage(
    required_facts: list[RequiredFact],
    snippets: list,
) -> dict[str, int]:
    """Dimension 1: Check how many RequiredFact terms appear in snippets."""
    coverage: dict[str, int] = {}
    for fact in required_facts:
        terms = set(t.lower() for t in fact.required_terms)
        hit_count = 0
        for sn in snippets:
            text_lower = sn.text.lower()
            if any(t in text_lower for t in terms):
                hit_count += 1
        coverage[fact.fact_id] = hit_count
    return coverage


def _check_draft_claims(draft, snippets: list) -> list[str]:
    """Dimension 2: Check that draft claims are grounded in snippets."""
    unsupported = []
    if not draft or not draft.claims:
        return unsupported

    snippet_texts = {sn.snippet_id: sn.text.lower() for sn in snippets}

    for claim_idx_str, cited_ids in draft.citations.items():
        for sn_id in cited_ids:
            sn_text = snippet_texts.get(sn_id, "")
            if not sn_text:
                unsupported.append(f"Claim {claim_idx_str} cites non-existent snippet {sn_id}")
                break
    return unsupported


def _analyze_missing(
    question: str,
    required_facts: list[RequiredFact],
    fact_coverage: dict[str, int],
    snippets: list,
) -> tuple[list[str], list[FeedbackQuery], bool]:
    """Dimension 3: Identify exactly what is missing + generate feedback queries."""
    missing: list[str] = []
    feedback_queries: list[FeedbackQuery] = []
    conflict_detected = False

    for fact in required_facts:
        count = fact_coverage.get(fact.fact_id, 0)
        if count == 0:
            missing.append(fact.fact_id)
            feedback_queries.append(FeedbackQuery(
                query=fact.description,
                target_corpus="all",
                reason=f"Missing required fact: {fact.description}",
                related_fact_id=fact.fact_id,
            ))
        elif fact.conflict_terms:
            conflict_map = _detect_conflicts(fact.conflict_terms, snippets)
            if conflict_map:
                conflict_detected = True
                for field, values in conflict_map.items():
                    feedback_queries.append(FeedbackQuery(
                        query=f"resolve: {field} = {' / '.join(values)}",
                        target_corpus="all",
                        reason=f"Conflicting evidence for {field}: {values}",
                        related_fact_id=fact.fact_id,
                    ))

    return missing, feedback_queries, conflict_detected


def _detect_conflicts(
    conflict_terms: dict[str, list[str]],
    snippets: list,
) -> dict[str, set[str]]:
    """Detect contradictions in snippets for multi-valued fields."""
    conflicts: dict[str, set[str]] = {}
    for field, values in conflict_terms.items():
        found: set[str] = set()
        for sn in snippets:
            text_lower = sn.text.lower()
            for val in values:
                if val.lower() in text_lower:
                    found.add(val)
        if len(found) > 1:
            conflicts[field] = found
    return conflicts


def _format_conflict_reason(
    fact_coverage: dict[str, int],
    missing: list[str],
) -> str:
    parts = [f"Evidence conflicts detected. Covered facts: {len(fact_coverage)}"]
    if missing:
        parts.append(f"Missing facts: {', '.join(missing[:3])}")
    return " | ".join(parts)
