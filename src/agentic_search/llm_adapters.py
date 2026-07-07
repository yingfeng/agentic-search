"""LLM-based implementations of all Protocol components.

Each adapter wraps a StructuredLLM with a specific prompt template,
converting natural language to structured JSON outputs.
"""

from __future__ import annotations

from .contracts import Planner, QueryRewriter, SufficiencyJudge, Synthesizer, StructuredLLM
from .schema import SchemaRegistry, parse_with_repair
from .state import (
    RetrievalPlan,
    SearchRoute,
    SubQuery,
    RequiredFact,
    Snippet,
    DraftAnswer,
    ContextAssessment,
    FeedbackQuery,
    GroundedAnswer,
    AnswerStatus,
    JudgementVerdict,
)


class LLMPlanner:
    """LLM-based planner — decomposes question into search routes."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def plan(
        self,
        question: str,
        corpora: dict[str, str],
        dead_corpora: set[str],
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> RetrievalPlan:
        if not corpora:
            return RetrievalPlan(question=question, routes=[], required_facts=[])

        # Build corpus context
        corpus_list = "\n".join(
            f"- {cid}: {desc}" for cid, desc in corpora.items()
            if cid not in dead_corpora
        )

        feedback_context = ""
        if prior_assessment and prior_assessment.feedback_queries:
            fq_text = "\n".join(
                f"  - Missing: {fq.reason}. Try: {fq.query}"
                for fq in prior_assessment.feedback_queries[:5]
            )
            feedback_context = f"\nPrevious iteration feedback:\n{fq_text}\n"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Available corpora:\n{corpus_list}\n"
            f"{feedback_context}\n"
            f"Create a search plan. For each required fact, specify which corpus to search."
        )

        result = await self.llm.complete_json(
            system_prompt="You are a search planner. Break down questions into search routes.",
            user_prompt=user_prompt,
            output_schema=self.schema.get("RetrievalPlan").to_json_schema(),
        )

        # Parse routes from result
        routes_raw = result.get("routes", [])
        routes = []
        for i, r in enumerate(routes_raw):
            if isinstance(r, dict):
                routes.append(SearchRoute(
                    corpus=r.get("corpus", "docs"),
                    query=r.get("query", question),
                    rationale=r.get("rationale", ""),
                    required_fact_ids=[f"fact_{i}"],
                ))

        facts = [RequiredFact(
            fact_id=f"fact_{i}",
            description=r.get("description", r.get("query", "")),
            required_terms=r.get("query", "").lower().split(),
        ) for i, r in enumerate(routes_raw)]

        return RetrievalPlan(question=question, routes=routes, required_facts=facts)


class LLMQueryRewriter:
    """LLM-based query rewriter — optimizes queries for each route."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def rewrite(
        self,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        tried_queries: set[str],
    ) -> list[SubQuery]:
        subqueries = []
        for route in plan.routes:
            query_key = f"{route.corpus}::{route.query}"
            if query_key in tried_queries:
                continue
            subqueries.append(SubQuery(
                query_id=f"q_{len(subqueries)}",
                text=route.query,
                target_corpus=route.corpus,
                required_fact_ids=route.required_fact_ids,
            ))

        # Add feedback queries if present
        if prior_assessment:
            for fq in prior_assessment.feedback_queries:
                qk = f"{fq.target_corpus}::{fq.query}"
                if qk not in tried_queries:
                    subqueries.append(SubQuery(
                        query_id=f"fq_{len(subqueries)}",
                        text=fq.query,
                        target_corpus=fq.target_corpus,
                        required_fact_ids=[fq.related_fact_id],
                    ))

        return subqueries


class LLMSufficiencyJudge:
    """LLM-based sufficiency judge — full 3-dimension check."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
        draft: DraftAnswer | None,
    ) -> ContextAssessment:
        if not plan.required_facts:
            return ContextAssessment(
                status=JudgementVerdict.UNANSWERABLE,
                sufficiency_score=0.0,
                evidence_counts={},
                unsupported_claims=[],
                missing_facts=[],
                feedback_queries=[],
                reason="No required facts defined.",
            )

        context_text = "\n\n".join(f"[{sn.snippet_id}] {sn.text[:300]}" for sn in snippets[:15])
        fact_list = "\n".join(f"- {f.fact_id}: {f.description}" for f in plan.required_facts)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Required facts:\n{fact_list}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Determine if the context is sufficient to answer.\n"
            f"Status must be one of: sufficient, partial, insufficient, conflicting, unanswerable.\n"
            f"If insufficient, list missing facts and suggest feedback queries."
        )

        result = await self.llm.complete_json(
            system_prompt="You assess whether retrieved context is sufficient.",
            user_prompt=user_prompt,
            output_schema=self.schema.get("ContextAssessment").to_json_schema(),
        )

        status = result.get("status", "insufficient")
        score = result.get("sufficiency_score", 0.0)
        missing = result.get("missing_facts", [])
        feedback_raw = result.get("feedback_queries", [])

        feedback_queries = [
            FeedbackQuery(
                query=fq.get("query", ""),
                target_corpus=fq.get("target_corpus", "all"),
                reason=fq.get("reason", ""),
                related_fact_id=fq.get("related_fact_id", ""),
            ) for fq in feedback_raw if isinstance(fq, dict)
        ] if isinstance(feedback_raw, list) else []

        return ContextAssessment(
            status=JudgementVerdict(status),
            sufficiency_score=score,
            evidence_counts={f.fact_id: 0 for f in plan.required_facts},
            unsupported_claims=result.get("unsupported_claims", []),
            missing_facts=missing if isinstance(missing, list) else [],
            feedback_queries=feedback_queries,
            reason=result.get("reason", ""),
        )


class LLMSynthesizer:
    """LLM-based final answer generator."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def synthesize(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
        assessment: ContextAssessment,
        confidence: float,
        status: AnswerStatus,
    ) -> GroundedAnswer:
        context_text = "\n\n".join(f"[{sn.snippet_id}] {sn.text}" for sn in snippets[:10])
        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Answer based ONLY on the provided context. Cite sources."
        )

        result = await self.llm.complete_json(
            system_prompt="You generate grounded answers with citations.",
            user_prompt=user_prompt,
            output_schema=self.schema.get("GroundedAnswer").to_json_schema(),
        )

        citations_raw = result.get("citations", [])
        citations = {}
        if isinstance(citations_raw, list):
            for i, c in enumerate(citations_raw):
                if isinstance(c, dict):
                    citations[c.get("claim", f"claim_{i}")] = c.get("sources", [])
                elif isinstance(c, str):
                    citations[f"claim_{i}"] = [c]

        return GroundedAnswer(
            question=question,
            answer=result.get("answer"),
            status=status,
            confidence=confidence,
            citations=citations,
            iteration_count=0,
            evidence_path=[assessment.reason],
            missing_info=None,
        )
