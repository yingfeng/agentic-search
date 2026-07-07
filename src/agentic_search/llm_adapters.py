"""LLM-based implementations of all Protocol components.

Each adapter wraps a StructuredLLM with a specific prompt template,
converting natural language to structured JSON outputs.
"""

from __future__ import annotations

from .contracts import Planner, QueryRewriter, SufficiencyJudge, Synthesizer, Drafter, StructuredLLM
from .prompts import load as P
from .schema import SchemaRegistry
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


# ═══════════════════════════════════════════════════════════════
# Root Agent — task delegation
# ═══════════════════════════════════════════════════════════════


class LLMRootAgent:
    """Top-level coordinator — delegates to sub-agents."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def delegate(self, question: str, corpora: dict[str, str]) -> str:
        corpus_list = "\n".join(f"- {cid}: {desc}" for cid, desc in corpora.items())
        result = await self.llm.complete_json(
            system_prompt=P("root_agent.system"),
            user_prompt=P("root_agent.user").format(question=question, corpora=corpus_list),
            output_schema={
                "type": "object",
                "properties": {
                    "question_type": {"type": "string", "enum": ["simple", "multi-hop", "analytical", "summarization"]},
                    "relevant_corpora": {"type": "array", "items": {"type": "string"}},
                    "strategy": {"type": "string"},
                    "success_criteria": {"type": "string"},
                },
                "required": ["question_type", "relevant_corpora", "strategy"],
            },
        )
        return result.get("strategy", "")


# ═══════════════════════════════════════════════════════════════
# Planner — cross-corpus route
# ═══════════════════════════════════════════════════════════════


class LLMPlanner:
    """LLM-based planner — decomposes question into required facts and
    routes each fact to the best corpus."""

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

        corpus_list = "\n".join(
            f"- {cid}: {desc}" for cid, desc in corpora.items()
            if cid not in dead_corpora
        )

        feedback_hint = ""
        if prior_assessment and prior_assessment.feedback_queries:
            hints = "\n".join(f"  Missing: {fq.reason} — try: {fq.query}" for fq in prior_assessment.feedback_queries[:5])
            feedback_hint = f"\nPrevious iteration identified these gaps:\n{hints}\nFocus new routes on filling these gaps."

        result = await self.llm.complete_json(
            system_prompt=P("planner.system"),
            user_prompt=P("planner.user").format(
                question=question, corpora=corpus_list, feedback_hint=feedback_hint,
            ),
            output_schema=self.schema.get("SearchPlan").to_json_schema(),
        )

        routes_raw = result.get("routes", [])
        facts_raw = result.get("required_facts", [])

        facts = []
        for i, f in enumerate(facts_raw):
            if isinstance(f, dict):
                facts.append(RequiredFact(
                    fact_id=f.get("fact_id", f"fact_{i}"),
                    description=f.get("description", ""),
                    required_terms=(f.get("description", "") + " " + f.get("key_terms", "")).lower().split(),
                ))

        routes = []
        for i, r in enumerate(routes_raw):
            if isinstance(r, dict):
                routes.append(SearchRoute(
                    corpus=r.get("target_corpus", "docs"),
                    query=r.get("query", question),
                    rationale=r.get("reasoning", r.get("rationale", "")),
                    required_fact_ids=[r.get("fact_id", f"fact_{i}")],
                ))

        # Fallback: if no routes were generated, create a probe route
        if not routes and corpora:
            best = list(corpora.keys())[0]
            routes.append(SearchRoute(
                corpus=best, query=question,
                rationale="Probe: no explicit route found",
                required_fact_ids=[f.fact_id for f in facts],
            ))

        return RetrievalPlan(question=question, routes=routes, required_facts=facts)


# ═══════════════════════════════════════════════════════════════
# Query Rewriter — multi-query decomposition
# ═══════════════════════════════════════════════════════════════


class LLMQueryRewriter:
    """LLM-based query rewriter — multi-query decomposition."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def rewrite(
        self,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        tried_queries: set[str],
    ) -> list[SubQuery]:
        subqueries: list[SubQuery] = []

        # Add untried routes
        for route in plan.routes:
            if route.corpus in getattr(plan, "_dead_corpora", set()):
                continue
            qk = f"{route.corpus}::{route.query}"
            if qk not in tried_queries:
                subqueries.append(SubQuery(
                    query_id=f"r_{len(subqueries)}",
                    text=route.query,
                    target_corpus=route.corpus,
                    required_fact_ids=route.required_fact_ids,
                ))

        # Expand into multi-formulations via LLM
        if subqueries and len(plan.required_facts) > 1:
            route_text = "\n".join(
                f"- fact(s) {sq.required_fact_ids} in {sq.target_corpus}: {sq.text}"
                for sq in subqueries[:5]
            )
            result = await self.llm.complete_json(
                system_prompt=P("rewriter.system"),
                user_prompt=P("rewriter.user").format(
                    routes=route_text, tried=str(tried_queries or "none"),
                ),
                output_schema=self.schema.get("QueryRewriteResult").to_json_schema(),
            )
            expanded = result.get("subqueries", [])
            if isinstance(expanded, list):
                for eq in expanded:
                    if isinstance(eq, dict):
                        qk = f"{eq.get('target_corpus', 'docs')}::{eq.get('query', '')}"
                        if qk not in tried_queries:
                            subqueries.append(SubQuery(
                                query_id=f"e_{len(subqueries)}",
                                text=eq.get("query", ""),
                                target_corpus=eq.get("target_corpus", "docs"),
                                required_fact_ids=[eq.get("fact_id", "fact_0")],
                            ))

        # Third, add feedback queries from prior assessment
        if prior_assessment and prior_assessment.feedback_queries:
            for fq in prior_assessment.feedback_queries:
                qk = f"{fq.target_corpus}::{fq.query}"
                if qk not in tried_queries:
                    subqueries.append(SubQuery(
                        query_id=f"f_{len(subqueries)}",
                        text=fq.query,
                        target_corpus=fq.target_corpus,
                        required_fact_ids=[fq.related_fact_id],
                    ))

        return subqueries


# ═══════════════════════════════════════════════════════════════
# Drafter — intermediate draft with citations
# ═══════════════════════════════════════════════════════════════


class LLMDrafter:
    """LLM-based drafter — intermediate draft with claim-level citations."""

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def draft(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: list[Snippet],
    ) -> DraftAnswer:
        if not snippets:
            return DraftAnswer(text="No context retrieved yet.", claims=[], citations={})

        context_text = "\n\n".join(
            f"[{sn.snippet_id}] {sn.text[:500]}" for sn in snippets[:15]
        )

        result = await self.llm.complete_json(
            system_prompt=P("drafter.system"),
            user_prompt=P("drafter.user").format(question=question, context=context_text),
            output_schema=self.schema.get("DraftAnswer").to_json_schema(),
        )

        claims_raw = result.get("claims", [])
        draft_text = result.get("draft_text", "")

        claims: list[str] = []
        citations: dict[str, list[str]] = {}

        if isinstance(claims_raw, list):
            for i, c in enumerate(claims_raw):
                if isinstance(c, dict):
                    claim_text = c.get("claim", c.get("text", ""))
                    claim_sources = c.get("snippet_ids", c.get("sources", []))
                    if claim_text:
                        key = f"claim_{len(claims)}"
                        claims.append(claim_text)
                        if isinstance(claim_sources, list):
                            citations[key] = [str(s) for s in claim_sources]
                elif isinstance(c, str):
                    claims.append(c)
                    citations[f"claim_{len(claims) - 1}"] = []

        return DraftAnswer(
            text=draft_text or " ".join(claims),
            claims=claims,
            citations=citations,
        )


# ═══════════════════════════════════════════════════════════════
# Sufficiency Judge (existing, prompt improved)
# ═══════════════════════════════════════════════════════════════


class LLMSufficiencyJudge:
    """LLM-based sufficiency judge — full 3-dimension check (Draft First)."""

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

        context_text = "\n\n".join(
            f"[{sn.snippet_id}] {sn.text[:300]}" for sn in snippets[:15]
        )
        fact_list = "\n".join(
            f"- {f.fact_id}: {f.description}" for f in plan.required_facts
        )

        result = await self.llm.complete_json(
            system_prompt=P("judge.system"),
            user_prompt=P("judge.user").format(
                question=question, facts=fact_list, context=context_text,
            ),
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
            draft_answer=result.get("draft_answer", ""),
        )


# ═══════════════════════════════════════════════════════════════
# Synthesizer (existing)
# ═══════════════════════════════════════════════════════════════


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
        result = await self.llm.complete_json(
            system_prompt=P("synthesizer.system"),
            user_prompt=P("synthesizer.user").format(question=question, context=context_text),
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
