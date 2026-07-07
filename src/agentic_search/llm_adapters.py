"""LLM-based implementations of all Protocol components.

Each adapter wraps a StructuredLLM with a specific prompt template,
converting natural language to structured JSON outputs.
"""

from __future__ import annotations

from .contracts import Planner, QueryRewriter, SufficiencyJudge, Synthesizer, Drafter, StructuredLLM
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
    """Top-level coordinator — analyzes the question, decides strategy,
    delegates to sub-agents (planner, rewriter, drafter, judge, synthesizer).

    Per the blog: the Root Agent parses the user's request and delegates
    to specialized sub-agents for each phase of the pipeline.
    """

    _SYSTEM_PROMPT = (
        "You are a Root Agent in a multi-agent RAG system. "
        "Given a user question and available data corpora, determine: "
        "1) What type of question this is (simple fact, multi-hop comparison, "
        "analytical, summarization) "
        "2) Which corpora are relevant "
        "3) What search strategy to use "
        "4) What the success criteria are.\n\n"
        "Output a delegation plan."
    )

    def __init__(self, llm: StructuredLLM, schema_registry: SchemaRegistry):
        self.llm = llm
        self.schema = schema_registry

    async def delegate(self, question: str, corpora: dict[str, str]) -> str:
        """Analyze question and return delegation strategy description."""
        corpus_list = "\n".join(f"- {cid}: {desc}" for cid, desc in corpora.items())
        result = await self.llm.complete_json(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=f"Question: {question}\n\nAvailable corpora:\n{corpus_list}",
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
    routes each fact to the best corpus.

    Per the blog: the Planner Agent maps out information pathways,
    deciding which databases/indices to search for each fact.
    """

    _SYSTEM_PROMPT = (
        "You are a Planner Agent in a multi-agent RAG system. "
        "Given a user question and a set of available corpora (each with a description):\n\n"
        "1. Decompose the question into the specific facts needed to answer it.\n"
        "2. For each fact, determine which corpus is most likely to contain it.\n"
        "3. Create a search route: a targeted query for each fact to its best corpus.\n\n"
        "Rules:\n"
        "- If a fact may exist in multiple corpora, create a route for each.\n"
        "- If no corpus seems relevant for a fact, pick the most plausible one as a probe.\n"
        "- Each route must have a specific, search-optimized query (not the original question).\n"
        "- Output routes and facts in the specified JSON schema."
    )

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

        user_prompt = (
            f"Question: {question}\n"
            f"Available corpora:\n{corpus_list}\n"
            f"{feedback_hint}\n\n"
            f"Decompose into required facts and routes. "
            f"For each fact: assign it to the best corpus and write a targeted search query."
        )

        result = await self.llm.complete_json(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
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
    """LLM-based query rewriter — decomposes each route's query into
    multiple optimized search queries.

    Per the blog: the Query Rewriter translates the request into multiple
    search queries, e.g. turning "What's up with Project X?" into
    "Status report for Project X Q3" and "Key blockers for Project X team."
    """

    _SYSTEM_PROMPT = (
        "You are a Query Rewriter Agent. "
        "Given a question and a set of search routes (each targeting a specific fact and corpus):\n\n"
        "For each route, expand the query into 1–2 different formulations. "
        "Use synonyms, different phrasings, and specific terms to maximize recall.\n\n"
        "Rules:\n"
        "- Each subquery must target exactly one fact.\n"
        "- Queries should be concise and search-optimized (3-8 words).\n"
        "- Avoid repeating queries that have already been attempted.\n"
        "- If feedback_queries from a prior assessment exist, prioritize those."
    )

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

        # First, add untried routes
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

        # Second, expand each route into multiple formulations via LLM
        if subqueries and len(plan.required_facts) > 1:
            route_text = "\n".join(
                f"- fact(s) {sq.required_fact_ids} in {sq.target_corpus}: {sq.text}"
                for sq in subqueries[:5]
            )
            result = await self.llm.complete_json(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=(
                    f"Original routes:\n{route_text}\n\n"
                    f"Expand each into alternate search formulations. "
                    f"Already tried: {tried_queries or 'none'}."
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
    """LLM-based drafter — generates intermediate draft with claim-level citations.

    Per the blog: the Drafter creates an intermediate answer from retrieved
    snippets. Each claim maps to its source snippet IDs for traceability.
    The draft is then used by the Sufficient Context Agent to verify coverage.
    """

    _SYSTEM_PROMPT = (
        "You are a Drafter Agent. "
        "Given a question and a set of retrieved snippets (each with an ID):\n\n"
        "1. Extract all factual claims that can support an answer.\n"
        "2. For each claim, cite the exact snippet IDs that support it.\n"
        "3. Write a coherent draft answer synthesizing all claims.\n\n"
        "Rules:\n"
        "- ONLY use information present in the provided snippets.\n"
        "- Each claim must cite at least one snippet ID.\n"
        "- If you cannot make a claim without speculating, leave it out.\n"
        "- Claims should be atomic: one fact per claim."
    )

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
            return DraftAnswer(
                text="No context retrieved yet.",
                claims=[],
                citations={},
            )

        context_text = "\n\n".join(
            f"[{sn.snippet_id}] {sn.text[:500]}" for sn in snippets[:15]
        )

        result = await self.llm.complete_json(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=f"Question: {question}\n\nRetrieved snippets:\n{context_text}",
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
    """LLM-based sufficiency judge — full 3-dimension check.

    Prompt design incorporates the "Draft First" technique: force the LLM
    to mentally draft an answer before judging sufficiency.
    """

    _SYSTEM_PROMPT = (
        "You are a Sufficient Context Agent in an agentic RAG system.\n\n"
        "Follow these steps:\n\n"
        "Step 1 — Mentally draft a potential answer.\n"
        "   Based ONLY on the retrieved snippets, draft the best answer you can.\n\n"
        "Step 2 — Identify gaps.\n"
        "   For each required fact, check: is the information present in the snippets?\n"
        "   Are there any parts of the question left unanswered?\n"
        "   Is any critical information missing or contradictory?\n\n"
        "Step 3 — Classify the verdict.\n"
        "   - sufficient: All required facts are covered.\n"
        "   - partial: Some facts covered, but not all.\n"
        "   - insufficient: Key facts are missing.\n"
        "   - conflicting: Multiple contradictory values for the same fact.\n"
        "   - unanswerable: No relevant information found.\n\n"
        "Step 4 — Generate feedback.\n"
        "   For each missing fact, provide a specific feedback query.\n\n"
        "Respond using the provided JSON schema."
    )

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

        user_prompt = (
            f"Question: {question}\n\n"
            f"Required facts:\n{fact_list}\n\n"
            f"Retrieved context:\n{context_text}"
        )

        result = await self.llm.complete_json(
            system_prompt=self._SYSTEM_PROMPT,
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
