"""Query Rewriter node — transforms plan routes into optimized search queries."""

from __future__ import annotations

from uuid import uuid4

from ..state import AgenticSearchState, SubQuery


class QueryRewriterNode:
    """Rewrite each search route into one or more optimized queries."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        if not state.plan:
            return {"subqueries": []}

        plan = state.plan
        feedback = state.assessment  # None on first iteration
        subqueries: list[SubQuery] = []

        for route in plan.routes:
            # Skip routes to dead corpora
            if route.corpus in state.dead_corpora:
                continue

            # Skip if this exact query has been tried
            query_key = f"{route.corpus}::{route.query}"
            if query_key in state.tried_queries:
                continue

            sub = SubQuery(
                query_id=uuid4().hex[:8],
                text=route.query,
                target_corpus=route.corpus,
                required_fact_ids=route.required_fact_ids,
            )
            subqueries.append(sub)

        # If feedback_queries exist from prior assessment, convert them
        if feedback and feedback.feedback_queries:
            for fq in feedback.feedback_queries:
                query_key = f"{fq.target_corpus}::{fq.query}"
                if query_key not in state.tried_queries:
                    sub = SubQuery(
                        query_id=uuid4().hex[:8],
                        text=fq.query,
                        target_corpus=fq.target_corpus,
                        required_fact_ids=[fq.related_fact_id],
                    )
                    subqueries.append(sub)

        return {"subqueries": subqueries}
