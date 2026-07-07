"""Protocol-aware orchestrator — high-level API for running the full pipeline."""

from __future__ import annotations

import logging

from .config import OrchestratorConfig
from .contracts import (
    Planner,
    QueryRewriter,
    Retriever,
    Drafter,
    SufficiencyJudge,
    ConfidenceScorer,
    Synthesizer,
)
from .state import (
    AgenticSearchState,
    ContextAssessment,
    FeedbackQuery,
    JudgementVerdict,
    AnswerStatus,
    GroundedAnswer,
)
from .schema import SchemaRegistry, build_default_registry

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    """Protocol-driven orchestrator for the Agentic Search pipeline."""

    def __init__(
        self,
        planner: Planner,
        rewriter: QueryRewriter,
        retrievers: list[Retriever],
        drafter: Drafter,
        judge: SufficiencyJudge,
        scorer: ConfidenceScorer,
        synthesizer: Synthesizer,
        config: OrchestratorConfig | None = None,
    ):
        self.planner = planner
        self.rewriter = rewriter
        self.retrievers = retrievers
        self.drafter = drafter
        self.judge = judge
        self.scorer = scorer
        self.synthesizer = synthesizer
        self.config = config or OrchestratorConfig()
        self.registry = build_default_registry()

    async def run(self, question: str, corpora: dict[str, str]) -> GroundedAnswer:
        """Run the full Agentic RAG pipeline."""
        state = AgenticSearchState(
            question=question,
            corpora=corpora,
            max_iterations=self.config.max_iterations,
            search_config={
                "mode": self.config.selective_gen.mode,
                "max_snippets": self.config.max_snippets,
            },
        )

        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            # ── 1. Plan ──
            state.plan = await self.planner.plan(
                question, corpora, state.dead_corpora,
                state.assessment, iteration,
            )

            # ── 2. Rewrite ──
            state.subqueries = await self.rewriter.rewrite(
                state.plan, state.assessment, state.tried_queries,
            )

            if not state.subqueries:
                logger.info("No subqueries to execute")
                break

            # ── 3. Search (parallel fan-out) ──
            search_tasks = []
            for sub in state.subqueries:
                qk = f"{sub.target_corpus}::{sub.text}"
                state.tried_queries.add(qk)
                for retriever in self.retrievers:
                    if retriever.supports_corpus(sub.target_corpus):
                        search_tasks.append(self._search_one(retriever, sub))

            import asyncio
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Search error: {result}")
                    continue
                for sn in result:
                    if sn.snippet_id not in state.snippets:
                        state.snippets[sn.snippet_id] = sn

            state.iteration = iteration + 1

            # ── 4. Draft ──
            state.draft = await self.drafter.draft(
                question, state.plan, state.get_all_snippets(),
            )

            # ── 5. Sufficiency Check ──
            state.assessment = await self.judge.assess(
                question, state.plan, state.get_all_snippets(), state.draft,
            )

            # ── 6. Decision ──
            verdict = state.assessment.status
            if verdict == JudgementVerdict.SUFFICIENT:
                confidence = await self.scorer.score(question, state.get_all_snippets())
                return await self.synthesizer.synthesize(
                    question, state.plan, state.get_all_snippets(),
                    state.assessment, confidence, AnswerStatus.ANSWERED,
                )

            if verdict == JudgementVerdict.USEFUL_BUT_INCOMPLETE:
                confidence = await self.scorer.score(question, state.get_all_snippets())
                if confidence >= self.config.selective_gen.threshold:
                    return await self.synthesizer.synthesize(
                        question, state.plan, state.get_all_snippets(),
                        state.assessment, confidence, AnswerStatus.PARTIAL,
                    )
                # Low confidence → continue searching

            if verdict == JudgementVerdict.INSUFFICIENT:
                if iteration < self.config.max_iterations - 1:
                    continue  # Iterate
                break  # Exhausted

            if verdict == JudgementVerdict.CONFLICTING:
                if iteration < self.config.max_iterations - 1:
                    continue  # Try to resolve
                break

            if verdict == JudgementVerdict.UNANSWERABLE:
                break

        # ── Fallback: abstain ──
        return GroundedAnswer(
            question=question,
            answer=None,
            status=AnswerStatus.ABSTAINED,
            confidence=0.0,
            citations={},
            iteration_count=state.iteration,
            evidence_path=[state.assessment.reason] if state.assessment else [],
            missing_info=state.assessment.reason if state.assessment else "Unable to find sufficient information.",
        )

    async def _search_one(self, retriever: Retriever, sub) -> list:
        try:
            return await retriever.search(sub.text, corpus=sub.target_corpus, top_k=10)
        except Exception as e:
            logger.warning(f"Search error on {sub.target_corpus}/{sub.text}: {e}")
            return []
