"""Planner node — decomposes question into search plan."""

from __future__ import annotations

from ..state import AgenticSearchState, RequiredFact, RetrievalPlan, SearchRoute


class PlanNode:
    """Decompose user question into required facts and search routes."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        question = state.question
        corpora = state.corpora
        feedback = state.assessment  # None on first iteration

        # Build fact list from question decomposition
        required_facts = _decompose_question(question)

        # If no corpora available, return empty plan
        if not corpora:
            return {
                "plan": RetrievalPlan(
                    question=question,
                    routes=[],
                    required_facts=required_facts,
                ),
                "iteration": state.iteration + 1,
            }

        # Build search routes: map each fact to one or more corpora
        routes: list[SearchRoute] = []
        for fact in required_facts:
            for corpus_id, description in corpora.items():
                if corpus_id in state.dead_corpora:
                    continue
                if _is_corpus_relevant(fact, description):
                    query = _build_search_query(fact, question)
                    routes.append(SearchRoute(
                        corpus=corpus_id,
                        query=query,
                        rationale=f"Find: {fact.description}",
                        required_fact_ids=[fact.fact_id],
                    ))

        # If no route found, add a probe route to the most promising corpus
        if not routes and corpora:
            best_corpus = _pick_best_corpus(question, corpora, state.dead_corpora)
            if best_corpus:
                routes.append(SearchRoute(
                    corpus=best_corpus,
                    query=question,
                    rationale="Probe: no explicit route found",
                    required_fact_ids=[f.fact_id for f in required_facts],
                ))

        plan = RetrievalPlan(
            question=question,
            routes=routes,
            required_facts=required_facts,
        )
        return {"plan": plan, "iteration": state.iteration + 1}


def _decompose_question(question: str) -> list[RequiredFact]:
    """Heuristic fact decomposition — can be replaced with LLM call."""
    # Simple keyword-based decomposition
    # In production, replace with LLM-based planner
    import re
    entities = re.findall(r'"([^"]+)"', question)
    if not entities:
        # Fallback: treat whole question as one fact
        return [RequiredFact(
            fact_id="fact_0",
            description=question,
            required_terms=[w.lower() for w in question.split() if len(w) > 3],
        )]

    facts = []
    for i, entity in enumerate(entities):
        facts.append(RequiredFact(
            fact_id=f"fact_{i}",
            description=f"Information about: {entity}",
            required_terms=entity.lower().split(),
        ))
    return facts


def _is_corpus_relevant(fact: RequiredFact, description: str) -> bool:
    """Check if a corpus description mentions any required terms."""
    desc_lower = description.lower()
    return any(term in desc_lower for term in fact.required_terms)


def _build_search_query(fact: RequiredFact, question: str) -> str:
    """Build a search query from a fact."""
    return fact.description


def _pick_best_corpus(question: str, corpora: dict[str, str], dead: set[str]) -> str | None:
    """Fallback: pick the most relevant corpus for a probe."""
    q_lower = question.lower()
    best_score = -1
    best_id = None
    for cid, desc in corpora.items():
        if cid in dead:
            continue
        score = sum(1 for w in q_lower.split() if w in desc.lower())
        if score > best_score:
            best_score = score
            best_id = cid
    return best_id
