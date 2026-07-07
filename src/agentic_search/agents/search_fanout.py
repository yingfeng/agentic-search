"""Search Fanout node — parallel multi-source retrieval."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from ..state import AgenticSearchState, Snippet, SubQuery


class SearchFanoutNode:
    """Execute multiple subqueries in parallel across all configured retrievers."""

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        subqueries = state.subqueries
        if not subqueries:
            return {}

        # Get configured retrievers
        retrievers = _get_retrievers(state.search_config)

        # Fan out: each subquery → all retrievers that match its target corpus
        tasks = []
        for sub in subqueries:
            for retriever in retrievers:
                if retriever.supports_corpus(sub.target_corpus):
                    tasks.append(_search_one(retriever, sub))

        # Parallel execution
        results: list[list[Snippet]] = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge new snippets into existing state
        new_snippets: dict[str, Snippet] = {}
        tried: set[str] = set(state.tried_queries)

        for sub in subqueries:
            tried.add(f"{sub.target_corpus}::{sub.text}")

        for result in results:
            if isinstance(result, Exception):
                continue
            for sn in result:
                if sn.snippet_id not in state.snippets:
                    new_snippets[sn.snippet_id] = sn

        return {
            "snippets": {**state.snippets, **new_snippets},
            "tried_queries": tried,
        }


async def _search_one(retriever: "BaseRetriever", sub: SubQuery) -> list[Snippet]:
    """Single retriever call with error handling."""
    try:
        return await retriever.search(sub.text, corpus=sub.target_corpus, top_k=10)
    except Exception:
        return []


def _get_retrievers(config: dict) -> list["BaseRetriever"]:
    """Load configured retrievers."""
    # In production, load from config. For now, return defaults.
    from ..retrieval.grep_retriever import GrepRetriever
    from ..retrieval.rag_retriever import RAGRetriever

    retrievers: list["BaseRetriever"] = []
    if "workspace" in config.get("sources", []):
        retrievers.append(GrepRetriever(workspace=config.get("workspace_path", ".")))
    if "rag" in config.get("sources", []):
        retrievers.append(RAGRetriever(
            db_path=config.get("rag_db_path"),
            embed_model=config.get("embed_model"),
        ))
    # Always add a basic text searcher as fallback
    if not retrievers:
        from ..retrieval.grep_retriever import GrepRetriever
        retrievers.append(GrepRetriever(workspace="."))
    return retrievers


from ..retrieval.base import BaseRetriever
