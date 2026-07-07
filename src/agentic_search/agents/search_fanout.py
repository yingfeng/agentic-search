"""Search Fanout node — parallel multi-source retrieval.

Features:
  - Neighbor stitching: extend KNN hits to sequential neighborhoods
  - Mechanical search statistics: code-calculated coverage and novelty
  - Hierarchical degradation: primary → cache → graceful empty
"""

from __future__ import annotations

import asyncio
from collections import Counter
from uuid import uuid4

from ..state import AgenticSearchState, Snippet, SubQuery


class SearchFanoutNode:
    """Execute multiple subqueries in parallel across all configured retrievers."""

    NEIGHBOR_STITCH_PAD = 3          # Extend each hit by ±3 seq positions
    NEIGHBOR_BRIDGE_GAP = 5          # Merge neighborhoods within 5 positions

    @staticmethod
    async def run(state: AgenticSearchState) -> dict:
        subqueries = state.subqueries
        if not subqueries:
            return {}

        retrievers = _get_retrievers(state.search_config)
        dead_corpora: set[str] = set(state.dead_corpora)
        previous_count = len(state.snippets)

        # ── 1. Fan-out: each subquery → matching retrievers with degradation ──
        tasks = []
        retriever_map: dict[int, str] = {}  # task_id → corpus
        for sub in subqueries:
            if sub.target_corpus in dead_corpora:
                continue
            qk = f"{sub.target_corpus}::{sub.text}"
            if qk in state.tried_queries:
                continue
            state.tried_queries.add(qk)

            for retriever in retrievers:
                if retriever.supports_corpus(sub.target_corpus):
                    idx = len(tasks)
                    tasks.append(_search_hierarchical(retriever, sub))
                    retriever_map[idx] = sub.target_corpus

        results: list[list[Snippet]] = await asyncio.gather(*tasks, return_exceptions=True)

        # ── 2. Collect new snippets with neighbor stitching ──
        new_snippets: dict[str, Snippet] = {}
        corpus_hits: Counter = Counter()
        corpus_queries: Counter = Counter()

        for idx, result in enumerate(results):
            corpus = retriever_map.get(idx, "unknown")
            corpus_queries[corpus] += 1

            if isinstance(result, Exception):
                dead_corpora.add(corpus)
                continue

            # Stitch neighbors
            stitched = _stitch_neighbors(result, SearchFanoutNode.NEIGHBOR_STITCH_PAD)
            corpus_hits[corpus] += len(stitched) if isinstance(stitched, list) else 0

            for sn in (stitched if isinstance(stitched, list) else result):
                if sn.snippet_id not in state.snippets and sn.snippet_id not in new_snippets:
                    new_snippets[sn.snippet_id] = sn

        # ── 3. Mechanical search statistics ──
        total_chunks = previous_count + len(new_snippets)
        novelty = len(new_snippets)
        coverage_pct = 0.0
        if state.plan and state.plan.required_facts:
            covered = sum(1 for f in state.plan.required_facts
                          if any(f.fact_id in sn.text for sn in state.snippets.values()))
            coverage_pct = covered / len(state.plan.required_facts)

        return {
            "snippets": {**state.snippets, **new_snippets},
            "tried_queries": set(state.tried_queries),
            "dead_corpora": dead_corpora,
            "search_stats": {
                "total_chunks": total_chunks,
                "new_chunks_this_iteration": novelty,
                "coverage_fraction": coverage_pct,
                "novelty_delta": novelty,
                "corpora_queried": dict(corpus_queries),
                "corpora_hits": dict(corpus_hits),
            },
        }


# ── Hierarchical degradation chain ──

async def _search_hierarchical(retriever, sub: SubQuery) -> list[Snippet]:
    """Search with hierarchical degradation: primary → cache → graceful empty."""
    # Level 1: Primary search
    try:
        results = await retriever.search(sub.text, corpus=sub.target_corpus, top_k=10)
        if results:
            return results
    except Exception:
        pass

    # Level 2: Cached/fallback (if retriever has a cache method)
    if hasattr(retriever, "search_cached") and callable(getattr(retriever, "search_cached")):
        try:
            results = await retriever.search_cached(sub.text, corpus=sub.target_corpus, top_k=5)
            if results:
                return results
        except Exception:
            pass

    # Level 3: Graceful empty
    return []


# ── Neighbor stitching ──

def _stitch_neighbors(snippets: list[Snippet], pad: int = 3) -> list[Snippet]:
    """Extend each KNN hit to its sequential neighborhood.

    Each hit's seq position is read from metadata.get('seq', 0).
    Neighborhoods within 'bridge_gap' of each other are merged.
    Returns the stitched set with original snippet_ids + extended coverage.
    """
    if not snippets:
        return []

    # Collect positions per document
    doc_groups: dict[str, list[tuple[int, Snippet]]] = {}
    for sn in snippets:
        doc_id = sn.document_id
        seq = _get_seq(sn)
        seq_start = max(0, seq - pad)
        seq_end = seq + pad
        doc_groups.setdefault(doc_id, []).append((seq_start, seq_end, seq, sn))

    stitched = []
    seen_ids = set()

    for doc_id, groups in doc_groups.items():
        # Sort by original seq position
        groups.sort(key=lambda x: x[2])

        # Merge overlapping neighborhoods
        merged_starts, merged_ends = _merge_ranges(
            [(s, e) for s, e, _, _ in groups],
            gap=SearchFanoutNode.NEIGHBOR_BRIDGE_GAP,
        )

        for start, end in merged_starts:
            for _, _, _, sn in groups:
                if sn.snippet_id not in seen_ids:
                    stitched.append(sn)
                    seen_ids.add(sn.snippet_id)

    return stitched


def _get_seq(sn: Snippet) -> int:
    """Get sequence position from snippet metadata."""
    raw = sn.metadata.get("seq", 0)
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _merge_ranges(
    ranges: list[tuple[int, int]],
    gap: int = 5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Merge overlapping/adjacent ranges."""
    if not ranges:
        return [], []

    sorted_r = sorted(ranges)
    merged = [sorted_r[0]]

    for start, end in sorted_r[1:]:
        last_s, last_e = merged[-1]
        if start <= last_e + gap:
            merged[-1] = (last_s, max(last_e, end))
        else:
            merged.append((start, end))

    return merged, merged


# ── Retriever loading ──

def _get_retrievers(config: dict) -> list["BaseRetriever"]:
    """Load configured retrievers with hierarchical fallback paths."""
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
    if not retrievers:
        retrievers.append(GrepRetriever(workspace="."))
    return retrievers


from ..retrieval.base import BaseRetriever
