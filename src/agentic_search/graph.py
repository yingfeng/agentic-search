"""LangGraph graph construction for Agentic Search.

Architecture (6-node, fully-edged cycle):

    planner ──→ query_rewriter ──→ search_fanout ──→ drafter ──→ sufficient_context
                                                                        │
                                                       ┌────────────────┤
                                                       ▼                ▼
                                                selective_gen    planner (iterate)
                                                       │
                                              ┌────────┴────────┐
                                              ▼                 ▼
                                         synthesizer       abstain
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from .state import AgenticSearchState, AnswerStatus, JudgementVerdict

from .agents.planner import PlanNode
from .agents.query_rewriter import QueryRewriterNode
from .agents.search_fanout import SearchFanoutNode
from .agents.drafter import DrafterNode
from .agents.sufficient_context import SufficientContextNode
from .agents.selective_gen import SelectiveGenerationNode
from .agents.synthesizer import SynthesizerNode
from .agents.abstain import AbstainNode


def build_graph() -> StateGraph:
    """Build the complete Agentic Search LangGraph."""

    builder = StateGraph(AgenticSearchState)

    # ── Register nodes ──
    builder.add_node("planner", PlanNode.run)
    builder.add_node("query_rewriter", QueryRewriterNode.run)
    builder.add_node("search_fanout", SearchFanoutNode.run)
    builder.add_node("drafter", DrafterNode.run)
    builder.add_node("sufficient_context", SufficientContextNode.run)
    builder.add_node("selective_gen", SelectiveGenerationNode.run)
    builder.add_node("synthesizer", SynthesizerNode.run)
    builder.add_node("abstain", AbstainNode.run)

    # ── Edges ──
    builder.set_entry_point("planner")

    # Pipeline edge
    builder.add_edge("planner", "query_rewriter")
    builder.add_edge("query_rewriter", "search_fanout")
    builder.add_edge("search_fanout", "drafter")
    builder.add_edge("drafter", "sufficient_context")

    # Sufficient Context → route based on verdict
    builder.add_conditional_edges(
        "sufficient_context",
        _route_after_sufficiency,
        {
            "selective_gen": "selective_gen",
            "planner": "planner",
            "abstain": "abstain",
        },
    )

    # Selective Gen → synthesis or abstain
    builder.add_conditional_edges(
        "selective_gen",
        _route_after_selective_gen,
        {
            "synthesizer": "synthesizer",
            "abstain": "abstain",
        },
    )

    builder.add_edge("synthesizer", END)
    builder.add_edge("abstain", END)

    # Compile (no checkpointer needed for single-run queries)
    graph = builder.compile()
    return graph


def _route_after_sufficiency(state: AgenticSearchState) -> str:
    """Route based on the Sufficient Context Agent verdict."""
    assessment = state.assessment
    if assessment is None:
        return "abstain"

    status = assessment.status
    if status == JudgementVerdict.SUFFICIENT:
        return "selective_gen"

    if status == JudgementVerdict.USEFUL_BUT_INCOMPLETE:
        return "selective_gen"

    if status == JudgementVerdict.INSUFFICIENT:
        if state.iteration < state.max_iterations:
            return "planner"
        return "abstain"

    if status == JudgementVerdict.CONFLICTING:
        if state.iteration < state.max_iterations:
            return "planner"
        return "abstain"

    return "abstain"


def _route_after_selective_gen(state: AgenticSearchState) -> str:
    """Route based on Selective Generation decision."""
    if state.answer is not None and state.answer.status == AnswerStatus.ANSWERED:
        return "synthesizer"
    return "abstain"
