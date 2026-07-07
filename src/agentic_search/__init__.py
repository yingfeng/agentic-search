"""Agentic Search — Complete Google Agentic RAG Implementation.

Architecture:
  planner → query_rewriter → search_fanout → drafter → sufficient_context → decision
                                                          │            │
                                                   selective_gen    planner(iterate)
                                                          │
                                                   synthesizer/abstain

Infrastructure layers:
  - contracts:   Protocol interfaces for all pluggable components
  - schema:      JSON Schema validation + repair
  - config:      Configuration management
  - evaluation:  FRAMES-style evaluation framework
  - utils/retry: Exponential backoff retry
  - llm_adapters: LLM-based implementations of all Protocol components
  - graph:       LangGraph StateGraph with conditional routing
  - orchestrator: Protocol-driven high-level API
"""

from .graph import build_graph
from .state import (
    AgenticSearchState,
    AnswerStatus,
    GroundedAnswer,
    JudgementVerdict,
)
from .config import OrchestratorConfig, SearchConfig, SelectiveGenConfig
from .contracts import (
    Planner,
    QueryRewriter,
    Retriever,
    Drafter,
    SufficiencyJudge,
    ConfidenceScorer,
    Synthesizer,
    StructuredLLM,
)
from .schema import SchemaRegistry, build_default_registry, SchemaSpec
from .evaluation import EvaluationFixture, evaluate_run, compare_runs
from .orchestrator import AgenticRAGOrchestrator
from .llm_adapters import (
    LLMRootAgent,
    LLMPlanner,
    LLMQueryRewriter,
    LLMDrafter,
    LLMSufficiencyJudge,
    LLMSynthesizer,
)
