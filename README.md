# Agentic Search

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-orange)](https://langchain-ai.github.io/langgraph/)

Complete implementation of **Google Agentic RAG** with **Sufficient Context Agent**, built on LangGraph. A production-ready Python scaffold for building RAG systems that can decide whether retrieved context is sufficient, identify missing facts, generate targeted follow-up queries, apply confidence-aware selective generation, and produce claim-level citations.

```text
                    ┌─ plan → route → rewrite ─┐
                    │                           ▼
                    │                    search_fanout
                    │                           │
                    │                    (parallel multi-source)
                    │                           │
                    │                    ┌──────┘
                    │                    ▼
               (iterate)              drafter
                    ▲                    │
                    │                    ▼
                    │            sufficient_context
                    │               │          │
                    │          (verdict)    (feedback)
                    │               │          │
                    └─────── plan ───┤          │
                                     ▼          ▼
                              selective_gen
                              │           │
                          (answer)    (abstain)
```

This repository focuses on the product gap between ordinary "retrieve once, answer once" RAG and a more dependable loop. Key innovation: **dual-signal selective generation** combining sufficiency verdict with self-rated confidence, achieving 2–10% accuracy improvements over confidence-only methods (per the ICLR 2025 paper).

## Key Features

- **8-node LangGraph pipeline** with conditional routing (plan → rewrite → search → draft → judge → selective gen → synthesize/abstain)
- **Sufficient Context Agent** with 3-dimension check:
  - ① Retrieved snippet coverage
  - ② Intermediate draft claim verification
  - ③ Missing pieces analysis with structured feedback queries
- **5-way verdict system**: SUFFICIENT, USEFUL_BUT_INCOMPLETE, INSUFFICIENT, CONFLICTING, UNANSWERABLE
- **Selective Generation**: dual-signal logistic regression fusion (confidence + sufficiency), with configurable strict/balanced/lenient thresholds
- **Autorater implementations**: FLAMe (online, 88%), Gemini (offline, 93%), and Fallback heuristic — all without ground truth answers
- **Protocol-driven plugin architecture**: all 8 components are Protocols — swap in LLM-based, deterministic, or custom implementations
- **Structured output schema registry**: JSON schema validation + automatic repair for LLM outputs
- **FRAMES-style evaluation**: 5-metric evaluation (fact coverage, fetch coverage, reasoning correctness, citation completeness, iteration count) with baseline comparison
- **Iterative retrieval**: multiple search iterations with `prior_assessment.feedback_queries` for precision gap-filling
- **Full traceability**: every iteration recorded, evidence path preserved in final answer

## Contents

- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Core Contracts](#core-contracts)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [License](#license)

## Repository Layout

```text
.
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── agentic_search/
│       ├── __init__.py            # Public API
│       ├── state.py               # LangGraph state + data contracts
│       ├── graph.py               # LangGraph graph construction
│       ├── config.py              # Configuration management
│       ├── contracts.py           # Protocol interfaces (8 components)
│       ├── schema.py              # JSON schema registry + validation + repair
│       ├── evaluation.py          # FRAMES-style 5-metric evaluation
│       ├── orchestrator.py        # Protocol-driven high-level API
│       ├── main.py                # CLI entry point
│       ├── agents/                # LangGraph node implementations
│       │   ├── planner.py         # Task decomposition
│       │   ├── query_rewriter.py  # Query optimization
│       │   ├── search_fanout.py   # Parallel multi-source search
│       │   ├── drafter.py         # Intermediate draft generation
│       │   ├── sufficient_context.py  # Core sufficiency judge
│       │   ├── selective_gen.py   # Dual-signal decision
│       │   ├── synthesizer.py     # Final answer synthesis
│       │   └── abstain.py         # Structured abstention
│       ├── autorater/             # Sufficient context classifiers
│       │   └── __init__.py        # FLAMe / Gemini / Fallback
│       ├── llm_adapters.py        # LLM implementations of all Protocols
│       ├── retrieval/             # Multi-source retrieval backends
│       │   ├── base.py            # Retriever Protocol
│       │   ├── grep_retriever.py  # Codebase grep
│       │   └── rag_retriever.py   # Vector search stub
│       └── utils/
│           ├── confidence.py      # P(Correct) / P(True)
│           └── retry.py           # Exponential backoff
└── tests/
    ├── test_complete_loop.py      # LangGraph end-to-end tests (12 cases)
    ├── test_evaluation.py         # FRAMES evaluation tests
    ├── test_orchestrator.py       # Protocol-driven pipeline tests
    └── test_schema.py             # Schema validation tests
```

## Quick Start

### Install

```bash
pip install -e .

# With optional dependency groups:
pip install -e ".[all]"     # Everything
pip install -e ".[openai]"  # OpenAI LLM support
pip install -e ".[grep]"    # Codebase grep retrieval
```

### Run the LangGraph Pipeline

```python
import asyncio
from agentic_search import build_graph, AgenticSearchState

graph = build_graph()

state = AgenticSearchState(
    question="What is the architecture of the search system?",
    corpora={
        "docs": "Technical documentation about the system architecture",
        "wiki": "Company knowledge base and internal wiki",
    },
    max_iterations=3,
    search_config={"mode": "balanced"},
)

result = asyncio.run(graph.ainvoke(state))
answer = result["answer"]
print(f"Status: {answer.status.value}")
print(f"Confidence: {answer.confidence:.0%}")
print(f"Answer: {answer.answer or '(abstained)'}")
```

### Run via CLI

```bash
python -m agentic_search.main "What is the main function of the planner node?"
```

### Deterministic Demo (No API Keys)

```python
from agentic_search import AgenticRAGOrchestrator, OrchestratorConfig
from agentic_search.state import *

# Mock implementations for testing (see tests/test_orchestrator.py)
answer = await orchestrator.run("test question", corpora={"docs": "Documentation"})
```

### Run Tests

```bash
pytest tests/ -v
```

## How It Works

The scaffold separates orchestration from provider-specific implementation through Protocol interfaces:

1. **Planner** decomposes the original question into required facts and routes each fact to candidate corpora.
2. **QueryRewriter** creates targeted subqueries for each routed fact.
3. **Search Fanout** executes parallel multi-source retrieval (grep, RAG, web).
4. **Drafter** creates an intermediate answer from retrieved snippets.
5. **Sufficient Context Agent** performs the 3-dimension check:
   - Checks required-fact coverage in raw snippets
   - Verifies draft claims are grounded in sources
   - Analyzes missing pieces and generates structured feedback queries
6. **Selective Generation** fuses sufficiency score with self-rated confidence via logistic regression, applying the paper's decision matrix:

   | | Confidence High | Confidence Low |
   |---|---|---|
   | **Sufficient** | ✅ Answer | ⚠️ Answer or re-search |
   | **Insufficient** | ❌ **Must abstain** (most dangerous) | 🔄 Re-search or abstain |

7. **Synthesizer** emits a grounded answer with claim-level citations.
8. **Abstain** returns structured abstention explaining what information is missing.

## Architecture

### LangGraph Node Flow

```
                    planner
                       │
                       ▼
               query_rewriter
                       │
                       ▼
                search_fanout
                  (parallel)
                       │
                       ▼
                   drafter
                       │
                       ▼
             sufficient_context
              ┌────────┴────────┐
              │                 │
         selective_gen      planner
              │              (iterate)
        ┌─────┴─────┐           ▲
        │           │           │
   synthesizer   abstain    (feedback)
        │           │
        └─────┬─────┘
              ▼
             END
```

### Component Protocols

```
Planner            ── protocol: plan(question, corpora, dead, prior, iteration) → RetrievalPlan
QueryRewriter      ── protocol: rewrite(plan, prior_assessment, tried) → list[SubQuery]
Retriever          ── protocol: supports_corpus(corpus) → bool
                       search(query, corpus, top_k) → list[Snippet]
Drafter            ── protocol: draft(question, plan, snippets) → DraftAnswer
SufficiencyJudge   ── protocol: assess(question, plan, snippets, draft) → ContextAssessment
ConfidenceScorer   ── protocol: score(question, snippets) → float (0..1)
Synthesizer        ── protocol: synthesize(...) → GroundedAnswer
StructuredLLM      ── protocol: complete_json(system, user, schema) → dict
```

## Core Contracts

Important contracts live in `src/agentic_search/state.py` and `src/agentic_search/contracts.py`:

- `RequiredFact`: a fact that must be answered, with `required_terms` and optional `conflict_terms`.
- `SearchRoute`: one search direction (corpus + query + rationale).
- `RetrievalPlan`: required facts + search routes.
- `SubQuery`: rewritten query with target corpus and required fact lineage.
- `Snippet`: retrieved evidence with corpus id, document id, score, and metadata.
- `JudgementVerdict`: 5-way enum (SUFFICIENT, USEFUL_BUT_INCOMPLETE, INSUFFICIENT, CONFLICTING, UNANSWERABLE).
- `ContextAssessment`: verdict, sufficiency score, covered/missing facts, feedback queries.
- `FeedbackQuery`: directed query with reason and search hints for missing facts.
- `GroundedAnswer`: final answer with confidence, citations, evidence trail, and missing info.
- `IterationTrace`: full trace for one iteration (plan, queries, snippets, draft, assessment).

## Design Decisions

| Decision | Rationale |
|---|---|
| **Dual-signal selective generation** | Per the paper: combining confidence + sufficiency outperforms either signal alone by 2–10% |
| **5-way verdict** | Binary sufficient/insufficient loses nuance; named cases (conflicting, partial) enable better routing |
| **Protocol-driven plugin architecture** | Every component is a Protocol — swap implementations without changing the orchestration logic |
| **Schema registry with repair** | LLM structured output is never perfect; schema validation + auto-repair ensures reliability |
| **Fact-coverage, not vector-score threshold** | Sufficiency is a fact-level check, not a similarity cutoff — prevents false positives from topically related but insufficient snippets |
| **No built-in LLM/vector DB** | All LLM and retrieval interactions go through Protocols — user provides their own implementations |
| **LangGraph for state management** | Conditional edges, iteration cycles, and checkpointing are built-in graph primitives |
| **Configurable selective gen threshold** | `strict` / `balanced` / `lenient` modes map to different production risk profiles |

## Evaluation

FRAMES-style 5-metric evaluation per `src/agentic_search/evaluation.py`:

| Metric | Description |
|---|---|
| `fact_coverage` | Fraction of required facts found in assessment |
| `fetch_coverage` | Fraction of subqueries that returned actual snippets |
| `reasoning_correctness` | Fraction of expected answer terms present in output |
| `citation_completeness` | Fraction of answer sentences with source citations |
| `iteration_count` | Number of iterations used |

Compare baseline vs candidate runs:

```python
from agentic_search import evaluate_run, compare_runs, EvaluationFixture

fixture = EvaluationFixture(
    question="multi-hop question",
    corpora={"docs": "Docs"},
    expected_facts=["fact about X"],
    expected_answer_terms=["X"],
)

baseline = evaluate_run(baseline_result, fixture)
candidate = evaluate_run(candidate_result, fixture)
delta = compare_runs(baseline, candidate)
```

## Configuration

```python
from agentic_search import OrchestratorConfig

config = OrchestratorConfig(
    max_iterations=3,             # Maximum search iterations
    max_snippets=100,             # Maximum snippets to keep
    search={
        "workspace_path": ".",
        "sources": ["workspace"],  # "workspace", "rag", "web"
    },
    selective_gen={
        "mode": "balanced",       # strict | balanced | lenient
    },
    autorater_mode="fallback",    # flame | gemini | fallback
    corpora={
        "docs": "Technical documentation",
        "wiki": "Internal wiki",
    },
    log_level="INFO",
)
```

## License

MIT. See `LICENSE`.
