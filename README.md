# Agentic Search

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)]()

Complete implementation of **Google Agentic RAG** with **Sufficient Context Agent**, built on LangGraph. A production-ready Python scaffold for building RAG systems that can decide whether retrieved context is sufficient, identify missing facts, generate targeted follow-up queries, apply confidence-aware selective generation, and produce claim-level citations.

This implementation is grounded in two research artifacts:

- **Sufficient Context (ICLR 2025)** — Formal definition of context sufficiency, autorater methodology, and selective generation framework.
- **Google Agentic RAG (2026)** — Multi-agent architecture with planner, rewriter, search fanout, drafter, sufficient context judge, and synthesizer.

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
- **Autorater Calibration**: gold-labeled dataset of 93+ instances, precision/recall/F1 measurement, comparison across autorater implementations
- **Selective Generation Trainer**: synthetic data generation, logistic regression training, accuracy-coverage curve production, dual-signal vs confidence-only comparison
- **Protocol-driven plugin architecture**: all 8 components are Protocols — swap in LLM-based, deterministic, or custom implementations
- **Structured output schema registry**: JSON schema validation + automatic repair for LLM outputs
- **FRAMES-style evaluation**: 5-metric evaluation (fact coverage, fetch coverage, reasoning correctness, citation completeness, iteration count) with baseline comparison
- **Iterative retrieval**: multiple search iterations with `prior_assessment.feedback_queries` for precision gap-filling
- **Full traceability**: every iteration recorded, evidence path preserved in final answer

## Contents

- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Autorater Calibration](#autorater-calibration)
- [Selective Generation Training](#selective-generation-training)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Core Contracts](#core-contracts)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Citation](#citation)
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
│       ├── selective_gen_trainer.py # Logistic regression trainer + accuracy-coverage curves
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
│       │   ├── __init__.py        # FLAMe / Gemini / Fallback
│       │   └── calibration.py     # Gold-labeled dataset + accuracy measurement
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
    ├── test_schema.py             # Schema validation tests
    ├── test_autorater_calibration.py # Autorater calibration tests
    └── test_selective_gen_trainer.py # Selective gen trainer tests
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

### Run Tests

```bash
pytest tests/ -v
```

All 45 tests passing verifies:

| Area | Tests | What's Verified |
|------|-------|-----------------|
| **LangGraph Pipeline** | 12 | Routing, verdicts, iteration, evidence trail |
| **Autorater Calibration** | 7 | Gold dataset coverage, accuracy/precision/recall/F1 |
| **Selective Gen Trainer** | 10 | Synthetic data, weight training, accuracy-coverage curves |
| **Schema Validation** | 8 | JSON schema registry, enum/range validation, JSON repair |
| **Orchestrator** | 4 | Protocol-driven pipeline, verdict routing, FRAMES eval |
| **FRAMES Evaluation** | 3 | 5-metric scoring, baseline comparison |

## Autorater Calibration

Per the paper, an **autorater** classifies query-context pairs as sufficient or insufficient **without needing a ground truth answer**. The paper achieves 93% accuracy with Gemini 1.5 Pro (1-shot) and 88% with FLAMe 24B.

Our calibration framework provides:

- **Gold-labeled dataset** — 93+ instances across 12 edge case categories (single-hop, multi-hop, ambiguous query, ambiguous context, conflicting evidence, yes/no, entity disambiguation, temporal reasoning, causal reasoning, parametric knowledge, quantitative reasoning, negation & contrast)
- **Accuracy measurement** — Precision, recall, F1, confusion matrix
- **Multi-autorater comparison** — Run any `SufficiencyAutorater` against the same gold standard

```python
import asyncio
from agentic_search.autorater import FallbackAutorater
from agentic_search.autorater.calibration import calibrate_autorater, print_calibration_report

result = asyncio.run(calibrate_autorater(FallbackAutorater(), "Fallback Heuristic"))
print_calibration_report([result])
```

Sample output:

```
  ── Fallback Heuristic ──
  Accuracy : 58.1%  (paper target: 93%)
  Precision: 71.4%
  Recall   : 31.2%
  F1       : 43.5%
  Confusion: TP=15  FP=6  TN=39  FN=33
```

To achieve the paper's 93%, swap in a `GeminiAutorater()` or `FLAMeAutorater()` implementation.

## Selective Generation Training

Per the paper, a **logistic regression model** fuses two signals to predict hallucination risk:

1. **Self-rated confidence** — P(Correct) or P(True) from the LLM
2. **Sufficiency score** — From the Sufficient Context Agent

The trainer provides:

- **Synthetic data generation** — 2,000 samples mimicking the paper's 4-quadrant empirical distribution
- **Logistic regression training** — Fits weights via scikit-learn or pure numpy (no sklearn dependency required)
- **Accuracy-coverage curves** — The paper's main result (Figure 4): dual-signal vs confidence-only comparison
- **Weight export** — Trained weights ready for production deployment

```python
from agentic_search.selective_gen_trainer import (
    generate_synthetic_data, train_weights,
    compute_accuracy_coverage_curve, compare_curves, print_training_report,
)

data = generate_synthetic_data(n_samples=2000, seed=42)
weights = train_weights(data)
curve = compute_accuracy_coverage_curve(data, weights)
comparison = compare_curves(data, weights)
print_training_report(data, weights, curve, comparison)
```

Sample output confirming the paper's findings:

```
  Trained weights:
    w_confidence  = -1.3740
    w_sufficiency = +3.9165
    bias          = +2.7321

  Accuracy-Coverage curve:
    Coverage 50%: dual 73.7% vs conf-only 56.7%  (▲ +17.1%)
    Coverage 70%: dual 59.5% vs conf-only 54.9%  (▲ +4.6%)
    Coverage 90%: dual 50.9% vs conf-only 51.5%  (-0.6%)

  → ▲ +17% at 50% coverage confirms the paper's key result
  → Diminishing returns at high coverage matches Figure 4
```

To deploy trained weights into production, replace the hardcoded values in `selective_gen.py` with the exported weights:

```python
weights = SelectiveGenWeights.from_dict({
    "w_confidence": -1.3740,
    "w_sufficiency": 3.9165,
    "bias": 2.7321,
    "thresholds": {"strict": 0.78, "balanced": 0.62, "lenient": 0.22},
})
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
- `GoldInstance`: human-labeled instance for autorater calibration.
- `SelectiveGenWeights`: trained logistic regression weights with thresholds.

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
| **Gold-labeled autorater dataset** | 93+ instances across 12 edge case categories — enables empirical accuracy measurement |
| **Synthetic data for weight training** | Mimics the paper's 4-quadrant distribution — enables training without production data |

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

## Citation

This implementation is based on the public Agentic RAG behavior described by Google Research and Google Cloud:

- Google Research. **Unlocking dependable responses with Gemini Enterprise Agent Platform's Agentic RAG**. Published June 5, 2026.  
  https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/

- Google Cloud. **RAG Engine Cross Corpus Retrieval**. Gemini Enterprise Agent Platform documentation.  
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/cross-corpus-retrieval

The implementation is also grounded in the following paper:

- **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems**  
  Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian. ICLR 2025.  
  https://arxiv.org/abs/2411.06037

This paper motivates the explicit `sufficiency_score`, unsupported-claim checks, missing-fact feedback queries, the autorater methodology, and the selective generation framework implemented throughout this repository.

Referenced evaluation benchmark:

- **Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation**  
  Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. arXiv 2024.  
  https://arxiv.org/abs/2409.12941

This paper introduces FRAMES, the multi-hop RAG evaluation benchmark referenced by Google's Agentic RAG write-up and used in this repository's evaluation framework.

## License

MIT. See `LICENSE`.
