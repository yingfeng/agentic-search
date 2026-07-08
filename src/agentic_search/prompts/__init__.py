"""Independent prompt template library.

All LLM prompts live here, separate from adapter code.
Each template uses `{variable}` placeholders (PEP 3101 str.format).
Load via `load()` — cached, zero-dependency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def load(name: str) -> str:
    """Load a prompt template by name (cached)."""
    return _load_all()[name]


@lru_cache(maxsize=1)
def _load_all() -> dict[str, str]:
    return {
        # ── Root Agent ──
        "root_agent.system": (
            "You are a Root Agent in a multi-agent RAG system.\n"
            "Given a user question and available data corpora, determine:\n"
            "1) What type of question this is (simple fact, multi-hop comparison,\n"
            "   analytical, summarization)\n"
            "2) Which corpora are relevant\n"
            "3) What search strategy to use\n"
            "4) What the success criteria are.\n\n"
            "Output a delegation plan."
        ),
        "root_agent.user": (
            "Question: {question}\n\nAvailable corpora:\n{corpora}"
        ),

        # ── Planner ──
        "planner.system": (
            "You are a Planner Agent in a multi-agent RAG system.\n"
            "Given a user question and available corpora (each with a description):\n\n"
            "1. Decompose the question into specific facts needed to answer it.\n"
            "2. For each fact, determine which corpus is most likely to contain it.\n"
            "3. Create a search route: a targeted query for each fact to its best corpus.\n\n"
            "Rules:\n"
            "- If a fact may exist in multiple corpora, create a route for each.\n"
            "- If no corpus seems relevant, pick the most plausible one as a probe.\n"
            "- Each route must have a specific, search-optimized query.\n"
            "- Output routes and facts in the specified JSON schema."
        ),
        "planner.user": (
            "Question: {question}\n"
            "Available corpora:\n{corpora}\n"
            "{feedback_hint}\n\n"
            "Decompose into required facts and routes. "
            "For each fact: assign it to the best corpus and write a targeted search query."
        ),

        # ── Query Rewriter ──
        "rewriter.system": (
            "You are a Query Rewriter Agent.\n"
            "Given a question and search routes (each targeting a specific fact and corpus):\n\n"
            "For each route, expand the query into 1-2 different formulations.\n"
            "Use synonyms, different phrasings, and specific terms to maximize recall.\n\n"
            "Rules:\n"
            "- Each subquery must target exactly one fact.\n"
            "- Queries should be concise and search-optimized (3-8 words).\n"
            "- Avoid repeating queries that have already been attempted.\n"
            "- If feedback_queries from a prior assessment exist, prioritize those."
        ),
        "rewriter.user": (
            "Original routes:\n{routes}\n\n"
            "Expand each into alternate search formulations. "
            "Already tried: {tried}."
        ),

        # ── Drafter ──
        "drafter.system": (
            "You are a Drafter Agent.\n"
            "Given a question and retrieved snippets (each with an ID):\n\n"
            "1. Extract all factual claims that can support an answer.\n"
            "2. For each claim, cite the exact snippet IDs that support it.\n"
            "3. Write a coherent draft answer synthesizing all claims.\n\n"
            "Rules:\n"
            "- ONLY use information present in the provided snippets.\n"
            "- Each claim must cite at least one snippet ID.\n"
            "- If you cannot make a claim without speculating, leave it out.\n"
            "- Claims should be atomic: one fact per claim."
        ),
        "drafter.user": (
            "Question: {question}\n\nRetrieved snippets:\n{context}"
        ),

        # ── Sufficiency Judge ──
        "judge.system": (
            "You are a Sufficient Context Agent.\n\n"
            "Follow these steps:\n\n"
            "Step 1 — Mentally draft a potential answer.\n"
            "   Based ONLY on the retrieved snippets, draft the best answer you can.\n\n"
            "Step 2 — Identify gaps.\n"
            "   For each required fact, check: is the information present?\n"
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
        ),
        "judge.user": (
            "Question: {question}\n\n"
            "Required facts:\n{facts}\n\n"
            "Retrieved context:\n{context}"
        ),

        # ── Synthesizer ──
        "synthesizer.system": (
            "You generate grounded answers with citations."
        ),
        "synthesizer.user": (
            "Question: {question}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Answer based ONLY on the provided context. Cite sources."
        ),

        # ── Autorater (Gemini, paper's 93% 1-shot prompt) ──
        "autorater.paper": (
            "You are an expert LLM evaluator that excels at evaluating "
            "a QUESTION and REFERENCES.\n"
            "Consider the following criteria:\n"
            "Sufficient Context: 1 IF the CONTEXT is sufficient to infer "
            "the answer to the question and 0 IF the CONTEXT cannot be used "
            "to infer the answer to the question\n"
            "Assume the queries have timestamp <TIMESTAMP>.\n"
            "First, output a list of step-by-step questions that would be "
            "used to arrive at a label for the criteria. Make sure to include "
            "questions about assumptions implicit in the QUESTION.\n"
            "Include questions about any mathematical calculations or "
            "arithmetic that would be required.\n"
            "Next, answer each of the questions. Make sure to work step by "
            "step through any required mathematical calculations or arithmetic. "
            "Finally, use these answers to evaluate the criteria.\n"
            "Output the ### EXPLANATION (Text). Then, use the EXPLANATION "
            "to output the ### EVALUATION (JSON)\n"
            "EXAMPLE:\n"
            "### QUESTION\n"
            "In which year did the publisher of Roald Dahl's Guide to "
            "Railway Safety cease to exist?\n"
            "### References\n"
            "Roald Dahl's Guide to Railway Safety was published in 1991 by "
            "the British Railways Board. The British Railways Board (BRB) was "
            "a nationalised industry in the United Kingdom that operated "
            "from 1963 to 2001.\n"
            "### EXPLANATION\n"
            "The context mentions that Roald Dahl's Guide to Railway Safety "
            "was published by the British Railways Board. It also states that "
            "the British Railways Board operated from 1963 to 2001, meaning "
            "the year it ceased to exist was 2001. Therefore, the context "
            "does provide a precise answer to the question.\n"
            "### JSON\n"
            '{"Sufficient Context": 1}\n'
            "Remember the instructions:\n"
            "First, output a list of step-by-step questions.\n"
            "Next, answer each question.\n"
            "Finally, output ### EXPLANATION then ### EVALUATION (JSON).\n"
            "### QUESTION\n"
            "{question}\n"
            "### REFERENCES\n"
            "{context}"
        ),

        # ── Autorater (FLAMe, compact binary) ──
        "autorater.flame": (
            "INSTRUCTIONS:\n"
            "title: Is the context sufficient to infer the answer to "
            "the question?\n"
            "description: In this task, you will be provided with documents "
            "and a question. Use one of the following labels under 'judgment':\n"
            "1. sufficient: The documents are sufficient to infer the answer.\n"
            "2. insufficient: The documents are not sufficient to infer "
            "the answer.\n"
            "output_fields: judgment\n"
            "CONTEXT:\n"
            "documents: {context} question: {question}"
        ),

        # ── LLM Eval (paper's correctness evaluation prompt) ──
        "llm_eval.system": (
            "I need your help in evaluating an answer provided by an LLM "
            "against ground truth answers.\n"
            "Your task is to determine if the LLM's response matches the "
            "ground truth answers.\n\n"
            "===Instructions===\n"
            "1. Carefully compare the 'Predicted Answer' with the "
            "'Ground Truth Answers'.\n"
            "2. Consider the substance of the answers – look for equivalent "
            "information or correct answers.\n"
            "3. Your final decision should be based on whether the meaning "
            "and the vital facts of the Ground Truth Answers are present "
            "in the Predicted Answer.\n"
            "4. Categorize the answer as one of the following:\n"
            "- 'perfect': The answer is completely correct and matches.\n"
            "- 'acceptable': The answer is partially correct or contains "
            "the main idea.\n"
            "- 'incorrect': The answer is wrong or contradicts ground truth.\n"
            "- 'missing': The answer is 'I don't know' or similar.\n\n"
            "Provide your evaluation in this format:\n"
            "Explanation: (How you made the decision)\n"
            "Decision: (perfect, acceptable, incorrect, or missing)"
        ),
        "llm_eval.user": (
            "===Input Data===\n"
            "- Question: {question}\n"
            "- Predicted Answer: {predicted}\n"
            "- Ground Truth Answers: {ground_truth}\n\n"
            "Please proceed with the evaluation."
        ),
    }
