#!/usr/bin/env python3
"""CLI entry point for Agentic Search."""

import asyncio
import json
import sys
from pathlib import Path

from agentic_search import build_graph, AgenticSearchState


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the main function of the planner node?"
    workspace = sys.argv[2] if len(sys.argv) > 2 else "."

    graph = build_graph()

    # Configure corpora
    corpora = {
        "workspace": "Source code and documentation in the workspace",
        "docs": "Technical documentation and specs",
    }

    state = AgenticSearchState(
        question=question,
        corpora=corpora,
        max_iterations=3,
        search_config={
            "mode": "balanced",
            "sources": ["workspace"],
            "workspace_path": workspace,
        },
    )

    # Run the full loop
    result = await graph.ainvoke(state)

    if result.answer:
        answer = result.answer
        print(f"\n{'='*60}")
        print(f"Status: {answer.status.value}")
        print(f"Confidence: {answer.confidence:.2%}")
        print(f"Iterations: {answer.iteration_count}")
        print(f"\nAnswer: {answer.answer or '(abstained)'}")
        if answer.missing_info:
            print(f"\nDetails: {answer.missing_info}")
        print(f"\nEvidence Trail:")
        for step in answer.evidence_path:
            print(f"  • {step}")
    else:
        print("No answer produced.")


if __name__ == "__main__":
    asyncio.run(main())
