"""Base retriever protocol for multi-source search."""

from __future__ import annotations

from typing import Protocol

from ..state import Snippet


class BaseRetriever(Protocol):
    """Protocol for retriever implementations."""

    def supports_corpus(self, corpus: str) -> bool:
        raise NotImplementedError

    async def search(self, query: str, corpus: str, top_k: int = 10) -> list[Snippet]:
        raise NotImplementedError
