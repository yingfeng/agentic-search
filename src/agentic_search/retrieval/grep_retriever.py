"""Grep-based retriever — for codebase search (inspired by claw-code)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

from .base import BaseRetriever
from ..state import Snippet


class GrepRetriever(BaseRetriever):
    """Retrieve snippets via ripgrep (grep -r)."""

    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()

    def supports_corpus(self, corpus: str) -> bool:
        return corpus in ("workspace", "code", "all")

    async def search(self, query: str, corpus: str = "all", top_k: int = 10) -> list[Snippet]:
        """Run ripgrep search over workspace files."""
        try:
            result = subprocess.run(
                ["rg", "-i", "-l", query, str(self.workspace)],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        files = [f for f in result.stdout.strip().split("\n") if f.strip()][:top_k]
        snippets = []
        for filepath in files[:5]:  # Read first 5 files
            try:
                text = Path(filepath).read_text(errors="replace")[:1000]
                snippets.append(Snippet(
                    snippet_id=uuid4().hex[:8],
                    corpus="workspace",
                    document_id=filepath,
                    text=text,
                    score=1.0,
                    metadata={"path": filepath},
                ))
            except Exception:
                continue
        return snippets
