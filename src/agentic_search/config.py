"""Configuration management for Agentic Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchConfig:
    """Search backend configuration."""
    workspace_path: str = "."
    rag_db_path: str | None = None
    embed_model: str | None = None
    sources: list[str] = field(default_factory=lambda: ["workspace"])
    web_search_enabled: bool = False
    max_snippets_per_query: int = 10


@dataclass
class SelectiveGenConfig:
    """Selective Generation threshold configuration."""
    mode: str = "balanced"  # strict | balanced | lenient

    @property
    def threshold(self) -> float:
        return {"strict": 0.70, "balanced": 0.50, "lenient": 0.30}[self.mode]


@dataclass
class OrchestratorConfig:
    """Top-level orchestrator config."""
    max_iterations: int = 3
    max_snippets: int = 100
    search: SearchConfig = field(default_factory=SearchConfig)
    selective_gen: SelectiveGenConfig = field(default_factory=SelectiveGenConfig)
    autorater_mode: str = "fallback"  # flame | gemini | fallback
    corpus_descriptions: dict[str, str] = field(default_factory=dict)
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, d: dict) -> OrchestratorConfig:
        """Merge a dict into defaults (yaml/env friendly)."""
        search = SearchConfig(**{**SearchConfig().__dict__, **d.get("search", {})})
        sg = SelectiveGenConfig(**{**SelectiveGenConfig().__dict__, **d.get("selective_gen", {})})
        return cls(
            max_iterations=d.get("max_iterations", 3),
            max_snippets=d.get("max_snippets", 100),
            search=search,
            selective_gen=sg,
            autorater_mode=d.get("autorater_mode", "fallback"),
            corpus_descriptions=d.get("corpora", {}),
            log_level=d.get("log_level", "INFO"),
        )
