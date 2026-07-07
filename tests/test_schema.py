"""Tests for the schema validation and repair framework."""

import pytest

from agentic_search import SchemaRegistry, SchemaSpec, build_default_registry
from agentic_search.schema import parse_with_repair


def test_schema_registry_build():
    reg = build_default_registry()
    for name in ["RetrievalPlan", "QueryRewriteResult", "ContextAssessment", "GroundedAnswer"]:
        assert reg.get(name) is not None


def test_schema_validation_passes():
    reg = build_default_registry()
    data = {
        "status": "sufficient",
        "sufficiency_score": 0.9,
        "draft_answer": "A complete answer based on the context.",
        "missing_facts": [],
        "feedback_queries": [],
        "reason": "Everything found.",
    }
    errors = reg.validate("ContextAssessment", data)
    assert len(errors) == 0


def test_schema_validation_fails():
    reg = build_default_registry()
    errors = reg.validate("ContextAssessment", {})
    assert len(errors) > 0  # Missing required fields


def test_schema_enum_validation():
    reg = build_default_registry()
    data = {
        "status": "invalid_value",
        "sufficiency_score": 0.5,
        "draft_answer": "test draft",
        "missing_facts": [],
        "feedback_queries": [],
        "reason": "test",
    }
    errors = reg.validate("ContextAssessment", data)
    assert any("status" in str(e) for e in errors)


def test_schema_range_validation():
    reg = build_default_registry()
    data = {
        "status": "sufficient",
        "sufficiency_score": 99.0,
        "draft_answer": "test draft",
        "missing_facts": [],
        "feedback_queries": [],
        "reason": "test",
    }
    errors = reg.validate("ContextAssessment", data)
    assert any("sufficiency_score" in str(e) for e in errors)


def test_schema_to_json():
    reg = build_default_registry()
    js = reg.to_json_schema("ContextAssessment")
    assert js["type"] == "object"
    assert "status" in js["properties"]
    assert "sufficient" in js["properties"]["status"].get("enum", [])


def test_parse_json_block():
    from agentic_search.schema import _extract_json_block
    text = 'Some text\n```json\n{"key": "value"}\n```\nmore text'
    result = _extract_json_block(text)
    assert result == {"key": "value"}


def test_parse_no_json_block():
    from agentic_search.schema import _extract_json_block
    assert _extract_json_block("plain text") is None


def test_register_custom():
    reg = SchemaRegistry()
    spec = SchemaSpec(
        name="Custom", fields={"val": {"type": "number", "required": True, "min": 0, "max": 100}},
        description="Custom schema",
    )
    reg.register(spec)
    assert reg.get("Custom") is not None

    # Valid
    assert len(reg.validate("Custom", {"val": 50})) == 0
    # Invalid: missing
    assert len(reg.validate("Custom", {})) > 0
    # Invalid: out of range
    assert len(reg.validate("Custom", {"val": 999})) > 0
