"""JSON Schema validation and structured output repair.

Schema registry with automatic validation and LLM-based repair.
Inspired by agentic-rag's SchemaSpec approach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when structured output fails schema validation."""

    def __init__(self, schema_name: str, field: str, expected: str, got: Any):
        self.schema_name = schema_name
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(f"{schema_name}.{field}: expected {expected}, got {got!r}")


@dataclass
class SchemaSpec:
    """One registered schema with validation rules."""
    name: str
    fields: dict[str, dict]  # field_name → {type, required, min, max, enum, ...}
    description: str

    def validate(self, data: dict) -> list[SchemaValidationError]:
        """Validate a dict against the schema. Returns list of errors."""
        errors = []
        for field_name, rules in self.fields.items():
            if rules.get("required", False) and field_name not in data:
                errors.append(SchemaValidationError(
                    self.name, field_name, "required field", None,
                ))
                continue
            if field_name not in data:
                continue
            value = data[field_name]
            expected_type = rules.get("type", "any")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(SchemaValidationError(self.name, field_name, "string", value))
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(SchemaValidationError(self.name, field_name, "number", value))
            elif expected_type == "list" and not isinstance(value, list):
                errors.append(SchemaValidationError(self.name, field_name, "list", value))
            elif expected_type == "dict" and not isinstance(value, dict):
                errors.append(SchemaValidationError(self.name, field_name, "dict", value))

            # Enum check
            enum_values = rules.get("enum")
            if enum_values is not None and value not in enum_values:
                errors.append(SchemaValidationError(
                    self.name, field_name, f"one of {enum_values}", value,
                ))

            # Range checks
            if isinstance(value, (int, float)):
                min_val = rules.get("min")
                max_val = rules.get("max")
                if min_val is not None and value < min_val:
                    errors.append(SchemaValidationError(self.name, field_name, f">= {min_val}", value))
                if max_val is not None and value > max_val:
                    errors.append(SchemaValidationError(self.name, field_name, f"<= {max_val}", value))

        return errors

    def to_json_schema(self) -> dict:
        """Convert to JSON Schema for LLM prompting."""
        properties = {}
        required = []
        for field_name, rules in self.fields.items():
            if rules.get("required", False):
                required.append(field_name)
            type_map = {"string": "string", "number": "number", "list": "array", "dict": "object"}
            prop = {"type": type_map.get(rules.get("type", "string"), "string")}
            if "description" in rules:
                prop["description"] = rules["description"]
            if "enum" in rules:
                prop["enum"] = rules["enum"]
            properties[field_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


# ── Schema Registry ──

class SchemaRegistry:
    """Registry of all structured output schemas."""

    def __init__(self):
        self._schemas: dict[str, SchemaSpec] = {}

    def register(self, spec: SchemaSpec):
        self._schemas[spec.name] = spec

    def get(self, name: str) -> SchemaSpec:
        if name not in self._schemas:
            raise KeyError(f"Schema '{name}' not registered. Available: {list(self._schemas)}")
        return self._schemas[name]

    def validate(self, name: str, data: dict) -> list[SchemaValidationError]:
        return self.get(name).validate(data)

    def to_json_schema(self, name: str) -> dict:
        return self.get(name).to_json_schema()


def build_default_registry() -> SchemaRegistry:
    """Build the default schema registry with all 4 core schemas."""
    reg = SchemaRegistry()

    reg.register(SchemaSpec(
        name="RetrievalPlan",
        fields={
            "routes": {"type": "list", "required": True, "description": "Search routes"},
            "analysis": {"type": "string", "required": True, "description": "Reasoning behind the plan"},
        },
        description="Search plan with routes and analysis",
    ))

    reg.register(SchemaSpec(
        name="QueryRewriteResult",
        fields={
            "queries": {"type": "list", "required": True, "description": "Rewritten search queries"},
            "reasoning": {"type": "string", "required": False},
        },
        description="Rewritten search queries",
    ))

    reg.register(SchemaSpec(
        name="ContextAssessment",
        fields={
            "status": {
                "type": "string", "required": True,
                "enum": ["sufficient", "partial", "insufficient", "conflicting", "unanswerable"],
                "description": "Sufficiency verdict",
            },
            "sufficiency_score": {
                "type": "number", "required": True, "min": 0.0, "max": 1.0,
                "description": "Fraction of required facts covered",
            },
            "draft_answer": {
                "type": "string", "required": True,
                "description": "A rough draft answer based on current context, used to identify gaps",
            },
            "missing_facts": {"type": "list", "required": True, "description": "Missing fact IDs"},
            "feedback_queries": {"type": "list", "required": True, "description": "Queries for next iteration"},
            "reason": {"type": "string", "required": True, "description": "Judgment explanation"},
        },
        description="Context sufficiency assessment with feedback",
    ))

    reg.register(SchemaSpec(
        name="GroundedAnswer",
        fields={
            "answer": {"type": "string", "required": True, "description": "The final answer text"},
            "citations": {"type": "list", "required": True, "description": "Source citations"},
            "confidence": {"type": "number", "required": True, "min": 0.0, "max": 1.0},
        },
        description="Final grounded answer",
    ))

    return reg


# ── JSON Repair ──

def parse_with_repair(
    text: str,
    schema_name: str,
    registry: SchemaRegistry,
    llm: Any = None,
    max_repair_attempts: int = 1,
) -> dict:
    """Parse JSON text, validate against schema, repair if needed."""
    # Attempt 1: direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_json_block(text)

    if data is None:
        raise SchemaValidationError(schema_name, "<root>", "valid JSON", text[:200])

    errors = registry.validate(schema_name, data)
    if not errors:
        return data

    # Attempt 2: LLM repair
    if llm is not None and max_repair_attempts > 0:
        schema = registry.get(schema_name)
        repair_prompt = (
            f"The following JSON for '{schema_name}' has validation errors:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
            f"Errors:\n" + "\n".join(str(e) for e in errors) + "\n\n"
            f"Schema description: {schema.description}\n"
            f"Return a corrected JSON that conforms to the schema."
        )
        try:
            import asyncio
            repaired = asyncio.run(llm.complete_json(
                system_prompt="You fix JSON to match a schema.",
                user_prompt=repair_prompt,
                output_schema=schema.to_json_schema(),
            ))
            repaired_errors = registry.validate(schema_name, repaired)
            if not repaired_errors:
                return repaired
        except Exception:
            pass

    raise SchemaValidationError(schema_name, "<root>", "valid after repair", data)


def _extract_json_block(text: str) -> dict | None:
    """Extract JSON from a text block (```json ... ```)."""
    import re
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None
