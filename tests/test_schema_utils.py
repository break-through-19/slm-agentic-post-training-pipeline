"""Tests for pipeline/formatting/schema_utils.py (shared by steps 3 and 5)."""
from __future__ import annotations

from pipeline.formatting.schema_utils import (
    build_tool_call_schema,
    coerce_call_arguments,
    coerce_calls_to_schema,
    coerce_scalar,
    tool_name,
    tool_parameter_types,
)

FLAT_TOOL = {
    "name": "set_timer",
    "parameters": {"type": "object",
                   "properties": {"minutes": {"type": "integer"},
                                  "loud": {"type": "boolean"},
                                  "label": {"type": "string"}},
                   "required": ["minutes"]},
}
WRAPPED_TOOL = {"type": "function", "function": FLAT_TOOL}


def test_coerce_scalar_types():
    assert coerce_scalar("10", "integer") == 10 and isinstance(coerce_scalar("10", "integer"), int)
    assert coerce_scalar("3.5", "number") == 3.5
    assert coerce_scalar("true", "boolean") is True
    assert coerce_scalar("false", "boolean") is False
    # Uncastable values are returned unchanged
    assert coerce_scalar("soon", "integer") == "soon"
    # Already-correct values pass through
    assert coerce_scalar(7, "integer") == 7


def test_tool_helpers_handle_both_shapes():
    assert tool_name(FLAT_TOOL) == "set_timer"
    assert tool_name(WRAPPED_TOOL) == "set_timer"
    assert tool_parameter_types(FLAT_TOOL) == {"minutes": "integer", "loud": "boolean", "label": "string"}
    assert tool_parameter_types(WRAPPED_TOOL)["minutes"] == "integer"


def test_coerce_call_arguments():
    call = {"name": "set_timer", "arguments": {"minutes": "10", "loud": "true"}}
    out = coerce_call_arguments(call, tool_parameter_types(FLAT_TOOL))
    assert out["arguments"]["minutes"] == 10
    assert out["arguments"]["loud"] is True


def test_coerce_calls_to_schema_unknown_name_untouched():
    calls = [{"name": "nope", "arguments": {"minutes": "10"}}]
    out = coerce_calls_to_schema(calls, [FLAT_TOOL])
    # Unknown function -> no schema -> value left as the string "10"
    assert out[0]["arguments"]["minutes"] == "10"


def test_build_tool_call_schema_enumerates_names():
    schema = build_tool_call_schema([FLAT_TOOL])
    assert schema["properties"]["name"]["enum"] == ["set_timer"]
    assert schema["required"] == ["name", "arguments"]
