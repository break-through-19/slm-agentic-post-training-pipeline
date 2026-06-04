"""
Tool-schema helpers shared by relabeling (step 3) and constrained decoding (step 5).

A "tool" here is a function definition in either of the two shapes the pipeline
handles:
  * flat   : {"name", "description", "parameters": {json-schema}}
  * wrapped: {"type": "function", "function": {"name", ...}}

These helpers read the parameter types out of whichever shape is given and use
them to (a) coerce predicted argument values to their declared JSON types and
(b) build a JSON schema that constrains generation to a valid tool call.
"""
from __future__ import annotations

from typing import Any

# JSON-schema primitive type -> Python constructor used for coercion
_JSON_TYPE_TO_PY = {"integer": int, "number": float, "boolean": bool, "string": str}


def _function_block(tool: dict) -> dict:
    """Return the {name, description, parameters} block regardless of nesting."""
    if isinstance(tool, dict) and "function" in tool and isinstance(tool["function"], dict):
        return tool["function"]
    return tool


def tool_name(tool: dict) -> str:
    """The function name of a tool, whatever shape it is in."""
    return _function_block(tool).get("name", "")


def tool_parameter_types(tool: dict) -> dict[str, str]:
    """Map each parameter name to its declared JSON-schema type (default 'string')."""
    params = _function_block(tool).get("parameters", {}) or {}
    properties = params.get("properties", {}) or {}
    return {
        key: (spec.get("type", "string") if isinstance(spec, dict) else "string")
        for key, spec in properties.items()
    }


def coerce_scalar(value: Any, json_type: str) -> Any:
    """
    Cast a scalar to the schema's declared JSON type. Returns the value
    unchanged when it is already correct or cannot be cast cleanly.
    """
    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return value
    if json_type in ("integer", "number"):
        if isinstance(value, bool):
            return value
        try:
            return int(value) if json_type == "integer" else float(value)
        except (TypeError, ValueError):
            return value
    if json_type == "string":
        return value if isinstance(value, str) else value
    return value


def coerce_call_arguments(call: dict, parameter_types: dict[str, str]) -> dict:
    """Return a copy of `call` with its argument values coerced to schema types."""
    if not isinstance(call, dict):
        return call
    arguments = call.get("arguments", call.get("args", {}))
    if not isinstance(arguments, dict):
        return call
    coerced = {
        key: coerce_scalar(value, parameter_types[key]) if key in parameter_types else value
        for key, value in arguments.items()
    }
    return {"name": call.get("name", ""), "arguments": coerced}


def coerce_calls_to_schema(calls: list[dict], tools: list[dict]) -> list[dict]:
    """
    Coerce every predicted call's argument values to the declared types of the
    matching tool. Calls whose name is not among the tools are left untouched.
    """
    types_by_name = {tool_name(t): tool_parameter_types(t) for t in tools}
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "")
        out.append(coerce_call_arguments(call, types_by_name.get(name, {})))
    return out


def build_tool_call_schema(tools: list[dict]) -> dict:
    """
    Build a JSON schema that validates a single tool call against `tools`.

    The function name is constrained to the available tools (an enum), which is
    the property a constrained decoder uses to make hallucinated function names
    impossible. Arguments are validated as an object; per-parameter typing is
    left permissive so optional-argument omission stays legal.
    """
    names = [tool_name(t) for t in tools if tool_name(t)]
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
