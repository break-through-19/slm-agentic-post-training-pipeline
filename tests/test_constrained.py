"""Tests for pipeline/evaluation/constrained.py structured-output decoding (step 5)."""
from __future__ import annotations

import json

from pipeline.evaluation.constrained import (
    canonical_tool_call_text,
    looks_like_failed_call,
    normalise_prediction,
)
from pipeline.formatting.chat_template import TOOL_CALL_CLOSE_TAG, TOOL_CALL_OPEN_TAG
from pipeline.reward.bfcl_grader import grade

TOOLS = [{
    "name": "set_timer",
    "parameters": {"type": "object",
                   "properties": {"minutes": {"type": "integer"}},
                   "required": ["minutes"]},
}]
EXPECTED = [{"name": "set_timer", "arguments": {"minutes": [10]}}]


def _block(payload: str) -> str:
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


def test_normalise_coerces_argument_types_and_strips_prose():
    raw = "Sure, I'll set that. " + _block(json.dumps(
        {"name": "set_timer", "arguments": {"minutes": "10"}}))
    out = normalise_prediction(raw, TOOLS)
    # The "minutes" string is coerced to an int in the canonical output
    assert '"minutes": 10' in out
    assert grade(out, EXPECTED).correct


def test_normalise_leaves_abstention_untouched():
    raw = "None of the available tools can answer that."
    assert normalise_prediction(raw, TOOLS) == raw
    # Still graded as a correct abstention against empty expected calls
    assert grade(normalise_prediction(raw, TOOLS), []).correct


def test_looks_like_failed_call():
    # Opened a tool_call block but the JSON is unparseable -> repairable
    assert looks_like_failed_call(_block("{not valid json"))
    # A clean abstention (no opener) is never flagged for repair
    assert not looks_like_failed_call("No tool applies here.")
    # A valid call is not a failed call
    assert not looks_like_failed_call(_block(json.dumps(
        {"name": "set_timer", "arguments": {"minutes": 10}})))


def test_canonical_tool_call_text_roundtrips_through_grader():
    text = canonical_tool_call_text([{"name": "set_timer", "arguments": {"minutes": 10}}])
    assert TOOL_CALL_OPEN_TAG in text and TOOL_CALL_CLOSE_TAG in text
    assert grade(text, EXPECTED).correct
