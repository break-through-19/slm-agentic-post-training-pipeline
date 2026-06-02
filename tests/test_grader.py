"""Tests for pipeline/reward/bfcl_grader.py"""
from __future__ import annotations

import json

import pytest

from pipeline.reward.bfcl_grader import (
    FAILURE_EXTRA_TOOL_CALL,
    FAILURE_MISSING_ARGUMENT,
    FAILURE_NO_TOOL_CALL,
    FAILURE_WRONG_ARGUMENT_TYPE,
    FAILURE_WRONG_FUNCTION,
    grade,
)
from pipeline.formatting.chat_template import TOOL_CALL_OPEN_TAG, TOOL_CALL_CLOSE_TAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_output(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


def _make_parallel_output(calls: list[tuple[str, dict]]) -> str:
    blocks = []
    for name, arguments in calls:
        payload = json.dumps({"name": name, "arguments": arguments})
        blocks.append(f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Correct calls
# ---------------------------------------------------------------------------

def test_correct_single_call_returns_reward_1():
    output = _make_model_output("get_weather", {"city": "Seattle", "days": 3})
    expected = [{"name": "get_weather", "arguments": {"city": "Seattle", "days": 3}}]
    result = grade(output, expected)
    assert result.correct is True
    assert result.reward == 1.0
    assert result.failure_category is None


def test_correct_parallel_calls():
    output = _make_parallel_output([
        ("get_weather", {"city": "London"}),
        ("get_time", {"timezone": "UTC"}),
    ])
    expected = [
        {"name": "get_weather", "arguments": {"city": "London"}},
        {"name": "get_time", "arguments": {"timezone": "UTC"}},
    ]
    result = grade(output, expected)
    assert result.correct is True
    assert result.reward == 1.0


# ---------------------------------------------------------------------------
# No tool call
# ---------------------------------------------------------------------------

def test_no_tool_call_tag():
    output = "I can help you with that."
    expected = [{"name": "get_weather", "arguments": {}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_NO_TOOL_CALL


def test_empty_output():
    result = grade("", [{"name": "fn", "arguments": {}}])
    assert result.correct is False
    assert result.failure_category == FAILURE_NO_TOOL_CALL


# ---------------------------------------------------------------------------
# Wrong function name
# ---------------------------------------------------------------------------

def test_wrong_function_name():
    output = _make_model_output("get_current_weather", {"city": "Seattle"})
    expected = [{"name": "get_weather_forecast", "arguments": {"city": "Seattle"}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_WRONG_FUNCTION


# ---------------------------------------------------------------------------
# Missing arguments
# ---------------------------------------------------------------------------

def test_missing_required_argument():
    output = _make_model_output("book_flight", {"destination": "NYC"})
    expected = [{"name": "book_flight", "arguments": {"destination": "NYC", "passengers": 2}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_MISSING_ARGUMENT


# ---------------------------------------------------------------------------
# Wrong argument type
# ---------------------------------------------------------------------------

def test_string_value_where_int_expected():
    output = _make_model_output("set_timer", {"minutes": "ten"})
    expected = [{"name": "set_timer", "arguments": {"minutes": 10}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_WRONG_ARGUMENT_TYPE


def test_wrong_string_value():
    output = _make_model_output("get_weather", {"city": "London"})
    expected = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_WRONG_ARGUMENT_TYPE


# ---------------------------------------------------------------------------
# Type coercion — models often emit numbers or booleans as strings
# ---------------------------------------------------------------------------

def test_integer_as_string_is_coerced():
    output = _make_model_output("set_timer", {"minutes": "30"})
    expected = [{"name": "set_timer", "arguments": {"minutes": 30}}]
    result = grade(output, expected)
    assert result.correct is True


def test_boolean_as_string_is_coerced():
    output = _make_model_output("send_alert", {"urgent": "true"})
    expected = [{"name": "send_alert", "arguments": {"urgent": True}}]
    result = grade(output, expected)
    assert result.correct is True


def test_float_as_string_is_coerced():
    output = _make_model_output("set_threshold", {"value": "0.5"})
    expected = [{"name": "set_threshold", "arguments": {"value": 0.5}}]
    result = grade(output, expected)
    assert result.correct is True


# ---------------------------------------------------------------------------
# Parallel call count mismatch
# ---------------------------------------------------------------------------

def test_too_many_predicted_calls():
    output = _make_parallel_output([
        ("fn_a", {}),
        ("fn_b", {}),
    ])
    expected = [{"name": "fn_a", "arguments": {}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_EXTRA_TOOL_CALL


def test_too_few_predicted_calls():
    output = _make_model_output("fn_a", {})
    expected = [
        {"name": "fn_a", "arguments": {}},
        {"name": "fn_b", "arguments": {}},
    ]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_EXTRA_TOOL_CALL


# ---------------------------------------------------------------------------
# GradeResult fields
# ---------------------------------------------------------------------------

def test_grade_result_includes_predicted_and_expected():
    output = _make_model_output("wrong_fn", {})
    expected = [{"name": "right_fn", "arguments": {}}]
    result = grade(output, expected)
    assert len(result.predicted_calls) == 1
    assert len(result.expected_calls) == 1
    assert result.predicted_calls[0]["name"] == "wrong_fn"


# ---------------------------------------------------------------------------
# Irrelevance: expected_calls=[] means model should produce no tool call
# ---------------------------------------------------------------------------

def test_irrelevance_correct_when_no_tool_call_produced():
    result = grade("I cannot help with that using any available tool.", [])
    assert result.correct is True
    assert result.reward == 1.0


def test_irrelevance_wrong_when_tool_call_produced():
    output = _make_model_output("some_fn", {})
    result = grade(output, [])
    assert result.correct is False
    assert result.failure_category == FAILURE_EXTRA_TOOL_CALL


# ---------------------------------------------------------------------------
# BFCL possible-answer format: argument values as lists of acceptable answers
# ---------------------------------------------------------------------------

def test_possible_answer_accepts_first_value_in_list():
    output = _make_model_output("calc_area", {"unit": "units"})
    expected = [{"name": "calc_area", "arguments": {"unit": ["units", ""]}}]
    result = grade(output, expected)
    assert result.correct is True


def test_possible_answer_accepts_alternate_value_in_list():
    output = _make_model_output("calc_area", {"unit": ""})
    expected = [{"name": "calc_area", "arguments": {"unit": ["units", ""]}}]
    result = grade(output, expected)
    assert result.correct is True


def test_possible_answer_rejects_value_not_in_list():
    output = _make_model_output("calc_area", {"unit": "meters"})
    expected = [{"name": "calc_area", "arguments": {"unit": ["units", ""]}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_WRONG_ARGUMENT_TYPE


def test_possible_answer_integer_list():
    output = _make_model_output("set_timer", {"minutes": 30})
    expected = [{"name": "set_timer", "arguments": {"minutes": [30, 29]}}]
    result = grade(output, expected)
    assert result.correct is True


# ---------------------------------------------------------------------------
# Phase 0 fix: optional arguments ("" in acceptable list) may be omitted
# ---------------------------------------------------------------------------

def test_optional_argument_may_be_omitted():
    # 'unit' is optional (its acceptable list contains ""), so leaving it out
    # of the prediction must NOT count as a missing argument.
    output = _make_model_output("calc_area", {"base": 10, "height": 5})
    expected = [{
        "name": "calc_area",
        "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
    }]
    result = grade(output, expected)
    assert result.correct is True


def test_required_argument_still_enforced_when_omitted():
    # 'height' is required (no "" in its acceptable list) — omitting it fails.
    output = _make_model_output("calc_area", {"base": 10})
    expected = [{
        "name": "calc_area",
        "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
    }]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_MISSING_ARGUMENT


def test_optional_argument_still_graded_when_present():
    # If the model DOES supply the optional arg, it must be an acceptable value.
    output = _make_model_output("calc_area", {"base": 10, "height": 5, "unit": "meters"})
    expected = [{
        "name": "calc_area",
        "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
    }]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_WRONG_ARGUMENT_TYPE


# ---------------------------------------------------------------------------
# Phase 0 fix: parallel calls matched order-independently
# ---------------------------------------------------------------------------

def test_parallel_calls_matched_regardless_of_order():
    # Predicted order is the reverse of expected — should still be correct.
    output = _make_parallel_output([
        ("get_time", {"timezone": "UTC"}),
        ("get_weather", {"city": "London"}),
    ])
    expected = [
        {"name": "get_weather", "arguments": {"city": "London"}},
        {"name": "get_time", "arguments": {"timezone": "UTC"}},
    ]
    result = grade(output, expected)
    assert result.correct is True


def test_parallel_calls_reordered_with_wrong_arg_still_fails():
    output = _make_parallel_output([
        ("get_time", {"timezone": "PST"}),          # wrong value
        ("get_weather", {"city": "London"}),
    ])
    expected = [
        {"name": "get_weather", "arguments": {"city": "London"}},
        {"name": "get_time", "arguments": {"timezone": "UTC"}},
    ]
    result = grade(output, expected)
    assert result.correct is False


def test_parallel_calls_in_order_still_correct():
    # Regression: in-order parallel matching must keep working.
    output = _make_parallel_output([
        ("get_weather", {"city": "London"}),
        ("get_time", {"timezone": "UTC"}),
    ])
    expected = [
        {"name": "get_weather", "arguments": {"city": "London"}},
        {"name": "get_time", "arguments": {"timezone": "UTC"}},
    ]
    result = grade(output, expected)
    assert result.correct is True
