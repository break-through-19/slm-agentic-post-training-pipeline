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
    score,
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


def test_non_object_tool_call_does_not_crash():
    # A <tool_call> whose JSON is a bare string previously crashed the grader
    # ('str' object has no attribute 'get') mid-GRPO-run. It must now grade
    # cleanly as "no valid call produced".
    output = f'{TOOL_CALL_OPEN_TAG}"weather in Paris"{TOOL_CALL_CLOSE_TAG}'
    expected = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    result = grade(output, expected)
    assert result.correct is False
    assert result.failure_category == FAILURE_NO_TOOL_CALL  # nothing extractable


def test_non_object_tool_call_on_irrelevance_is_abstention():
    # For irrelevance, an unparseable/non-object call yields no tool call → correct
    output = f"{TOOL_CALL_OPEN_TAG}42{TOOL_CALL_CLOSE_TAG}"
    result = grade(output, [])
    assert result.correct is True


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


# ---------------------------------------------------------------------------
# Phase 2: shaped partial-credit reward score()
# ---------------------------------------------------------------------------

EXPECTED_TWO_ARGS = [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]


def test_score_fully_correct_is_one():
    output = _make_model_output("get_weather", {"city": "Paris", "days": 3})
    assert score(output, EXPECTED_TWO_ARGS) == 1.0


def test_score_matches_grade_at_extremes():
    # score() == 1.0 exactly when grade() is correct
    output = _make_model_output("get_weather", {"city": "Paris", "days": 3})
    assert grade(output, EXPECTED_TWO_ARGS).correct is True
    assert score(output, EXPECTED_TWO_ARGS) == 1.0


def test_score_no_tool_call_is_zero():
    assert score("I can't help.", EXPECTED_TWO_ARGS) == 0.0


def test_score_wrong_function_gets_format_credit_only():
    output = _make_model_output("totally_wrong_fn", {"city": "Paris", "days": 3})
    # 0.2 format credit, no name/argument credit
    assert score(output, EXPECTED_TWO_ARGS) == pytest.approx(0.2)


def test_score_right_function_partial_args_is_between():
    # correct name + 1 of 2 args correct -> 0.2 + 0.3 + 0.5*(1/2) = 0.75
    output = _make_model_output("get_weather", {"city": "Paris", "days": 99})
    assert score(output, EXPECTED_TWO_ARGS) == pytest.approx(0.75)


def test_score_right_function_all_args_wrong():
    # correct name + 0 of 2 args -> 0.2 + 0.3 = 0.5
    output = _make_model_output("get_weather", {"city": "London", "days": 99})
    assert score(output, EXPECTED_TWO_ARGS) == pytest.approx(0.5)


def test_score_is_monotonic_more_correct_scores_higher():
    none_right = score(_make_model_output("get_weather", {"city": "X", "days": 9}), EXPECTED_TWO_ARGS)
    one_right = score(_make_model_output("get_weather", {"city": "Paris", "days": 9}), EXPECTED_TWO_ARGS)
    all_right = score(_make_model_output("get_weather", {"city": "Paris", "days": 3}), EXPECTED_TWO_ARGS)
    assert none_right < one_right < all_right


def test_score_irrelevance_is_binary():
    assert score("No suitable tool here.", []) == 1.0
    assert score(_make_model_output("some_fn", {}), []) == 0.0


def test_score_penalises_extra_calls():
    # One correct call plus a spurious extra call: total 1.0 over denom 2 = 0.5
    output = _make_parallel_output([
        ("get_weather", {"city": "Paris", "days": 3}),
        ("spurious_fn", {}),
    ])
    assert score(output, EXPECTED_TWO_ARGS) == pytest.approx(0.5)
