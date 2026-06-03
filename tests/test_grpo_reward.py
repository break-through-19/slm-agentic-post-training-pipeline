"""Tests for pipeline/reward/grpo_reward.py (the TRL reward-function wrapper).

The default reward is SHAPED (partial credit, Phase 2); a binary ablation mode
is available via build_bfcl_reward_function(shaped=False).
"""
from __future__ import annotations

import json

import pytest

from pipeline.formatting.chat_template import TOOL_CALL_CLOSE_TAG, TOOL_CALL_OPEN_TAG
from pipeline.reward.grpo_reward import build_bfcl_reward_function


def _completion(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


def _gt(calls: list[dict]) -> str:
    return json.dumps(calls)


# ---------------------------------------------------------------------------
# Shared behaviour (independent of shaping)
# ---------------------------------------------------------------------------

def test_reward_function_has_stable_name():
    # TRL uses __name__ as the metric label; it must be the expected string
    assert build_bfcl_reward_function().__name__ == "bfcl_reward"
    assert build_bfcl_reward_function(shaped=False).__name__ == "bfcl_reward"


def test_fully_correct_scores_one_in_both_modes():
    completions = [_completion("get_weather", {"city": "Paris"})]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])]
    for shaped in (True, False):
        reward_fn = build_bfcl_reward_function(shaped=shaped)
        assert reward_fn(completions=completions, expected_calls_json=expected) == [1.0]


def test_irrelevance_is_binary_in_both_modes():
    completions = [
        "I cannot help with the available tools.",   # correct: abstains
        _completion("some_fn", {}),                   # wrong: calls a tool
    ]
    expected = [_gt([]), _gt([])]
    for shaped in (True, False):
        reward_fn = build_bfcl_reward_function(shaped=shaped)
        assert reward_fn(completions=completions, expected_calls_json=expected) == [1.0, 0.0]


def test_missing_ground_truth_column_degrades_to_zero():
    reward_fn = build_bfcl_reward_function()
    completions = [_completion("get_weather", {"city": "Paris"})]
    assert reward_fn(completions=completions, expected_calls_json=None) == [0.0]


def test_malformed_ground_truth_json_treated_as_irrelevance():
    reward_fn = build_bfcl_reward_function()
    completions = ["No tool needed here."]
    assert reward_fn(completions=completions, expected_calls_json=["{not json"]) == [1.0]


# ---------------------------------------------------------------------------
# Binary ablation mode
# ---------------------------------------------------------------------------

def test_binary_mode_wrong_function_scores_zero():
    reward_fn = build_bfcl_reward_function(shaped=False)
    completions = [_completion("wrong_fn", {})]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])]
    assert reward_fn(completions=completions, expected_calls_json=expected) == [0.0]


# ---------------------------------------------------------------------------
# Shaped (default) mode — partial credit creates within-group variance
# ---------------------------------------------------------------------------

def test_shaped_mode_wrong_function_gets_format_credit():
    reward_fn = build_bfcl_reward_function(shaped=True)
    completions = [_completion("wrong_fn", {})]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])]
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards[0] == pytest.approx(0.2)  # format credit only


def test_shaped_mode_produces_within_group_variance():
    # The whole point of Phase 2: a mixed group must NOT have identical rewards.
    reward_fn = build_bfcl_reward_function(shaped=True)
    completions = [
        _completion("get_weather", {"city": "Paris", "days": 3}),   # fully correct
        _completion("get_weather", {"city": "Paris", "days": 9}),   # half args
        _completion("get_weather", {"city": "London", "days": 9}),  # name only
        "I cannot help.",                                            # nothing
    ]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"], "days": [3]}}])] * 4
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards[0] > rewards[1] > rewards[2] > rewards[3]
    assert len(set(rewards)) == 4          # all distinct → non-zero variance
    assert rewards[0] == 1.0 and rewards[3] == 0.0


def test_binary_mode_same_group_has_zero_variance():
    # Contrast: under binary reward the same mixed group collapses to {0,1}
    reward_fn = build_bfcl_reward_function(shaped=False)
    completions = [
        _completion("get_weather", {"city": "Paris", "days": 9}),   # half args
        _completion("get_weather", {"city": "London", "days": 9}),  # name only
    ]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"], "days": [3]}}])] * 2
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards == [0.0, 0.0]  # both "wrong" → no gradient signal