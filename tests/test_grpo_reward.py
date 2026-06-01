"""Tests for pipeline/reward/grpo_reward.py (the TRL reward-function wrapper)."""
from __future__ import annotations

import json

from pipeline.formatting.chat_template import TOOL_CALL_CLOSE_TAG, TOOL_CALL_OPEN_TAG
from pipeline.reward.grpo_reward import build_bfcl_reward_function


def _completion(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


def _gt(calls: list[dict]) -> str:
    return json.dumps(calls)


def test_reward_function_has_stable_name():
    # TRL uses __name__ as the metric label; it must be the expected string
    reward_fn = build_bfcl_reward_function()
    assert reward_fn.__name__ == "bfcl_reward"


def test_correct_completion_scores_one():
    reward_fn = build_bfcl_reward_function()
    completions = [_completion("get_weather", {"city": "Paris"})]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])]
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards == [1.0]


def test_incorrect_completion_scores_zero():
    reward_fn = build_bfcl_reward_function()
    completions = [_completion("wrong_fn", {})]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])]
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards == [0.0]


def test_batch_of_mixed_completions():
    reward_fn = build_bfcl_reward_function()
    completions = [
        _completion("get_weather", {"city": "Paris"}),   # correct
        _completion("get_weather", {"city": "London"}),  # wrong value
        "I cannot help with that.",                        # no tool call
    ]
    expected = [_gt([{"name": "get_weather", "arguments": {"city": ["Paris"]}}])] * 3
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards == [1.0, 0.0, 0.0]


def test_irrelevance_ground_truth_rewards_abstention():
    reward_fn = build_bfcl_reward_function()
    completions = [
        "I cannot help with the available tools.",   # correct: abstains
        _completion("some_fn", {}),                   # wrong: calls a tool
    ]
    expected = [_gt([]), _gt([])]  # empty ground truth = irrelevance
    rewards = reward_fn(completions=completions, expected_calls_json=expected)
    assert rewards == [1.0, 0.0]


def test_missing_ground_truth_column_degrades_to_zero():
    reward_fn = build_bfcl_reward_function()
    completions = [_completion("get_weather", {"city": "Paris"})]
    rewards = reward_fn(completions=completions, expected_calls_json=None)
    assert rewards == [0.0]


def test_malformed_ground_truth_json_treated_as_irrelevance():
    reward_fn = build_bfcl_reward_function()
    # Unparseable GT → empty expected_calls → abstention is the correct answer
    completions = ["No tool needed here."]
    rewards = reward_fn(completions=completions, expected_calls_json=["{not json"])
    assert rewards == [1.0]
