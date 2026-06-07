"""Tests for pipeline/generation/pair_generator.py preference-pairing logic."""
from __future__ import annotations

import json

from pipeline.formatting.chat_template import TOOL_CALL_CLOSE_TAG, TOOL_CALL_OPEN_TAG
from pipeline.generation.pair_generator import _build_pairs_for_query


def _completion(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


EXPECTED = [{"name": "get_weather", "arguments": {"city": ["Paris"]}}]
CORRECT = _completion("get_weather", {"city": "Paris"})
WRONG = _completion("get_weather", {"city": "London"})


def test_pairs_built_when_both_correct_and_incorrect_present():
    completions = [CORRECT, WRONG]
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=completions,
        expected_calls=EXPECTED, max_pairs=1,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["chosen"] == CORRECT
    assert pair["rejected"] == WRONG
    assert pair["prompt"] == "P"
    assert pair["chosen_reward"] == 1.0
    assert pair["rejected_reward"] == 0.0


def test_no_pairs_when_all_correct():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[CORRECT, CORRECT],
        expected_calls=EXPECTED, max_pairs=1,
    )
    assert pairs == []


def test_no_pairs_when_all_incorrect():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[WRONG, WRONG],
        expected_calls=EXPECTED, max_pairs=1,
    )
    assert pairs == []


def test_max_pairs_caps_emitted_pairs():
    completions = [CORRECT, CORRECT, WRONG, WRONG, WRONG]
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=completions,
        expected_calls=EXPECTED, max_pairs=2,
    )
    # capped by min(max_pairs, #correct=2, #incorrect=3) = 2
    assert len(pairs) == 2


def test_rejected_failure_category_is_recorded():
    completions = [CORRECT, WRONG]
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=completions,
        expected_calls=EXPECTED, max_pairs=1,
    )
    # WRONG has a valid name but the wrong argument value
    assert pairs[0]["rejected_failure"] == "wrong_argument_type"
