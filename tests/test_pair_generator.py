"""Tests for pipeline/generation/pair_generator.py preference-pairing logic."""
from __future__ import annotations

import json
import random

from pipeline.formatting.chat_template import (
    NO_TOOL_RESPONSE,
    TOOL_CALL_CLOSE_TAG,
    TOOL_CALL_OPEN_TAG,
)
from pipeline.generation.pair_generator import _build_pairs_for_query
from pipeline.reward.bfcl_grader import grade


def _completion(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


EXPECTED = [{"name": "get_weather", "arguments": {"city": ["Paris"]}}]
CORRECT = _completion("get_weather", {"city": "Paris"})
WRONG = _completion("get_weather", {"city": "London"})
TOOLS = [{
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]},
}]
ABSTAIN = "None of the available tools can answer that."


# ---------------------------------------------------------------------------
# Case A — natural pairs (correct + incorrect both present)
# ---------------------------------------------------------------------------

def test_pairs_built_when_both_correct_and_incorrect_present():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[CORRECT, WRONG],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["chosen"] == CORRECT
    assert pair["rejected"] == WRONG
    assert pair["chosen_reward"] == 1.0
    assert pair["rejected_reward"] == 0.0
    assert pair["pair_kind"] == "natural"


def test_max_pairs_caps_emitted_pairs():
    completions = [CORRECT, CORRECT, WRONG, WRONG, WRONG]
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=completions,
        expected_calls=EXPECTED, max_pairs=2, tools=TOOLS,
    )
    assert len(pairs) == 2


def test_rejected_failure_category_is_recorded():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[CORRECT, WRONG],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS,
    )
    assert pairs[0]["rejected_failure"] == "wrong_argument_type"


# ---------------------------------------------------------------------------
# Legacy behaviour preserved with synthesize=False
# ---------------------------------------------------------------------------

def test_no_pairs_when_all_correct_and_synthesis_off():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[CORRECT, CORRECT],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS, synthesize=False,
    )
    assert pairs == []


def test_no_pairs_when_all_incorrect_and_synthesis_off():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[WRONG, WRONG],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS, synthesize=False,
    )
    assert pairs == []


# ---------------------------------------------------------------------------
# Case B — all rollouts correct -> synthesise a rejected
# ---------------------------------------------------------------------------

def test_all_correct_positive_synthesises_corrupted_negative():
    rng = random.Random(0)
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[CORRECT, CORRECT],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS, rng=rng, synthesize=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_kind"] == "synth_corrupted_negative"
    # chosen still grades correct, synthesised rejected grades wrong
    assert grade(pair["chosen"], EXPECTED).correct
    assert not grade(pair["rejected"], EXPECTED).correct


def test_all_correct_irrelevance_synthesises_abstention_pair():
    rng = random.Random(0)
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[ABSTAIN, ABSTAIN],
        expected_calls=[], max_pairs=1, tools=TOOLS, rng=rng, synthesize=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_kind"] == "synth_abstention_negative"
    # chosen = abstention (correct for irrelevance), rejected = a hallucinated call
    assert grade(pair["chosen"], []).correct
    assert not grade(pair["rejected"], []).correct


# ---------------------------------------------------------------------------
# Case C — all rollouts incorrect -> synthesise the chosen from ground truth
# ---------------------------------------------------------------------------

def test_all_incorrect_positive_synthesises_gold_chosen():
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[WRONG, WRONG],
        expected_calls=EXPECTED, max_pairs=1, tools=TOOLS, synthesize=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_kind"] == "synth_gold_positive"
    assert grade(pair["chosen"], EXPECTED).correct      # gold answer
    assert pair["rejected"] == WRONG


def test_all_incorrect_irrelevance_synthesises_abstention_gold():
    # For irrelevance, every call is "incorrect"; chosen should be the abstention.
    call = _completion("get_weather", {"city": "Paris"})
    pairs = _build_pairs_for_query(
        prompt="P", query="Q", completions=[call, call],
        expected_calls=[], max_pairs=1, tools=TOOLS, synthesize=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_kind"] == "synth_abstention_gold"
    assert pair["chosen"] == NO_TOOL_RESPONSE
    assert grade(pair["chosen"], []).correct
