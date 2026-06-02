"""Tests for Phase 1 irrelevance synthesis (pipeline/data/irrelevance.py)
and the abstention SFT target (chat_template.NO_TOOL_RESPONSE)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from datasets import Dataset

from pipeline.data.irrelevance import inject_irrelevance


def _toy_xlam(n: int = 20) -> Dataset:
    """A tiny xLAM-shaped dataset with JSON-string tools/answers columns."""
    rows = {
        "query": [f"query number {i}" for i in range(n)],
        "tools": [json.dumps([{"name": f"fn_{i}", "parameters": {}}]) for i in range(n)],
        "answers": [json.dumps([{"name": f"fn_{i}", "arguments": {}}]) for i in range(n)],
    }
    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# inject_irrelevance
# ---------------------------------------------------------------------------

def test_zero_fraction_returns_dataset_unchanged():
    ds = _toy_xlam(10)
    out = inject_irrelevance(ds, fraction=0.0, seed=1)
    assert out is ds


def test_fraction_sets_expected_number_of_abstention_rows():
    ds = _toy_xlam(20)
    out = inject_irrelevance(ds, fraction=0.25, seed=1)
    assert len(out) == 20
    abstain = [a for a in out["answers"] if a == "[]"]
    assert len(abstain) == 5  # 25% of 20


def test_abstention_rows_have_mismatched_tools():
    ds = _toy_xlam(20)
    out = inject_irrelevance(ds, fraction=0.5, seed=7)
    for i in range(len(out)):
        if out["answers"][i] == "[]":
            # The tools were swapped from another row, so fn_{i} should not be present
            assert f'"fn_{i}"' not in out["tools"][i]


def test_positive_rows_are_preserved():
    ds = _toy_xlam(20)
    out = inject_irrelevance(ds, fraction=0.25, seed=3)
    # Non-abstention rows keep a non-empty answer
    positives = [a for a in out["answers"] if a != "[]"]
    assert len(positives) == 15
    for a in positives:
        assert json.loads(a)  # parses to a non-empty list


def test_deterministic_with_seed():
    ds = _toy_xlam(20)
    out1 = inject_irrelevance(ds, fraction=0.3, seed=42)
    out2 = inject_irrelevance(ds, fraction=0.3, seed=42)
    assert out1["answers"] == out2["answers"]
    assert out1["tools"] == out2["tools"]


def test_handles_tiny_dataset():
    ds = _toy_xlam(1)
    out = inject_irrelevance(ds, fraction=0.5, seed=1)
    assert len(out) == 1  # too small to swap, returned unchanged


# ---------------------------------------------------------------------------
# Abstention SFT target
# ---------------------------------------------------------------------------

def _capturing_tokenizer(captured: dict):
    """
    Mock tokenizer that records the message list passed to the full (non-prompt)
    apply_chat_template call, so we can assert what the assistant target was.

    Note: MagicMock does not honour an instance-level __call__ assignment, so we
    avoid depending on tokenisation output and inspect the messages instead.
    """
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0

    def apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=False):
        if not add_generation_prompt:
            # This is the full conversation including the assistant target
            captured["messages"] = messages
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    tok.apply_chat_template = apply_chat_template
    return tok


def test_empty_expected_calls_uses_refusal_target():
    from pipeline.formatting.chat_template import NO_TOOL_RESPONSE, format_sft_example

    captured: dict = {}
    tok = _capturing_tokenizer(captured)
    format_sft_example(
        query="Do something the tools cannot do",
        tools=[{"name": "unrelated_fn", "description": "x", "parameters": {}}],
        expected_calls=[],          # irrelevance example
        tokenizer=tok,
    )
    assistant_message = captured["messages"][-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == NO_TOOL_RESPONSE
    assert "<tool_call>" not in assistant_message["content"]


def test_nonempty_expected_calls_uses_tool_call_target():
    from pipeline.formatting.chat_template import format_sft_example

    captured: dict = {}
    tok = _capturing_tokenizer(captured)
    format_sft_example(
        query="What is the weather in Paris?",
        tools=[{"name": "get_weather", "description": "x", "parameters": {}}],
        expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        tokenizer=tok,
    )
    assistant_message = captured["messages"][-1]
    assert "<tool_call>" in assistant_message["content"]
    assert "get_weather" in assistant_message["content"]
