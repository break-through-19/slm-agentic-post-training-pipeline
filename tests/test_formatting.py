"""Tests for pipeline/formatting/chat_template.py"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pipeline.formatting.chat_template import (
    TOOL_CALL_CLOSE_TAG,
    TOOL_CALL_OPEN_TAG,
    _normalise_tools_for_tokenizer,
    _render_tool_call_response,
    extract_tool_calls,
    format_inference_prompt,
    format_sft_example,
)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer — avoids loading any real weights.
# apply_chat_template accepts the tools= kwarg so the normalisation path
# is exercised without needing a real Qwen2.5 tokenizer.
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(vocab_size: int = 100):
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<eos>"

    def apply_chat_template(
        messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if tools:
            parts.insert(0, f"<tools>{json.dumps(tools)}</tools>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def tokenize_fn(text, truncation=True, max_length=2048, return_tensors=None):
        ids = [ord(c) % vocab_size for c in text[:max_length]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.__call__ = tokenize_fn
    return tokenizer


# ---------------------------------------------------------------------------
# _normalise_tools_for_tokenizer
# ---------------------------------------------------------------------------

def test_normalise_wraps_bare_tool_in_function_schema():
    tools = [{"name": "get_weather", "description": "Gets weather", "parameters": {}}]
    result = _normalise_tools_for_tokenizer(tools)
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"


def test_normalise_does_not_double_wrap_already_schema_tool():
    tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
    result = _normalise_tools_for_tokenizer(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    # Should not be nested twice
    assert "function" not in result[0]["function"]


def test_normalise_handles_multiple_tools():
    tools = [
        {"name": "tool_a", "description": "a", "parameters": {}},
        {"name": "tool_b", "description": "b", "parameters": {}},
    ]
    result = _normalise_tools_for_tokenizer(tools)
    assert len(result) == 2
    names = {r["function"]["name"] for r in result}
    assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# _render_tool_call_response
# ---------------------------------------------------------------------------

def test_render_single_tool_call():
    calls = [{"name": "get_weather", "arguments": {"city": "Seattle"}}]
    rendered = _render_tool_call_response(calls)
    assert TOOL_CALL_OPEN_TAG in rendered
    assert TOOL_CALL_CLOSE_TAG in rendered
    assert "get_weather" in rendered
    assert "Seattle" in rendered


def test_render_multiple_tool_calls():
    calls = [
        {"name": "fn_a", "arguments": {"x": 1}},
        {"name": "fn_b", "arguments": {"y": "hello"}},
    ]
    rendered = _render_tool_call_response(calls)
    assert rendered.count(TOOL_CALL_OPEN_TAG) == 2
    assert "fn_a" in rendered
    assert "fn_b" in rendered


def test_render_picks_first_value_from_bfcl_possible_answer_list():
    """BFCL possible-answer format stores acceptable values as lists;
    the renderer should pick the first concrete value for the training target."""
    calls = [{"name": "calc", "arguments": {"unit": ["units", ""], "base": [10]}}]
    rendered = _render_tool_call_response(calls)
    parsed = extract_tool_calls(rendered)
    assert parsed[0]["arguments"]["unit"] == "units"
    assert parsed[0]["arguments"]["base"] == 10


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------

def test_extract_single_valid_tool_call():
    payload = json.dumps({"name": "get_weather", "arguments": {"city": "Seattle"}})
    output = f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"
    calls = extract_tool_calls(output)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"]["city"] == "Seattle"


def test_extract_multiple_tool_calls():
    p1 = json.dumps({"name": "fn_a", "arguments": {}})
    p2 = json.dumps({"name": "fn_b", "arguments": {"z": 99}})
    output = (
        f"{TOOL_CALL_OPEN_TAG}\n{p1}\n{TOOL_CALL_CLOSE_TAG}\n"
        f"{TOOL_CALL_OPEN_TAG}\n{p2}\n{TOOL_CALL_CLOSE_TAG}"
    )
    calls = extract_tool_calls(output)
    assert len(calls) == 2
    assert calls[0]["name"] == "fn_a"
    assert calls[1]["name"] == "fn_b"


def test_extract_returns_empty_for_no_tags():
    calls = extract_tool_calls("I cannot help with that.")
    assert calls == []


def test_extract_silently_drops_malformed_json():
    output = f"{TOOL_CALL_OPEN_TAG}not valid json{TOOL_CALL_CLOSE_TAG}"
    calls = extract_tool_calls(output)
    assert calls == []


def test_extract_handles_whitespace_around_json():
    payload = json.dumps({"name": "ping", "arguments": {}})
    output = f"{TOOL_CALL_OPEN_TAG}   \n  {payload}  \n  {TOOL_CALL_CLOSE_TAG}"
    calls = extract_tool_calls(output)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# format_sft_example
# ---------------------------------------------------------------------------

def test_format_sft_example_returns_required_keys():
    tokenizer = _make_mock_tokenizer()
    result = format_sft_example(
        query="What is the weather in Paris?",
        tools=[{"name": "get_weather", "description": "Gets weather", "parameters": {}}],
        expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        tokenizer=tokenizer,
    )
    assert set(result.keys()) == {"input_ids", "labels", "attention_mask"}


def test_format_sft_example_labels_match_input_ids_length():
    tokenizer = _make_mock_tokenizer()
    result = format_sft_example(
        query="Book a flight",
        tools=[{"name": "book_flight", "description": "Books a flight", "parameters": {}}],
        expected_calls=[{"name": "book_flight", "arguments": {"destination": "NYC"}}],
        tokenizer=tokenizer,
    )
    if result["input_ids"]:
        assert len(result["input_ids"]) == len(result["labels"])
        assert len(result["input_ids"]) == len(result["attention_mask"])


def test_format_sft_example_prompt_tokens_are_masked():
    tokenizer = _make_mock_tokenizer()
    result = format_sft_example(
        query="Search for something",
        tools=[{"name": "search", "description": "Searches", "parameters": {}}],
        expected_calls=[{"name": "search", "arguments": {"query": "python"}}],
        tokenizer=tokenizer,
    )
    if result["input_ids"]:
        assert -100 in result["labels"]


# ---------------------------------------------------------------------------
# format_inference_prompt
# ---------------------------------------------------------------------------

def test_format_inference_prompt_contains_query():
    tokenizer = _make_mock_tokenizer()
    prompt = format_inference_prompt(
        query="What time is it in Tokyo?",
        tools=[{"name": "get_time", "description": "Gets time", "parameters": {}}],
        tokenizer=tokenizer,
    )
    assert "Tokyo" in prompt


def test_format_inference_prompt_contains_tool_name():
    tokenizer = _make_mock_tokenizer()
    prompt = format_inference_prompt(
        query="Search for cats",
        tools=[{"name": "web_search", "description": "Searches web", "parameters": {}}],
        tokenizer=tokenizer,
    )
    assert "web_search" in prompt


def test_format_inference_prompt_passes_tools_to_tokenizer():
    """Verify tools= is forwarded to apply_chat_template (not a manual system prompt)."""
    tokenizer = _make_mock_tokenizer()
    prompt = format_inference_prompt(
        query="Do something",
        tools=[{"name": "my_fn", "description": "Does something", "parameters": {}}],
        tokenizer=tokenizer,
    )
    # The mock embeds tools as <tools>...</tools> when tools= kwarg is received
    assert "<tools>" in prompt
