"""Tests for pipeline/formatting/chat_template.py"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pipeline.formatting.chat_template import (
    TOOL_CALL_CLOSE_TAG,
    TOOL_CALL_OPEN_TAG,
    _build_system_prompt_with_tools,
    _render_tool_call_response,
    extract_tool_calls,
    format_inference_prompt,
    format_sft_example,
)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer that avoids loading any real weights
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(vocab_size: int = 100):
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<eos>"

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def tokenize_fn(text, truncation=True, max_length=2048, return_tensors=None):
        # Deterministic fake token IDs based on character positions
        ids = [ord(c) % vocab_size for c in text[:max_length]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.__call__ = tokenize_fn
    return tokenizer


# ---------------------------------------------------------------------------
# _build_system_prompt_with_tools
# ---------------------------------------------------------------------------

def test_system_prompt_contains_tool_name():
    tools = [{"name": "get_weather", "description": "Gets weather", "parameters": {}}]
    prompt = _build_system_prompt_with_tools(tools)
    assert "get_weather" in prompt
    assert "<tools>" in prompt
    assert "</tools>" in prompt


def test_system_prompt_wraps_tool_in_function_schema():
    tools = [{"name": "search", "description": "Search the web", "parameters": {}}]
    prompt = _build_system_prompt_with_tools(tools)
    assert '"type": "function"' in prompt


def test_system_prompt_passthrough_for_already_wrapped_tool():
    tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
    prompt = _build_system_prompt_with_tools(tools)
    # Should not double-wrap
    assert prompt.count('"type": "function"') == 1


def test_system_prompt_multiple_tools():
    tools = [
        {"name": "tool_a", "description": "a", "parameters": {}},
        {"name": "tool_b", "description": "b", "parameters": {}},
    ]
    prompt = _build_system_prompt_with_tools(tools)
    assert "tool_a" in prompt
    assert "tool_b" in prompt


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
    if result["input_ids"]:  # non-empty (not filtered out)
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
        # At least the prompt tokens should be masked
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
