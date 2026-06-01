"""Tests for pipeline/data modules (registry, xlam, bfcl parsers)."""
from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def test_registry_raises_on_unknown_dataset():
    from pipeline.data.registry import get_dataset

    cfg = OmegaConf.create({"training": {}, "data": {"seed": 42}})
    with pytest.raises(KeyError, match="not registered"):
        get_dataset("nonexistent_dataset_xyz", cfg)


def test_registry_register_and_retrieve():
    from datasets import Dataset

    from pipeline.data.registry import _REGISTRY, get_dataset, register

    def mock_load(cfg):
        return Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]})

    def mock_format(example):
        return {"processed_text": example["text"].upper()}

    register("test_mock_dataset", mock_load, mock_format)
    try:
        cfg = OmegaConf.create({"training": {}, "data": {"seed": 42}})
        dataset = get_dataset("test_mock_dataset", cfg)
        assert "processed_text" in dataset.column_names
        assert dataset[0]["processed_text"] == "HELLO"
    finally:
        del _REGISTRY["test_mock_dataset"]


def test_registry_format_fn_receives_tokenizer_when_provided():
    from datasets import Dataset

    from pipeline.data.registry import _REGISTRY, get_dataset, register

    received_tokenizer = []

    def mock_load(cfg):
        return Dataset.from_dict({"value": [42]})

    def mock_format_with_tokenizer(example, tokenizer):
        received_tokenizer.append(tokenizer)
        return {"output": example["value"]}

    register("test_tokenizer_dataset", mock_load, mock_format_with_tokenizer)
    try:
        cfg = OmegaConf.create({"training": {}, "data": {}})
        fake_tokenizer = object()
        get_dataset("test_tokenizer_dataset", cfg, tokenizer=fake_tokenizer)
        assert received_tokenizer[0] is fake_tokenizer
    finally:
        del _REGISTRY["test_tokenizer_dataset"]


# ---------------------------------------------------------------------------
# xLAM field parsing
# ---------------------------------------------------------------------------

def test_parse_json_field_from_string():
    from pipeline.data.xlam import _parse_json_field

    tools_str = json.dumps([{"name": "search", "parameters": {}}])
    result = _parse_json_field(tools_str)
    assert isinstance(result, list)
    assert result[0]["name"] == "search"


def test_parse_json_field_from_native_list():
    from pipeline.data.xlam import _parse_json_field

    tools = [{"name": "get_time"}, {"name": "get_weather"}]
    result = _parse_json_field(tools)
    assert result == tools


def test_parse_json_field_wraps_single_dict_in_list():
    from pipeline.data.xlam import _parse_json_field

    single_tool = {"name": "fn", "parameters": {}}
    result = _parse_json_field(json.dumps(single_tool))
    assert isinstance(result, list)
    assert result[0]["name"] == "fn"


def test_is_valid_example_accepts_well_formed():
    from pipeline.data.xlam import _is_valid_example

    example = {
        "query": "Get weather for London",
        "tools": json.dumps([{"name": "get_weather", "parameters": {}}]),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "London"}}]),
    }
    assert _is_valid_example(example) is True


def test_is_valid_example_rejects_empty_tools():
    from pipeline.data.xlam import _is_valid_example

    example = {
        "query": "Do something",
        "tools": json.dumps([]),
        "answers": json.dumps([{"name": "fn", "arguments": {}}]),
    }
    assert _is_valid_example(example) is False


def test_is_valid_example_rejects_empty_answers():
    from pipeline.data.xlam import _is_valid_example

    example = {
        "query": "Do something",
        "tools": json.dumps([{"name": "fn", "parameters": {}}]),
        "answers": json.dumps([]),
    }
    assert _is_valid_example(example) is False


# ---------------------------------------------------------------------------
# BFCL v3 internal helpers
# ---------------------------------------------------------------------------

def test_extract_query_from_plain_string():
    from pipeline.data.bfcl import _extract_query

    assert _extract_query("What is the weather in Seattle?") == "What is the weather in Seattle?"


def test_extract_query_from_nested_list_of_messages():
    # BFCL v3 format: list[list[dict]]
    from pipeline.data.bfcl import _extract_query

    question = [[
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Response"},
    ], [
        {"role": "user", "content": "Final user query"},
    ]]
    assert _extract_query(question) == "Final user query"


def test_extract_query_flat_list_of_messages():
    # Older format: list[dict] (flat, not nested)
    from pipeline.data.bfcl import _extract_query

    question = [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Last query"},
    ]
    assert _extract_query(question) == "Last query"


def test_normalise_ground_truth_bfcl_possible_answer_format():
    from pipeline.data.bfcl import _normalise_ground_truth

    raw = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    result = _normalise_ground_truth(raw)
    assert len(result) == 1
    assert result[0]["name"] == "calculate_triangle_area"
    assert result[0]["arguments"]["base"] == [10]
    assert result[0]["arguments"]["unit"] == ["units", ""]


def test_normalise_ground_truth_empty():
    from pipeline.data.bfcl import _normalise_ground_truth

    assert _normalise_ground_truth([]) == []


# ---------------------------------------------------------------------------
# parse_bfcl_example pass-through (already-normalised rows)
# ---------------------------------------------------------------------------

def test_parse_bfcl_example_passthrough_normalised_row():
    from pipeline.data.bfcl import parse_bfcl_example

    row = {
        "query": "What is the weather in Seattle?",
        "tools": [{"name": "get_weather", "parameters": {}}],
        "expected_calls": [{"name": "get_weather", "arguments": {"city": ["Seattle"]}}],
        "category": "simple",
    }
    parsed = parse_bfcl_example(row, category="simple")
    assert parsed["query"] == "What is the weather in Seattle?"
    assert len(parsed["tools"]) == 1
    assert len(parsed["expected_calls"]) == 1
    assert parsed["category"] == "simple"


def test_parse_bfcl_example_handles_missing_fields_gracefully():
    from pipeline.data.bfcl import parse_bfcl_example

    parsed = parse_bfcl_example({})
    assert parsed["query"] == ""
    assert parsed["tools"] == []
    assert parsed["expected_calls"] == []
