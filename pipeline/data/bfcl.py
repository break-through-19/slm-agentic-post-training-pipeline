"""
Berkeley Function Calling Leaderboard (BFCL v3) dataset loader.

Source: gorilla-llm/Berkeley-Function-Calling-Leaderboard on Hugging Face.
Used for Stage 0 evaluation (baseline) and Stage 1 post-training evaluation.
Also provides the reward signal for Stage 2 (DPO / GRPO).

Each normalised example exposes:
  query         str           — the user's natural language request
  tools         list[dict]    — available function definitions (JSON Schema)
  expected_calls list[dict]   — ground-truth {name, arguments} dicts
  category      str           — BFCL sub-category label
"""
from __future__ import annotations

import json
import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

BFCL_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Maps our canonical category names to the HuggingFace dataset config names.
CATEGORY_TO_HF_CONFIG: dict[str, str] = {
    "simple": "gorilla_openfunctions_v1_test_simple",
    "multiple": "gorilla_openfunctions_v1_test_multiple_function",
    "parallel": "gorilla_openfunctions_v1_test_parallel_function",
    "irrelevance": "gorilla_openfunctions_v1_test_relevance",
}


def load_bfcl_category(category: str, max_samples: int | None = None) -> Dataset:
    """Load a single BFCL category split from Hugging Face."""
    hf_config = CATEGORY_TO_HF_CONFIG.get(category)
    if hf_config is None:
        raise ValueError(
            f"Unknown BFCL category '{category}'. "
            f"Valid options: {list(CATEGORY_TO_HF_CONFIG)}"
        )

    logger.info("Loading BFCL category '%s' (config=%s)", category, hf_config)
    dataset = load_dataset(BFCL_HF_REPO, hf_config, split="train", trust_remote_code=True)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info("  Loaded %d examples for category '%s'", len(dataset), category)
    return dataset


def _coerce_to_list(value: str | list | dict | None) -> list:
    """Normalise a field that may be a JSON string, dict, or list into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []
    return []


def parse_bfcl_example(raw_example: dict, category: str = "unknown") -> dict:
    """
    Normalise a raw BFCL HuggingFace row into a consistent structure.

    The BFCL dataset stores the user question as either a plain string or
    a list-of-message dicts (multi-turn format).  This function extracts
    the final user message in both cases.
    """
    raw_question = raw_example.get("question", "")
    if isinstance(raw_question, list):
        # Multi-turn: take the last user message content
        user_messages = [m for m in raw_question if m.get("role") == "user"]
        query = user_messages[-1]["content"] if user_messages else ""
    else:
        query = str(raw_question)

    tools = _coerce_to_list(raw_example.get("function"))
    expected_calls = _coerce_to_list(raw_example.get("ground_truth"))

    return {
        "query": query,
        "tools": tools,
        "expected_calls": expected_calls,
        "category": category,
    }
