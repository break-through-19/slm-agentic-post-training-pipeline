"""
Berkeley Function Calling Leaderboard (BFCL v3) dataset loader.

Source: gorilla-llm/Berkeley-Function-Calling-Leaderboard on Hugging Face.

The dataset stores question files and answer files separately:
  BFCL_v3_<category>.json          — queries + tool definitions (one JSON per line)
  possible_answer/BFCL_v3_<category>.json — ground-truth acceptable answers

Each answer entry uses the BFCL possible-answer format where argument values
are lists of acceptable responses rather than a single expected value.

The irrelevance category has no answer file — the correct model behaviour is
to produce no tool call at all.

Each normalised example exposes:
  query          str            — the user's natural language request
  tools          list[dict]     — available function definitions (JSON Schema)
  expected_calls list[dict]     — {name, arguments} where argument values are
                                  lists of acceptable answers; [] for irrelevance
  category       str            — BFCL sub-category label
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

BFCL_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Maps our canonical category names to their JSON file names in the HF repo.
CATEGORY_TO_QUESTION_FILE: dict[str, str] = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}

CATEGORY_TO_ANSWER_FILE: dict[str, str | None] = {
    "simple": "possible_answer/BFCL_v3_simple.json",
    "multiple": "possible_answer/BFCL_v3_multiple.json",
    "parallel": "possible_answer/BFCL_v3_parallel.json",
    "irrelevance": None,  # correct answer is no tool call
}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _download_jsonl(filename: str) -> list[dict]:
    """Download a JSONL file from the BFCL HF repo and parse it line by line."""
    local_path = hf_hub_download(
        repo_id=BFCL_HF_REPO,
        filename=filename,
        repo_type="dataset",
    )
    rows = []
    with open(local_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_query(raw_question: list | str) -> str:
    """
    Extract the user query string from the BFCL question field.

    BFCL v3 stores questions as list[list[dict]], where each inner list is one
    conversation turn and each dict has 'role' and 'content' keys.
    Single-turn example: [[{"role": "user", "content": "..."}]]
    """
    if isinstance(raw_question, str):
        return raw_question

    # Flatten all turns and return the last user message content
    all_messages: list[dict] = []
    for turn in raw_question:
        if isinstance(turn, list):
            all_messages.extend(turn)
        elif isinstance(turn, dict):
            all_messages.append(turn)

    user_messages = [m for m in all_messages if m.get("role") == "user"]
    return user_messages[-1]["content"] if user_messages else ""


def _normalise_ground_truth(raw_ground_truth: list[dict]) -> list[dict]:
    """
    Convert BFCL possible-answer format to our internal format.

    BFCL format:   [{"fn_name": {"arg1": [val1, val2], "arg2": [val3]}}]
    Internal fmt:  [{"name": "fn_name", "arguments": {"arg1": [val1, val2], ...}}]

    Argument values remain as lists so the grader can accept any acceptable value.
    """
    normalised = []
    for entry in raw_ground_truth:
        if not isinstance(entry, dict):
            continue
        for fn_name, raw_args in entry.items():
            args = raw_args if isinstance(raw_args, dict) else {}
            normalised.append({"name": fn_name, "arguments": args})
    return normalised


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_bfcl_category(category: str, max_samples: int | None = None) -> list[dict]:
    """
    Load a single BFCL v3 category and return a plain list of dicts.

    Each dict contains: query, tools, expected_calls, category.
    For the irrelevance category expected_calls is always an empty list.

    A plain list is used intentionally — Apache Arrow (used by HuggingFace
    Dataset.from_list) infers a strict schema from the first row and rejects
    subsequent rows whose tool parameter types differ, which happens routinely
    in BFCL because tool definitions are heterogeneous by design.
    """
    question_file = CATEGORY_TO_QUESTION_FILE.get(category)
    if question_file is None:
        raise ValueError(
            f"Unknown BFCL category '{category}'. "
            f"Valid options: {list(CATEGORY_TO_QUESTION_FILE)}"
        )

    logger.info("Downloading BFCL '%s' question file: %s", category, question_file)
    question_rows = _download_jsonl(question_file)

    # Build id → ground_truth lookup from the answer file (if one exists)
    answer_by_id: dict[str, list[dict]] = {}
    answer_file = CATEGORY_TO_ANSWER_FILE.get(category)
    if answer_file is not None:
        logger.info("Downloading BFCL '%s' answer file: %s", category, answer_file)
        for row in _download_jsonl(answer_file):
            row_id = row.get("id", "")
            raw_gt = row.get("ground_truth", [])
            answer_by_id[row_id] = _normalise_ground_truth(raw_gt)

    # Merge questions with answers
    examples = []
    for row in question_rows:
        row_id = row.get("id", "")
        query = _extract_query(row.get("question", ""))
        tools = row.get("function", [])
        if not isinstance(tools, list):
            tools = [tools]

        if answer_file is None:
            # Irrelevance: correct response is no tool call
            expected_calls: list[dict] = []
        else:
            expected_calls = answer_by_id.get(row_id, [])

        examples.append({
            "query": query,
            "tools": tools,
            "expected_calls": expected_calls,
            "category": category,
        })

    if max_samples is not None:
        examples = examples[:max_samples]

    logger.info("Loaded %d examples for BFCL category '%s'", len(examples), category)
    return examples


# ---------------------------------------------------------------------------
# Example normalisation (kept for backward-compatibility with evaluator)
# ---------------------------------------------------------------------------


def parse_bfcl_example(raw_example: dict, category: str = "unknown") -> dict:
    """
    Pass-through normaliser for already-parsed BFCL rows.

    When iterating over a Dataset returned by load_bfcl_category the rows are
    already in the correct format; this function exists so the evaluator loop
    does not need to know whether data came from load_bfcl_category or some
    other source.
    """
    return {
        "query": raw_example.get("query", ""),
        "tools": raw_example.get("tools", []),
        "expected_calls": raw_example.get("expected_calls", []),
        "category": raw_example.get("category", category),
    }
