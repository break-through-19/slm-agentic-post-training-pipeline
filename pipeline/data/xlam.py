"""
xLAM-60K dataset loader for Stage 1 SFT.

Source: Salesforce/xlam-function-calling-60k on Hugging Face.
Each example contains: query (str), tools (JSON str or list), answers (JSON str or list).

NOTE: This is a gated dataset. Before using it you must:
  1. Accept the terms at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
  2. Authenticate via one of:
       export HF_TOKEN="hf_..."          # environment variable (recommended)
       huggingface-cli login             # interactive login (persists to ~/.cache)

Registers the "xlam_sft" dataset in the global registry on module import.
"""
from __future__ import annotations

import json
import logging
import os

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from pipeline.data.registry import register
from pipeline.formatting.chat_template import format_sft_example

logger = logging.getLogger(__name__)

XLAM_HF_REPO = "Salesforce/xlam-function-calling-60k"
XLAM_TERMS_URL = "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"


def _resolve_hf_token() -> str | None:
    """Return the HuggingFace token from the environment, or None if not set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        logger.info("HF_TOKEN found in environment — using for authenticated requests")
    return token


def _parse_json_field(value: str | list | dict) -> list:
    """Coerce a field that may be a JSON string or a native Python object to a list."""
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, list):
        return value
    return [value]


def _is_valid_example(example: dict) -> bool:
    """Reject examples with empty or unparseable tools/answers fields."""
    try:
        tools = _parse_json_field(example.get("tools", []))
        answers = _parse_json_field(example.get("answers", []))
        return len(tools) > 0 and len(answers) > 0
    except (json.JSONDecodeError, TypeError):
        return False


def _load_xlam(cfg: DictConfig) -> Dataset:
    max_samples = cfg.training.get("max_samples", None)
    seed = cfg.data.get("seed", 42)
    token = _resolve_hf_token()

    logger.info("Downloading xLAM dataset from %s", XLAM_HF_REPO)
    try:
        dataset = load_dataset(XLAM_HF_REPO, split="train", token=token)
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"\n\n{'='*65}\n"
            f"  xLAM dataset requires authentication.\n\n"
            f"  Steps to fix:\n"
            f"    1. Accept the dataset terms at:\n"
            f"       {XLAM_TERMS_URL}\n\n"
            f"    2. Then authenticate using either:\n"
            f"       export HF_TOKEN='hf_your_token_here'\n"
            f"       — or —\n"
            f"       huggingface-cli login\n"
            f"{'='*65}\n"
        ) from exc

    original_size = len(dataset)
    dataset = dataset.filter(_is_valid_example, desc="Filtering invalid examples")
    logger.info("Kept %d / %d examples after validity filter", len(dataset), original_size)

    if max_samples is not None:
        num_to_select = min(max_samples, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(num_to_select))
        logger.info("Subsampled to %d examples (max_samples=%d)", len(dataset), max_samples)

    return dataset


def _format_xlam_for_sft(example: dict, tokenizer: PreTrainedTokenizer) -> dict:
    tools = _parse_json_field(example["tools"])
    expected_calls = _parse_json_field(example["answers"])

    return format_sft_example(
        query=example["query"],
        tools=tools,
        expected_calls=expected_calls,
        tokenizer=tokenizer,
    )


register("xlam_sft", _load_xlam, _format_xlam_for_sft)
