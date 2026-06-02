"""
Prompt dataset for Stage 2B (GRPO, online RL).

Unlike DPO (which trains on pre-generated pairs) GRPO generates completions
on-policy during training. The dataset therefore only needs:
  prompt              — the chat-templated query string the policy rolls out from
  expected_calls_json — ground-truth calls (JSON string) consumed by the reward fn

The reward function (pipeline.reward.grpo_reward) receives every non-prompt
column as a keyword list, so `expected_calls_json` is how the verifier learns
what the correct answer was for each prompt in the batch.

Source queries come from xLAM (same distribution as SFT). The ground truth is
stored as a JSON string to sidestep Arrow schema inference over heterogeneous
tool-argument types.

Registers the "xlam_grpo" dataset in the global registry on module import.
"""
from __future__ import annotations

import json
import logging

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from pipeline.data.registry import register
from pipeline.data.xlam import _parse_json_field, _resolve_hf_token
from pipeline.formatting.chat_template import format_inference_prompt

logger = logging.getLogger(__name__)


def _load_grpo_prompts(cfg: DictConfig) -> Dataset:
    """Load an xLAM subset to serve as GRPO rollout prompts."""
    max_samples = cfg.training.get("max_samples", None)
    seed = cfg.data.get("seed", 42)

    logger.info("Loading xLAM prompts for GRPO rollouts")
    dataset = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split="train",
        token=_resolve_hf_token(),
    )
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    # Phase 1: blend in abstention prompts so GRPO sees irrelevance cases. The
    # BFCL reward already returns 1.0 for correctly producing no tool call, so
    # these prompts give the policy a signal to recover the irrelevance category.
    irrelevance_fraction = cfg.training.get("irrelevance_fraction", 0.0)
    if irrelevance_fraction > 0:
        from pipeline.data.irrelevance import inject_irrelevance

        dataset = inject_irrelevance(dataset, irrelevance_fraction, seed=seed)

    return dataset


def _format_grpo_example(example: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """
    Build the rollout prompt and stash the ground truth as a JSON string.

    Returns:
      prompt              — chat-templated string the policy generates from
      expected_calls_json — JSON-encoded list of ground-truth calls for grading
    """
    tools = _parse_json_field(example["tools"])
    expected_calls = _parse_json_field(example["answers"])

    prompt = format_inference_prompt(
        query=example["query"],
        tools=tools,
        tokenizer=tokenizer,
    )
    return {
        "prompt": prompt,
        "expected_calls_json": json.dumps(expected_calls, ensure_ascii=False),
    }


register("xlam_grpo", _load_grpo_prompts, _format_grpo_example)
