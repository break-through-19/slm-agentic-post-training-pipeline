"""
Self-generated preference-pair generation for Stage 2A (DPO).

Pipeline (mirrors slide 8 of the proposal):
  1. Sample `rollouts_per_query` completions from the SFT model per query.
  2. Auto-grade each completion with the BFCL verifier (name + args + types).
  3. Pair a correct completion (chosen) with an incorrect one (rejected).
  4. Drop queries where every rollout passes or every rollout fails — those
     carry no preference signal.

Source queries come from xLAM, which ships ground-truth answers we can grade
against using the exact same `bfcl_grader.grade()` used at evaluation time.

Generation uses plain HuggingFace `model.generate` so it runs unchanged on
CUDA, MPS, and CPU. On a GPU you can later swap in vLLM for a large speed-up;
the rest of the pipeline is agnostic to how completions are produced.

The output is a JSONL file with one record per preference pair:
  {"prompt": str, "chosen": str, "rejected": str, "query": str,
   "chosen_reward": float, "rejected_reward": float,
   "rejected_failure": str | None}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from pipeline.data.xlam import _parse_json_field, _resolve_hf_token
from pipeline.formatting.chat_template import format_inference_prompt
from pipeline.reward.bfcl_grader import grade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source query loading
# ---------------------------------------------------------------------------


def _load_source_queries(cfg: DictConfig) -> list[dict]:
    """
    Load xLAM examples to use as the source of preference-generation prompts.

    Returns a list of dicts with: query, tools (list), expected_calls (list).
    Reuses xLAM's helpers so parsing and auth behaviour stay identical to SFT.
    """
    from datasets import load_dataset

    gen_cfg = cfg.generation
    num_queries = gen_cfg.get("num_source_queries", 2000)
    seed = cfg.data.get("seed", 42)

    logger.info("Loading xLAM source queries for preference generation")
    dataset = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split="train",
        token=_resolve_hf_token(),
    )
    dataset = dataset.shuffle(seed=seed).select(range(min(num_queries, len(dataset))))

    # Phase 1: blend in abstention queries. For these the SFT model that
    # sometimes abstains and sometimes calls a tool yields natural preference
    # pairs (chosen = abstain, rejected = hallucinated call), teaching DPO to
    # recover the irrelevance category.
    irrelevance_fraction = gen_cfg.get("irrelevance_fraction", 0.0)
    if irrelevance_fraction > 0:
        from pipeline.data.irrelevance import inject_irrelevance

        dataset = inject_irrelevance(dataset, irrelevance_fraction, seed=seed)

    source_examples: list[dict] = []
    for row in dataset:
        source_examples.append(
            {
                "query": row["query"],
                "tools": _parse_json_field(row["tools"]),
                "expected_calls": _parse_json_field(row["answers"]),
            }
        )
    logger.info("Prepared %d source queries", len(source_examples))
    return source_examples


# ---------------------------------------------------------------------------
# Rollout sampling
# ---------------------------------------------------------------------------


def _sample_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """
    Sample `num_samples` stochastic completions for a single prompt.

    Returns the decoded completion strings with special tokens stripped
    (the <tool_call> tags are ordinary text, so they survive decoding and
    remain visible to the grader).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    with torch.inference_mode():
        output_token_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    completions = [
        tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        for sequence in output_token_ids
    ]
    return completions


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _build_pairs_for_query(
    prompt: str,
    query: str,
    completions: list[str],
    expected_calls: list[dict],
    max_pairs: int,
) -> list[dict]:
    """
    Grade every completion and form (chosen, rejected) pairs.

    A query yields pairs only when it has at least one correct and at least
    one incorrect completion. Up to `max_pairs` pairs are emitted, each pairing
    a distinct correct completion with a distinct incorrect one.
    """
    graded = [(completion, grade(completion, expected_calls)) for completion in completions]

    correct_completions = [c for c, result in graded if result.correct]
    incorrect = [(c, result) for c, result in graded if not result.correct]

    # No preference signal if everything passed or everything failed
    if not correct_completions or not incorrect:
        return []

    pairs: list[dict] = []
    for index in range(min(max_pairs, len(correct_completions), len(incorrect))):
        chosen_completion = correct_completions[index]
        rejected_completion, rejected_result = incorrect[index]
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen_completion,
                "rejected": rejected_completion,
                "query": query,
                "chosen_reward": 1.0,
                "rejected_reward": 0.0,
                "rejected_failure": rejected_result.failure_category,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_preference_pairs(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> list[dict]:
    """
    Run the full self-generation loop and return the list of preference pairs.

    Reads its knobs from cfg.generation: rollouts_per_query, temperature,
    top_p, max_new_tokens, max_pairs_per_query.
    """
    gen_cfg = cfg.generation
    rollouts_per_query = gen_cfg.get("rollouts_per_query", 8)
    temperature = gen_cfg.get("temperature", 0.8)
    top_p = gen_cfg.get("top_p", 0.95)
    max_new_tokens = gen_cfg.get("max_new_tokens", 256)
    max_pairs_per_query = gen_cfg.get("max_pairs_per_query", 1)

    source_examples = _load_source_queries(cfg)
    model.eval()

    all_pairs: list[dict] = []
    queries_with_signal = 0

    for index, example in enumerate(source_examples):
        # Tools are always required to build a prompt. expected_calls MAY be
        # empty — that is a valid irrelevance example where abstention is the
        # correct (chosen) behaviour, so we must NOT skip it.
        if not example["tools"]:
            continue

        prompt = format_inference_prompt(
            query=example["query"],
            tools=example["tools"],
            tokenizer=tokenizer,
        )
        completions = _sample_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            num_samples=rollouts_per_query,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        pairs = _build_pairs_for_query(
            prompt=prompt,
            query=example["query"],
            completions=completions,
            expected_calls=example["expected_calls"],
            max_pairs=max_pairs_per_query,
        )
        if pairs:
            queries_with_signal += 1
            all_pairs.extend(pairs)

        if (index + 1) % 50 == 0:
            logger.info(
                "Processed %d/%d queries — %d pairs from %d informative queries",
                index + 1, len(source_examples), len(all_pairs), queries_with_signal,
            )

    logger.info(
        "Preference generation complete — %d pairs from %d/%d informative queries",
        len(all_pairs), queries_with_signal, len(source_examples),
    )
    return all_pairs


def save_pairs_jsonl(pairs: list[dict], output_path: str | Path) -> None:
    """Write preference pairs to a JSONL file, one record per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Wrote %d preference pairs to %s", len(pairs), output_path)
