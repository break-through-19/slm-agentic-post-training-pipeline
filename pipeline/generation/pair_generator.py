"""
Self-generated preference-pair generation for Stage 2A (DPO).

Pipeline (mirrors slide 8 of the proposal):
  1. Sample `rollouts_per_query` completions from the SFT model per query.
  2. Auto-grade each completion with the BFCL verifier (name + args + types).
  3. Pair a correct completion (chosen) with an incorrect one (rejected).

Sprint step 1 — balanced / synthesised pairs
--------------------------------------------
A query yields a natural pair only when the model produced BOTH a correct and an
incorrect rollout. On the first DPO run that left only ~20% of queries usable and
just 4 / 1355 abstention pairs, so DPO drifted toward always-calling and the
irrelevance category collapsed. With `synthesize_pairs` on (the default) we now
also rescue the uniform cases:

  * all rollouts correct  -> synthesise a plausible REJECTED:
      - irrelevance query: a hallucinated tool call (the wrong thing to do)
      - positive query:    the gold call with one argument corrupted
  * all rollouts incorrect -> synthesise the CHOSEN from the ground truth:
      - irrelevance query: the abstention response (no tool call)
      - positive query:    the gold call rendered from the expected answer

This drives the pair yield toward ~100% of queries and guarantees an abstention
pair for every irrelevance query, which is the fix for the irrelevance crash.

Source queries come from xLAM (graded against its ground truth) plus the
synthetic irrelevance queries injected upstream (Phase 1).

Output: a JSONL file with one record per preference pair:
  {"prompt", "chosen", "rejected", "query", "chosen_reward", "rejected_reward",
   "rejected_failure", "pair_kind"}
"""
from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from pipeline.data.xlam import _parse_json_field, _resolve_hf_token
from pipeline.formatting.chat_template import (
    NO_TOOL_RESPONSE,
    _render_tool_call_response,
    format_inference_prompt,
)
from pipeline.reward.bfcl_grader import grade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source query loading
# ---------------------------------------------------------------------------


def _load_source_queries(cfg: DictConfig) -> list[dict]:
    """
    Load xLAM examples to use as the source of preference-generation prompts.

    Returns a list of dicts with: query, tools (list), expected_calls (list).
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

    # Phase 1: blend in abstention queries (real query + mismatched tools).
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
    """Sample `num_samples` stochastic completions for a single prompt."""
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
    return [
        tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        for sequence in output_token_ids
    ]


# ---------------------------------------------------------------------------
# Synthesis helpers (used when a query's rollouts are all-correct/all-incorrect)
# ---------------------------------------------------------------------------

_JUNK_STRINGS = ["placeholder", "example", "sample", "unknown", "n/a"]


def _tool_params(tool: dict) -> dict:
    """Return the JSON-schema parameters block of a tool, however it is nested."""
    if "function" in tool:
        return tool["function"].get("parameters", {}) or {}
    return tool.get("parameters", {}) or {}


def _tool_name(tool: dict) -> str:
    if "function" in tool:
        return tool["function"].get("name", "")
    return tool.get("name", "")


def _junk_value(json_type: str, rng: random.Random):
    if json_type in ("integer", "number"):
        return rng.randint(2, 999)
    if json_type == "boolean":
        return rng.choice([True, False])
    return rng.choice(_JUNK_STRINGS)


def _synthesize_hallucinated_call(tools: list[dict], rng: random.Random) -> str | None:
    """
    Build a plausible-but-wrong tool call for an irrelevance query, where the
    correct behaviour is to abstain. Used as the REJECTED side of an
    abstention pair.
    """
    if not tools:
        return None
    tool = rng.choice(tools)
    name = _tool_name(tool)
    if not name:
        return None
    params = _tool_params(tool)
    properties = params.get("properties", {})
    required = params.get("required") or list(properties.keys())[:1]
    arguments = {
        key: _junk_value(properties.get(key, {}).get("type", "string"), rng)
        for key in required
    }
    return _render_tool_call_response([{"name": name, "arguments": arguments}])


def _render_gold_call(expected_calls: list[dict]) -> str | None:
    """Render the ground-truth call(s) as the CHOSEN side (positive queries)."""
    if not expected_calls:
        return None
    return _render_tool_call_response(expected_calls)


def _corrupt_expected_call(expected_calls: list[dict], rng: random.Random) -> str | None:
    """
    Render the gold call with exactly one argument corrupted (type-flipped or
    wrong value). A hard negative: right function, wrong argument.
    """
    if not expected_calls:
        return None
    calls = copy.deepcopy(expected_calls)
    candidates = [c for c in calls if c.get("arguments")]
    if not candidates:
        # No arguments to corrupt — corrupt the function name instead
        calls[0]["name"] = (calls[0].get("name", "fn") or "fn") + "_wrong"
        return _render_tool_call_response(calls)

    call = rng.choice(candidates)
    key = rng.choice(list(call["arguments"].keys()))
    value = call["arguments"][key]
    first = value[0] if isinstance(value, list) and value else value
    if isinstance(first, bool):
        call["arguments"][key] = not first
    elif isinstance(first, (int, float)):
        call["arguments"][key] = f"{first}_wrong"   # wrong type (string where number expected)
    else:
        call["arguments"][key] = "__wrong_value__"
    return _render_tool_call_response(calls)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _make_pair(prompt, query, chosen, rejected, failure, kind) -> dict:
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "query": query,
        "chosen_reward": 1.0,
        "rejected_reward": 0.0,
        "rejected_failure": failure,
        "pair_kind": kind,
    }


def _build_pairs_for_query(
    prompt: str,
    query: str,
    completions: list[str],
    expected_calls: list[dict],
    max_pairs: int,
    tools: list[dict] | None = None,
    rng: random.Random | None = None,
    synthesize: bool = True,
) -> list[dict]:
    """
    Grade every completion and form (chosen, rejected) pairs.

    With `synthesize=False` this keeps the original behaviour (a pair only when
    the rollouts contain both a correct and an incorrect completion). With
    `synthesize=True` (default) the all-correct and all-incorrect cases are
    rescued by synthesising the missing side, which both raises the pair yield
    and guarantees abstention pairs for irrelevance queries.
    """
    tools = tools or []
    rng = rng or random.Random(0)
    is_irrelevance = not expected_calls

    graded = [(c, grade(c, expected_calls)) for c in completions]
    correct = [c for c, r in graded if r.correct]
    incorrect = [(c, r) for c, r in graded if not r.correct]

    # Case A — natural signal: real correct vs real incorrect completions
    if correct and incorrect:
        pairs = []
        for i in range(min(max_pairs, len(correct), len(incorrect))):
            pairs.append(_make_pair(
                prompt, query, correct[i], incorrect[i][0],
                incorrect[i][1].failure_category, "natural"))
        return pairs

    if not synthesize:
        return []

    # Case B — all rollouts correct: synthesise a rejected
    if correct and not incorrect:
        chosen = correct[0]
        if is_irrelevance:
            rejected = _synthesize_hallucinated_call(tools, rng)
            kind = "synth_abstention_negative"
        else:
            rejected = _corrupt_expected_call(expected_calls, rng)
            kind = "synth_corrupted_negative"
        if rejected:
            return [_make_pair(prompt, query, chosen, rejected, "synthetic_negative", kind)]
        return []

    # Case C — all rollouts incorrect: synthesise the chosen from the ground truth
    if incorrect and not correct:
        if is_irrelevance:
            chosen = NO_TOOL_RESPONSE
            kind = "synth_abstention_gold"
        else:
            chosen = _render_gold_call(expected_calls)
            kind = "synth_gold_positive"
        rejected, rejected_result = incorrect[0]
        if chosen:
            return [_make_pair(prompt, query, chosen, rejected,
                               rejected_result.failure_category, kind)]
        return []

    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_preference_pairs(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> list[dict]:
    """Run the full self-generation loop and return the list of preference pairs."""
    gen_cfg = cfg.generation
    rollouts_per_query = gen_cfg.get("rollouts_per_query", 8)
    temperature = gen_cfg.get("temperature", 0.8)
    top_p = gen_cfg.get("top_p", 0.95)
    max_new_tokens = gen_cfg.get("max_new_tokens", 256)
    max_pairs_per_query = gen_cfg.get("max_pairs_per_query", 1)
    synthesize = gen_cfg.get("synthesize_pairs", True)
    rng = random.Random(cfg.data.get("seed", 42))

    source_examples = _load_source_queries(cfg)
    model.eval()

    all_pairs: list[dict] = []
    kind_counts: dict[str, int] = {}

    for index, example in enumerate(source_examples):
        if not example["tools"]:
            continue

        prompt = format_inference_prompt(
            query=example["query"], tools=example["tools"], tokenizer=tokenizer)
        completions = _sample_completions(
            model, tokenizer, prompt, device,
            rollouts_per_query, temperature, top_p, max_new_tokens)
        pairs = _build_pairs_for_query(
            prompt=prompt, query=example["query"], completions=completions,
            expected_calls=example["expected_calls"], max_pairs=max_pairs_per_query,
            tools=example["tools"], rng=rng, synthesize=synthesize)
        for p in pairs:
            kind_counts[p["pair_kind"]] = kind_counts.get(p["pair_kind"], 0) + 1
        all_pairs.extend(pairs)

        if (index + 1) % 50 == 0:
            logger.info("Processed %d/%d queries — %d pairs so far",
                        index + 1, len(source_examples), len(all_pairs))

    abstention = sum(v for k, v in kind_counts.items() if "abstention" in k)
    logger.info("Preference generation complete — %d pairs total", len(all_pairs))
    logger.info("  pair kinds: %s", kind_counts)
    logger.info("  abstention pairs: %d (%.0f%%)", abstention,
                100 * abstention / max(1, len(all_pairs)))
    return all_pairs


def save_pairs_jsonl(pairs: list[dict], output_path: str | Path) -> None:
    """Write preference pairs to a JSONL file, one record per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Wrote %d preference pairs to %s", len(pairs), output_path)
