"""
Constrained / structured-output decoding for evaluation (sprint step 5).

Design note — why not hard schema-forced decoding?
---------------------------------------------------
A naive constrained decoder forces the output to match a tool-call JSON schema
on *every* prompt. That destroys the irrelevance category: the correct answer
there is to emit NO tool call, which a "must produce a call" grammar makes
impossible. So instead of forcing a call, this module applies two safe,
abstention-preserving interventions:

  1. normalise_prediction() — extract the tool call(s), coerce argument values to
     the declared schema types, and re-render a single canonical <tool_call>
     block. Empty output (genuine abstention) is left untouched. This is the
     structured-output cleanup a constrained decoder would have produced for
     argument types, with zero risk to irrelevance.

  2. repair_prediction() — a one-shot re-ask used ONLY when the model clearly
     *tried* to call a tool (it emitted a <tool_call> opener) but produced
     unparseable JSON. Genuine abstentions (no opener at all) are never
     repaired, so the irrelevance category is preserved.

Both are pure-Python and require no extra dependency. The grader already coerces
scalar types, so the measured lift on the current 1.5B checkpoint is small; the
value is robustness for weaker/larger checkpoints whose surface form is noisier.
"""
from __future__ import annotations

import json
import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pipeline.formatting.chat_template import (
    TOOL_CALL_CLOSE_TAG,
    TOOL_CALL_OPEN_TAG,
    _normalise_tools_for_tokenizer,
    extract_tool_calls,
)
from pipeline.formatting.schema_utils import coerce_calls_to_schema

logger = logging.getLogger(__name__)


def canonical_tool_call_text(calls: list[dict]) -> str:
    """Render coerced calls back into canonical <tool_call> blocks for grading."""
    blocks = []
    for call in calls:
        payload = json.dumps(
            {"name": call.get("name", ""), "arguments": call.get("arguments", {})},
            ensure_ascii=False,
        )
        blocks.append(f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}")
    return "\n".join(blocks)


def normalise_prediction(raw_output: str, tools: list[dict]) -> str:
    """
    Structured-output cleanup: type-coerce arguments to the schema and re-render
    a canonical tool-call string. Outputs with no parseable call (genuine
    abstention or plain prose) are returned unchanged, preserving irrelevance.
    """
    calls = extract_tool_calls(raw_output)
    if not calls:
        return raw_output
    coerced = coerce_calls_to_schema(calls, tools)
    return canonical_tool_call_text(coerced)


def looks_like_failed_call(raw_output: str) -> bool:
    """
    True when the model opened a <tool_call> block but produced no parseable
    call — a malformed attempt worth repairing. False when there is no opener at
    all (a genuine abstention, which must NOT be repaired).
    """
    if TOOL_CALL_OPEN_TAG not in raw_output:
        return False
    return len(extract_tool_calls(raw_output)) == 0


def build_repair_prompt(
    query: str, tools: list[dict], broken_output: str, tokenizer: PreTrainedTokenizer
) -> str:
    """Construct a one-shot re-ask prompt that nudges the model to valid JSON."""
    instruction = (
        "Your previous reply was not a valid tool call. Reply with ONLY a single "
        f"{TOOL_CALL_OPEN_TAG} ... {TOOL_CALL_CLOSE_TAG} block containing a JSON "
        "object {\"name\": ..., \"arguments\": ...} that uses one of the available "
        "tools. If no tool applies, say so in plain text with no tool call."
    )
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": broken_output},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=_normalise_tools_for_tokenizer(tools),
        tokenize=False,
        add_generation_prompt=True,
    )


def repair_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query: str,
    tools: list[dict],
    raw_output: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """
    Run one repair pass when `raw_output` is a malformed call attempt; otherwise
    return it unchanged. The returned text is still passed through the grader.
    """
    if not looks_like_failed_call(raw_output):
        return raw_output

    prompt = build_repair_prompt(query, tools, raw_output, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    repaired = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
    # Only accept the repair if it actually parses; else keep the original
    return repaired if extract_tool_calls(repaired) else raw_output
