"""
Prompt formatting for Qwen2.5-Instruct tool-calling.

All stages (SFT, DPO, GRPO, evaluation) share the same formatting functions so
that train-time and inference-time prompts are byte-for-byte identical.

Key design: the tokenizer's native tools= parameter is used in apply_chat_template
rather than a manually written system prompt. This guarantees the model sees the
exact prompt format it was trained on, reliably triggering <tool_call> output.

Tool-call format used by Qwen2.5:
    <tool_call>
    {"name": "fn_name", "arguments": {...}}
    </tool_call>
"""
from __future__ import annotations

import json
import logging
import re

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN_TAG = "<tool_call>"
TOOL_CALL_CLOSE_TAG = "</tool_call>"

# Assistant target for irrelevance (abstention) SFT examples: a plain-text
# refusal with NO <tool_call> block. Training on these teaches the model to
# abstain when none of the available tools fit the request, which is what the
# BFCL `irrelevance` category rewards.
NO_TOOL_RESPONSE = (
    "None of the available tools can fulfil this request, so no function should be called."
)

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN_TAG)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE_TAG)}",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_tools_for_tokenizer(tools: list[dict]) -> list[dict]:
    """
    Ensure every tool dict is in the OpenAI function-schema format that
    Qwen2.5's apply_chat_template(tools=...) expects:
      {"type": "function", "function": {name, description, parameters}}

    Tools from xLAM arrive with just name/description/parameters at the top
    level; this wraps them so the tokenizer recognises them correctly.
    """
    normalised = []
    for tool in tools:
        if "type" in tool and "function" in tool:
            normalised.append(tool)
        else:
            normalised.append({"type": "function", "function": tool})
    return normalised


def _render_tool_call_response(expected_calls: list[dict]) -> str:
    """Render a list of expected calls into the assistant response string."""
    rendered_calls = []
    for call in expected_calls:
        name = call.get("name", "")
        # BFCL possible-answer format stores argument values as lists;
        # take the first acceptable value for each argument when building
        # the training target so the model learns a single concrete output.
        raw_arguments = call.get("arguments", call.get("args", {}))
        concrete_arguments = {
            key: (vals[0] if isinstance(vals, list) and vals else vals)
            for key, vals in raw_arguments.items()
        }
        payload = json.dumps(
            {"name": name, "arguments": concrete_arguments}, ensure_ascii=False
        )
        rendered_calls.append(f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}")
    return "\n".join(rendered_calls)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_sft_example(
    query: str,
    tools: list[dict],
    expected_calls: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 2048,
) -> dict:
    """
    Tokenize one xLAM example for SFT.

    Uses the tokenizer's native tools= parameter so the system prompt matches
    exactly what the model was pre-trained on.  Returns a dict with
    input_ids, attention_mask, and labels where prompt tokens are masked to
    -100 so cross-entropy is computed on the assistant response only.

    Returns empty lists when the example is too long or produces no
    learnable tokens after masking.
    """
    normalised_tools = _normalise_tools_for_tokenizer(tools)
    # Empty expected_calls = an irrelevance example: the target is a plain-text
    # refusal (no tool call), teaching the model to abstain.
    if expected_calls:
        assistant_response = _render_tool_call_response(expected_calls)
    else:
        assistant_response = NO_TOOL_RESPONSE

    full_messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_response},
    ]
    prompt_only_messages = [{"role": "user", "content": query}]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tools=normalised_tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages,
        tools=normalised_tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_encoding = tokenizer(
        full_text, truncation=True, max_length=max_seq_len, return_tensors=None
    )
    prompt_encoding = tokenizer(
        prompt_text, truncation=True, max_length=max_seq_len, return_tensors=None
    )

    input_ids = full_encoding["input_ids"]
    prompt_length = len(prompt_encoding["input_ids"])

    # Mask prompt tokens so loss is computed on the assistant response only
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    # Discard examples where the response was truncated away entirely
    has_learnable_tokens = any(label != -100 for label in labels)
    if not has_learnable_tokens:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_encoding["attention_mask"],
    }


def format_inference_prompt(
    query: str,
    tools: list[dict],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Build a prompt string for generation (user turn only; no assistant turn).

    Uses the tokenizer's native tools= parameter so the system prompt the
    model sees at inference time is identical to the one used during SFT.
    Used by the evaluator and the Stage 2 preference pair generator.
    """
    normalised_tools = _normalise_tools_for_tokenizer(tools)
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        tools=normalised_tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_tool_calls(model_output: str) -> list[dict]:
    """
    Parse all <tool_call>...</tool_call> blocks from a model's raw output text.
    Returns a list of dicts with 'name' and 'arguments' keys.
    Silently drops blocks that contain invalid JSON.
    """
    parsed_calls = []
    for raw_json in _TOOL_CALL_RE.findall(model_output):
        try:
            parsed_calls.append(json.loads(raw_json))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed tool_call block: %.80s", raw_json)
    return parsed_calls
