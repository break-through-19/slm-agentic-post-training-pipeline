"""
Prompt formatting for Qwen2.5-Instruct tool-calling.

All stages (SFT, DPO, GRPO, evaluation) share the same formatting functions so
that train-time and inference-time prompts are byte-for-byte identical.

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

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN_TAG)}\s*(.*?)\s*{re.escape(TOOL_CALL_CLOSE_TAG)}",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_system_prompt_with_tools(tools: list[dict]) -> str:
    """
    Render a list of tool dicts into the Qwen2.5 system-prompt format.

    Each tool is wrapped in the OpenAI function-call schema convention:
      {"type": "function", "function": {name, description, parameters}}
    If a tool already has a "function" key it is kept as-is.
    """
    tool_json_lines = []
    for tool in tools:
        if "function" in tool:
            tool_json_lines.append(json.dumps(tool, ensure_ascii=False))
        else:
            tool_json_lines.append(
                json.dumps({"type": "function", "function": tool}, ensure_ascii=False)
            )

    tools_block = "\n".join(tool_json_lines)
    return (
        "You are a helpful assistant with access to the following tools.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "Function signatures are provided within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_block}\n</tools>"
    )


def _render_tool_call_response(expected_calls: list[dict]) -> str:
    """Render a list of expected calls into the assistant response string."""
    rendered_calls = []
    for call in expected_calls:
        name = call.get("name", "")
        arguments = call.get("arguments", call.get("args", {}))
        payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
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

    Returns a dict with input_ids, attention_mask, and labels where the
    prompt tokens are masked to -100 so cross-entropy is only computed on
    the assistant tool-call response.

    Returns empty lists for input_ids/labels/attention_mask when the example
    is too long or the assistant turn would produce no learnable tokens.
    """
    system_prompt = _build_system_prompt_with_tools(tools)
    assistant_response = _render_tool_call_response(expected_calls)

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_response},
    ]
    prompt_only_messages = full_messages[:2]

    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages, tokenize=False, add_generation_prompt=True
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
    Build a prompt string for generation (system + user turns only).
    Used by the evaluator and the Stage 2 preference pair generator.
    """
    system_prompt = _build_system_prompt_with_tools(tools)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
