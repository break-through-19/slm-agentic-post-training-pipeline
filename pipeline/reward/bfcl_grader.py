"""
BFCL-style verifiable reward / grader.

Pure functions — no model required.  Used by:
  - Stage 0 / Stage 1 evaluation (evaluator.py)
  - Stage 2B GRPO reward function (grpo_trainer.py)
  - Stage 2A DPO preference pair generation (generate_pairs.py)

Grading checks, in order:
  1. At least one <tool_call> block is present (else: no_tool_call)
  2. Number of predicted calls matches expected (else: extra_tool_call)
  3. For each (predicted, expected) pair:
     a. Function name matches exactly (else: wrong_function)
     b. All expected argument keys are present (else: missing_argument)
     c. Argument values match after type coercion (else: wrong_argument_type)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pipeline.formatting.chat_template import extract_tool_calls

logger = logging.getLogger(__name__)

# Failure category constants — used as string tags in GradeResult and metrics
FAILURE_NO_TOOL_CALL = "no_tool_call"
FAILURE_EXTRA_TOOL_CALL = "extra_tool_call"
FAILURE_WRONG_FUNCTION = "wrong_function"
FAILURE_MISSING_ARGUMENT = "missing_argument"
FAILURE_WRONG_ARGUMENT_TYPE = "wrong_argument_type"
FAILURE_MALFORMED_JSON = "malformed_json"


@dataclass
class GradeResult:
    correct: bool
    reward: float                           # 1.0 if correct, 0.0 otherwise
    failure_category: str | None = None     # one of the FAILURE_* constants above
    predicted_calls: list[dict] = field(default_factory=list)
    expected_calls: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _coerce_value(value: Any) -> Any:
    """
    Coerce string representations of scalars to their native Python types.
    Models sometimes emit integers or booleans as quoted strings.
    """
    if not isinstance(value, str):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _load_args(raw: Any) -> dict:
    """Ensure arguments are a dict, parsing from JSON string if necessary."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Per-call grading
# ---------------------------------------------------------------------------


def _grade_arguments(
    predicted_args: dict, expected_args: dict
) -> tuple[bool, str | None]:
    """
    Compare predicted arguments against expected.
    Returns (all_correct, failure_category_or_None).
    """
    for expected_key, expected_value in expected_args.items():
        if expected_key not in predicted_args:
            return False, FAILURE_MISSING_ARGUMENT

        predicted_value = _coerce_value(predicted_args[expected_key])
        expected_value = _coerce_value(expected_value)

        if type(predicted_value) is not type(expected_value):
            return False, FAILURE_WRONG_ARGUMENT_TYPE
        if predicted_value != expected_value:
            return False, FAILURE_WRONG_ARGUMENT_TYPE

    return True, None


def _grade_single_call(
    predicted_call: dict, expected_call: dict
) -> tuple[bool, str | None]:
    """Grade one predicted call against one expected call."""
    predicted_name = predicted_call.get("name", "")
    expected_name = expected_call.get("name", "")

    if predicted_name != expected_name:
        return False, FAILURE_WRONG_FUNCTION

    predicted_args = _load_args(predicted_call.get("arguments", predicted_call.get("args", {})))
    expected_args = _load_args(expected_call.get("arguments", expected_call.get("args", {})))

    return _grade_arguments(predicted_args, expected_args)


# ---------------------------------------------------------------------------
# Public grading entry point
# ---------------------------------------------------------------------------


def grade(model_output: str, expected_calls: list[dict]) -> GradeResult:
    """
    Grade the raw text output of a model against a list of expected tool calls.

    Handles single-call and parallel (multi-call) scenarios.
    Returns a GradeResult with correct=True only when every call matches.
    """
    predicted_calls = extract_tool_calls(model_output)

    if not predicted_calls:
        return GradeResult(
            correct=False,
            reward=0.0,
            failure_category=FAILURE_NO_TOOL_CALL,
            predicted_calls=[],
            expected_calls=expected_calls,
        )

    if len(predicted_calls) != len(expected_calls):
        return GradeResult(
            correct=False,
            reward=0.0,
            failure_category=FAILURE_EXTRA_TOOL_CALL,
            predicted_calls=predicted_calls,
            expected_calls=expected_calls,
        )

    for predicted_call, expected_call in zip(predicted_calls, expected_calls):
        call_correct, failure = _grade_single_call(predicted_call, expected_call)
        if not call_correct:
            return GradeResult(
                correct=False,
                reward=0.0,
                failure_category=failure,
                predicted_calls=predicted_calls,
                expected_calls=expected_calls,
            )

    return GradeResult(
        correct=True,
        reward=1.0,
        failure_category=None,
        predicted_calls=predicted_calls,
        expected_calls=expected_calls,
    )
