"""
BFCL-style verifiable reward / grader.

Pure functions — no model required.  Used by:
  - Stage 0 / Stage 1 evaluation (evaluator.py)
  - Stage 2B GRPO reward function (grpo_trainer.py)
  - Stage 2A DPO preference pair generation (generate_pairs.py)

Grading checks, in order:
  0. Irrelevance: if expected_calls is empty, correct iff no tool call produced
  1. At least one <tool_call> block is present (else: no_tool_call)
  2. Number of predicted calls matches expected (else: extra_tool_call)
  3. Each expected call is matched to a predicted call (order-independent, so
     parallel calls may appear in any order). A call matches when:
     a. Function name matches exactly (else: wrong_function)
     b. All REQUIRED argument keys are present (else: missing_argument).
        Optional arguments — those whose acceptable list contains "" — may be
        omitted without penalty, matching official BFCL scoring.
     c. Argument values match after type coercion (else: wrong_argument_type).
        When the expected value is a list, any element in that list is accepted
        (BFCL possible-answer format).
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


def _is_optional_argument(expected_value) -> bool:
    """
    True if this argument may be omitted, per the BFCL possible-answer format.

    BFCL signals an optional argument by including an empty string "" among its
    acceptable values (e.g. "unit": ["units", ""]). When "" is acceptable, the
    model is allowed to leave the argument out entirely.
    """
    return isinstance(expected_value, list) and "" in expected_value


def _grade_arguments(
    predicted_args: dict, expected_args: dict
) -> tuple[bool, str | None]:
    """
    Compare predicted arguments against expected.
    Returns (all_correct, failure_category_or_None).

    When an expected argument value is a list (BFCL possible-answer format),
    the predicted value is accepted if it matches any element in that list.
    When it is a scalar, exact equality after type coercion is required.

    Optional arguments (those whose acceptable list contains "") may be omitted
    from the prediction without penalty — this matches official BFCL scoring,
    where "" denotes "this argument need not be supplied".
    """
    for expected_key, expected_value in expected_args.items():
        if expected_key not in predicted_args:
            # Omitting an optional argument is acceptable; only required
            # arguments count as missing.
            if _is_optional_argument(expected_value):
                continue
            return False, FAILURE_MISSING_ARGUMENT

        predicted_value = _coerce_value(predicted_args[expected_key])

        if isinstance(expected_value, list):
            # BFCL possible-answer: accept any value in the acceptable list
            acceptable_values = [_coerce_value(v) for v in expected_value]
            if predicted_value not in acceptable_values:
                return False, FAILURE_WRONG_ARGUMENT_TYPE
        else:
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
    # Defensive: extract_tool_calls should only return dicts, but never let a
    # malformed prediction (e.g. a bare JSON string/number) crash a long run.
    if not isinstance(predicted_call, dict):
        return False, FAILURE_MALFORMED_JSON

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

    Handles single-call, parallel (multi-call), and irrelevance scenarios.
    Returns a GradeResult with correct=True only when every call matches.

    Irrelevance: when expected_calls is empty the correct response is to
    produce no tool call.  Any tool call produced is marked as extra_tool_call.
    """
    predicted_calls = extract_tool_calls(model_output)

    # Irrelevance: model should abstain from calling any tool
    if not expected_calls:
        if not predicted_calls:
            return GradeResult(
                correct=True,
                reward=1.0,
                failure_category=None,
                predicted_calls=[],
                expected_calls=[],
            )
        return GradeResult(
            correct=False,
            reward=0.0,
            failure_category=FAILURE_EXTRA_TOOL_CALL,
            predicted_calls=predicted_calls,
            expected_calls=[],
        )

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

    # Order-independent matching: BFCL accepts parallel calls in any order, so
    # greedily pair each expected call with an as-yet-unmatched predicted call
    # that fully satisfies it. (For the single-call case this reduces to the
    # obvious one-to-one check.)
    unmatched_predicted_indices = list(range(len(predicted_calls)))
    representative_failure: str | None = None

    for expected_call in expected_calls:
        matched_index = None
        for predicted_index in unmatched_predicted_indices:
            call_correct, failure = _grade_single_call(
                predicted_calls[predicted_index], expected_call
            )
            if call_correct:
                matched_index = predicted_index
                break
            # Remember the first concrete failure to report if nothing matches
            if representative_failure is None:
                representative_failure = failure

        if matched_index is None:
            return GradeResult(
                correct=False,
                reward=0.0,
                failure_category=representative_failure or FAILURE_WRONG_FUNCTION,
                predicted_calls=predicted_calls,
                expected_calls=expected_calls,
            )
        unmatched_predicted_indices.remove(matched_index)

    return GradeResult(
        correct=True,
        reward=1.0,
        failure_category=None,
        predicted_calls=predicted_calls,
        expected_calls=expected_calls,
    )


# ---------------------------------------------------------------------------
# Shaped (partial-credit) reward — Phase 2, for GRPO training only
# ---------------------------------------------------------------------------
#
# grade() above returns a binary 1/0 used for EVALUATION (the BFCL metric). For
# GRPO *training* a binary reward is too sparse: most sampled groups come back
# all-correct or all-wrong (zero within-group variance => zero advantage => no
# gradient). score() returns a continuous reward in [0, 1] that gives partial
# credit for getting the format / function name / individual arguments right,
# so groups have variance and the policy receives a learning signal.
#
# By construction score() == 1.0 exactly when grade() is correct, so the shaped
# reward never disagrees with the metric at the extremes — it only fills in the
# middle.

SCORE_FORMAT_CREDIT = 0.2      # produced a parseable tool-call object
SCORE_NAME_CREDIT = 0.3        # correct function name
SCORE_ARGUMENT_CREDIT = 0.5    # split across the expected arguments


def _score_single_call(predicted_call, expected_call: dict) -> float:
    """Continuous [0, 1] partial-credit score for one predicted vs expected call."""
    if not isinstance(predicted_call, dict):
        return 0.0

    # Format credit: a well-formed call object was produced
    call_score = SCORE_FORMAT_CREDIT

    if predicted_call.get("name", "") != expected_call.get("name", ""):
        return call_score  # wrong function — format credit only

    call_score += SCORE_NAME_CREDIT  # correct function name

    expected_args = _load_args(expected_call.get("arguments", expected_call.get("args", {})))
    if not expected_args:
        return call_score + SCORE_ARGUMENT_CREDIT  # nothing to get wrong

    predicted_args = _load_args(predicted_call.get("arguments", predicted_call.get("args", {})))
    correct_args = 0
    for expected_key, expected_value in expected_args.items():
        if expected_key not in predicted_args:
            if _is_optional_argument(expected_value):
                correct_args += 1  # optional arg may be omitted
            continue
        predicted_value = _coerce_value(predicted_args[expected_key])
        if isinstance(expected_value, list):
            if predicted_value in [_coerce_value(v) for v in expected_value]:
                correct_args += 1
        else:
            coerced_expected = _coerce_value(expected_value)
            if type(predicted_value) is type(coerced_expected) and predicted_value == coerced_expected:
                correct_args += 1

    call_score += SCORE_ARGUMENT_CREDIT * (correct_args / len(expected_args))
    return call_score


def score(model_output: str, expected_calls: list[dict]) -> float:
    """
    Continuous partial-credit reward in [0, 1] for GRPO training.

    - Irrelevance (expected_calls empty): binary — 1.0 for abstaining, else 0.0.
    - Otherwise: greedily match each expected call to its best predicted call,
      sum the per-call partial scores, and normalise by max(#expected,
      #predicted) so that extra or missing calls are penalised.
    """
    predicted_calls = extract_tool_calls(model_output)

    # Irrelevance stays binary: abstention is inherently all-or-nothing
    if not expected_calls:
        return 1.0 if not predicted_calls else 0.0

    if not predicted_calls:
        return 0.0

    remaining = list(predicted_calls)
    total_score = 0.0
    for expected_call in expected_calls:
        if not remaining:
            break
        best_index, best_value = 0, -1.0
        for index, candidate in enumerate(remaining):
            candidate_score = _score_single_call(candidate, expected_call)
            if candidate_score > best_value:
                best_index, best_value = index, candidate_score
        total_score += best_value
        remaining.pop(best_index)

    # Normalise by the larger count so extra predicted calls dilute the reward
    denominator = max(len(expected_calls), len(predicted_calls))
    return total_score / denominator
