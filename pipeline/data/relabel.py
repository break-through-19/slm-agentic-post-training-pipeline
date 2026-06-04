"""
Teacher-relabeling core logic (sprint step 3).

Problem this solves
--------------------
xLAM's gold answers follow xLAM's argument conventions, which differ from the
ones BFCL grades against. Training the 1.5B student directly on xLAM therefore
teaches a slightly wrong "dialect": across iterations the dominant residual
failure was `wrong_argument_type`, not function selection.

Approach
--------
Distil from a stronger teacher (default Qwen2.5-7B-Instruct). For each xLAM
query we let the teacher answer the same function-calling task, then keep its
answer only when it selects the SAME function(s) as the xLAM gold — so the task
is unchanged and only the argument surface form is upgraded. Teacher arguments
are coerced to the declared schema types as a final safety net.

This module holds the pure, testable decision logic. The model-running CLI that
uses it lives in scripts/relabel_xlam.py.
"""
from __future__ import annotations

from pipeline.formatting.schema_utils import coerce_calls_to_schema, tool_name


def gold_function_names(calls: list[dict]) -> list[str]:
    """Sorted multiset of function names in a list of calls (for set comparison)."""
    return sorted(call.get("name", "") for call in calls if isinstance(call, dict))


def names_match(predicted_calls: list[dict], gold_calls: list[dict]) -> bool:
    """
    True when the teacher selected exactly the same functions as the gold answer
    (same names, same multiplicity, order-independent). This keeps relabeling
    faithful: we only adopt the teacher's *arguments*, never a different task.
    """
    return gold_function_names(predicted_calls) == gold_function_names(gold_calls)


def build_relabeled_answer(
    predicted_calls: list[dict],
    gold_calls: list[dict],
    tools: list[dict],
    keep_on_mismatch: bool = False,
) -> list[dict] | None:
    """
    Decide the answer to store for one example.

    Returns:
      * the teacher's calls (type-coerced to the schema) when they select the
        same functions as gold — the relabeled, BFCL-style answer;
      * the original gold calls when the teacher disagrees and
        keep_on_mismatch is True (so the example is not lost);
      * None when the teacher disagrees and keep_on_mismatch is False (skip).
    """
    valid_predicted = [c for c in predicted_calls if isinstance(c, dict) and c.get("name")]

    if valid_predicted and names_match(valid_predicted, gold_calls):
        return coerce_calls_to_schema(valid_predicted, tools)

    if keep_on_mismatch:
        return gold_calls
    return None


def summarise_relabeling(total: int, relabeled: int, fell_back: int, skipped: int) -> str:
    """Human-readable one-line summary of a relabeling run."""
    pct = (100 * relabeled / total) if total else 0.0
    return (
        f"relabeled {relabeled}/{total} ({pct:.0f}%) with teacher answers, "
        f"{fell_back} kept original on mismatch, {skipped} skipped"
    )
