"""Tests for pipeline/data/relabel.py teacher-relabeling decision logic (step 3)."""
from __future__ import annotations

from pipeline.data.relabel import (
    build_relabeled_answer,
    gold_function_names,
    names_match,
)

TOOLS = [{
    "name": "set_timer",
    "parameters": {"type": "object",
                   "properties": {"minutes": {"type": "integer"}},
                   "required": ["minutes"]},
}]
GOLD = [{"name": "set_timer", "arguments": {"minutes": [10]}}]


def test_names_match_is_order_independent():
    a = [{"name": "f"}, {"name": "g"}]
    b = [{"name": "g"}, {"name": "f"}]
    assert names_match(a, b)


def test_names_match_respects_multiplicity():
    assert not names_match([{"name": "f"}], [{"name": "f"}, {"name": "f"}])
    assert not names_match([{"name": "f"}], [{"name": "g"}])


def test_gold_function_names_filters_non_dicts():
    assert gold_function_names([{"name": "f"}, "junk", {"name": "a"}]) == ["a", "f"]


def test_relabel_adopts_teacher_when_functions_match_and_coerces_types():
    teacher = [{"name": "set_timer", "arguments": {"minutes": "10"}}]  # string arg
    answer = build_relabeled_answer(teacher, GOLD, TOOLS)
    assert answer is not None
    # Adopted teacher answer with the argument coerced to the schema's int type
    assert answer[0]["arguments"]["minutes"] == 10


def test_relabel_skips_on_mismatch_by_default():
    teacher = [{"name": "other_fn", "arguments": {}}]
    assert build_relabeled_answer(teacher, GOLD, TOOLS) is None


def test_relabel_falls_back_to_gold_when_keep_on_mismatch():
    teacher = [{"name": "other_fn", "arguments": {}}]
    answer = build_relabeled_answer(teacher, GOLD, TOOLS, keep_on_mismatch=True)
    assert answer is GOLD


def test_relabel_skips_when_teacher_emits_nothing():
    assert build_relabeled_answer([], GOLD, TOOLS) is None
