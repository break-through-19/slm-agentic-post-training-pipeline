"""Tests for pipeline/data/preference.py (DPO preference-pair dataset)."""
from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf


def test_format_preference_example_selects_three_columns():
    from pipeline.data.preference import _format_preference_example

    raw = {
        "prompt": "P",
        "chosen": "C",
        "rejected": "R",
        "query": "Q",            # extra metadata that must be dropped
        "chosen_reward": 1.0,
    }
    formatted = _format_preference_example(raw)
    assert formatted == {"prompt": "P", "chosen": "C", "rejected": "R"}


def test_load_preference_pairs_reads_jsonl(tmp_path):
    from pipeline.data.preference import _load_preference_pairs

    pairs_file = tmp_path / "pairs.jsonl"
    rows = [
        {"prompt": "P1", "chosen": "C1", "rejected": "R1"},
        {"prompt": "P2", "chosen": "C2", "rejected": "R2"},
    ]
    pairs_file.write_text("\n".join(json.dumps(r) for r in rows))

    cfg = OmegaConf.create({"training": {"pairs_path": str(pairs_file)}})
    dataset = _load_preference_pairs(cfg)
    assert len(dataset) == 2
    assert dataset[0]["prompt"] == "P1"
    assert dataset[1]["rejected"] == "R2"


def test_load_preference_pairs_respects_max_samples(tmp_path):
    from pipeline.data.preference import _load_preference_pairs

    pairs_file = tmp_path / "pairs.jsonl"
    rows = [{"prompt": f"P{i}", "chosen": "C", "rejected": "R"} for i in range(10)]
    pairs_file.write_text("\n".join(json.dumps(r) for r in rows))

    cfg = OmegaConf.create({"training": {"pairs_path": str(pairs_file), "max_samples": 3}})
    dataset = _load_preference_pairs(cfg)
    assert len(dataset) == 3


def test_load_preference_pairs_missing_file_raises_actionable_error(tmp_path):
    from pipeline.data.preference import _load_preference_pairs

    cfg = OmegaConf.create({"training": {"pairs_path": str(tmp_path / "missing.jsonl")}})
    with pytest.raises(FileNotFoundError, match="generate-pairs"):
        _load_preference_pairs(cfg)
