"""
Preference-pair dataset loader for Stage 2A (DPO).

Loads the JSONL file written by `pipeline.generation.pair_generator` and exposes
it in the (prompt, chosen, rejected) "standard" format that TRL's DPOTrainer
expects. TRL tokenises these text columns internally, so — unlike the SFT
dataset — no tokenizer is needed here.

Registers the "xlam_dpo" dataset in the global registry on module import.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig

from pipeline.data.registry import register

logger = logging.getLogger(__name__)


def _load_preference_pairs(cfg: DictConfig) -> Dataset:
    """
    Read the generated preference pairs JSONL into a HuggingFace Dataset.

    The pairs file path comes from cfg.training.pairs_path. Each line is a JSON
    object with at least prompt / chosen / rejected keys.
    """
    pairs_path = Path(cfg.training.pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"\n\n{'='*65}\n"
            f"  Preference pairs not found at: {pairs_path}\n\n"
            f"  Generate them first with:\n"
            f"    python scripts/run_pipeline.py generate-pairs --device <dev>\n"
            f"{'='*65}\n"
        )

    rows: list[dict] = []
    with open(pairs_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    max_samples = cfg.training.get("max_samples", None)
    if max_samples is not None:
        rows = rows[:max_samples]

    logger.info("Loaded %d preference pairs from %s", len(rows), pairs_path)
    # All columns are plain strings/floats — no Arrow schema concerns
    return Dataset.from_list(rows)


def _format_preference_example(example: dict) -> dict:
    """Select the three columns TRL's DPOTrainer consumes (standard format)."""
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


register("xlam_dpo", _load_preference_pairs, _format_preference_example)
