"""
Relabeled-xLAM dataset for Stage 1 SFT (sprint step 3).

Loads the JSONL written by scripts/relabel_xlam.py — BFCL-style answers distilled
from a stronger teacher — and feeds it through the exact same SFT formatting as
plain xLAM, so swapping datasets is a one-line config change:

    training:
      dataset: "xlam_relabeled"
      relabeled_path: "outputs/relabel/xlam_relabeled.jsonl"

Each JSONL record has the same schema as a raw xLAM row, so the SFT formatter is
reused verbatim:
    {"query": str, "tools": json-str|list, "answers": json-str|list}

Phase 1 irrelevance injection still applies (controlled by training.irrelevance_fraction),
keeping abstention coverage identical to the xLAM path.

Registers the "xlam_relabeled" dataset in the global registry on module import.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig

from pipeline.data.registry import register
from pipeline.data.xlam import _format_xlam_for_sft  # reuse the SFT formatter

logger = logging.getLogger(__name__)


def _load_relabeled(cfg: DictConfig) -> Dataset:
    path = cfg.training.get("relabeled_path", "outputs/relabel/xlam_relabeled.jsonl")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n\n{'='*65}\n"
            f"  Relabeled dataset not found at: {path}\n\n"
            f"  Produce it first with the teacher-relabeling script:\n"
            f"    python scripts/relabel_xlam.py --device cuda \\\n"
            f"        --teacher Qwen/Qwen2.5-7B-Instruct \\\n"
            f"        --num-samples 20000 --output {path}\n"
            f"{'='*65}\n"
        )

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d relabeled examples from %s", len(rows), path)
    dataset = Dataset.from_list(rows)

    max_samples = cfg.training.get("max_samples", None)
    seed = cfg.data.get("seed", 42)
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    # Phase 1: keep abstention coverage identical to the xLAM path
    irrelevance_fraction = cfg.training.get("irrelevance_fraction", 0.0)
    if irrelevance_fraction > 0:
        from pipeline.data.irrelevance import inject_irrelevance

        dataset = inject_irrelevance(dataset, irrelevance_fraction, seed=seed)

    return dataset


register("xlam_relabeled", _load_relabeled, _format_xlam_for_sft)
