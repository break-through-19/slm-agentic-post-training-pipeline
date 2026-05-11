"""
Dataset registry: maps dataset names to (load_fn, format_fn) pairs.

Datasets self-register by importing their module (e.g. `import pipeline.data.xlam`).
Training scripts only ever call get_dataset() — they never reference loaders directly.
This makes swapping or extending datasets transparent to the training code.
"""
from __future__ import annotations

from typing import Callable

from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

_REGISTRY: dict[str, tuple[Callable, Callable]] = {}


def register(name: str, load_fn: Callable, format_fn: Callable) -> None:
    """Register a dataset under a given name."""
    _REGISTRY[name] = (load_fn, format_fn)


def get_dataset(
    name: str,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> Dataset:
    """
    Load and format a registered dataset.

    format_fn receives (example, tokenizer) when a tokenizer is provided,
    or just (example,) when it is not — allowing raw datasets to be returned
    for inspection without tokenization.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{name}' is not registered. "
            f"Did you import its module? Available datasets: {available}"
        )

    load_fn, format_fn = _REGISTRY[name]
    raw_dataset = load_fn(cfg)

    if tokenizer is not None:
        formatted = raw_dataset.map(
            lambda example: format_fn(example, tokenizer),
            remove_columns=raw_dataset.column_names,
            desc=f"Formatting '{name}'",
        )
    else:
        formatted = raw_dataset.map(
            format_fn,
            remove_columns=raw_dataset.column_names,
            desc=f"Formatting '{name}'",
        )

    return formatted
