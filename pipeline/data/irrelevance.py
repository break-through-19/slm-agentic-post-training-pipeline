"""
Irrelevance (abstention) example synthesis — Phase 1.

xLAM contains only positive examples (every query has a correct tool call), so
SFT on it teaches the model to ALWAYS emit a tool call and destroys the
ability to abstain. On BFCL this collapses the `irrelevance` category from
~0.72 (base model) to ~0.02.

To counteract this we synthesise irrelevance examples from xLAM itself, with no
dependence on the BFCL test set: take a query but pair it with the tools from a
DIFFERENT example. Because none of those swapped tools match the query, the
correct behaviour is to call no function at all. We mark this by setting the
answer to an empty list ("[]"), which every downstream consumer already
interprets as "abstain":
  - SFT formatting renders the NO_TOOL_RESPONSE refusal as the target.
  - The BFCL grader rewards producing no tool call.
  - DPO pair generation naturally yields (abstain = chosen, call = rejected).

Everything is kept as JSON strings to avoid Arrow schema inference over
heterogeneous tool definitions.
"""
from __future__ import annotations

import json
import logging
import random

from datasets import Dataset

logger = logging.getLogger(__name__)


def _as_json_string(value) -> str:
    """Return value unchanged if it is already a JSON string, else encode it."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def inject_irrelevance(dataset: Dataset, fraction: float, seed: int = 42) -> Dataset:
    """
    Convert a fraction of xLAM rows into synthetic irrelevance examples.

    For each selected row, its tools are replaced with another row's tools
    (guaranteeing a mismatch) and its answer is set to "[]" (abstain). The
    returned Dataset has exactly three columns — query, tools, answers — all as
    JSON strings, matching what the xLAM / GRPO / pair-generation loaders read.

    fraction <= 0 returns the dataset unchanged.
    """
    if fraction <= 0:
        return dataset

    num_rows = len(dataset)
    if num_rows < 2:
        return dataset

    queries = list(dataset["query"])
    tools = [_as_json_string(t) for t in dataset["tools"]]
    answers = [_as_json_string(a) for a in dataset["answers"]]

    num_irrelevance = min(int(round(num_rows * fraction)), num_rows)
    rng = random.Random(seed)
    irrelevance_indices = set(rng.sample(range(num_rows), num_irrelevance))

    out_queries: list[str] = []
    out_tools: list[str] = []
    out_answers: list[str] = []

    for index in range(num_rows):
        if index in irrelevance_indices:
            # Pick a different row's tools so the query cannot be answered
            donor_index = (index + rng.randint(1, num_rows - 1)) % num_rows
            out_queries.append(queries[index])
            out_tools.append(tools[donor_index])   # mismatched tools
            out_answers.append("[]")                # correct response: abstain
        else:
            out_queries.append(queries[index])
            out_tools.append(tools[index])
            out_answers.append(answers[index])

    logger.info(
        "Injected %d synthetic irrelevance examples (%.0f%%) into %d rows",
        num_irrelevance, fraction * 100, num_rows,
    )
    return Dataset.from_dict(
        {"query": out_queries, "tools": out_tools, "answers": out_answers}
    )
