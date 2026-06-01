"""
GRPO reward function — adapts the BFCL verifier to TRL's reward-function API.

TRL's GRPOTrainer calls each reward function with:
  - completions: list[str]  — the group of sampled completions for the batch
  - plus every extra dataset column as a same-length keyword list
    (here: expected_calls_json, supplied by pipeline.data.grpo_prompts)

and expects a list[float] of per-completion rewards in return.

We delegate scoring to the exact same `bfcl_grader.grade()` used during
offline evaluation, so the online RL signal and the reported BFCL metric are
the same function — they cannot drift apart.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from pipeline.reward.bfcl_grader import grade

logger = logging.getLogger(__name__)


def build_bfcl_reward_function() -> Callable:
    """
    Return a TRL-compatible reward function closure.

    The returned callable scores each completion with the BFCL verifier
    (reward 1.0 for a fully correct tool call, 0.0 otherwise).
    """

    def bfcl_reward(
        completions: list[str],
        expected_calls_json: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        # GRPO always supplies the ground-truth column; guard anyway so the
        # function degrades to zero reward rather than crashing a long run.
        if expected_calls_json is None:
            logger.warning("Reward fn received no expected_calls_json — returning 0 rewards")
            return [0.0] * len(completions)

        rewards: list[float] = []
        for completion, ground_truth_json in zip(completions, expected_calls_json):
            try:
                expected_calls = json.loads(ground_truth_json) if ground_truth_json else []
            except (json.JSONDecodeError, TypeError):
                expected_calls = []
            rewards.append(grade(completion, expected_calls).reward)
        return rewards

    # TRL uses the function name as the metric label in its logs
    bfcl_reward.__name__ = "bfcl_reward"
    return bfcl_reward
