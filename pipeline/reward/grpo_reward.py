"""
GRPO reward function — adapts the BFCL verifier to TRL's reward-function API.

TRL's GRPOTrainer calls each reward function with:
  - completions: list[str]  — the group of sampled completions for the batch
  - plus every extra dataset column as a same-length keyword list
    (here: expected_calls_json, supplied by pipeline.data.grpo_prompts)

and expects a list[float] of per-completion rewards in return.

Two reward variants (both from bfcl_grader, so the training signal stays tied
to the evaluation metric):

  - shaped=True  (default, Phase 2): bfcl_grader.score() — continuous partial
    credit in [0, 1] for format / function name / individual arguments. This
    gives within-group reward variance so GRPO actually receives a gradient.
    A binary reward leaves ~88% of groups with zero variance (zero advantage).
  - shaped=False (ablation): bfcl_grader.grade().reward — the binary 1/0 metric.

By construction score() == 1.0 exactly when grade() is correct, so the shaped
reward agrees with the binary metric at the extremes and only fills the middle.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from pipeline.reward.bfcl_grader import grade, score

logger = logging.getLogger(__name__)


def build_bfcl_reward_function(shaped: bool = True) -> Callable:
    """
    Return a TRL-compatible reward function closure.

    shaped=True  -> continuous partial-credit reward (recommended for GRPO).
    shaped=False -> binary reward identical to the BFCL evaluation metric.
    """
    score_fn = score if shaped else (lambda output, expected: grade(output, expected).reward)

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
            rewards.append(score_fn(completion, expected_calls))
        return rewards

    # TRL uses the function name as the metric label in its logs
    bfcl_reward.__name__ = "bfcl_reward"
    return bfcl_reward
