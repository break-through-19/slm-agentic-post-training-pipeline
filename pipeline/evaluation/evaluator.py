"""
BFCL evaluation runner.

Iterates over each requested category, generates predictions with the model,
grades them with the BFCL grader, and returns an EvalSummary.

Used for both Stage 0 (baseline) and Stage 1 (post-SFT) evaluations.
The same grading logic is reused by the Stage 2 reward function — keeping
train-time and eval-time reward signals identical.
"""
from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from pipeline.data.bfcl import load_bfcl_category, parse_bfcl_example  # returns list[dict]
from pipeline.evaluation.metrics import (
    CategoryMetrics,
    EvalSummary,
    aggregate_grade_results,
)
from pipeline.formatting.chat_template import format_inference_prompt
from pipeline.reward.bfcl_grader import GradeResult, grade

logger = logging.getLogger(__name__)


def _generate_one_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """Run greedy decoding for a single prompt and return the raw generated text."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    with torch.inference_mode():
        output_token_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt tokens to get only the newly generated tokens
    generated_token_ids = output_token_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=False)


def evaluate_category(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    category: str,
    device: str,
    max_eval_samples: int | None,
    max_new_tokens: int,
    constrained: bool = False,
) -> CategoryMetrics:
    """Evaluate the model on one BFCL category and return per-category metrics."""
    dataset = load_bfcl_category(category, max_samples=max_eval_samples)
    logger.info("Evaluating %d examples in category '%s'%s", len(dataset), category,
                " (constrained decoding)" if constrained else "")

    grade_results: list[GradeResult] = []

    for step, raw_row in enumerate(dataset):
        example = parse_bfcl_example(raw_row, category=category)

        # expected_calls is intentionally empty for irrelevance examples —
        # only skip rows that have no tool definitions at all.
        if not example["tools"]:
            logger.debug("Skipping row %d — no tool definitions present", step)
            continue

        prompt = format_inference_prompt(
            query=example["query"],
            tools=example["tools"],
            tokenizer=tokenizer,
        )
        model_output = _generate_one_response(
            model, tokenizer, prompt, device, max_new_tokens
        )
        # Sprint step 5: repair a malformed call (abstention-preserving) and
        # normalise argument types to the schema before grading.
        if constrained:
            from pipeline.evaluation.constrained import (
                normalise_prediction,
                repair_prediction,
            )

            model_output = repair_prediction(
                model, tokenizer, example["query"], example["tools"],
                model_output, device, max_new_tokens)
            model_output = normalise_prediction(model_output, example["tools"])
        result = grade(model_output, example["expected_calls"])
        grade_results.append(result)

        # Log every example on small runs; every 50 on full-scale runs
        log_every = max(1, min(50, len(dataset) // 5))
        if (step + 1) % log_every == 0:
            running_accuracy = sum(r.correct for r in grade_results) / len(grade_results)
            logger.info(
                "  [%s] %d/%d — running accuracy: %.3f",
                category, step + 1, len(dataset), running_accuracy,
            )

    metrics = aggregate_grade_results(category, grade_results)
    logger.info(
        "Category '%s' done — accuracy=%.4f (%d/%d)",
        category, metrics.accuracy, metrics.correct, metrics.total,
    )
    return metrics


def evaluate_bfcl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    device: str,
) -> EvalSummary:
    """
    Evaluate the model across all configured BFCL categories.

    cfg.evaluation.bfcl_categories — list of categories to run
    cfg.evaluation.max_eval_samples — optional per-category cap (None = full set)
    cfg.evaluation.max_new_tokens   — generation budget per example
    """
    categories: list[str] = list(cfg.evaluation.bfcl_categories)
    max_eval_samples: int | None = cfg.evaluation.get("max_eval_samples", None)
    max_new_tokens: int = cfg.evaluation.get("max_new_tokens", 256)
    constrained: bool = cfg.evaluation.get("constrained_decoding", False)

    model.eval()
    summary = EvalSummary()

    for category in categories:
        try:
            metrics = evaluate_category(
                model=model,
                tokenizer=tokenizer,
                category=category,
                device=device,
                max_eval_samples=max_eval_samples,
                max_new_tokens=max_new_tokens,
                constrained=constrained,
            )
            summary.per_category[category] = metrics
        except Exception as exc:
            logger.warning("Skipping category '%s' — %s: %s", category, type(exc).__name__, exc)

    logger.info("Overall BFCL accuracy: %.4f", summary.overall_accuracy)
    return summary
