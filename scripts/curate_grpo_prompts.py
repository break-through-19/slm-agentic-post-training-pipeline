#!/usr/bin/env python3
"""
Curate "informative" GRPO prompts (sprint step 4).

Why
---
Even after the Phase 2 shaped reward, ~85% of GRPO groups had zero within-group
reward variance: the SFT model already solves most xLAM prompts, so all G
rollouts score the same and the advantage (and gradient) is zero. The shaped
reward fixed the *measurement*; this fixes the *data*.

What it does
------------
For each candidate prompt, sample G rollouts from the SFT model, grade them, and
keep the prompt only when the rollouts DISAGREE (some pass, some fail) — i.e.
0 < fraction_correct < 1. Those are exactly the prompts that carry a non-zero
advantage and therefore actually teach the policy. The kept prompts are written
in the raw xLAM schema so GRPO consumes them via training.curated_prompts_path.

Examples
--------
    python scripts/curate_grpo_prompts.py --device cuda \
        --sft-checkpoint outputs/sft/checkpoint-final \
        --num-candidates 8000 --rollouts 8 \
        --output outputs/grpo/curated_prompts.jsonl

    # then train GRPO on the curated set:
    python scripts/run_pipeline.py grpo --device cuda \
        --curated-prompts outputs/grpo/curated_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent.parent / ".env")

from datasets import load_dataset
from omegaconf import OmegaConf

from pipeline.data.irrelevance import inject_irrelevance
from pipeline.data.xlam import _parse_json_field, _resolve_hf_token
from pipeline.formatting.chat_template import format_inference_prompt
from pipeline.generation.pair_generator import _sample_completions
from pipeline.model.loader import resolve_device
from pipeline.reward.bfcl_grader import grade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate informative GRPO prompts.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--sft-checkpoint", default="outputs/sft/checkpoint-final")
    parser.add_argument("--from-base", action="store_true",
                        help="Sample from the base model when no SFT checkpoint exists.")
    parser.add_argument("--num-candidates", type=int, default=8000,
                        help="xLAM prompts to screen.")
    parser.add_argument("--rollouts", type=int, default=8,
                        help="Rollouts sampled per candidate when measuring variance.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--irrelevance-fraction", type=float, default=0.25,
                        help="Blend in abstention candidates (kept only if also informative).")
    parser.add_argument("--max-keep", type=int, default=None,
                        help="Optional cap on the number of curated prompts written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/grpo/curated_prompts.jsonl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.num_candidates = min(args.num_candidates, 16)
        args.rollouts = 4
        args.max_new_tokens = 128

    cfg = OmegaConf.load(args.config)
    if args.device != "auto":
        cfg.model.device = args.device
    device = resolve_device(cfg.model.device)

    # Load the SFT policy (or base fallback)
    from pipeline.model.loader import load_model, load_model_from_checkpoint

    ckpt = Path(args.sft_checkpoint)
    if ckpt.exists():
        logger.info("Loading SFT policy from %s", ckpt)
        model, tokenizer = load_model_from_checkpoint(cfg, ckpt, is_trainable=False)
    elif args.from_base or args.smoke:
        logger.warning("No SFT checkpoint at %s — sampling from the base model.", ckpt)
        model, tokenizer = load_model(cfg, apply_lora=False)
    else:
        raise FileNotFoundError(
            f"SFT checkpoint not found at {ckpt}. Run sft first, pass "
            f"--sft-checkpoint PATH, or use --from-base.")
    model.eval()

    # Candidate prompts: xLAM + injected abstention cases
    dataset = load_dataset(
        "Salesforce/xlam-function-calling-60k", split="train", token=_resolve_hf_token())
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_candidates, len(dataset))))
    if args.irrelevance_fraction > 0:
        dataset = inject_irrelevance(dataset, args.irrelevance_fraction, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    informative = saturated_pass = saturated_fail = no_tools = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for index, row in enumerate(dataset):
            tools = _parse_json_field(row["tools"])
            expected_calls = _parse_json_field(row["answers"])
            if not tools:
                no_tools += 1
                continue

            prompt = format_inference_prompt(row["query"], tools, tokenizer)
            completions = _sample_completions(
                model, tokenizer, prompt, device,
                num_samples=args.rollouts, temperature=args.temperature,
                top_p=args.top_p, max_new_tokens=args.max_new_tokens)
            num_correct = sum(grade(c, expected_calls).correct for c in completions)
            fraction = num_correct / len(completions)

            if 0 < num_correct < len(completions):
                informative += 1
                out_fh.write(json.dumps({
                    "query": row["query"],
                    "tools": json.dumps(tools, ensure_ascii=False),
                    "answers": json.dumps(expected_calls, ensure_ascii=False),
                }, ensure_ascii=False) + "\n")
            elif fraction == 1.0:
                saturated_pass += 1
            else:
                saturated_fail += 1

            if (index + 1) % 100 == 0:
                logger.info(
                    "Screened %d/%d — informative=%d  saturated(all-pass)=%d  "
                    "saturated(all-fail)=%d", index + 1, len(dataset),
                    informative, saturated_pass, saturated_fail)
            if args.max_keep is not None and informative >= args.max_keep:
                logger.info("Reached --max-keep=%d; stopping early.", args.max_keep)
                break

    total_screened = informative + saturated_pass + saturated_fail
    keep_pct = 100 * informative / max(1, total_screened)
    logger.info(
        "Curation complete — kept %d informative prompts (%.0f%% of %d screened); "
        "dropped %d all-pass + %d all-fail (zero-variance). %d had no tools.",
        informative, keep_pct, total_screened, saturated_pass, saturated_fail, no_tools)
    logger.info("Wrote curated GRPO prompts to %s", output_path)


if __name__ == "__main__":
    main()
