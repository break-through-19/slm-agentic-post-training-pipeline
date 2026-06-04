#!/usr/bin/env python3
"""
Teacher-relabel xLAM into BFCL-style SFT data (sprint step 3).

Runs a stronger teacher model (default Qwen2.5-1.5B's bigger sibling,
Qwen2.5-7B-Instruct) over xLAM queries, keeps the teacher's answer when it
selects the same function(s) as the xLAM gold, coerces its arguments to the
declared schema types, and writes a JSONL the SFT stage can train on directly:

    {"query": ..., "tools": <json>, "answers": <json>}

Train on it by pointing SFT at the relabeled dataset:

    python scripts/run_pipeline.py sft --dataset xlam_relabeled \
        --relabeled-path outputs/relabel/xlam_relabeled.jsonl

Examples
--------
    # Full relabel on a GPU (a few hours for 20k examples)
    python scripts/relabel_xlam.py --device cuda \
        --teacher Qwen/Qwen2.5-7B-Instruct \
        --num-samples 20000 --output outputs/relabel/xlam_relabeled.jsonl

    # Tiny smoke check (no big download if the teacher is the 1.5B itself)
    python scripts/relabel_xlam.py --device mps \
        --teacher Qwen/Qwen2.5-1.5B-Instruct --num-samples 8 --smoke
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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.data.relabel import build_relabeled_answer, summarise_relabeling
from pipeline.data.xlam import _parse_json_field, _resolve_hf_token
from pipeline.formatting.chat_template import extract_tool_calls, format_inference_prompt
from pipeline.model.loader import resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("relabel")


def _generate(model, tokenizer, prompt, device, max_new_tokens) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-relabel xLAM into BFCL-style SFT data.")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HF model id of the teacher (default: Qwen2.5-7B-Instruct).")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="How many xLAM rows to relabel.")
    parser.add_argument("--output", default="outputs/relabel/xlam_relabeled.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-on-mismatch", action="store_true",
                        help="Keep the original xLAM answer when the teacher picks "
                             "different functions (instead of skipping the example).")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--smoke", action="store_true", help="Tiny run for a quick sanity check.")
    args = parser.parse_args()

    if args.smoke:
        args.num_samples = min(args.num_samples, 8)
        args.max_new_tokens = 128

    device = resolve_device(args.device)
    dtype = torch.float32 if device == "cpu" else getattr(torch, args.dtype)
    logger.info("Teacher relabel — model=%s device=%s dtype=%s", args.teacher, device, dtype)

    dataset = load_dataset(
        "Salesforce/xlam-function-calling-60k", split="train", token=_resolve_hf_token())
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_samples, len(dataset))))

    logger.info("Loading teacher (this may download several GB the first time)…")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, token=_resolve_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=dtype, token=_resolve_hf_token())
    model.to(device).eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    relabeled = fell_back = skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for index, row in enumerate(dataset):
            tools = _parse_json_field(row["tools"])
            gold_calls = _parse_json_field(row["answers"])
            if not tools or not gold_calls:
                skipped += 1
                continue

            prompt = format_inference_prompt(row["query"], tools, tokenizer)
            teacher_output = _generate(model, tokenizer, prompt, device, args.max_new_tokens)
            predicted_calls = extract_tool_calls(teacher_output)

            answer = build_relabeled_answer(
                predicted_calls, gold_calls, tools, keep_on_mismatch=args.keep_on_mismatch)
            if answer is None:
                skipped += 1
            else:
                used_teacher = answer is not gold_calls
                relabeled += int(used_teacher)
                fell_back += int(not used_teacher)
                out_fh.write(json.dumps({
                    "query": row["query"],
                    "tools": json.dumps(tools, ensure_ascii=False),
                    "answers": json.dumps(answer, ensure_ascii=False),
                }, ensure_ascii=False) + "\n")

            if (index + 1) % 100 == 0:
                logger.info("Processed %d/%d — %s", index + 1, len(dataset),
                            summarise_relabeling(index + 1, relabeled, fell_back, skipped))

    logger.info("Done. %s", summarise_relabeling(len(dataset), relabeled, fell_back, skipped))
    logger.info("Wrote relabeled SFT data to %s", output_path)


if __name__ == "__main__":
    main()
