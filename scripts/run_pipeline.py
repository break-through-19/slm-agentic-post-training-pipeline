#!/usr/bin/env python3
"""
End-to-end baseline pipeline runner.

Two sub-commands:

  baseline   Stage 0 — evaluate the raw Qwen2.5-1.5B-Instruct on BFCL.
             Produces the floor score that all post-training stages compare against.

  sft        Stage 1 — supervised fine-tuning on xLAM-60K with LoRA,
             followed by BFCL evaluation to measure improvement.

Device options
--------------
  --device auto   Pick the best available device (default)
  --device cuda   Force NVIDIA GPU
  --device mps    Apple Silicon (M1/M2/M3)
  --device cpu    CPU only (slow; fine for smoke tests)

Quick-start examples
--------------------
  # Full baseline evaluation (auto device):
  python scripts/run_pipeline.py baseline

  # Smoke test on CPU — 10 examples per BFCL category, no download wait:
  python scripts/run_pipeline.py baseline --smoke

  # Baseline on Apple Silicon:
  python scripts/run_pipeline.py baseline --device mps

  # SFT smoke test (64 training examples, 1 epoch):
  python scripts/run_pipeline.py sft --smoke --device mps

  # Full SFT run, skip BFCL eval after training:
  python scripts/run_pipeline.py sft --skip-eval

  # Evaluate specific BFCL categories only:
  python scripts/run_pipeline.py baseline --categories simple irrelevance
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf


def _load_dotenv(env_path: Path) -> None:
    """
    Load key=value pairs from a .env file into os.environ.

    - Skips blank lines and lines starting with #
    - Does NOT override variables already set in the shell environment
      (shell wins over .env, consistent with standard dotenv behaviour)
    """
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env from repo root before any pipeline imports that read env vars
_load_dotenv(Path(__file__).parent.parent / ".env")

from pipeline.evaluation.evaluator import evaluate_bfcl
from pipeline.model.loader import load_model, resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared argument parsing
# ---------------------------------------------------------------------------

VALID_DEVICES = ["auto", "cuda", "mps", "cpu"]
ALL_BFCL_CATEGORIES = ["simple", "multiple", "parallel", "irrelevance"]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the base YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--device",
        choices=VALID_DEVICES,
        default=None,
        help="Compute device. Use 'mps' for Apple Silicon. Default: auto-detect.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_BFCL_CATEGORIES,
        default=None,
        help="BFCL categories to evaluate (default: all four).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap evaluation samples per category. Useful for dev runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory from config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: 5 eval samples per category, 128 max new tokens. "
            "Use --device mps on Apple Silicon (~1-2 min total). "
            "CPU is supported but slow (~3 min per example)."
        ),
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_smoke_settings_for_eval(cfg) -> None:
    """
    Shrink evaluation to a tiny slice for fast smoke testing.

    Device priority for smoke runs (when no --device flag is given):
      MPS  — Apple Silicon; fast enough for a smoke run in ~1–2 min
      CUDA — any NVIDIA GPU
      CPU  — last resort; a 1.5B model in float32 takes ~3 min per example
    """
    cfg.evaluation.max_eval_samples = 5
    cfg.evaluation.max_new_tokens = 128   # shorter generation = faster per step
    # Keep bfloat16 on accelerated hardware; fall back to float32 on CPU only
    from pipeline.model.loader import resolve_device
    effective_device = resolve_device(cfg.model.device)
    if effective_device == "cpu":
        cfg.model.torch_dtype = "float32"
        logger.info(
            "Smoke mode active: 5 samples/category, float32, CPU — "
            "expect ~3 min per example. Use --device mps on Apple Silicon for faster runs."
        )
    else:
        logger.info("Smoke mode active: 5 samples/category, device=%s", effective_device)


def _apply_common_overrides(cfg, args) -> None:
    """Apply CLI overrides that are common to both sub-commands."""
    if args.device:
        cfg.model.device = args.device
    if args.max_eval_samples is not None:
        cfg.evaluation.max_eval_samples = args.max_eval_samples
    if args.categories:
        cfg.evaluation.bfcl_categories = args.categories
    if args.output_dir:
        cfg.output.dir = args.output_dir


def _print_results_table(results: dict) -> None:
    print("\n" + "=" * 52)
    print(f"  Overall accuracy : {results['overall_accuracy']:.4f}")
    print("=" * 52)
    for category, stats in results.items():
        if category == "overall_accuracy":
            continue
        acc = stats.get("accuracy", 0.0)
        correct = stats.get("correct", 0)
        total = stats.get("total", 0)
        failures = stats.get("failure_breakdown", {})
        top_failure = max(failures, key=failures.get) if failures else "—"
        print(f"  {category:<12}  acc={acc:.4f}  ({correct}/{total})  top failure: {top_failure}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Sub-command: baseline (Stage 0)
# ---------------------------------------------------------------------------

def run_baseline(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    # Apply device override before smoke settings so the smoke logger can
    # report the effective device and choose the right dtype.
    if args.device:
        cfg.model.device = args.device

    if args.smoke:
        _apply_smoke_settings_for_eval(cfg)

    _apply_common_overrides(cfg, args)

    output_path = Path(cfg.output.dir) / "baseline_results.json"
    device = resolve_device(cfg.model.device)
    logger.info("Stage 0 — baseline evaluation on device: %s", device)

    model, tokenizer = load_model(cfg, apply_lora=False)
    summary = evaluate_bfcl(model, tokenizer, cfg, device)
    results = summary.to_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    _print_results_table(results)


# ---------------------------------------------------------------------------
# Sub-command: sft (Stage 1)
# ---------------------------------------------------------------------------

def run_sft(args: argparse.Namespace) -> None:
    # Import here to avoid pulling in heavy training deps during baseline runs
    import pipeline.data.xlam  # registers "xlam_sft" in the dataset registry
    from pipeline.data.registry import get_dataset
    from pipeline.training.sft_trainer import SFTTrainer

    base_cfg = OmegaConf.load("configs/base.yaml")
    sft_cfg = OmegaConf.load("configs/sft.yaml")
    cfg = OmegaConf.merge(base_cfg, sft_cfg)

    if args.smoke:
        cfg.training.max_samples = 32
        cfg.training.max_steps = 5            # cap at 5 steps regardless of epochs
        cfg.training.num_epochs = 1
        cfg.training.per_device_batch_size = 1  # keep MPS VRAM footprint minimal
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.gradient_checkpointing = True  # trade compute for memory
        cfg.training.learning_rate = 5.0e-5    # gentle LR avoids weight collapse
        cfg.training.save_steps = 5
        cfg.training.eval_steps = 0            # no in-training eval (2 samples = no signal)
        cfg.training.disable_in_training_eval = True
        cfg.data.max_seq_len = 512             # shorter sequences = faster steps
        cfg.output.dir = "outputs/sft_smoke"
        # Skip BFCL eval after smoke training unless explicitly requested;
        # 5 steps cannot produce a meaningfully trained model.
        if not args.run_eval:
            args.skip_eval = True
        # Force float32 on MPS — bf16 + LoRA + gradient checkpointing on MPS
        # produces silent NaN gradients, corrupting LoRA weights. With batch=1
        # and gradient checkpointing the fp32 footprint fits in the 9 GB MPS budget.
        effective_device = resolve_device(cfg.model.device)
        if effective_device == "mps":
            cfg.model.torch_dtype = "float32"
            logger.info("MPS detected — using float32 for numerical stability")
        _apply_smoke_settings_for_eval(cfg)
        logger.info(
            "Smoke mode: 32 samples, max 5 steps, batch=1, "
            "grad_ckpt=on, lr=5e-5, seq_len=512, post_train_eval=%s",
            not args.skip_eval,
        )

    _apply_common_overrides(cfg, args)

    if hasattr(args, "max_samples") and args.max_samples is not None:
        cfg.training.max_samples = args.max_samples

    device = resolve_device(cfg.model.device)
    logger.info("Stage 1 — SFT on device: %s", device)

    model, tokenizer = load_model(cfg, apply_lora=True)

    logger.info("Loading and tokenising xLAM dataset…")
    full_dataset = get_dataset("xlam_sft", cfg, tokenizer=tokenizer)
    # Drop examples that were filtered out during tokenisation (empty input_ids)
    full_dataset = full_dataset.filter(
        lambda ex: len(ex["input_ids"]) > 0, desc="Dropping empty examples"
    )

    split_ratio = cfg.training.get("train_split", 0.95)
    splits = full_dataset.train_test_split(
        test_size=1.0 - split_ratio, seed=cfg.data.seed
    )
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    logger.info("Dataset split — train: %d | eval: %d", len(train_dataset), len(eval_dataset))

    bfcl_eval_fn = None
    if not args.skip_eval:
        def bfcl_eval_fn(eval_model, eval_tokenizer):
            summary = evaluate_bfcl(eval_model, eval_tokenizer, cfg, device)
            return summary.to_dict()

    trainer = SFTTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_fn=bfcl_eval_fn,
    )
    results = trainer.train()

    if results:
        output_path = Path(cfg.output.dir) / "sft_bfcl_results.json"
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("BFCL results saved to %s", output_path)
        _print_results_table(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="SLM agentic post-training pipeline — baseline stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- baseline ---
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Stage 0: evaluate Qwen2.5-1.5B-Instruct on BFCL without any fine-tuning.",
    )
    _add_common_args(baseline_parser)

    # --- sft ---
    sft_parser = subparsers.add_parser(
        "sft",
        help="Stage 1: fine-tune with LoRA on xLAM-60K, then evaluate on BFCL.",
    )
    _add_common_args(sft_parser)
    sft_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override max training samples (default: from configs/sft.yaml).",
    )
    sft_parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip BFCL evaluation after training (saves time on smoke runs).",
    )
    sft_parser.add_argument(
        "--run-eval",
        action="store_true",
        help=(
            "Force BFCL evaluation even in smoke mode. By default smoke mode "
            "skips eval because 5 steps does not produce a meaningfully trained model."
        ),
    )

    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    parsed_args = arg_parser.parse_args()

    if parsed_args.command == "baseline":
        run_baseline(parsed_args)
    elif parsed_args.command == "sft":
        run_sft(parsed_args)
