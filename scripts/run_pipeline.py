#!/usr/bin/env python3
"""
End-to-end post-training pipeline runner.

Sub-commands (one per pipeline stage):

  baseline        Stage 0  — evaluate the raw Qwen2.5-1.5B-Instruct on BFCL.
                  Produces the floor score every later stage compares against.

  evaluate        Any stage — re-score a saved LoRA checkpoint (or the base
                  model) on BFCL without re-running training.

  sft             Stage 1  — supervised fine-tuning on xLAM-60K with LoRA,
                  followed by BFCL evaluation.

  generate-pairs  Stage 2A data prep — sample completions from the SFT model,
                  BFCL-grade them, and write (chosen, rejected) preference pairs.

  dpo             Stage 2A — Direct Preference Optimisation on the generated
                  pairs, followed by BFCL evaluation.

  grpo            Stage 2B — Group Relative Policy Optimisation (online RL with
                  the verifiable BFCL reward), followed by BFCL evaluation.

Device options
--------------
  --device auto   Pick the best available device (default)
  --device cuda   Force NVIDIA GPU
  --device mps    Apple Silicon (M1/M2/M3)
  --device cpu    CPU only (slow; fine for smoke tests)

Typical Stage 2 order
---------------------
  python scripts/run_pipeline.py sft                       # produce SFT adapter
  python scripts/run_pipeline.py generate-pairs            # produce DPO pairs
  python scripts/run_pipeline.py dpo                        # Stage 2A
  python scripts/run_pipeline.py grpo                       # Stage 2B (GPU advised)

Quick-start examples
--------------------
  # Baseline on Apple Silicon:
  python scripts/run_pipeline.py baseline --device mps

  # SFT smoke test:
  python scripts/run_pipeline.py sft --smoke --device mps

  # End-to-end Stage 2 smoke test on Apple Silicon (no SFT checkpoint needed):
  python scripts/run_pipeline.py generate-pairs --smoke --device mps
  python scripts/run_pipeline.py dpo  --smoke --device mps
  python scripts/run_pipeline.py grpo --smoke --device mps

  # Real DPO run from a specific SFT checkpoint and pairs file:
  python scripts/run_pipeline.py dpo \\
      --sft-checkpoint outputs/sft/checkpoint-final \\
      --pairs-path outputs/pairs/dpo_pairs.jsonl
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


def _add_batch_size_arg(parser: argparse.ArgumentParser) -> None:
    """Per-device batch-size override — the primary lever for fixing CUDA OOM."""
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Override per-device batch size (lower this first if you hit CUDA OOM).",
    )


def _add_stage2_training_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the dpo and grpo sub-commands."""
    parser.add_argument(
        "--sft-checkpoint", default=None, metavar="PATH",
        help="Stage 1 SFT adapter to start from (default: from the stage config).",
    )
    parser.add_argument(
        "--from-base", action="store_true",
        help="Start from a fresh LoRA on the base model when no SFT checkpoint exists.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip BFCL evaluation after training.",
    )
    parser.add_argument(
        "--run-eval", action="store_true",
        help="Force BFCL evaluation even in smoke mode (off by default in smoke).",
    )
    _add_batch_size_arg(parser)


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
# Shared helpers for the Stage 2 sub-commands (dpo, grpo, generate-pairs)
# ---------------------------------------------------------------------------

def _maybe_build_eval_fn(args, cfg, device):
    """
    Return a post-training BFCL eval closure, or None when eval is skipped.

    Used by the dpo/grpo trainers via BaseTrainer's eval hook.
    """
    if getattr(args, "skip_eval", False):
        return None

    def bfcl_eval_fn(eval_model, eval_tokenizer):
        summary = evaluate_bfcl(eval_model, eval_tokenizer, cfg, device)
        return summary.to_dict()

    return bfcl_eval_fn


def _load_policy_model(cfg, sft_checkpoint, args, *, trainable):
    """
    Load the Stage 2 policy: the SFT adapter when available, else a fallback.

    - If the SFT checkpoint exists, it is loaded (trainable for dpo/grpo).
    - Otherwise, in --smoke or --from-base mode, we fall back to the base model
      (with a fresh LoRA adapter when a trainable policy is required) so the
      pipeline can be exercised without first running a full SFT.
    - Otherwise we raise with actionable guidance.
    """
    from pipeline.model.loader import load_model, load_model_from_checkpoint

    checkpoint_path = Path(sft_checkpoint) if sft_checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        logger.info("Loading SFT adapter (trainable=%s) from %s", trainable, checkpoint_path)
        return load_model_from_checkpoint(cfg, checkpoint_path, is_trainable=trainable)

    allow_fallback = getattr(args, "smoke", False) or getattr(args, "from_base", False)
    if allow_fallback:
        logger.warning(
            "No SFT checkpoint at %s — falling back to %s the base model.",
            checkpoint_path,
            "a fresh LoRA adapter on" if trainable else "plain",
        )
        return load_model(cfg, apply_lora=trainable)

    raise FileNotFoundError(
        f"\n\n{'='*65}\n"
        f"  SFT checkpoint not found at: {checkpoint_path}\n\n"
        f"  Options:\n"
        f"    1. Run Stage 1 first:  python scripts/run_pipeline.py sft\n"
        f"    2. Point at a checkpoint:  --sft-checkpoint PATH\n"
        f"    3. Start from the base model:  --from-base\n"
        f"{'='*65}\n"
    )


def _apply_training_smoke_overrides(cfg, args, output_dir: str) -> None:
    """
    Shrink a Stage 2 training run to a few fast steps for validation.

    Shared by dpo and grpo; each then layers on its stage-specific knobs
    (e.g. GRPO's num_generations). Mirrors the SFT smoke settings: tiny sample
    count, 5-step cap, gradient checkpointing, float32 on MPS for stability,
    and eval skipped unless --run-eval is passed.
    """
    cfg.training.max_samples = 32
    cfg.training.max_steps = 5
    cfg.training.num_epochs = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.gradient_checkpointing = True
    cfg.training.save_steps = 5
    cfg.data.max_seq_len = 512
    cfg.output.dir = output_dir

    if not args.run_eval:
        args.skip_eval = True

    effective_device = resolve_device(cfg.model.device)
    if effective_device == "mps":
        cfg.model.torch_dtype = "float32"
        logger.info("MPS detected — using float32 for numerical stability")

    _apply_smoke_settings_for_eval(cfg)


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
# Sub-command: evaluate (re-score any checkpoint, no training)
# ---------------------------------------------------------------------------

def run_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate a saved LoRA checkpoint (or the base model) on BFCL.

    Lets you re-score existing Stage-1/2 checkpoints — e.g. after a grader
    change — without re-running any training. Results are written next to the
    checkpoint as evaluate_results.json (or to --output-dir if given).
    """
    cfg = OmegaConf.load(args.config)

    if args.device:
        cfg.model.device = args.device
    if args.smoke:
        _apply_smoke_settings_for_eval(cfg)

    _apply_common_overrides(cfg, args)

    device = resolve_device(cfg.model.device)

    if args.checkpoint:
        from pipeline.model.loader import load_model_from_checkpoint

        model, tokenizer = load_model_from_checkpoint(
            cfg, args.checkpoint, is_trainable=False
        )
        default_output_dir = Path(args.checkpoint)
        label = args.checkpoint
    else:
        # No checkpoint → evaluate the unmodified base model
        model, tokenizer = load_model(cfg, apply_lora=False)
        default_output_dir = Path(cfg.output.dir)
        label = f"base model ({cfg.model.name_or_path})"

    logger.info("Evaluating %s on device: %s", label, device)
    summary = evaluate_bfcl(model, tokenizer, cfg, device)
    results = summary.to_dict()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluate_results.json"
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
    if args.batch_size is not None:
        cfg.training.per_device_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.no_train_eval:
        # Skip the periodic in-training (LM-loss) evaluation to save wall-clock.
        # The final BFCL evaluation still runs unless --skip-eval is also passed.
        cfg.training.disable_in_training_eval = True

    device = resolve_device(cfg.model.device)
    logger.info("Stage 1 — SFT on device: %s", device)

    model, tokenizer = load_model(cfg, apply_lora=True)

    logger.info("Loading and tokenising xLAM dataset…")
    full_dataset = get_dataset("xlam_sft", cfg, tokenizer=tokenizer)
    # Drop examples that were filtered out during tokenisation (empty input_ids)
    full_dataset = full_dataset.filter(
        lambda ex: len(ex["input_ids"]) > 0, desc="Dropping empty examples"
    )

    disable_in_training_eval = cfg.training.get("disable_in_training_eval", False)

    if disable_in_training_eval:
        train_dataset = full_dataset
        eval_dataset = None
        logger.info("Dataset — train: %d | in-training eval: disabled", len(train_dataset))
    else:
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
# Sub-command: generate-pairs (Stage 2A data prep)
# ---------------------------------------------------------------------------

def run_generate_pairs(args: argparse.Namespace) -> None:
    from pipeline.generation.pair_generator import (
        generate_preference_pairs,
        save_pairs_jsonl,
    )

    base_cfg = OmegaConf.load("configs/base.yaml")
    gen_cfg = OmegaConf.load("configs/generate_pairs.yaml")
    cfg = OmegaConf.merge(base_cfg, gen_cfg)

    if args.device:
        cfg.model.device = args.device

    if args.smoke:
        cfg.generation.num_source_queries = 8
        cfg.generation.rollouts_per_query = 4
        cfg.generation.max_new_tokens = 128
        cfg.generation.output_path = "outputs/pairs_smoke/dpo_pairs.jsonl"
        if resolve_device(cfg.model.device) == "mps":
            cfg.model.torch_dtype = "float32"
        logger.info("Smoke mode: 8 queries x 4 rollouts, float32 on MPS")

    # CLI overrides
    if args.num_queries is not None:
        cfg.generation.num_source_queries = args.num_queries
    if args.rollouts is not None:
        cfg.generation.rollouts_per_query = args.rollouts
    if args.max_pairs_per_query is not None:
        cfg.generation.max_pairs_per_query = args.max_pairs_per_query
    if args.output_path:
        cfg.generation.output_path = args.output_path

    sft_checkpoint = args.sft_checkpoint or cfg.generation.get("sft_checkpoint")
    device = resolve_device(cfg.model.device)
    logger.info("Stage 2A data prep — generating preference pairs on device: %s", device)

    # trainable=False: we only sample from the SFT model, never update it here
    model, tokenizer = _load_policy_model(cfg, sft_checkpoint, args, trainable=False)

    pairs = generate_preference_pairs(cfg, model, tokenizer, device)
    if not pairs:
        logger.warning(
            "No preference pairs produced — every rollout for every query was "
            "uniformly correct or uniformly incorrect. Try more queries/rollouts."
        )
    save_pairs_jsonl(pairs, cfg.generation.output_path)


# ---------------------------------------------------------------------------
# Sub-command: dpo (Stage 2A)
# ---------------------------------------------------------------------------

def run_dpo(args: argparse.Namespace) -> None:
    import pipeline.data.preference  # registers "xlam_dpo" in the registry
    from pipeline.data.registry import get_dataset
    from pipeline.training.dpo_trainer import DPOTrainer

    base_cfg = OmegaConf.load("configs/base.yaml")
    dpo_cfg = OmegaConf.load("configs/dpo.yaml")
    cfg = OmegaConf.merge(base_cfg, dpo_cfg)

    if args.device:
        cfg.model.device = args.device

    if args.smoke:
        _apply_training_smoke_overrides(cfg, args, output_dir="outputs/dpo_smoke")
        cfg.training.per_device_batch_size = 2
        # Consume the pairs produced by `generate-pairs --smoke`
        cfg.training.pairs_path = "outputs/pairs_smoke/dpo_pairs.jsonl"
        logger.info("Smoke mode: DPO on the smoke preference pairs, max 5 steps")

    _apply_common_overrides(cfg, args)

    if args.batch_size is not None:
        cfg.training.per_device_batch_size = args.batch_size
    if args.pairs_path:
        cfg.training.pairs_path = args.pairs_path
    # Phase 3.2: per-run overrides for the beta sweep / data scaling
    if args.beta is not None:
        cfg.training.beta = args.beta
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs

    sft_checkpoint = args.sft_checkpoint or cfg.training.get("sft_checkpoint")
    device = resolve_device(cfg.model.device)
    logger.info(
        "Stage 2A — DPO on device: %s | beta=%s epochs=%s",
        device, cfg.training.beta, cfg.training.num_epochs,
    )

    model, tokenizer = _load_policy_model(cfg, sft_checkpoint, args, trainable=True)

    # DPO data is text (prompt/chosen/rejected) — TRL tokenises it internally
    train_dataset = get_dataset("xlam_dpo", cfg)
    logger.info("Loaded %d preference pairs", len(train_dataset))

    trainer = DPOTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_fn=_maybe_build_eval_fn(args, cfg, device),
    )
    results = trainer.train()

    if results:
        output_path = Path(cfg.output.dir) / "dpo_bfcl_results.json"
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("BFCL results saved to %s", output_path)
        _print_results_table(results)


# ---------------------------------------------------------------------------
# Sub-command: grpo (Stage 2B)
# ---------------------------------------------------------------------------

def run_grpo(args: argparse.Namespace) -> None:
    import pipeline.data.grpo_prompts  # registers "xlam_grpo" in the registry
    from pipeline.data.registry import get_dataset
    from pipeline.reward.grpo_reward import build_bfcl_reward_function
    from pipeline.training.grpo_trainer import GRPOTrainer

    base_cfg = OmegaConf.load("configs/base.yaml")
    grpo_cfg = OmegaConf.load("configs/grpo.yaml")
    cfg = OmegaConf.merge(base_cfg, grpo_cfg)

    if args.device:
        cfg.model.device = args.device

    if args.smoke:
        _apply_training_smoke_overrides(cfg, args, output_dir="outputs/grpo_smoke")
        # GRPO needs per_device_batch_size to be a multiple of num_generations,
        # and online rollouts are expensive — keep the group tiny for smoke.
        cfg.training.num_generations = 2
        cfg.training.per_device_batch_size = 2
        cfg.training.max_completion_length = 128
        cfg.training.max_steps = 2
        logger.info("Smoke mode: GRPO with group size 2, max 2 steps")

    _apply_common_overrides(cfg, args)

    if args.batch_size is not None:
        # GRPO requires per_device_batch_size to be a multiple of num_generations
        num_generations = cfg.training.get("num_generations", 8)
        if args.batch_size % num_generations != 0:
            raise ValueError(
                f"--batch-size {args.batch_size} must be a multiple of "
                f"num_generations ({num_generations}) for GRPO."
            )
        cfg.training.per_device_batch_size = args.batch_size

    sft_checkpoint = args.sft_checkpoint or cfg.training.get("sft_checkpoint")
    device = resolve_device(cfg.model.device)
    logger.info("Stage 2B — GRPO on device: %s", device)

    model, tokenizer = _load_policy_model(cfg, sft_checkpoint, args, trainable=True)

    # GRPO prompts need the tokenizer to build the chat-templated prompt string
    train_dataset = get_dataset("xlam_grpo", cfg, tokenizer=tokenizer)
    logger.info("Loaded %d GRPO rollout prompts", len(train_dataset))

    # Phase 2: shaped (partial-credit) reward by default — gives GRPO within-group
    # variance so it actually learns. Set training.reward_shaping=false for the
    # binary-reward ablation.
    use_shaped_reward = cfg.training.get("reward_shaping", True)
    logger.info("GRPO reward: %s", "shaped (partial credit)" if use_shaped_reward else "binary")
    reward_fn = build_bfcl_reward_function(shaped=use_shaped_reward)

    trainer = GRPOTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_fn=reward_fn,
        eval_fn=_maybe_build_eval_fn(args, cfg, device),
    )
    results = trainer.train()

    if results:
        output_path = Path(cfg.output.dir) / "grpo_bfcl_results.json"
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

    # --- evaluate ---
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate any saved LoRA checkpoint (or the base model) on BFCL — no training.",
    )
    _add_common_args(evaluate_parser)
    evaluate_parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="LoRA adapter checkpoint to evaluate (e.g. outputs/dpo/checkpoint-final). "
             "Omit to evaluate the base model.",
    )

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
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override number of training epochs (lower this to expedite; e.g. 1).",
    )
    sft_parser.add_argument(
        "--no-train-eval",
        action="store_true",
        help=(
            "Skip the periodic in-training LM-loss evaluation to save time. "
            "The final BFCL evaluation still runs unless --skip-eval is set."
        ),
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
    _add_batch_size_arg(sft_parser)

    # --- generate-pairs (Stage 2A data prep) ---
    pairs_parser = subparsers.add_parser(
        "generate-pairs",
        help="Stage 2A prep: sample from the SFT model and write DPO preference pairs.",
    )
    pairs_parser.add_argument("--config", default="configs/base.yaml", help=argparse.SUPPRESS)
    pairs_parser.add_argument(
        "--device", choices=VALID_DEVICES, default=None,
        help="Compute device. Use 'mps' for Apple Silicon. Default: auto-detect.",
    )
    pairs_parser.add_argument("--smoke", action="store_true", help="Tiny run: 8 queries x 4 rollouts.")
    pairs_parser.add_argument(
        "--sft-checkpoint", default=None, metavar="PATH",
        help="SFT adapter to sample from (default: from configs/generate_pairs.yaml).",
    )
    pairs_parser.add_argument(
        "--from-base", action="store_true",
        help="Sample from the base model when no SFT checkpoint exists.",
    )
    pairs_parser.add_argument("--num-queries", type=int, default=None, metavar="N",
                              help="Number of source queries to sample from.")
    pairs_parser.add_argument("--rollouts", type=int, default=None, metavar="N",
                              help="Completions sampled per query.")
    pairs_parser.add_argument("--max-pairs-per-query", type=int, default=None, metavar="N",
                              help="Max (chosen, rejected) pairs per query (Phase 3.1; more pairs, "
                                   "no extra generation cost).")
    pairs_parser.add_argument("--output-path", default=None, metavar="PATH",
                              help="Where to write the pairs JSONL.")

    # --- dpo (Stage 2A) ---
    dpo_parser = subparsers.add_parser(
        "dpo",
        help="Stage 2A: Direct Preference Optimisation on generated pairs, then evaluate.",
    )
    _add_common_args(dpo_parser)
    _add_stage2_training_args(dpo_parser)
    dpo_parser.add_argument(
        "--pairs-path", default=None, metavar="PATH",
        help="Preference pairs JSONL (default: from configs/dpo.yaml).",
    )
    dpo_parser.add_argument(
        "--beta", type=float, default=None, metavar="B",
        help="DPO KL-regularisation strength (Phase 3.2 sweep; default from configs/dpo.yaml).",
    )
    dpo_parser.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="Override number of DPO epochs (default from configs/dpo.yaml).",
    )

    # --- grpo (Stage 2B) ---
    grpo_parser = subparsers.add_parser(
        "grpo",
        help="Stage 2B: online GRPO with the verifiable BFCL reward, then evaluate.",
    )
    _add_common_args(grpo_parser)
    _add_stage2_training_args(grpo_parser)

    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    parsed_args = arg_parser.parse_args()

    # Dispatch table keeps the entry point flat as stages are added
    COMMAND_DISPATCH = {
        "baseline": run_baseline,
        "evaluate": run_evaluate,
        "sft": run_sft,
        "generate-pairs": run_generate_pairs,
        "dpo": run_dpo,
        "grpo": run_grpo,
    }
    COMMAND_DISPATCH[parsed_args.command](parsed_args)
