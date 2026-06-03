# SLM Agentic Post-Training Pipeline

Frontier-lab post-training techniques at small scale — teaching a 1.5B model to call functions like a much larger one.

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Benchmark:** [Berkeley Function Calling Leaderboard (BFCL v3)](https://gorilla.cs.berkeley.edu/leaderboard.html)

---

## Overview

The pipeline implements a two-stage post-training recipe for improving agentic function-calling in small language models, following the SFT → preference optimisation approach used by frontier labs.

```
Stage 0  ──►  Stage 1  ──►  Stage 2A  ──►  Stage 2B
Baseline       SFT            DPO            GRPO
(floor)       (LoRA)        (offline)       (online)
```

All four model variants are evaluated on BFCL to measure incremental improvement.

### Stage 0 — Baseline
Evaluate the raw `Qwen2.5-1.5B-Instruct` on all BFCL sub-categories to establish the floor score.

### Stage 1 — Supervised Fine-Tuning (SFT)
Fine-tune with LoRA (rank 16) on the [xLAM-60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset — 10–20K high-quality function-calling demonstrations. Teaches the model the correct output format and basic tool-selection logic.

### Stage 2A — Direct Preference Optimisation (DPO)
First, `generate-pairs` samples several completions per query from the SFT model, grades each with the BFCL verifier, and pairs a correct completion (chosen) with an incorrect one (rejected). `dpo` then trains the SFT adapter offline on those pairs. The reference model is the adapter-disabled base network, so LoRA makes it free.

### Stage 2B — Group Relative Policy Optimisation (GRPO)
Online RL using the verifiable BFCL reward. For each prompt the policy samples a group of completions, scores each, and shifts toward the above-average completions in the group. Group-relative advantages replace a value function, following the DeepSeek-R1 approach at 1.5B scale.

The reward is a **shaped, partial-credit** signal (`bfcl_grader.score`, controlled by `reward_shaping` in `configs/grpo.yaml`): `0.2` for a well-formed call, `+0.3` for the correct function name, `+0.5 ×` the fraction of correct arguments. A plain binary 0/1 reward leaves most sampled groups with identical rewards (zero variance → zero advantage → no gradient); partial credit gives within-group variance so GRPO actually learns. By construction the shaped score equals `1.0` exactly when the binary metric is correct, so training and evaluation never disagree at the extremes. Irrelevance stays binary (abstention is all-or-nothing).

### Irrelevance augmentation (recovering abstention)
xLAM contains **only positive** examples, so training on it teaches the model to always emit a tool call and collapses the BFCL `irrelevance` category (base ~0.72 → post-SFT ~0.02). To counter this, every training stage can blend in synthetic **abstention** examples: a real query paired with the tools from a *different* example, so the correct behaviour is to call no function. This is controlled by `irrelevance_fraction` (default `0.25`) in each stage config and requires no BFCL data (no test contamination). See `pipeline/data/irrelevance.py`.

---

## Repo Structure

```
slm-agentic-post-training-pipeline/
├── .env.example           # token config template — copy to .env and fill in
├── configs/
│   ├── base.yaml          # shared defaults (model, LoRA, evaluation)
│   ├── sft.yaml           # Stage 1 training hyperparameters
│   ├── generate_pairs.yaml# Stage 2A preference-pair generation knobs
│   ├── dpo.yaml           # Stage 2A DPO hyperparameters
│   └── grpo.yaml          # Stage 2B GRPO hyperparameters
│
├── pipeline/
│   ├── model/
│   │   ├── loader.py      # load model + tokenizer; apply/load/merge LoRA
│   │   └── lora_config.py # build LoraConfig from YAML
│   │
│   ├── data/
│   │   ├── registry.py    # DatasetRegistry — name → (load_fn, format_fn)
│   │   ├── xlam.py        # xLAM-60K loader + SFT formatter (Stage 1)
│   │   ├── bfcl.py        # BFCL test-set loader + example parser (eval)
│   │   ├── irrelevance.py # synthesise abstention examples from xLAM (Phase 1)
│   │   ├── preference.py  # DPO (prompt, chosen, rejected) loader (xlam_dpo)
│   │   └── grpo_prompts.py# GRPO rollout-prompt loader (xlam_grpo)
│   │
│   ├── formatting/
│   │   └── chat_template.py  # Qwen2.5 tool-call prompt builder; shared by
│   │                         # SFT formatting, inference, and Stage 2 reward
│   │
│   ├── generation/
│   │   └── pair_generator.py # sample from SFT model + grade → preference pairs
│   │
│   ├── reward/
│   │   ├── bfcl_grader.py    # verifiable grader: grade() binary metric +
│   │   │                     # score() shaped partial-credit reward
│   │   └── grpo_reward.py    # wraps the grader as a TRL GRPO reward function
│   │
│   ├── training/
│   │   ├── base_trainer.py   # abstract: config persistence, checkpointing,
│   │   │                     # eval hooks — shared by all stages
│   │   ├── sft_trainer.py    # Stage 1: wraps TRL SFTTrainer
│   │   ├── dpo_trainer.py    # Stage 2A: wraps TRL DPOTrainer
│   │   └── grpo_trainer.py   # Stage 2B: wraps TRL GRPOTrainer
│   │
│   └── evaluation/
│       ├── evaluator.py      # runs BFCL categories, returns EvalSummary
│       └── metrics.py        # CategoryMetrics, EvalSummary, aggregation
│
├── scripts/
│   ├── run_pipeline.py    # single CLI entry point (6 sub-commands)
│   └── sweep_dpo_beta.sh  # Phase 3.2 — DPO beta sweep (one run per beta)
│
├── tests/
│   ├── test_formatting.py    # prompt builder + tool-call extraction
│   ├── test_grader.py        # all grader failure modes + type coercion
│   ├── test_data.py          # registry, xLAM parser, BFCL parser
│   ├── test_preference.py    # DPO pair dataset loader + formatter
│   ├── test_pair_generator.py# preference-pairing logic
│   └── test_grpo_reward.py   # GRPO reward function wrapper
│
├── outputs/               # results and checkpoints (git-ignored)
└── pyproject.toml
```

### Design Principles
- **Shared reward signal** — `bfcl_grader.grade()` is used identically during evaluation (Stage 0/1) and RL training (Stage 2B). Train-time and eval-time rewards cannot diverge.
- **Dataset registry** — adding a new dataset for Stage 2 (preference pairs) requires only a new file; no training code changes.
- **Single formatter** — all prompt construction goes through `pipeline/formatting/chat_template.py`, so SFT demonstrations and inference prompts are byte-for-byte identical.
- **Pluggable trainers** — `DPOTrainer` and `GRPOTrainer` will override only `_run_training()`; checkpointing, config persistence, and eval hooks are inherited from `BaseTrainer`.

---

## Setup

**Requirements:**

- **Python ≥ 3.10 (hard requirement).** The training stack (`trl≥1.0`, `transformers≥4.49`, `tokenizers≥0.22`) ships wheels only for cp310+. On Python 3.9 pip silently backtracks to ancient, incompatible versions (trl 0.12, transformers 4.46) — this is slow *and* installs a TRL whose API the trainers don't target. Check with `python --version`.
- **PyTorch ≥ 2.5, installed first** (see below). `torch` is deliberately not a declared dependency, so you control which CUDA build is used.

```bash
git clone https://github.com/break-through-19/slm-agentic-post-training-pipeline.git
cd slm-agentic-post-training-pipeline

# 1. Install torch FIRST, matched to your platform:
pip install torch                                   # CPU / macOS (MPS)
# or, on a CUDA box, pick the build matching your driver, e.g. cu121:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch

# 2. Install the package:
pip install -e ".[dev]"

# Optional extras:
pip install -e ".[dev,quant]"     # 4-bit quantisation (CUDA Linux only)
pip install -e ".[dev,wandb]"     # experiment tracking
pip install -e ".[generation]"    # vLLM for fast GRPO rollouts (GPU only)
```

### Linux GPU clusters

Two things make cluster installs fast and reliable:

**1. Use a Python 3.10+ interpreter for the venv.** On Python 3.9 the install will backtrack for a very long time and pull mutually-incompatible old packages.

```bash
module load python/3.11    # or: conda create -n slm python=3.11
python3.11 -m venv .venv
source .venv/bin/activate
python --version           # must print 3.10+
```

**2. Install torch first, and redirect pip's cache/temp to a partition with space.** Shared `/tmp` is often only 10–20 GB; the torch + CUDA wheels need more.

```bash
# Redirect pip scratch space to your project/data partition
mkdir -p $HOME/.cache/pip $HOME/tmp_pip
export TMPDIR=$HOME/tmp_pip
export PIP_CACHE_DIR=$HOME/.cache/pip

# torch first, from PyTorch's CUDA index (the +cuXXX build, not PyPI's)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1

# then the package — pip leaves the installed torch untouched and never
# re-resolves the multi-GB nvidia-* CUDA wheels
pip install -e ".[dev,quant]"

pip cache purge        # reclaim space afterwards
```

If the editable install is *still* slow, add `--no-cache-dir`. To verify the
right stack landed: `python -c "import trl, transformers; print(trl.__version__, transformers.__version__)"` should report `trl ≥ 1.0` and `transformers ≥ 4.49`.

---

## Authentication

The xLAM-60K training dataset is gated on HuggingFace and requires a token. The BFCL evaluation dataset is public and needs no token.

### Step 1 — Create a HuggingFace Read token

1. Go to **https://huggingface.co/settings/tokens**
2. Click **New token**
3. Set type to **Read** (write access is not needed)
4. Copy the token (starts with `hf_...`)

### Step 2 — Accept the xLAM dataset terms

Visit **https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k** and click **"Agree and access repository"**. This is a one-time action per HuggingFace account.

### Step 3 — Configure your token

Copy the provided template and add your token:

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_your_token_here
```

The `.env` file is git-ignored and will never be committed. The pipeline loads it automatically on startup — no `export` command needed.

**Alternative:** set the token directly in your shell (takes priority over `.env`):

```bash
export HF_TOKEN="hf_your_token_here"
```

Or use the HuggingFace CLI for a persistent login across all projects:

```bash
huggingface-cli login   # paste token when prompted
```

---

## Running the Pipeline

All stages run through a single script with six sub-commands:

| Sub-command | Stage | What it does |
|---|---|---|
| `baseline` | 0 | Evaluate the raw model on BFCL (floor score) |
| `evaluate` | any | Re-score any saved checkpoint (or the base model) on BFCL — no training |
| `sft` | 1 | LoRA fine-tune on xLAM-60K, then evaluate |
| `generate-pairs` | 2A prep | Sample from the SFT model, BFCL-grade, write preference pairs |
| `dpo` | 2A | Direct Preference Optimisation on the pairs, then evaluate |
| `grpo` | 2B | Online GRPO with the verifiable BFCL reward, then evaluate |

### Stage 0 — Baseline Evaluation

No authentication required — BFCL is a public dataset.

```bash
# Auto-detect best device (CUDA → MPS → CPU)
python scripts/run_pipeline.py baseline

# Apple Silicon (M1/M2/M3)
python scripts/run_pipeline.py baseline --device mps

# Specific categories only
python scripts/run_pipeline.py baseline --categories simple irrelevance

# Cap samples per category (faster dev run)
python scripts/run_pipeline.py baseline --max-eval-samples 100

# Smoke test — 5 samples/category, ~2 min on Apple Silicon
python scripts/run_pipeline.py baseline --smoke --device mps
```

Results are written to `outputs/baseline_results.json`.

### Re-scoring a checkpoint — `evaluate`

Evaluate any saved LoRA checkpoint (or the base model) on BFCL without re-running training. Useful after a grader change, or to compare intermediate checkpoints.

```bash
# Re-score the SFT / DPO / GRPO final adapters
python scripts/run_pipeline.py evaluate --checkpoint outputs/sft/checkpoint-final  --device cuda
python scripts/run_pipeline.py evaluate --checkpoint outputs/dpo/checkpoint-final  --device cuda
python scripts/run_pipeline.py evaluate --checkpoint outputs/grpo/checkpoint-final --device cuda

# Evaluate the base model (no --checkpoint) — same as `baseline`
python scripts/run_pipeline.py evaluate --device cuda

# Quick check on a subset
python scripts/run_pipeline.py evaluate --checkpoint outputs/dpo/checkpoint-final --max-eval-samples 100
```

Results are written to `evaluate_results.json` inside the checkpoint directory (or to `--output-dir` if given).

### Stage 1 — Supervised Fine-Tuning

Requires HuggingFace authentication (see [Authentication](#authentication) above).

```bash
# Full run (15K samples, 3 epochs, auto device)
python scripts/run_pipeline.py sft

# Apple Silicon
python scripts/run_pipeline.py sft --device mps

# Override training sample count
python scripts/run_pipeline.py sft --max-samples 5000

# Lower the batch size if you hit CUDA out-of-memory
python scripts/run_pipeline.py sft --batch-size 1

# Skip BFCL eval after training
python scripts/run_pipeline.py sft --skip-eval

# Smoke test — 32 samples, 5 training steps, ~10 min on Apple Silicon
python scripts/run_pipeline.py sft --smoke --device mps

# Smoke test WITH post-training BFCL eval (adds ~20 min)
python scripts/run_pipeline.py sft --smoke --device mps --run-eval
```

Checkpoints and BFCL results are written to `outputs/sft/`.

**Smoke mode notes (`--smoke`):**

- Caps training at 5 steps regardless of `num_epochs` — designed to validate the pipeline end-to-end, not to produce a useful model.
- Defaults to skipping BFCL evaluation after training, since 5 steps cannot meaningfully change the baseline scores. Use `--run-eval` to opt back in.
- Forces `float32` precision on MPS to avoid known bf16 NaN-gradient issues on Apple Silicon. The default `bfloat16` is still used on CUDA.
- Sets `max_seq_len=512` so each training step finishes quickly.

### Stage 2A — Generate Preference Pairs

Samples completions from the SFT model and writes BFCL-graded `(chosen, rejected)` pairs to JSONL. Requires HuggingFace authentication (xLAM is the query source).

```bash
# Full run from the default SFT checkpoint (outputs/sft/checkpoint-final)
python scripts/run_pipeline.py generate-pairs --device cuda

# Point at a specific SFT adapter and control sampling
python scripts/run_pipeline.py generate-pairs \
    --sft-checkpoint outputs/sft/checkpoint-final \
    --num-queries 2000 --rollouts 8 \
    --output-path outputs/pairs/dpo_pairs.jsonl

# Smoke test (8 queries x 4 rollouts). --from-base lets it run with no SFT yet.
python scripts/run_pipeline.py generate-pairs --smoke --device mps --from-base
```

Pairs are written to `outputs/pairs/dpo_pairs.jsonl` (or `outputs/pairs_smoke/` in smoke mode).

### Stage 2A — DPO

Trains the SFT adapter on the generated pairs. The reference model is the adapter-disabled base network (free with LoRA).

```bash
# Full run (reads pairs_path and sft_checkpoint from configs/dpo.yaml)
python scripts/run_pipeline.py dpo --device cuda

# Override the pairs file and SFT checkpoint
python scripts/run_pipeline.py dpo \
    --sft-checkpoint outputs/sft/checkpoint-final \
    --pairs-path outputs/pairs/dpo_pairs.jsonl

# Override beta / epochs per run (Phase 3.2)
python scripts/run_pipeline.py dpo --device cuda --beta 0.05 --epochs 2 --output-dir outputs/dpo_beta0.05

# Smoke test (consumes the pairs from `generate-pairs --smoke`)
python scripts/run_pipeline.py generate-pairs --smoke --device mps --from-base
python scripts/run_pipeline.py dpo --smoke --device mps --from-base
```

Checkpoints and BFCL results are written to `outputs/dpo/`.

#### Phase 3.1 — scaling the DPO data

More preference pairs give DPO a stronger signal (the first run trained on only ~466). Two levers, set as defaults in `configs/generate_pairs.yaml`:

- `max_pairs_per_query` (default `3`) extracts more pairs from the **same** rollouts — no extra generation cost.
- `num_source_queries` (default `4000`) samples more queries — adds generation time (~50 min per 2000 queries).

```bash
# Generate a larger pair set, then DPO over it (2 epochs by default)
python scripts/run_pipeline.py generate-pairs --device cuda --num-queries 4000 --max-pairs-per-query 3
python scripts/run_pipeline.py dpo --device cuda
```

#### Phase 3.2 — beta sweep

`beta` controls how far DPO may move from the reference policy. Sweep it with the helper, which trains one run per value into its own output dir for clean comparison:

```bash
# Defaults: betas 0.05 0.1 0.3, from outputs/sft/checkpoint-final and outputs/pairs/dpo_pairs.jsonl
scripts/sweep_dpo_beta.sh

# Custom pairs / checkpoint / beta grid
scripts/sweep_dpo_beta.sh outputs/pairs/dpo_pairs.jsonl outputs/sft/checkpoint-final 0.05 0.1 0.2 0.5
```

Then compare `outputs/dpo_beta*/dpo_bfcl_results.json`.

### Stage 2B — GRPO

Online RL: samples a group of completions per prompt and scores them with the verifiable BFCL reward. GRPO is compute-heavy (multiple generations per prompt per step) and is realistically a **GPU** workload; the MPS/CPU paths exist for smoke-testing only.

```bash
# Full run (GPU strongly recommended; set use_vllm: true in configs/grpo.yaml
# if vLLM is installed for much faster rollouts)
python scripts/run_pipeline.py grpo --device cuda

# Start from a specific SFT checkpoint
python scripts/run_pipeline.py grpo --sft-checkpoint outputs/sft/checkpoint-final

# Smoke test (group size 2, 2 steps; --from-base needs no SFT checkpoint)
python scripts/run_pipeline.py grpo --smoke --device mps --from-base
```

Checkpoints and BFCL results are written to `outputs/grpo/`.

**Stage 2 smoke mode** applies the same safeguards as SFT smoke (5-step cap, gradient checkpointing, float32 on MPS, eval skipped unless `--run-eval`). When no SFT checkpoint exists, pass `--from-base` to start Stage 2 from a fresh LoRA on the base model so the pipeline can be exercised without a full SFT run first.

### Running Tests

No authentication or GPU required — tests use a mock tokenizer.

```bash
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=pipeline --cov-report=term-missing
```

---

## Configuration

All hyperparameters live in `configs/`. Each stage merges `base.yaml` with its own stage file at runtime (`sft.yaml`, `generate_pairs.yaml`, `dpo.yaml`, `grpo.yaml`), so only the overrides live in each stage file.

Key `base.yaml` settings:

| Key | Default | Description |
|-----|---------|-------------|
| `model.name_or_path` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `model.device` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `model.torch_dtype` | `bfloat16` | `bfloat16` \| `float16` \| `float32` (use float32 on MPS) |
| `model.load_in_4bit` | `false` | 4-bit NF4 quant (CUDA + bitsandbytes only) |
| `lora.r` | `16` | LoRA rank |
| `evaluation.max_eval_samples` | `null` | Per-category cap (`null` = full set) |

Key `sft.yaml` settings (Stage 1):

| Key | Default | Description |
|-----|---------|-------------|
| `training.max_samples` | `15000` | xLAM examples to use for training |
| `training.irrelevance_fraction` | `0.25` | Fraction converted to abstention examples (Phase 1) |
| `training.max_steps` | `-1` | Hard cap on training steps (`-1` = use `num_epochs`) |
| `training.num_epochs` | `3` | Training epochs (ignored when `max_steps > 0`) |
| `training.per_device_batch_size` | `2` | Per-GPU batch size (24 GB-safe) |
| `training.learning_rate` | `2e-4` | AdamW learning rate |

Key `dpo.yaml` settings (Stage 2A):

| Key | Default | Description |
|-----|---------|-------------|
| `training.pairs_path` | `outputs/pairs/dpo_pairs.jsonl` | Preference pairs from `generate-pairs` |
| `training.sft_checkpoint` | `outputs/sft/checkpoint-final` | SFT adapter to refine |
| `training.num_epochs` | `2` | Passes over the pair set (Phase 3.1) |
| `training.beta` | `0.1` | DPO KL-regularisation strength (sweepable, Phase 3.2) |
| `training.loss_type` | `sigmoid` | DPO loss variant (sigmoid, ipo, hinge, …) |
| `training.learning_rate` | `5e-6` | Much smaller than SFT |

Key `generate_pairs.yaml` settings (Stage 2A prep):

| Key | Default | Description |
|-----|---------|-------------|
| `generation.num_source_queries` | `4000` | xLAM queries to sample from (Phase 3.1) |
| `generation.irrelevance_fraction` | `0.25` | Fraction of abstention source queries (Phase 1) |
| `generation.rollouts_per_query` | `8` | Completions sampled per query |
| `generation.max_pairs_per_query` | `3` | `(chosen, rejected)` pairs per query (Phase 3.1) |
| `generation.temperature` | `0.8` | Sampling temperature for rollouts |

Key `grpo.yaml` settings (Stage 2B):

| Key | Default | Description |
|-----|---------|-------------|
| `training.num_generations` | `4` | Group size G (completions per prompt; 24 GB-safe) |
| `training.irrelevance_fraction` | `0.25` | Fraction of abstention rollout prompts (Phase 1) |
| `training.reward_shaping` | `true` | Shaped partial-credit reward vs binary (Phase 2) |
| `training.beta` | `0.04` | KL coefficient to the reference policy |
| `training.per_device_batch_size` | `4` | Must be a multiple of `num_generations` |
| `training.learning_rate` | `1e-6` | Tiny LR for RL fine-tuning |
| `training.use_vllm` | `false` | Set `true` on GPU with vLLM for fast rollouts |

CLI flags (sub-command specific):

| Flag | Sub-commands | Description |
|------|-------------|-------------|
| `--device {auto,cuda,mps,cpu}` | all | Override compute device |
| `--smoke` | all | Tiny fast run for validation, not a real experiment |
| `--max-eval-samples N` | baseline, sft, dpo, grpo | Cap eval samples per BFCL category |
| `--categories ...` | baseline, sft, dpo, grpo | Restrict eval to a subset of BFCL categories |
| `--max-samples N` | sft | Override training sample count |
| `--epochs N` | sft, dpo | Override number of training epochs |
| `--batch-size N` | sft, dpo, grpo | Per-device batch size (lower first on CUDA OOM) |
| `--beta B` | dpo | DPO KL strength (Phase 3.2 sweep) |
| `--skip-eval` / `--run-eval` | sft, dpo, grpo | Skip / force BFCL eval after training |
| `--checkpoint PATH` | evaluate | LoRA checkpoint to score (omit for the base model) |
| `--sft-checkpoint PATH` | generate-pairs, dpo, grpo | SFT adapter to start from |
| `--from-base` | generate-pairs, dpo, grpo | Start from base model when no SFT checkpoint exists |
| `--pairs-path PATH` | dpo | Preference pairs JSONL to train on |
| `--num-queries N` / `--rollouts N` | generate-pairs | Source queries / completions per query |
| `--max-pairs-per-query N` | generate-pairs | Pairs extracted per query (Phase 3.1) |
| `--output-path PATH` | generate-pairs | Where to write the pairs JSONL |

Environment variables (set in `.env` or shell):

| Variable | Required for | Description |
|----------|-------------|-------------|
| `HF_TOKEN` | Stage 1 SFT | Read token for gated xLAM-60K dataset |
| `WANDB_API_KEY` | Optional | Enables W&B logging when `wandb.enabled=true` |

---

## Troubleshooting: CUDA out of memory

The defaults target a single **24 GB** GPU. If you see `torch.OutOfMemoryError: CUDA out of memory`, apply these in order (each trades speed or effective batch size for memory):

1. **Lower the per-device batch size** — the fastest lever:
   ```bash
   python scripts/run_pipeline.py sft --batch-size 1
   ```
2. **Reduce sequence length** in `configs/sft.yaml` (`data.max_seq_len: 768` or `512`). Shorter sequences cut both activation and logit memory; a few long examples get dropped.
3. **Ensure gradient checkpointing is on** (it is by default in `sft.yaml`). Do not disable it on ≤24 GB cards.
4. **Reduce the fragmentation overhead** the error message itself suggests:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
5. **Check the GPU isn't already occupied** by a stale process: `nvidia-smi`. On a shared node another job may be holding most of the card.

Defaults that keep Stage 1 inside 24 GB: `per_device_batch_size=2`, `gradient_accumulation_steps=8` (effective batch 16), `gradient_checkpointing=true`, `max_seq_len=1024`. To go faster on a 40/80 GB card, raise `per_device_batch_size` and optionally set `gradient_checkpointing: false`.

---

## BFCL Evaluation Categories

| Category | Description |
|----------|-------------|
| `simple` | One query → one function call |
| `multiple` | Choose the correct function from several candidates |
| `parallel` | One query → multiple simultaneous calls |
| `irrelevance` | Recognise when no tool applies |

Results are reported as per-category accuracy with failure breakdowns:
`no_tool_call`, `wrong_function`, `missing_argument`, `wrong_argument_type`, `extra_tool_call`.

### Grading rules (`pipeline/reward/bfcl_grader.py`)

A prediction is correct only when every expected call is matched. The grader follows official BFCL semantics on two points that materially affect scores:

- **Optional arguments may be omitted.** BFCL marks an argument optional by including an empty string `""` in its acceptable-value list (e.g. `"unit": ["units", ""]`). Omitting such an argument is **not** a `missing_argument` failure; only required arguments are enforced. If the model *does* supply an optional argument, the value must still be acceptable.
- **Parallel calls are matched order-independently.** For multi-call (parallel) examples, each expected call is greedily matched to any predicted call that satisfies it, so a correct set of calls in a different order still scores as correct.

Argument values are compared after light type coercion (e.g. the string `"30"` matches the integer `30`, `"true"` matches `True`).

---

## How the Stages Compose

Every stage reuses the same building blocks, which is what keeps train-time and eval-time behaviour consistent:

| Building block | Used by |
|---|---|
| `chat_template.format_inference_prompt` | SFT formatting, evaluation, pair generation, GRPO prompts |
| `bfcl_grader.grade` (binary) | Stage 0/1 evaluation, pair-generation grading |
| `bfcl_grader.score` (shaped) | GRPO training reward (Phase 2) |
| `BaseTrainer` | SFT, DPO, and GRPO trainers (each overrides only `_run_training`) |
| Dataset registry | `xlam_sft`, `xlam_dpo`, `xlam_grpo` — added without touching trainers |
| `loader.load_model_from_checkpoint(..., is_trainable=True)` | DPO and GRPO load the SFT adapter as the policy |

**Adding another preference algorithm** (e.g. KTO, IPO, ORPO, SimPO) is a localised change:

1. Add `configs/<algo>.yaml` with the algorithm's knobs.
2. Add `pipeline/training/<algo>_trainer.py` subclassing `BaseTrainer` and overriding `_run_training` (most DPO-family losses are just a different `loss_type` on TRL's `DPOConfig`, so it can often reuse the DPO data path).
3. Add a `run_<algo>` function and an entry in the `COMMAND_DISPATCH` table in `scripts/run_pipeline.py`.

No existing stage needs to change.
