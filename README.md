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

### Stage 2A — Direct Preference Optimisation (DPO) *(planned)*
Generate preference pairs by sampling 8 rollouts per query from the SFT model, grading each with the BFCL verifier, then training offline on (chosen, rejected) pairs.

### Stage 2B — Group Relative Policy Optimisation (GRPO) *(planned)*
Online RL using BFCL-style verifiable rewards. Group-relative advantages replace a value function, following the DeepSeek-R1 approach at 1.5B scale.

---

## Repo Structure

```
slm-agentic-post-training-pipeline/
├── configs/
│   ├── base.yaml          # shared defaults (model, LoRA, evaluation)
│   └── sft.yaml           # Stage 1 training hyperparameters
│
├── pipeline/
│   ├── model/
│   │   ├── loader.py      # load model + tokenizer; apply/merge LoRA
│   │   └── lora_config.py # build LoraConfig from YAML
│   │
│   ├── data/
│   │   ├── registry.py    # DatasetRegistry — name → (load_fn, format_fn)
│   │   ├── xlam.py        # xLAM-60K loader + SFT formatter (Stage 1)
│   │   └── bfcl.py        # BFCL test-set loader + example parser (eval)
│   │
│   ├── formatting/
│   │   └── chat_template.py  # Qwen2.5 tool-call prompt builder; shared by
│   │                         # SFT formatting, inference, and Stage 2 reward
│   │
│   ├── reward/
│   │   └── bfcl_grader.py    # verifiable grader: name + args + types
│   │                         # returns GradeResult with failure_category
│   │
│   ├── training/
│   │   ├── base_trainer.py   # abstract: config persistence, checkpointing,
│   │   │                     # eval hooks — shared by all stages
│   │   └── sft_trainer.py    # Stage 1: wraps TRL SFTTrainer
│   │
│   └── evaluation/
│       ├── evaluator.py      # runs BFCL categories, returns EvalSummary
│       └── metrics.py        # CategoryMetrics, EvalSummary, aggregation
│
├── scripts/
│   └── run_pipeline.py    # single CLI entry point (baseline + sft sub-commands)
│
├── tests/
│   ├── test_formatting.py # prompt builder + tool-call extraction
│   ├── test_grader.py     # all grader failure modes + type coercion
│   └── test_data.py       # registry, xLAM parser, BFCL parser
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

**Requirements:** Python 3.10+, PyTorch 2.3+

```bash
git clone https://github.com/break-through-19/slm-agentic-post-training-pipeline.git
cd slm-agentic-post-training-pipeline
pip install -e ".[dev]"

# With 4-bit quantisation (CUDA + Linux only)
pip install -e ".[dev,quant]"
```

---

## Running the Pipeline

All commands use a single script with two sub-commands: `baseline` and `sft`.

### Stage 0 — Baseline Evaluation

```bash
# Auto-detect best device (CUDA → MPS → CPU)
python scripts/run_pipeline.py baseline

# Apple Silicon (M1/M2/M3)
python scripts/run_pipeline.py baseline --device mps

# Specific categories only
python scripts/run_pipeline.py baseline --categories simple irrelevance

# Cap samples per category (faster dev run)
python scripts/run_pipeline.py baseline --max-eval-samples 100

# Smoke test — 10 samples, float32, CPU (< 1 minute, no GPU needed)
python scripts/run_pipeline.py baseline --smoke
```

Results are written to `outputs/baseline_results.json`.

### Stage 1 — Supervised Fine-Tuning

```bash
# Full run (15K samples, 3 epochs, auto device)
python scripts/run_pipeline.py sft

# Apple Silicon
python scripts/run_pipeline.py sft --device mps

# Override training sample count
python scripts/run_pipeline.py sft --max-samples 5000

# Skip BFCL eval after training
python scripts/run_pipeline.py sft --skip-eval

# Smoke test — 64 samples, 1 epoch, 10 eval samples per category
python scripts/run_pipeline.py sft --smoke

# Smoke test on Apple Silicon
python scripts/run_pipeline.py sft --smoke --device mps
```

Checkpoints and BFCL results are written to `outputs/sft/`.

### Running Tests

```bash
# Full suite (no GPU required — uses a mock tokenizer)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=pipeline --cov-report=term-missing
```

---

## Configuration

All hyperparameters live in `configs/`. The SFT script merges `base.yaml` and `sft.yaml` at runtime.

Key `base.yaml` settings:

| Key | Default | Description |
|-----|---------|-------------|
| `model.name_or_path` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `model.device` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `model.torch_dtype` | `bfloat16` | `bfloat16` \| `float16` \| `float32` |
| `model.load_in_4bit` | `false` | 4-bit NF4 quant (CUDA + bitsandbytes only) |
| `lora.r` | `16` | LoRA rank |
| `evaluation.max_eval_samples` | `null` | Per-category cap (`null` = full set) |

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

---

## Extending to Stage 2

The baseline is designed so Stage 2 adds files without modifying existing ones:

| Component | What to create |
|---|---|
| Preference pair dataset | `pipeline/data/preference.py` — registers `"xlam_dpo"` in the registry |
| DPO trainer | `pipeline/training/dpo_trainer.py` — overrides `BaseTrainer._run_training()` |
| GRPO trainer | `pipeline/training/grpo_trainer.py` — overrides `_run_training()`, calls `bfcl_grader.grade()` as reward |
| Pair generation script | `scripts/generate_pairs.py` — uses `merge_lora_and_save()` + `format_inference_prompt()` + `grade()` |
