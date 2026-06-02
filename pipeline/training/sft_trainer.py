"""
Stage 1 — Supervised Fine-Tuning trainer.

Wraps TRL's SFTTrainer so that our BaseTrainer contract (config persistence,
checkpointing, eval hooks) is honoured while TRL handles the inner training loop,
gradient accumulation, LR scheduling, and HuggingFace Trainer integrations.

The dataset is expected to arrive pre-tokenised with input_ids / labels /
attention_mask columns (produced by pipeline.data.xlam._format_xlam_for_sft).
TRL's dataset preparation step is skipped accordingly.
"""
from __future__ import annotations

import logging
from typing import Callable

from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer

from pipeline.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        eval_fn: Callable[[PreTrainedModel, PreTrainedTokenizer], dict] | None = None,
    ) -> None:
        super().__init__(cfg, model, tokenizer, eval_fn)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _run_training(self) -> None:
        train_cfg = self.cfg.training
        model_cfg = self.cfg.model

        use_bf16 = model_cfg.torch_dtype == "bfloat16"
        use_fp16 = model_cfg.torch_dtype == "float16"

        # pin_memory is not supported on MPS and adds overhead — disable it
        from pipeline.model.loader import resolve_device
        effective_device = resolve_device(model_cfg.device)
        use_pin_memory = effective_device not in ("mps", "cpu")

        report_to = (
            ["wandb"] if self.cfg.get("wandb", {}).get("enabled", False) else ["none"]
        )

        # max_steps=-1 means "use num_epochs"; a positive value caps training
        # regardless of epochs (used by smoke mode to guarantee a short run)
        max_steps = int(train_cfg.get("max_steps", -1))

        # Gradient checkpointing trades extra forward compute for ~50% less
        # activation memory — essential for fitting a 1.5B model on MPS.
        gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))

        has_eval = self.eval_dataset is not None

        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=train_cfg.num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=train_cfg.per_device_batch_size,
            per_device_eval_batch_size=train_cfg.get(
                "per_device_eval_batch_size", train_cfg.per_device_batch_size
            ),
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            # use_reentrant=False is required by PEFT for correct grad flow
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            learning_rate=train_cfg.learning_rate,
            warmup_steps=0,                    # warmup_ratio deprecated in TRL 1.x
            lr_scheduler_type=train_cfg.lr_scheduler,
            weight_decay=train_cfg.weight_decay,
            max_grad_norm=train_cfg.max_grad_norm,
            logging_steps=self.cfg.output.log_steps,
            save_steps=train_cfg.save_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=train_cfg.eval_steps if has_eval else None,
            save_strategy="steps",
            load_best_model_at_end=has_eval,
            # IMPORTANT: rank the best checkpoint by token accuracy, NOT eval_loss.
            # SFT eval batches occasionally contain an example whose only learnable
            # tokens land on the truncation boundary; after the causal shift that
            # batch has zero valid targets, so the language-modeling eval_loss comes
            # back NaN. With the default metric ("loss"), NaN poisons best-model
            # tracking and transformers silently keeps the FIRST checkpoint —
            # loading a badly undertrained model at the end. eval_mean_token_accuracy
            # is always finite and is the meaningful SFT signal anyway.
            metric_for_best_model="eval_mean_token_accuracy" if has_eval else None,
            greater_is_better=True if has_eval else None,
            bf16=use_bf16,
            fp16=use_fp16,
            dataloader_pin_memory=use_pin_memory,
            report_to=report_to,
            max_length=self.cfg.data.max_seq_len,  # renamed in TRL >= 1.0
            # Dataset is already tokenised; skip TRL's internal prepare step
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        # When using gradient checkpointing with PEFT/LoRA, the base model's
        # input embeddings must produce tensors that require_grad=True so the
        # checkpointed graph can backprop through to the LoRA adapters.
        if gradient_checkpointing and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        trl_trainer = TRLSFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting SFT training — output dir: %s", self.output_dir)
        train_result = trl_trainer.train()
        logger.info(
            "SFT training complete — loss=%.4f, steps=%d",
            train_result.training_loss,
            train_result.global_step,
        )

        # Expose the trained model back on self so BaseTrainer can checkpoint it
        self.model = trl_trainer.model
