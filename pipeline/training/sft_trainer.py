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

        report_to = (
            ["wandb"] if self.cfg.get("wandb", {}).get("enabled", False) else ["none"]
        )

        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=train_cfg.num_epochs,
            per_device_train_batch_size=train_cfg.per_device_batch_size,
            per_device_eval_batch_size=train_cfg.get(
                "per_device_eval_batch_size", train_cfg.per_device_batch_size
            ),
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.learning_rate,
            warmup_ratio=train_cfg.warmup_ratio,
            lr_scheduler_type=train_cfg.lr_scheduler,
            weight_decay=train_cfg.weight_decay,
            max_grad_norm=train_cfg.max_grad_norm,
            logging_steps=self.cfg.output.log_steps,
            save_steps=train_cfg.save_steps,
            eval_strategy="steps" if self.eval_dataset is not None else "no",
            eval_steps=train_cfg.eval_steps if self.eval_dataset is not None else None,
            save_strategy="steps",
            load_best_model_at_end=self.eval_dataset is not None,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=report_to,
            max_seq_length=self.cfg.data.max_seq_len,
            # Dataset is already tokenised; skip TRL's internal prepare step
            dataset_kwargs={"skip_prepare_dataset": True},
        )

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
