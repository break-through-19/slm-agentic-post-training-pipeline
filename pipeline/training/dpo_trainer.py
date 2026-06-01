"""
Stage 2A — Direct Preference Optimisation trainer.

Wraps TRL's DPOTrainer behind the same BaseTrainer contract used by SFT, so
config persistence, checkpointing, and the post-training BFCL eval hook all
behave identically across stages.

The policy is the Stage 1 SFT adapter, loaded trainable. Because LoRA is used,
no separate reference model is materialised: passing ref_model=None lets TRL
use the same network with the adapter disabled as the frozen reference, which
is both correct and memory-free.

The dataset arrives in TRL's "standard" preference format with prompt / chosen
/ rejected string columns; TRL tokenises them internally.
"""
from __future__ import annotations

import logging
from typing import Callable

from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer

from pipeline.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
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

        # pin_memory is unsupported on MPS and adds overhead — disable off-CUDA
        from pipeline.model.loader import resolve_device
        effective_device = resolve_device(model_cfg.device)
        use_pin_memory = effective_device not in ("mps", "cpu")

        report_to = (
            ["wandb"] if self.cfg.get("wandb", {}).get("enabled", False) else ["none"]
        )

        max_steps = int(train_cfg.get("max_steps", -1))
        gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))

        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=train_cfg.num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=train_cfg.per_device_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if gradient_checkpointing else None
            ),
            learning_rate=train_cfg.learning_rate,
            warmup_steps=0,
            lr_scheduler_type=train_cfg.lr_scheduler,
            weight_decay=train_cfg.weight_decay,
            max_grad_norm=train_cfg.max_grad_norm,
            logging_steps=self.cfg.output.log_steps,
            save_steps=train_cfg.save_steps,
            save_strategy="steps",
            bf16=use_bf16,
            fp16=use_fp16,
            dataloader_pin_memory=use_pin_memory,
            report_to=report_to,
            max_length=self.cfg.data.max_seq_len,
            # DPO-specific objective knobs
            beta=train_cfg.get("beta", 0.1),
            loss_type=train_cfg.get("loss_type", "sigmoid"),
        )

        # PEFT + gradient checkpointing needs grad-enabled input embeddings so
        # the checkpointed graph can backprop into the LoRA adapters.
        if gradient_checkpointing and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        trl_trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,                 # LoRA: reference = adapter-disabled base
            args=dpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting DPO training — output dir: %s", self.output_dir)
        train_result = trl_trainer.train()
        logger.info(
            "DPO training complete — loss=%.4f, steps=%d",
            train_result.training_loss,
            train_result.global_step,
        )

        # Hand the trained model back so BaseTrainer can checkpoint it
        self.model = trl_trainer.model
