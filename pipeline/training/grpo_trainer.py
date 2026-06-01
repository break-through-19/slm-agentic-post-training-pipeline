"""
Stage 2B — Group Relative Policy Optimisation trainer.

Wraps TRL's GRPOTrainer behind the same BaseTrainer contract as SFT and DPO.

GRPO is on-policy: for every prompt it samples a group of `num_generations`
completions, scores each with the verifiable BFCL reward, and shifts the policy
toward the above-average completions in the group. Group-relative advantages
remove the need for a value network (DeepSeek-R1's recipe).

The policy is the Stage 1 SFT adapter loaded trainable; the KL reference is the
adapter-disabled base model (LoRA makes this free). The reward function is
supplied by the caller and is the exact same BFCL grader used at eval time.
"""
from __future__ import annotations

import logging
from typing import Callable

from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

from pipeline.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        reward_fn: Callable,
        eval_fn: Callable[[PreTrainedModel, PreTrainedTokenizer], dict] | None = None,
    ) -> None:
        super().__init__(cfg, model, tokenizer, eval_fn)
        self.train_dataset = train_dataset
        self.reward_fn = reward_fn

    def _run_training(self) -> None:
        train_cfg = self.cfg.training
        model_cfg = self.cfg.model

        use_bf16 = model_cfg.torch_dtype == "bfloat16"
        use_fp16 = model_cfg.torch_dtype == "float16"

        from pipeline.model.loader import resolve_device
        effective_device = resolve_device(model_cfg.device)
        use_pin_memory = effective_device not in ("mps", "cpu")

        report_to = (
            ["wandb"] if self.cfg.get("wandb", {}).get("enabled", False) else ["none"]
        )

        max_steps = int(train_cfg.get("max_steps", -1))
        gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))

        grpo_config = GRPOConfig(
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
            # GRPO-specific knobs
            beta=train_cfg.get("beta", 0.04),
            num_generations=train_cfg.get("num_generations", 8),
            max_completion_length=train_cfg.get("max_completion_length", 256),
            temperature=train_cfg.get("temperature", 0.9),
            scale_rewards=train_cfg.get("scale_rewards", True),
            use_vllm=train_cfg.get("use_vllm", False),
        )

        if gradient_checkpointing and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        trl_trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_fn,
            args=grpo_config,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting GRPO training — output dir: %s", self.output_dir)
        train_result = trl_trainer.train()
        logger.info(
            "GRPO training complete — loss=%.4f, steps=%d",
            train_result.training_loss,
            train_result.global_step,
        )

        self.model = trl_trainer.model
