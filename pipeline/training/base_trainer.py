"""
Abstract base trainer shared by all post-training stages.

Concrete trainers (SFTTrainer, DPOTrainer, GRPOTrainer) only override
_run_training().  Everything else — config persistence, checkpointing,
eval scheduling, W&B integration — is inherited from here.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        eval_fn: Callable[[PreTrainedModel, PreTrainedTokenizer], dict] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.output_dir = Path(cfg.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training cycle and return eval results (if any)."""
        self._persist_config()
        self._run_training()
        self._save_checkpoint("final")

        eval_results: dict = {}
        if self.eval_fn is not None:
            logger.info("Running post-training BFCL evaluation…")
            eval_results = self._run_eval()

        return eval_results

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _persist_config(self) -> None:
        config_path = self.output_dir / "train_config.yaml"
        OmegaConf.save(self.cfg, config_path)
        logger.info("Training config saved to %s", config_path)

    def _save_checkpoint(self, tag: str) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info("Checkpoint saved: %s", checkpoint_dir)

    def _run_eval(self) -> dict:
        if self.eval_fn is None:
            return {}
        results = self.eval_fn(self.model, self.tokenizer)
        results_path = self.output_dir / "eval_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        logger.info("Eval results written to %s", results_path)
        return results

    # ------------------------------------------------------------------
    # Abstract — implemented by each stage-specific trainer
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_training(self) -> None:
        """Execute the stage-specific training loop."""
        ...
