from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from pipeline.model.lora_config import build_lora_config

logger = logging.getLogger(__name__)


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device; pass through explicit choices."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(cfg: DictConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: DictConfig) -> PreTrainedModel:
    """Load the base model weights without any adapter."""
    device = resolve_device(cfg.model.device)
    torch_dtype = getattr(torch, cfg.model.torch_dtype)

    load_in_4bit = cfg.model.get("load_in_4bit", False)
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to full precision.")
            load_in_4bit = False

    if not load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path,
            dtype=torch_dtype,
        )
        if device != "cpu":
            model = model.to(device)

    logger.info(f"Base model loaded: {cfg.model.name_or_path} | device={device} | dtype={torch_dtype}")
    return model


def load_model(
    cfg: DictConfig, apply_lora: bool = True
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer, optionally wrapping with fresh LoRA adapters.
    Use apply_lora=False for Stage 0 baseline inference.
    """
    model = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)

    if apply_lora:
        lora_config = build_lora_config(cfg.lora)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_model_from_checkpoint(
    cfg: DictConfig, adapter_path: str | Path
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and attach a previously saved LoRA adapter checkpoint."""
    base_model = load_base_model(cfg)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = load_tokenizer(cfg)
    logger.info(f"LoRA adapter loaded from: {adapter_path}")
    return model, tokenizer


def merge_lora_and_save(
    cfg: DictConfig, adapter_path: str | Path, output_path: str | Path
) -> None:
    """
    Merge LoRA weights into the base model and save a full-weight checkpoint.
    Required before vLLM inference for Stage 2 preference pair generation.
    """
    model, tokenizer = load_model_from_checkpoint(cfg, adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info(f"Merged model saved to: {output_path}")
