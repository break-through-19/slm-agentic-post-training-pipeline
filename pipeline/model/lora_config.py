from omegaconf import DictConfig
from peft import LoraConfig, TaskType


def build_lora_config(lora_cfg: DictConfig) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
