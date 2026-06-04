#!/usr/bin/env python3
"""
Live tool-calling demo: Base vs post-trained models, side by side.

Loads the base Qwen2.5-1.5B-Instruct once and attaches the Stage-1/2 LoRA
adapters on top of it (no extra copies of the weights), then runs a curated set
of function-calling prompts through each variant and prints the tool call each
one produces, graded against the ground truth.

The headline example is the irrelevance case: a query where no available tool
fits. Watch the base model abstain, and watch how post-training shifts that
behaviour.

Examples
--------
    # Auto-detect adapters under outputs/ and compare Base vs SFT vs DPO vs GRPO
    python scripts/demo.py --device cuda

    # Compare specific checkpoints, with your own labels
    python scripts/demo.py --device cuda \
        --checkpoint SFT=outputs/sft/checkpoint-final \
        --checkpoint DPO=outputs/dpo/checkpoint-final

    # Base model only (no adapters needed)
    python scripts/demo.py --device mps --base-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent.parent / ".env")

logging.disable(logging.INFO)  # keep the demo output clean

import torch
from omegaconf import OmegaConf
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pipeline.model.loader import load_base_model, load_tokenizer, resolve_device
from pipeline.formatting.chat_template import format_inference_prompt, extract_tool_calls
from pipeline.reward.bfcl_grader import grade

console = Console()
PURPLE = "medium_purple3"
GOLD = "dark_goldenrod"


# ---------------------------------------------------------------------------
# Curated demo prompts (self-contained: query, tools, ground-truth answer)
# ---------------------------------------------------------------------------

def _tool(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}

DEMOS = [
    {
        "category": "Simple",
        "note": "one query, one function",
        "query": "Find the area of a triangle with a base of 10 and a height of 5.",
        "tools": [_tool("calculate_triangle_area", "Area of a triangle from base and height.",
                        {"base": {"type": "integer"}, "height": {"type": "integer"},
                         "unit": {"type": "string", "description": "optional unit"}},
                        ["base", "height"])],
        "expected": [{"name": "calculate_triangle_area",
                      "arguments": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    },
    {
        "category": "Function selection",
        "note": "pick the right one of several tools",
        "query": "Reverse the string 'hello'.",
        "tools": [_tool("reverse_string", "Reverse the characters in a string.",
                        {"text": {"type": "string"}}, ["text"]),
                  _tool("to_uppercase", "Uppercase a string.",
                        {"text": {"type": "string"}}, ["text"])],
        "expected": [{"name": "reverse_string", "arguments": {"text": ["hello"]}}],
    },
    {
        "category": "Parallel",
        "note": "multiple calls from one query",
        "query": "Get the current weather in both London and Tokyo.",
        "tools": [_tool("get_current_weather", "Current weather for a city.",
                        {"city": {"type": "string"}}, ["city"])],
        "expected": [{"name": "get_current_weather", "arguments": {"city": ["London"]}},
                     {"name": "get_current_weather", "arguments": {"city": ["Tokyo"]}}],
    },
    {
        "category": "Irrelevance  (the headline)",
        "note": "no tool fits, so the model should abstain",
        "query": "Tell me a fun fact about outer space.",
        "tools": [_tool("get_current_weather", "Current weather for a city.",
                        {"city": {"type": "string"}}, ["city"]),
                  _tool("set_timer", "Start a countdown timer.",
                        {"minutes": {"type": "integer"}}, ["minutes"])],
        "expected": [],  # correct behaviour: no tool call
    },
]


# ---------------------------------------------------------------------------
# Model wrapper: one base, many adapters, instant switching
# ---------------------------------------------------------------------------

class VariantRunner:
    """Holds the base model + named LoRA adapters; switches between them cheaply."""

    def __init__(self, cfg, adapter_specs):
        self.device = resolve_device(cfg.model.device)
        self.tokenizer = load_tokenizer(cfg)
        self.max_new_tokens = cfg.evaluation.get("max_new_tokens", 200)
        base = load_base_model(cfg)

        self.adapter_names = [name for name, _ in adapter_specs]
        self.model = base
        for i, (name, path) in enumerate(adapter_specs):
            if i == 0:
                self.model = PeftModel.from_pretrained(base, str(path), adapter_name=name)
            else:
                self.model.load_adapter(str(path), adapter_name=name)
        if self.adapter_names and self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        # Display order: Base first, then each adapter
        self.variants = ["Base"] + self.adapter_names

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def run(self, variant: str, prompt: str) -> str:
        if variant == "Base":
            if not self.adapter_names:
                return self._generate(prompt)
            with self.model.disable_adapter():
                return self._generate(prompt)
        self.model.set_adapter(variant)
        return self._generate(prompt)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_calls(calls: list[dict]) -> str:
    if not calls:
        return "[grey58](no tool call · abstains)[/]"
    parts = []
    for c in calls:
        name = c.get("name", "?")
        args = c.get("arguments", c.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        arg_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
        parts.append(f"{name}({arg_str})")
    return "   +   ".join(parts)


def verdict(output: str, expected: list[dict]) -> str:
    r = grade(output, expected)
    if r.correct:
        return "[bold green]correct[/]"
    return f"[bold red]{r.failure_category}[/]"


def render_demo(runner: VariantRunner, demo: dict, index: int, total: int) -> None:
    tool_names = ", ".join(t["name"] for t in demo["tools"])
    header = (f"[bold white]{demo['query']}[/]\n"
              f"[grey62]available tools:[/] [italic]{tool_names}[/]   "
              f"[grey50]·  {demo['note']}[/]")
    console.print()
    console.print(Panel(header, title=f"[bold {GOLD}]{index}/{total}  ·  {demo['category']}[/]",
                        title_align="left", border_style=PURPLE, box=box.ROUNDED, padding=(0, 2)))

    table = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {PURPLE}", expand=True, show_lines=False)
    table.add_column("Model", style="bold", width=8)
    table.add_column("Tool call produced", ratio=3)
    table.add_column("Verdict", width=20)
    for variant in runner.variants:
        output = runner.run(variant, format_inference_prompt(demo["query"], demo["tools"], runner.tokenizer))
        calls = extract_tool_calls(output)
        style = GOLD if variant == "Base" else "white"
        table.add_row(f"[{style}]{variant}[/]", format_calls(calls), verdict(output, demo["expected"]))
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_detect_adapters() -> list[tuple[str, Path]]:
    found = []
    for label, rel in [("SFT", "outputs/sft/checkpoint-final"),
                       ("DPO", "outputs/dpo/checkpoint-final"),
                       ("GRPO", "outputs/grpo/checkpoint-final")]:
        p = Path(rel)
        if (p / "adapter_config.json").exists():
            found.append((label, p))
    return found


def _parse_checkpoint(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    p = Path(spec)
    return p.parent.name, p  # fall back to the parent dir name as the label


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Base-vs-trained tool-calling demo.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="Adapter to compare (repeatable). Default: auto-detect under outputs/.")
    parser.add_argument("--base-only", action="store_true", help="Run only the base model (no adapters).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device != "auto":
        cfg.model.device = args.device

    if args.base_only:
        adapters = []
    elif args.checkpoint:
        adapters = [_parse_checkpoint(s) for s in args.checkpoint]
    else:
        adapters = _auto_detect_adapters()

    device = resolve_device(cfg.model.device)
    console.print()
    console.rule(f"[bold {PURPLE}]Agentic function-calling demo[/]  ·  Qwen2.5-1.5B  ·  device: {device}",
                 style=GOLD)
    if adapters:
        console.print(f"[grey62]Comparing:[/] Base  +  " + "  ·  ".join(n for n, _ in adapters))
    else:
        console.print("[grey62]Running the base model only (no adapters loaded).[/]")
    console.print("[grey50]Loading model and adapters…[/]")

    runner = VariantRunner(cfg, adapters)

    for i, demo in enumerate(DEMOS, 1):
        render_demo(runner, demo, i, len(DEMOS))

    console.print()
    console.rule(style=GOLD)
    console.print("[grey62]Note the irrelevance row: the right answer is to call "
                  "no tool at all. Post-training on all-positive data erodes that, "
                  "which Phase 1 abstention data restores.[/]\n")


if __name__ == "__main__":
    main()
