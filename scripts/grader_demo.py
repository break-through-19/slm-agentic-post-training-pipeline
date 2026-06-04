#!/usr/bin/env python3
"""
Grader & reward demo: no GPU, no model, runs instantly on any laptop.

Demonstrates the two functions at the heart of the pipeline:
  - bfcl_grader.grade() : the binary, verifiable BFCL metric (correct / failure)
  - bfcl_grader.score() : the shaped, partial-credit reward used to train GRPO

This is the safety-net demo: it has zero dependency on the GPU, the network, or
any downloaded weights, yet it still tells a core part of the story: how a
verifiable reward works and why partial credit (not binary) gives GRPO a
learning signal.

    python scripts/grader_demo.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pipeline.formatting.chat_template import TOOL_CALL_CLOSE_TAG, TOOL_CALL_OPEN_TAG
from pipeline.reward.bfcl_grader import grade, score

console = Console()

PURPLE = "medium_purple3"
GOLD = "dark_goldenrod"


# ---------------------------------------------------------------------------
# Helpers to build illustrative model outputs
# ---------------------------------------------------------------------------

def tool_call(name: str, arguments: dict) -> str:
    """Render a <tool_call> block exactly as the model would emit one."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN_TAG}\n{payload}\n{TOOL_CALL_CLOSE_TAG}"


def verdict(result) -> str:
    if result.correct:
        return "[bold green]correct[/]"
    return f"[bold red]{result.failure_category}[/]"


def section(title: str, subtitle: str) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold white]{title}[/]\n[grey70]{subtitle}[/]",
        border_style=GOLD, box=box.ROUNDED, padding=(0, 2)))


# ---------------------------------------------------------------------------
# A: verifiable grading: one ground truth, several model outputs
# ---------------------------------------------------------------------------

def demo_verifiable_grading() -> None:
    section("1 · Verifiable grading",
            "One query, one ground truth. The grader labels each output correct, "
            "or assigns a precise failure category.")
    ground_truth = [{"name": "get_weather", "arguments": {"city": ["Paris"]}}]
    console.print(f"   [grey62]ground truth:[/] [italic]get_weather(city=\"Paris\")[/]\n")

    cases = [
        ("Correct call", tool_call("get_weather", {"city": "Paris"})),
        ("Wrong argument value", tool_call("get_weather", {"city": "London"})),
        ("Wrong function", tool_call("get_forecast", {"city": "Paris"})),
        ("No tool call at all", "Sure, I can help you with the weather."),
        ("Malformed (non-object)", f"{TOOL_CALL_OPEN_TAG}\"Paris weather\"{TOOL_CALL_CLOSE_TAG}"),
    ]
    table = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {PURPLE}", expand=True)
    table.add_column("Model output", style="white", ratio=3)
    table.add_column("Reward", justify="center", width=8)
    table.add_column("Verdict", justify="left", width=22)
    for label, output in cases:
        r = grade(output, ground_truth)
        table.add_row(label, f"{r.reward:.1f}", verdict(r))
    console.print(table)


# ---------------------------------------------------------------------------
# B: shaped reward creates the variance GRPO needs
# ---------------------------------------------------------------------------

def demo_shaped_reward() -> None:
    section("2 · Shaped reward gives GRPO a learning signal",
            "When no completion in a group is fully correct (the common case "
            "that stalled GRPO), a binary reward gives every one a 0, zero "
            "variance, no gradient. Partial credit still ranks them.")
    ground_truth = [{"name": "get_weather",
                     "arguments": {"city": ["Paris"], "days": [3], "units": ["celsius"]}}]
    console.print("   [grey62]ground truth:[/] "
                  "[italic]get_weather(city=\"Paris\", days=3, units=\"celsius\")[/]\n")

    group = [
        ("Right name, 2 of 3 args", tool_call("get_weather", {"city": "Paris", "days": 3, "units": "fahrenheit"})),
        ("Right name, 1 of 3 args", tool_call("get_weather", {"city": "Paris", "days": 9, "units": "fahrenheit"})),
        ("Right name, 0 of 3 args", tool_call("get_weather", {"city": "London", "days": 9, "units": "fahrenheit"})),
        ("Wrong function", tool_call("lookup", {"q": "Paris"})),
    ]
    binary = [grade(o, ground_truth).reward for _, o in group]
    shaped = [score(o, ground_truth) for _, o in group]

    table = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {PURPLE}", expand=True)
    table.add_column("Completion in the group", style="white", ratio=3)
    table.add_column("Binary reward", justify="center", width=14)
    table.add_column("Shaped reward", justify="center", width=14)
    for (label, _), b, sc in zip(group, binary, shaped):
        table.add_row(label, f"[grey50]{b:.2f}[/]", f"[bold green]{sc:.2f}[/]")
    console.print(table)

    b_std = statistics.pstdev(binary)
    s_std = statistics.pstdev(shaped)
    console.print(
        f"\n   within-group spread (std):   "
        f"binary [grey50]{b_std:.3f}[/]   vs   shaped [bold green]{s_std:.3f}[/]")
    console.print(
        "   [italic grey70]Zero binary variance means zero advantage and no "
        "gradient. The shaped reward keeps the signal alive.[/]")


# ---------------------------------------------------------------------------
# C: irrelevance stays binary (abstention is all-or-nothing)
# ---------------------------------------------------------------------------

def demo_irrelevance() -> None:
    section("3 · Irrelevance is graded as abstention",
            "When no tool fits, the correct behaviour is to call nothing at all.")
    ground_truth: list = []  # empty = irrelevance
    console.print("   [grey62]ground truth:[/] [italic]no tool should be called[/]\n")

    cases = [
        ("Abstains (plain text reply)", "None of the available tools can answer that."),
        ("Calls a tool anyway", tool_call("get_weather", {"city": "Paris"})),
    ]
    table = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {PURPLE}", expand=True)
    table.add_column("Model output", style="white", ratio=3)
    table.add_column("Reward", justify="center", width=8)
    table.add_column("Verdict", justify="left", width=22)
    for label, output in cases:
        r = grade(output, ground_truth)
        table.add_row(label, f"{r.reward:.1f}", verdict(r))
    console.print(table)


# ---------------------------------------------------------------------------
# D: optional arguments may be omitted (BFCL fidelity, Phase 0)
# ---------------------------------------------------------------------------

def demo_optional_arguments() -> None:
    section("4 · Optional arguments may be omitted",
            "BFCL marks an argument optional with an empty acceptable value. "
            "Omitting it is correct; this Phase 0 fix lifted every score.")
    ground_truth = [{
        "name": "calculate_triangle_area",
        "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
    }]
    console.print("   [grey62]ground truth:[/] [italic]base=10, height=5, "
                  "unit is optional (acceptable values: \"units\" or omitted)[/]\n")

    cases = [
        ("Omits the optional 'unit'", tool_call("calculate_triangle_area", {"base": 10, "height": 5})),
        ("Supplies a valid 'unit'", tool_call("calculate_triangle_area", {"base": 10, "height": 5, "unit": "units"})),
        ("Supplies a wrong 'unit'", tool_call("calculate_triangle_area", {"base": 10, "height": 5, "unit": "meters"})),
        ("Omits a required arg", tool_call("calculate_triangle_area", {"base": 10})),
    ]
    table = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {PURPLE}", expand=True)
    table.add_column("Model output", style="white", ratio=3)
    table.add_column("Reward", justify="center", width=8)
    table.add_column("Verdict", justify="left", width=22)
    for label, output in cases:
        r = grade(output, ground_truth)
        table.add_row(label, f"{r.reward:.1f}", verdict(r))
    console.print(table)


def main() -> None:
    console.print()
    console.rule("[bold medium_purple3]BFCL grader & reward demo[/]  ·  no GPU required",
                 style=GOLD)
    demo_verifiable_grading()
    demo_shaped_reward()
    demo_irrelevance()
    demo_optional_arguments()
    console.print()
    console.rule(style=GOLD)
    console.print("[grey62]Same grader powers Stage 0/1 evaluation AND the GRPO "
                  "training reward, so the metric and the signal cannot drift apart.[/]\n")


if __name__ == "__main__":
    main()
