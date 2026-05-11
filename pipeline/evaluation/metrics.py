"""
Metric aggregation for BFCL evaluation results.

EvalSummary aggregates per-category CategoryMetrics objects and exposes a
to_dict() serialisation used for JSON logging and W&B reporting.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pipeline.reward.bfcl_grader import GradeResult


@dataclass
class CategoryMetrics:
    category: str
    total: int = 0
    correct: int = 0
    failure_counts: Counter = field(default_factory=Counter)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def record(self, result: GradeResult) -> None:
        self.total += 1
        if result.correct:
            self.correct += 1
        elif result.failure_category:
            self.failure_counts[result.failure_category] += 1


@dataclass
class EvalSummary:
    per_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float:
        total = sum(m.total for m in self.per_category.values())
        correct = sum(m.correct for m in self.per_category.values())
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        result: dict = {"overall_accuracy": round(self.overall_accuracy, 4)}
        for category_name, metrics in self.per_category.items():
            result[category_name] = {
                "accuracy": round(metrics.accuracy, 4),
                "correct": metrics.correct,
                "total": metrics.total,
                "failure_breakdown": dict(metrics.failure_counts),
            }
        return result


def aggregate_grade_results(
    category: str, grade_results: list[GradeResult]
) -> CategoryMetrics:
    metrics = CategoryMetrics(category=category)
    for result in grade_results:
        metrics.record(result)
    return metrics
