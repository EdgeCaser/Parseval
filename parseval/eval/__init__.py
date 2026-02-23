"""Evaluation utilities for benchmark datasets, splits, metrics, and reports."""

from .dataset import BenchmarkRecord, load_jsonl_dataset
from .metrics import classification_metrics
from .reports import write_report_json
from .splits import document_disjoint_split

__all__ = [
    "BenchmarkRecord",
    "load_jsonl_dataset",
    "classification_metrics",
    "write_report_json",
    "document_disjoint_split",
]
