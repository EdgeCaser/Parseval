"""Leakage-resistant split helpers for benchmark evaluation."""

from __future__ import annotations

from collections import defaultdict
import random

from .dataset import BenchmarkRecord


def document_disjoint_split(
    rows: list[BenchmarkRecord],
    test_ratio: float = 0.2,
    seed: int = 13,
) -> tuple[list[BenchmarkRecord], list[BenchmarkRecord]]:
    """Split records such that all segments from a document stay together."""
    if not rows:
        return [], []

    by_doc = defaultdict(list)
    for row in rows:
        by_doc[row.doc_id].append(row)

    doc_ids = list(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n_test_docs = max(1, int(round(len(doc_ids) * test_ratio)))
    test_docs = set(doc_ids[:n_test_docs])

    train, test = [], []
    for doc_id, doc_rows in by_doc.items():
        if doc_id in test_docs:
            test.extend(doc_rows)
        else:
            train.extend(doc_rows)

    if not train:
        # Ensure both sets non-empty for tiny datasets.
        train, test = test, train

    return train, test
