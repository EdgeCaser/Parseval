"""Simple contiguous anomaly detection utilities."""

from __future__ import annotations


def contiguous_anomaly_spans(scores: list[float], threshold: float) -> list[tuple[int, int]]:
    """Return index spans [start, end) where scores exceed threshold contiguously."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, score in enumerate(scores):
        if score >= threshold and start is None:
            start = i
        elif score < threshold and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(scores)))
    return spans
