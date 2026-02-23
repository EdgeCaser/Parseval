"""Segment/window utilities for segment-level attribution."""

from .windows import TextWindow, sliding_windows
from .changepoint import contiguous_anomaly_spans

__all__ = ["TextWindow", "sliding_windows", "contiguous_anomaly_spans"]
