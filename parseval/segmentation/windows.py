"""Token window segmentation for segment-level attribution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextWindow:
    start_token: int
    end_token: int
    text: str


def sliding_windows(text: str, window_tokens: int = 120, overlap_ratio: float = 0.5) -> list[TextWindow]:
    if window_tokens < 1:
        raise ValueError("window_tokens must be >= 1")
    if overlap_ratio < 0 or overlap_ratio >= 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    tokens = text.split()
    if not tokens:
        return []

    step = max(1, int(round(window_tokens * (1.0 - overlap_ratio))))
    windows: list[TextWindow] = []
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + window_tokens)
        windows.append(TextWindow(start_token=i, end_token=j, text=" ".join(tokens[i:j])))
        if j >= len(tokens):
            break
        i += step
    return windows
