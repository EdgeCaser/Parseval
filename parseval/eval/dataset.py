"""Benchmark dataset loading and validation.

Schema targets `data/benchmarks/*.jsonl` records described in
EXPERT_LEVEL_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


_REQUIRED_FIELDS = {
    "author_id",
    "doc_id",
    "segment_id",
    "text",
    "label",
    "source_type",
    "domain",
    "time_period",
    "length_tokens",
}


@dataclass(frozen=True)
class BenchmarkRecord:
    author_id: str
    doc_id: str
    segment_id: str
    text: str
    label: str
    source_type: str
    domain: str
    time_period: str
    length_tokens: int

    @property
    def y_true(self) -> int:
        """Binary target for 'not authored by reference author'."""
        label = self.label.strip().lower()
        if label in ("not_author", "non_author", "not-author"):
            return 1
        if label == "author":
            return 0
        raise ValueError(f"Unsupported label: {self.label!r}")


def _validate_record(raw: dict, line_number: int, path: Path) -> BenchmarkRecord:
    missing = _REQUIRED_FIELDS - set(raw.keys())
    if missing:
        missing_fmt = ", ".join(sorted(missing))
        raise ValueError(f"{path}:{line_number} missing required fields: {missing_fmt}")

    return BenchmarkRecord(
        author_id=str(raw["author_id"]),
        doc_id=str(raw["doc_id"]),
        segment_id=str(raw["segment_id"]),
        text=str(raw["text"]),
        label=str(raw["label"]),
        source_type=str(raw["source_type"]),
        domain=str(raw["domain"]),
        time_period=str(raw["time_period"]),
        length_tokens=int(raw["length_tokens"]),
    )


def load_jsonl_dataset(path: str | Path) -> list[BenchmarkRecord]:
    """Load and validate benchmark records from a JSONL file."""
    p = Path(path)
    rows: list[BenchmarkRecord] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i} invalid JSON: {e.msg}") from e
            rows.append(_validate_record(raw, i, p))
    if not rows:
        raise ValueError(f"No records found in dataset: {p}")
    return rows
