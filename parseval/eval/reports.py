"""Evaluation report writers."""

from __future__ import annotations

import json
from pathlib import Path


def write_report_json(path: str | Path, payload: dict) -> None:
    """Write evaluation payload to disk as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
