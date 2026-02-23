"""Decision policy profile loading for attribution claims."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    probability_threshold: float
    min_tokens: int
    min_contiguous_windows: int
    max_fpr: float


def load_policy_profiles(path: str | Path) -> dict[str, PolicyProfile]:
    """Load policy profiles from a JSON config file.

    JSON schema:
      {
        "profiles": [
          {"name": "conservative", ...}
        ]
      }
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[str, PolicyProfile] = {}
    for item in raw.get("profiles", []):
        prof = PolicyProfile(
            name=str(item["name"]),
            probability_threshold=float(item["probability_threshold"]),
            min_tokens=int(item["min_tokens"]),
            min_contiguous_windows=int(item["min_contiguous_windows"]),
            max_fpr=float(item["max_fpr"]),
        )
        out[prof.name] = prof

    if not out:
        raise ValueError(f"No policy profiles found in {p}")
    return out
