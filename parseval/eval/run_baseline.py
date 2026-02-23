"""Run a baseline benchmark from JSONL dataset using current Parseval scorer signals.

Usage:
    python -m parseval.eval.run_baseline --dataset data/benchmarks/default.jsonl --out reports/baseline.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from parseval.eval.dataset import load_jsonl_dataset
from parseval.eval.metrics import classification_metrics
from parseval.eval.reports import write_report_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_jsonl_dataset(args.dataset)

    # Placeholder baseline score:
    # until v2 model integration, use a neutral prior score to validate pipeline wiring.
    y_true = [r.y_true for r in rows]
    y_score = [0.5 for _ in rows]

    metrics = classification_metrics(y_true, y_score, threshold=args.threshold)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "n_records": len(rows),
        "scorer_version": "heuristic_v1_pipeline_scaffold",
        "metrics": metrics,
    }

    write_report_json(args.out, payload)


if __name__ == "__main__":
    main()
