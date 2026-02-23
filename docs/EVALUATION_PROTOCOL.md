# Parseval Evaluation Protocol (v0)

## Purpose

Define a reproducible benchmark process for segment-level authorship attribution.

## Dataset format

Use JSONL records with required fields:

- `author_id`
- `doc_id`
- `segment_id`
- `text`
- `label` (`author` or `not_author`)
- `source_type`
- `domain`
- `time_period`
- `length_tokens`

## Split strategy

Default split is **document-disjoint** (all segments from the same document in one partition).

## Minimum metrics

- ROC-AUC
- PR-AUC
- Brier score
- Precision/Recall/FPR at active threshold
- Confusion matrix

## Reporting

A benchmark run should write a JSON report containing:

- profile metadata
- dataset stats
- metrics
- scorer/model/calibration version fields

## Claim-ready note

Scores are *not* claim-ready unless thresholds are tied to validated low-FPR operating points in the target domain.
