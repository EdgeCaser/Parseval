# Parseval Expert-Level Authorship Attribution: Complete Implementation Plan

## Objective

Upgrade Parseval from a heuristic similarity analyzer to a **validated, calibrated, segment-level authorship attribution system** that can support high-confidence statements like:

> "This segment is highly inconsistent with the reference author under a validated model with known error bounds."

This plan is designed to reach a level where claims are **probabilistic, auditable, and benchmark-backed**, rather than impressionistic.

---

## 1) Current-state baseline (what exists today)

Parseval already has a strong baseline architecture:

- Hybrid scoring: stylometric + embedding weighted blend.
- Corpus-relative normalization for style and embeddings.
- Optional Mahalanobis stylometric distance.
- Self-mode with leave-one-out / leave-N-out style consistency checks.
- Low-confidence flag for small reference corpora.

These pieces are good foundations but currently lack rigorous model validation/calibration, uncertainty quantification, and explicit decision policy.

---

## 2) Target system capabilities (definition of "expert-level")

The target system must provide all of the following:

1. **Calibrated probabilities** (`prob_not_author`) rather than raw similarity only.
2. **Known operating characteristics** (FPR/TPR/EER/AUC) on representative held-out datasets.
3. **Segment-level inference** with temporal smoothing/change-point detection.
4. **Uncertainty + abstention**: clear "insufficient evidence" behavior.
5. **Reproducible forensics trail**: model version, dataset version, threshold version, run metadata.
6. **Policy-safe language** in UI/API (no absolute statements without caveats).

---

## 3) End-to-end architecture changes

### 3.1 Scoring stack refactor

Move from single blended score to a two-layer system:

- **Layer A: Signal extraction**
  - Keep existing `style_similarity`, `embedding_similarity`, feature outlier info.
  - Add window-level signals (token windows), change metrics, local neighborhood deviations.
- **Layer B: Decision model**
  - Train a supervised model over Layer A features for `author_match` vs `non_match`.
  - Calibrate predicted probabilities (isotonic or Platt scaling).

Output per segment:

- `raw_signals`
- `prob_not_author`
- `confidence_interval`
- `decision` (`consistent` / `inconsistent` / `insufficient_evidence`)
- `decision_rationale` (human-readable)

### 3.2 Segment engine

Implement segmentization independent from paragraph boundaries:

- Default windows: 80–200 tokens, 50% overlap.
- Preserve paragraph mapping for UX highlighting.
- Run attribution per window, then aggregate to paragraph/region.

### 3.3 Change-point and continuity model

Add sequence-aware detection:

- CUSUM/Bayesian online change-point (or simpler rolling z-threshold initially).
- Require contiguous anomalous windows before high-confidence non-author claims.

### 3.4 Calibration + decision policy service

Create explicit threshold profiles:

- Conservative policy (low false positive).
- Balanced policy.
- Investigative policy (high sensitivity).

Each policy stores:

- threshold(s), expected FPR/TPR, applicable domain, minimum evidence requirements.

---

## 4) Data program (critical path)

### 4.1 Dataset specification

Create `data/benchmarks/` manifest schema:

- `author_id`
- `doc_id`
- `segment_id`
- `text`
- `label` (`author` / `not_author`)
- `source_type` (human, edited, LLM-assisted, translated, etc.)
- `domain/topic`
- `time_period`
- `length_tokens`

### 4.2 Split strategy

Use leakage-safe splits:

- Author-disjoint where relevant for generalization tests.
- Document-disjoint for within-author tests.
- Temporal split for drift robustness.

### 4.3 Evaluation cohorts

Measure by:

- Segment length bins.
- Topic distance bins.
- Language/register bins.
- Editing intensity bins.

---

## 5) Modeling roadmap

### Phase A (fast uplift, low risk)

1. Keep existing Parseval raw features.
2. Add engineered features:
   - local z-score volatility
   - nearest-neighbor distances to reference embeddings
   - residual between style and semantic signals
   - fragment reliability indicators
3. Train baseline classifier:
   - logistic regression + gradient boosting comparison.
4. Calibrate probabilities.
5. Report reliability diagrams and Brier score.

### Phase B (higher performance)

1. Expand stylometric representation:
   - richer char n-grams (frequency hashed or selected vocab)
   - syntactic dependency motifs
   - punctuation rhythm profiles
2. Author-specific one-class detector as auxiliary signal.
3. Topic-confound correction:
   - estimate topic shift and adjust confidence.

### Phase C (advanced)

1. Attribution-oriented encoder fine-tuning (contrastive/triplet setup).
2. Ensemble blending with uncertainty-aware weighting.
3. Domain adaptation procedures for new corpora.

---

## 6) Metrics and acceptance criteria

## 6.1 Core metrics

- ROC-AUC, PR-AUC
- EER
- TPR@FPR in 
  - 1%
  - 0.1%
- Brier score + calibration error (ECE)
- Detection delay for change-point cases

### 6.2 “Claim-ready” gate (minimum)

For a selected domain/profile:

- Calibrated `prob_not_author >= 0.95`
- Estimated FPR <= 1% at active threshold
- Segment length >= configured minimum (e.g., 120 tokens)
- At least N contiguous anomalous windows (e.g., 3)
- Reference corpus quality pass (size/diversity checks)
- No abstention triggers

If any fail -> `insufficient_evidence`.

---

## 7) Product/API changes

### 7.1 API schema

Extend scorer response with:

- `prob_not_author`
- `confidence_level` (`low/medium/high`)
- `evidence_quality`
- `policy_profile`
- `model_version`
- `calibration_version`
- `decision_reason_codes`

### 7.2 UI updates

- Replace single scalar emphasis with probability + confidence badge.
- Add explicit warning when evidence is insufficient.
- Show region-level anomalies (contiguous segments), not isolated red paragraphs only.
- Add “Why flagged?” panel from reason codes and top contributing signals.

### 7.3 Language policy

UI copy must avoid absolute claims:

- Use: "high inconsistency with reference author"
- Avoid: "definitively not written by"

---

## 8) Reliability, governance, and compliance

1. **Versioning**
   - Model artifact ID
   - Calibration artifact ID
   - Dataset snapshot ID
2. **Audit logs**
   - request metadata
   - policy profile used
   - thresholds + evidence checks
3. **Red-team suite**
   - paraphrase attacks
   - style transfer attacks
   - truncation/fragment stress tests
4. **Documentation**
   - model card
   - known limitations
   - domain constraints

---

## 9) Implementation plan by sprint

## Sprint 0 — Design + scaffolding (1 week)

- Add benchmark data schema docs.
- Add eval package skeleton (`parseval/eval/`).
- Define output schema v2.
- Define decision policy config format.

Deliverables:

- RFC doc
- JSON schema files
- Empty evaluation pipeline commands wired

## Sprint 1 — Evaluation baseline (1–2 weeks)

- Build dataset loader and split manager.
- Compute baseline metrics from current scorer.
- Add report generation (tables/plots/json).

Deliverables:

- Reproducible benchmark report
- First public baseline numbers

## Sprint 2 — Probabilistic layer + calibration (1–2 weeks)

- Train classifier on existing signals.
- Add calibration step.
- Integrate into scorer output.

Deliverables:

- `prob_not_author`
- calibration plots
- policy thresholds v1

## Sprint 3 — Segment engine + continuity logic (1–2 weeks)

- Token-window segmentation.
- Change-point/contiguity detection.
- Region aggregation for UI/API.

Deliverables:

- region-level inconsistency output
- contiguous-evidence policy checks

## Sprint 4 — Robustness uplift (2+ weeks)

- Additional stylometric/syntactic features.
- confound-aware adjustments.
- abstention model improvements.

Deliverables:

- improved low-FPR performance
- reduced false positives under topic shift

## Sprint 5 — Production hardening (1 week)

- audit logging
- model/calibration version pinning
- model card + decision-language pass

Deliverables:

- claim-ready checklist
- release candidate

---

## 10) Repository-level task breakdown

### New modules/directories

- `parseval/eval/`
  - `dataset.py`
  - `splits.py`
  - `metrics.py`
  - `calibration.py`
  - `reports.py`
- `parseval/models/`
  - `decision_model.py`
  - `policy.py`
- `parseval/segmentation/`
  - `windows.py`
  - `changepoint.py`
- `configs/`
  - `policy_profiles.yaml`
  - `benchmark_profiles.yaml`
- `docs/`
  - model card template
  - evaluation protocol

### Modified existing modules

- `parseval/scorer.py`
  - expose raw feature bundle + v2 outputs
- `parseval/features.py`
  - additional feature channels + reliability flags
- `static/app.js` and `static/index.html`
  - UI for confidence, abstention, reason codes

---

## 11) Testing strategy

### Unit tests

- deterministic feature extraction tests
- calibration transform correctness
- policy threshold application
- abstention trigger logic

### Integration tests

- end-to-end scoring on synthetic mini benchmark
- segment aggregation correctness
- backwards compatibility for legacy thumbprints

### Regression tests

- model version upgrade does not silently change policy profile
- benchmark metric drift alarms

---

## 12) Risks and mitigations

1. **Topic confounding**
   - Mitigation: topic-distance-aware controls, matched-topic benchmarks.
2. **Insufficient reference text**
   - Mitigation: stricter evidence gates + abstention.
3. **Overfitting to benchmark corpus**
   - Mitigation: external validation sets, temporal splits.
4. **Interpretation misuse**
   - Mitigation: policy language constraints and explicit uncertainty display.

---

## 13) Definition of done for "reasonably claim specific segment not written by author"

A deployment may make high-confidence segment-level non-author claims only when all are true:

1. The active model/calibration/policy is benchmarked and versioned.
2. Operating point for claim mode has validated low FPR in matching domain.
3. Segment passes minimum evidence length + contiguous anomaly criteria.
4. Probability and uncertainty pass thresholds.
5. No abstention/quality flags are triggered.
6. Output language is probabilistic and references model limitations.

If not, system must return: **insufficient evidence for claim-level attribution**.

---

## 14) Recommended immediate next 3 actions

1. Implement Sprint 0 scaffolding + benchmark schema.
2. Produce first baseline benchmark report from current scorer.
3. Add probabilistic calibration layer behind a feature flag (`PARSEVAL_SCORER_V2=1`).

This sequence gives quick, measurable progress while preserving current functionality.
