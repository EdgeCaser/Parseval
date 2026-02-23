"""Similarity scoring engine.

Two modes:
  corpus_mode   — score each paragraph against a pre-built thumbprint
  self_mode     — score each paragraph against the rest of the same document
                  (intra-document consistency analysis)

Combined score formula:
  style_sim = dimensionality-aware exponential decay → [0, 1]
  emb_sim   = cosine similarity clamped to [0, 1]
  combined  = 0.40 * style_sim + 0.60 * emb_sim

Stylometric score detail:
  ratio = dist / expected_dist   (expected_dist = mean z-norm of corpus paragraphs)
  style_sim = exp(-max(0, ratio - 1.0) * SHARPNESS)
  → ratio ≤ 1.0 (as typical as an average corpus paragraph): score = 1.0
  → ratio = 1.5: exp(-0.75) ≈ 0.47
  → ratio = 2.0: exp(-1.5)  ≈ 0.22

Score interpretation:
  1.0 = very similar to reference style
  0.0 = very different from reference style
"""

import numpy as np

from parseval import features, embeddings
from parseval.corpus import MIN_PARAGRAPH_CHARS

# Weight split between stylometric and embedding scores
STYLE_WEIGHT = 0.40
EMBED_WEIGHT = 0.60

# Controls how fast stylometric scores decay for paragraphs further from the
# corpus mean than the average corpus paragraph (ratio > 1.0).
SHARPNESS = 1.5

# Controls how fast embedding scores decay for paragraphs less similar to the
# centroid than the average corpus paragraph.
EMB_SHARPNESS = 4.0

# Fallback expected_dist if thumbprint was built before this field was added
_FALLBACK_EXPECTED_DIST = float(np.sqrt(65))  # ≈ 8.06
# Fallback expected_emb_sim for old thumbprints (0.0 triggers the max(0,cos) path)
_FALLBACK_EXPECTED_EMB_SIM = 0.0

# Minimum paragraphs for intra-doc mode before switching to leave-N-out
INTRADOC_LOO_MIN = 10
# For leave-N-out, how many neighbours to exclude on each side
INTRADOC_LEAVE_N = 2


def _style_score(para_vec: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray,
                 expected_dist: float) -> float:
    """Compute stylometric similarity [0, 1] using dimensionality-aware exponential decay.

    Paragraphs whose z-score distance is ≤ expected_dist (i.e. as typical as an
    average corpus paragraph) score 1.0. Scores decay exponentially above that.
    """
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z = (para_vec - feature_mean) / safe_std
    dist = float(np.linalg.norm(z))
    ratio = dist / max(expected_dist, 1.0)
    return float(np.exp(-max(0.0, ratio - 1.0) * SHARPNESS))


def _emb_score(para_emb: np.ndarray, centroid: np.ndarray, expected_emb_sim: float) -> float:
    """Corpus-relative embedding similarity [0, 1].

    Paragraphs as similar to the centroid as the average corpus paragraph score 1.0.
    Scores decay exponentially for paragraphs less similar than the corpus average.

    This mirrors the stylometric approach: we normalize against the expected
    in-distribution similarity rather than using raw cosine, which would give
    misleadingly low scores for diverse corpora where no single paragraph is
    very close to the centroid.
    """
    cos = float(np.dot(para_emb, centroid))
    if expected_emb_sim > 0.01:
        # ratio = how this paragraph's similarity compares to the corpus average
        # ratio >= 1.0 → at least as similar as average → score 1.0
        # ratio < 1.0  → less similar than average → decay
        ratio = cos / expected_emb_sim
        return float(np.exp(-max(0.0, 1.0 - ratio) * EMB_SHARPNESS))
    else:
        # Fallback for old thumbprints without expected_emb_sim
        return float(max(0.0, cos))


def _combined(style_sim: float, emb_sim: float) -> float:
    return STYLE_WEIGHT * style_sim + EMBED_WEIGHT * emb_sim


def _is_fragment(text: str) -> bool:
    return len(text.strip()) < MIN_PARAGRAPH_CHARS


def score_corpus_mode(paragraphs: list, meta: dict, embedding_data: dict) -> dict:
    """Score each paragraph against a corpus thumbprint.

    Args:
        paragraphs:     List of paragraph strings from the target document.
        meta:           Thumbprint metadata dict (from storage).
        embedding_data: Thumbprint embedding dict (from storage).

    Returns:
        dict with keys: paragraphs (list of scored paragraph dicts), overall_score
    """
    profile = meta["stylometric_profile"]
    feature_mean = np.array(profile["feature_mean"], dtype=np.float64)
    feature_std = np.array(profile["feature_std"], dtype=np.float64)
    expected_dist = float(profile.get("expected_dist", _FALLBACK_EXPECTED_DIST))
    centroid = embedding_data["centroid"].astype(np.float64)
    expected_emb_sim = float(embedding_data.get("expected_emb_sim", _FALLBACK_EXPECTED_EMB_SIM))

    model = embeddings.get_model()
    para_embeddings = model.encode(paragraphs)

    results = []
    scored_values = []

    for i, (para, emb) in enumerate(zip(paragraphs, para_embeddings)):
        fragment = _is_fragment(para)

        if fragment:
            emb_sim = _emb_score(emb, centroid, expected_emb_sim)
            combined_score = emb_sim
            results.append({
                "index": i,
                "text": para,
                "style_similarity": None,
                "embedding_similarity": round(emb_sim, 4),
                "combined_score": round(combined_score, 4),
                "is_fragment": True,
            })
        else:
            try:
                para_vec = features.extract_paragraph_features(para)
                style_sim = _style_score(para_vec, feature_mean, feature_std, expected_dist)
            except Exception:
                style_sim = 0.5  # neutral fallback

            emb_sim = _emb_score(emb, centroid, expected_emb_sim)
            combined_score = _combined(style_sim, emb_sim)
            scored_values.append(combined_score)

            results.append({
                "index": i,
                "text": para,
                "style_similarity": round(style_sim, 4),
                "embedding_similarity": round(emb_sim, 4),
                "combined_score": round(combined_score, 4),
                "is_fragment": False,
            })

    overall_score = float(np.mean(scored_values)) if scored_values else 0.0
    return {
        "paragraphs": results,
        "overall_score": round(overall_score, 4),
    }


def score_self_mode(paragraphs: list) -> dict:
    """Score each paragraph against the rest of the same document.

    Uses leave-one-out (or leave-N-out for short documents) to build a
    local thumbprint from all other paragraphs, then scores the paragraph
    against it.

    Score interpretation:
      High score = paragraph is consistent with the rest of the document
      Low score  = paragraph is a style outlier (suspicious)

    Args:
        paragraphs: All paragraph strings from the document.

    Returns:
        dict with keys: paragraphs (list of scored paragraph dicts), overall_consistency
    """
    n = len(paragraphs)
    if n < 2:
        # Can't compare if there's only one paragraph
        return {
            "paragraphs": [{
                "index": 0,
                "text": paragraphs[0] if paragraphs else "",
                "style_similarity": None,
                "embedding_similarity": None,
                "combined_score": None,
                "is_fragment": True,
            }],
            "overall_consistency": None,
        }

    # Determine exclusion window for leave-N-out (short docs)
    if n < INTRADOC_LOO_MIN:
        exclude_radius = INTRADOC_LEAVE_N
    else:
        exclude_radius = 0  # pure leave-one-out

    # Pre-compute all feature vectors and embeddings
    model = embeddings.get_model()
    all_embeddings = model.encode(paragraphs)  # shape: (n, 384)

    all_feature_vecs = []
    for para in paragraphs:
        if _is_fragment(para):
            all_feature_vecs.append(None)
        else:
            try:
                all_feature_vecs.append(features.extract_paragraph_features(para))
            except Exception:
                all_feature_vecs.append(None)

    results = []
    scored_values = []

    for i in range(n):
        para = paragraphs[i]
        fragment = _is_fragment(para)

        # Build exclusion set: paragraph i and its neighbours (for short docs)
        exclude = set(range(
            max(0, i - exclude_radius),
            min(n, i + exclude_radius + 1)
        ))
        include_indices = [j for j in range(n) if j not in exclude]

        if not include_indices:
            # Edge case: all excluded (very tiny doc)
            results.append({
                "index": i,
                "text": para,
                "style_similarity": None,
                "embedding_similarity": None,
                "combined_score": None,
                "is_fragment": fragment,
            })
            continue

        # Build local profile from remaining paragraphs
        other_embs = all_embeddings[include_indices]
        local_centroid = embeddings.compute_centroid(other_embs)
        local_expected_emb_sim = embeddings.compute_expected_emb_sim(other_embs, local_centroid)

        if fragment:
            emb_sim = _emb_score(all_embeddings[i], local_centroid, local_expected_emb_sim)
            combined_score = emb_sim
            results.append({
                "index": i,
                "text": para,
                "style_similarity": None,
                "embedding_similarity": round(emb_sim, 4),
                "combined_score": round(combined_score, 4),
                "is_fragment": True,
            })
        else:
            # Build local stylometric profile from non-fragment others
            other_vecs = [all_feature_vecs[j] for j in include_indices if all_feature_vecs[j] is not None]

            if other_vecs:
                local_profile = features.build_profile_from_vectors(other_vecs)
                local_mean = np.array(local_profile["feature_mean"], dtype=np.float64)
                local_std = np.array(local_profile["feature_std"], dtype=np.float64)
                local_expected_dist = float(local_profile.get("expected_dist", _FALLBACK_EXPECTED_DIST))

                if all_feature_vecs[i] is not None:
                    style_sim = _style_score(all_feature_vecs[i], local_mean, local_std, local_expected_dist)
                else:
                    style_sim = 0.5
            else:
                style_sim = 0.5  # fallback: no other feature vectors available

            emb_sim = _emb_score(all_embeddings[i], local_centroid, local_expected_emb_sim)
            combined_score = _combined(style_sim, emb_sim)
            scored_values.append(combined_score)

            results.append({
                "index": i,
                "text": para,
                "style_similarity": round(style_sim, 4),
                "embedding_similarity": round(emb_sim, 4),
                "combined_score": round(combined_score, 4),
                "is_fragment": False,
            })

    overall_consistency = float(np.mean(scored_values)) if scored_values else None
    return {
        "paragraphs": results,
        "overall_consistency": round(overall_consistency, 4) if overall_consistency is not None else None,
    }
