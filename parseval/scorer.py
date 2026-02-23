"""Similarity scoring engine.

Two modes:
  corpus_mode   — score each paragraph against a pre-built thumbprint
  self_mode     — score each paragraph against the rest of the same document
                  (intra-document consistency analysis)

Combined score formula:
  style_sim = dimensionality-aware exponential decay → [0, 1]
  emb_sim   = cosine similarity clamped to [0, 1]
  combined  = STYLE_WEIGHT * style_sim + EMBED_WEIGHT * emb_sim  (configurable via env)

Stylometric score: Euclidean z-norm or optional Mahalanobis distance; ratio vs expected_dist.
"""

import os
import numpy as np

from parseval import features, embeddings
from parseval.corpus import MIN_PARAGRAPH_CHARS


def _get_weights():
    """Style/embedding weights from env; default 0.4/0.6; normalized to sum 1."""
    try:
        sw = float(os.environ.get("PARSEVAL_STYLE_WEIGHT", "0.4"))
        ew = float(os.environ.get("PARSEVAL_EMBED_WEIGHT", "0.6"))
    except (TypeError, ValueError):
        sw, ew = 0.4, 0.6
    if sw < 0 or ew < 0:
        sw, ew = 0.4, 0.6
    total = sw + ew
    if total <= 0:
        return 0.4, 0.6
    return sw / total, ew / total


STYLE_WEIGHT, EMBED_WEIGHT = _get_weights()

# Controls how fast stylometric scores decay for paragraphs further from the
# corpus mean than the average corpus paragraph (ratio > 1.0).
SHARPNESS = 1.5

# Controls how fast embedding scores decay for paragraphs less similar to the
# centroid than the average corpus paragraph.
EMB_SHARPNESS = 4.0

# Fallback expected_dist when not in profile (legacy): use sqrt(dim)
def _fallback_expected_dist(dim: int) -> float:
    return float(np.sqrt(max(1, dim)))


# Fallback expected_emb_sim for old thumbprints (0.0 triggers the max(0,cos) path)
_FALLBACK_EXPECTED_EMB_SIM = 0.0

# Minimum paragraphs for intra-doc mode before switching to leave-N-out
INTRADOC_LOO_MIN = 10
# For leave-N-out, how many neighbours to exclude on each side
INTRADOC_LEAVE_N = 2

# Top-K features to return for interpretability
TOP_DIFFERING_FEATURES_K = 5


def _style_score(para_vec: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray,
                 expected_dist: float) -> float:
    """Compute stylometric similarity [0, 1] using z-norm Euclidean distance and exponential decay."""
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z = (para_vec - feature_mean) / safe_std
    dist = float(np.linalg.norm(z))
    ratio = dist / max(expected_dist, 1.0)
    return float(np.exp(-max(0.0, ratio - 1.0) * SHARPNESS))


def _style_score_mahal(para_vec: np.ndarray, feature_mean: np.ndarray, cov_inv: np.ndarray,
                       expected_mahal_dist: float) -> float:
    """Compute stylometric similarity [0, 1] using Mahalanobis distance and exponential decay."""
    delta = para_vec - feature_mean
    d_sq = np.maximum(0.0, float(delta @ cov_inv @ delta))
    dist = np.sqrt(d_sq)
    ratio = dist / max(expected_mahal_dist, 1.0)
    return float(np.exp(-max(0.0, ratio - 1.0) * SHARPNESS))


def _top_differing_features(para_vec: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray,
                            feature_names: list, k: int = TOP_DIFFERING_FEATURES_K) -> list:
    """Return top-k feature names and absolute z-scores by contribution to distance."""
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z = np.abs((para_vec - feature_mean) / safe_std)
    n = min(k, len(z), len(feature_names))
    if n == 0:
        return []
    indices = np.argsort(z)[::-1][:n]
    return [
        {"name": feature_names[j], "z_abs": round(float(z[j]), 4)}
        for j in indices
    ]


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

    Supports legacy 65-dim profiles (feature_dim missing or 65) and new full-dim profiles.
    Optional Mahalanobis when profile has covariance_inv and expected_mahal_dist.
    """
    import os as _os
    profile = meta["stylometric_profile"]
    profile_dim = int(profile.get("feature_dim", 65))
    feature_mean_full = np.array(profile["feature_mean"], dtype=np.float64)
    feature_std_full = np.array(profile["feature_std"], dtype=np.float64)
    # Use only first profile_dim dimensions for backward compatibility
    feature_mean = feature_mean_full[:profile_dim]
    feature_std = feature_std_full[:profile_dim]
    expected_dist = float(profile.get("expected_dist", _fallback_expected_dist(profile_dim)))
    use_mahal = "covariance_inv" in profile and "expected_mahal_dist" in profile
    # #region agent log
    _log_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "debug-1a3ab6.log")
    try:
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(__import__("json").dumps({"sessionId": "1a3ab6", "location": "scorer.score_corpus_mode.entry", "message": "score_corpus_mode entry", "data": {"profile_dim": profile_dim, "use_mahal": use_mahal, "n_paragraphs": len(paragraphs)}, "hypothesisId": "H4", "timestamp": __import__("time").time() * 1000}) + "\n")
    except Exception:
        pass
    # #endregion
    if use_mahal:
        cov_inv = np.array(profile["covariance_inv"], dtype=np.float64)
        if cov_inv.shape[0] != profile_dim:
            cov_inv = cov_inv[:profile_dim, :profile_dim]
        expected_mahal_dist = float(profile["expected_mahal_dist"])
    else:
        cov_inv = None
        expected_mahal_dist = None

    feature_names = features.get_feature_names_for_profile(profile)

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
                "top_differing_features": None,
            })
        else:
            try:
                para_vec_full = features.extract_paragraph_features(para)
                para_vec = para_vec_full[:profile_dim]
                if use_mahal and cov_inv is not None:
                    style_sim = _style_score_mahal(para_vec, feature_mean, cov_inv, expected_mahal_dist)
                else:
                    style_sim = _style_score(para_vec, feature_mean, feature_std, expected_dist)
                names_dim = feature_names[:profile_dim]
                top_diff = _top_differing_features(para_vec, feature_mean, feature_std, names_dim, TOP_DIFFERING_FEATURES_K)
            except Exception:
                style_sim = 0.5
                top_diff = []

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
                "top_differing_features": top_diff,
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
                "top_differing_features": None,
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
                "top_differing_features": None,
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
                "top_differing_features": None,
            })
        else:
            # Build local stylometric profile from non-fragment others
            other_vecs = [all_feature_vecs[j] for j in include_indices if all_feature_vecs[j] is not None]

            if other_vecs:
                local_profile = features.build_profile_from_vectors(other_vecs)
                local_dim = int(local_profile.get("feature_dim", features.FEATURE_SIZE))
                local_mean = np.array(local_profile["feature_mean"], dtype=np.float64)[:local_dim]
                local_std = np.array(local_profile["feature_std"], dtype=np.float64)[:local_dim]
                local_expected_dist = float(local_profile.get("expected_dist", _fallback_expected_dist(local_dim)))

                if all_feature_vecs[i] is not None:
                    pv = all_feature_vecs[i][:local_dim]
                    use_mahal = "covariance_inv" in local_profile and "expected_mahal_dist" in local_profile
                    if use_mahal:
                        cov_inv = np.array(local_profile["covariance_inv"], dtype=np.float64)
                        if cov_inv.shape[0] != local_dim:
                            cov_inv = cov_inv[:local_dim, :local_dim]
                        style_sim = _style_score_mahal(pv, local_mean, cov_inv, float(local_profile["expected_mahal_dist"]))
                    else:
                        style_sim = _style_score(pv, local_mean, local_std, local_expected_dist)
                    local_names = features.get_feature_names_for_profile(local_profile)[:local_dim]
                    top_diff = _top_differing_features(pv, local_mean, local_std, local_names, TOP_DIFFERING_FEATURES_K)
                else:
                    style_sim = 0.5
                    top_diff = []
            else:
                style_sim = 0.5
                top_diff = []

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
                "top_differing_features": top_diff if other_vecs else [],
            })

    overall_consistency = float(np.mean(scored_values)) if scored_values else None
    return {
        "paragraphs": results,
        "overall_consistency": round(overall_consistency, 4) if overall_consistency is not None else None,
    }
