"""Sentence embedding model management.

Uses sentence-transformers/all-MiniLM-L6-v2 (~90MB, downloaded on first use).
Embeddings are L2-normalized so cosine similarity = dot product.
"""

import threading
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_model_lock = threading.Lock()


class EmbeddingModel:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        with self._lock:
            if self._model is None:
                print(f"[parseval] Loading embedding model '{MODEL_NAME}'...")
                print("[parseval] First run: may download ~90MB to HuggingFace cache.")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_NAME)
                print("[parseval] Embedding model ready.")

    def encode(self, texts: list) -> np.ndarray:
        """Encode a list of texts into L2-normalized embedding vectors.

        Returns np.ndarray of shape (len(texts), 384).
        """
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 384), dtype=np.float64)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize: cosine sim = dot product
            batch_size=32,
        )
        return embeddings.astype(np.float64)

    def is_cached(self) -> bool:
        """Check whether the model files are already in the HuggingFace cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(MODEL_NAME, "config.json")
            return result is not None and result != "no_cache"
        except Exception:
            return False


def get_model() -> EmbeddingModel:
    """Get the global EmbeddingModel singleton (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None:
            _model = EmbeddingModel()
    return _model


def compute_corpus_stats(embeddings: np.ndarray) -> tuple:
    """Compute centroid, std, and expected cosine similarity from L2-normalized embeddings.

    The centroid is renormalized after averaging (mean of normalized vectors
    is not itself normalized).

    Also computes expected_emb_sim: the mean cosine similarity of all corpus
    embeddings to the centroid. This is used by the scorer to calibrate the
    embedding score the same way expected_dist calibrates the stylometric score —
    paragraphs as similar to the centroid as the average corpus paragraph score 1.0.

    Returns:
      centroid:         np.ndarray shape (384,) — renormalized mean embedding
      std:              np.ndarray shape (384,) — per-dimension standard deviation
      expected_emb_sim: float — mean cosine similarity of corpus paragraphs to centroid
    """
    if embeddings.shape[0] == 0:
        return np.zeros(384, dtype=np.float64), np.ones(384, dtype=np.float64), 0.0

    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-8:
        centroid = centroid / norm

    std = np.std(embeddings, axis=0)

    # Cosine similarity of each corpus embedding to the centroid.
    # Since embeddings are L2-normalized, dot product = cosine similarity.
    cos_sims = embeddings @ centroid   # shape (N,)
    expected_emb_sim = float(np.mean(cos_sims))

    return centroid, std, expected_emb_sim


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute and renormalize centroid from embeddings (helper for intra-doc scorer)."""
    if embeddings.shape[0] == 0:
        return np.zeros(384, dtype=np.float64)
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-8:
        centroid = centroid / norm
    return centroid


def compute_expected_emb_sim(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    """Compute mean cosine similarity of a set of embeddings to a given centroid.

    Used in self_mode to calibrate the embedding score for each leave-one-out profile.
    """
    if embeddings.shape[0] == 0:
        return 0.0
    cos_sims = embeddings @ centroid
    return float(np.mean(cos_sims))
