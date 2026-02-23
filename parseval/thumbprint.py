"""Thumbprint creation and orchestration.

Coordinates file parsing, feature extraction, and embedding computation
to produce a complete writing thumbprint.
"""

import uuid
from datetime import datetime, timezone

from parseval import corpus, features, embeddings, storage

MIN_PARAGRAPHS = 5


def create_thumbprint(name: str, file_paths: list, file_names: list) -> dict:
    """Create a new thumbprint from one or more files.

    Args:
        name:       Human-readable label for this thumbprint.
        file_paths: List of absolute paths to uploaded files.
        file_names: Corresponding original filenames (used for display and type detection).

    Returns:
        The metadata dict (same as what is stored in meta.json).

    Raises:
        ValueError: If the corpus is too small or files cannot be parsed.
    """
    thumbprint_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # 1. Parse all files and collect paragraphs
    all_paragraphs = []
    for path, fname in zip(file_paths, file_names):
        text = corpus.extract_text(path, fname)
        paragraphs = corpus.split_paragraphs(text)
        all_paragraphs.extend(paragraphs)

    if len(all_paragraphs) < MIN_PARAGRAPHS:
        raise ValueError(
            f"Insufficient content: need at least {MIN_PARAGRAPHS} paragraphs, "
            f"found {len(all_paragraphs)}. Try providing longer or more files."
        )

    # 2. Extract stylometric profile
    stylometric_profile = features.extract_corpus_profile(all_paragraphs)

    # 3. Compute paragraph embeddings
    model = embeddings.get_model()
    emb_matrix = model.encode(all_paragraphs)
    centroid, emb_std, expected_emb_sim = embeddings.compute_corpus_stats(emb_matrix)

    # 4. Assemble metadata
    meta = {
        "id": thumbprint_id,
        "name": name,
        "created_at": created_at,
        "source_files": file_names,
        "paragraph_count": len(all_paragraphs),
        "stylometric_profile": stylometric_profile,
    }

    embedding_data = {
        "centroid": centroid,
        "std": emb_std,
        "all_embeddings": emb_matrix,
        "expected_emb_sim": expected_emb_sim,
    }

    # 5. Persist to disk
    storage.save_thumbprint(meta, embedding_data)

    return meta


def regenerate_thumbprint(thumbprint_id: str, file_paths: list, file_names: list) -> dict:
    """Regenerate an existing thumbprint with new (or the same) files.

    Overwrites the stored thumbprint while keeping the same UUID.
    """
    if not storage.thumbprint_exists(thumbprint_id):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")

    # Load existing metadata to preserve original name and created_at
    existing_meta = storage.load_thumbprint_meta(thumbprint_id)
    name = existing_meta.get("name", "Unknown")
    created_at = existing_meta.get("created_at", datetime.now(timezone.utc).isoformat())

    # Re-parse and re-compute
    all_paragraphs = []
    for path, fname in zip(file_paths, file_names):
        text = corpus.extract_text(path, fname)
        paragraphs = corpus.split_paragraphs(text)
        all_paragraphs.extend(paragraphs)

    if len(all_paragraphs) < MIN_PARAGRAPHS:
        raise ValueError(
            f"Insufficient content: need at least {MIN_PARAGRAPHS} paragraphs, "
            f"found {len(all_paragraphs)}."
        )

    stylometric_profile = features.extract_corpus_profile(all_paragraphs)

    model = embeddings.get_model()
    emb_matrix = model.encode(all_paragraphs)
    centroid, emb_std, expected_emb_sim = embeddings.compute_corpus_stats(emb_matrix)

    meta = {
        "id": thumbprint_id,
        "name": name,
        "created_at": created_at,
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": file_names,
        "paragraph_count": len(all_paragraphs),
        "stylometric_profile": stylometric_profile,
    }

    embedding_data = {
        "centroid": centroid,
        "std": emb_std,
        "all_embeddings": emb_matrix,
        "expected_emb_sim": expected_emb_sim,
    }

    storage.save_thumbprint(meta, embedding_data)

    return meta
