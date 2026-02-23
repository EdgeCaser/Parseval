"""Persistent storage for writing thumbprints.

Thumbprints are stored in ~/.parseval/thumbprints/<uuid>/
  meta.json      — JSON-serializable metadata and stylometric profile
  embeddings.pkl — numpy arrays (centroid, std, all_embeddings)
"""

import os
import json
import pickle
import shutil

STORAGE_ROOT = os.path.join(os.path.expanduser("~"), ".parseval", "thumbprints")


def _thumbprint_dir(thumbprint_id: str) -> str:
    return os.path.join(STORAGE_ROOT, thumbprint_id)


def _ensure_storage_root() -> None:
    os.makedirs(STORAGE_ROOT, exist_ok=True)


def thumbprint_exists(thumbprint_id: str) -> bool:
    return os.path.isdir(_thumbprint_dir(thumbprint_id))


def save_thumbprint(meta: dict, embedding_data: dict) -> None:
    """Save thumbprint metadata and embeddings atomically."""
    _ensure_storage_root()
    tdir = _thumbprint_dir(meta["id"])
    os.makedirs(tdir, exist_ok=True)

    # Atomic write for meta.json
    meta_path = os.path.join(tdir, "meta.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, meta_path)

    # Atomic write for embeddings.pkl
    emb_path = os.path.join(tdir, "embeddings.pkl")
    emb_tmp = emb_path + ".tmp"
    with open(emb_tmp, "wb") as f:
        pickle.dump(embedding_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(emb_tmp, emb_path)


def load_thumbprint(thumbprint_id: str) -> tuple:
    """Load and return (meta dict, embedding_data dict) for a thumbprint."""
    tdir = _thumbprint_dir(thumbprint_id)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")

    meta_path = os.path.join(tdir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emb_path = os.path.join(tdir, "embeddings.pkl")
    with open(emb_path, "rb") as f:
        embedding_data = pickle.load(f)

    return meta, embedding_data


def load_thumbprint_meta(thumbprint_id: str) -> dict:
    """Load only the metadata (no embeddings) for a thumbprint."""
    tdir = _thumbprint_dir(thumbprint_id)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")
    meta_path = os.path.join(tdir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_thumbprints() -> list:
    """Return a list of metadata dicts for all saved thumbprints."""
    _ensure_storage_root()
    results = []
    for entry in os.listdir(STORAGE_ROOT):
        tdir = os.path.join(STORAGE_ROOT, entry)
        meta_path = os.path.join(tdir, "meta.json")
        if os.path.isdir(tdir) and os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # Return lightweight listing: exclude the full stylometric profile
                results.append({
                    "id": meta.get("id"),
                    "name": meta.get("name"),
                    "created_at": meta.get("created_at"),
                    "source_files": meta.get("source_files", []),
                    "paragraph_count": meta.get("paragraph_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                pass  # Skip corrupted entries
    # Sort by creation date descending
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def delete_thumbprint(thumbprint_id: str) -> None:
    """Delete a thumbprint and all its associated files."""
    tdir = _thumbprint_dir(thumbprint_id)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")
    shutil.rmtree(tdir)
