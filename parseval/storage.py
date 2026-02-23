"""Persistent storage for writing thumbprints.

Thumbprints are stored in ~/.parseval/thumbprints/<uuid>/
  meta.json      — JSON-serializable metadata and stylometric profile
  embeddings.npz — numpy arrays (centroid, std, all_embeddings, expected_emb_sim)
  (embeddings.pkl — legacy; migrated to .npz on first load)
"""

import os
import json
import shutil
import uuid

import numpy as np

STORAGE_ROOT = os.path.join(os.path.expanduser("~"), ".parseval", "thumbprints")

_EMB_NPZ = "embeddings.npz"
_EMB_PKL = "embeddings.pkl"


def _validate_thumbprint_id(thumbprint_id: str) -> None:
    """Raise ValueError if thumbprint_id is not a valid UUID."""
    if not thumbprint_id or not isinstance(thumbprint_id, str):
        raise ValueError("Invalid thumbprint id")
    try:
        uuid.UUID(thumbprint_id)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid thumbprint id")


def _thumbprint_dir(thumbprint_id: str) -> str:
    return os.path.join(STORAGE_ROOT, thumbprint_id)


def _ensure_storage_root() -> None:
    os.makedirs(STORAGE_ROOT, exist_ok=True)


def thumbprint_exists(thumbprint_id: str) -> bool:
    _validate_thumbprint_id(thumbprint_id)
    return os.path.isdir(_thumbprint_dir(thumbprint_id))


def save_thumbprint(meta: dict, embedding_data: dict) -> None:
    """Save thumbprint metadata and embeddings atomically."""
    _ensure_storage_root()
    _validate_thumbprint_id(meta["id"])
    tdir = _thumbprint_dir(meta["id"])
    os.makedirs(tdir, exist_ok=True)

    # Atomic write for meta.json
    meta_path = os.path.join(tdir, "meta.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, meta_path)

    # Atomic write for embeddings.npz (no pickle)
    emb_path = os.path.join(tdir, _EMB_NPZ)
    emb_tmp = emb_path + ".tmp"
    with open(emb_tmp, "wb") as f:
        np.savez_compressed(
            f,
            centroid=embedding_data["centroid"],
            std=embedding_data["std"],
            all_embeddings=embedding_data["all_embeddings"],
            expected_emb_sim=np.array(embedding_data["expected_emb_sim"], dtype=np.float64),
        )
    os.replace(emb_tmp, emb_path)


def _load_embeddings_npz(emb_path: str) -> dict:
    """Load embedding_data dict from .npz file."""
    with np.load(emb_path, allow_pickle=False) as data:
        return {
            "centroid": data["centroid"],
            "std": data["std"],
            "all_embeddings": data["all_embeddings"],
            "expected_emb_sim": float(data["expected_emb_sim"]),
        }


def _migrate_pkl_to_npz(tdir: str) -> dict:
    """Load legacy .pkl, save as .npz, remove .pkl. Returns embedding_data."""
    import pickle
    pkl_path = os.path.join(tdir, _EMB_PKL)
    with open(pkl_path, "rb") as f:
        embedding_data = pickle.load(f)
    npz_path = os.path.join(tdir, _EMB_NPZ)
    npz_tmp = npz_path + ".tmp"
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(
            f,
            centroid=embedding_data["centroid"],
            std=embedding_data["std"],
            all_embeddings=embedding_data["all_embeddings"],
            expected_emb_sim=np.array(embedding_data.get("expected_emb_sim", 0.0), dtype=np.float64),
        )
    os.replace(npz_tmp, npz_path)
    try:
        os.unlink(pkl_path)
    except OSError:
        pass
    return embedding_data


def load_thumbprint(thumbprint_id: str) -> tuple:
    """Load and return (meta dict, embedding_data dict) for a thumbprint."""
    _validate_thumbprint_id(thumbprint_id)
    tdir = _thumbprint_dir(thumbprint_id)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")

    meta_path = os.path.join(tdir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    npz_path = os.path.join(tdir, _EMB_NPZ)
    pkl_path = os.path.join(tdir, _EMB_PKL)
    if os.path.isfile(npz_path):
        embedding_data = _load_embeddings_npz(npz_path)
    elif os.path.isfile(pkl_path):
        embedding_data = _migrate_pkl_to_npz(tdir)
        # Ensure expected_emb_sim key for old thumbprints
        if "expected_emb_sim" not in embedding_data:
            embedding_data["expected_emb_sim"] = 0.0
    else:
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")

    return meta, embedding_data


def load_thumbprint_meta(thumbprint_id: str) -> dict:
    """Load only the metadata (no embeddings) for a thumbprint."""
    _validate_thumbprint_id(thumbprint_id)
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
        try:
            _validate_thumbprint_id(entry)
        except ValueError:
            continue
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
    _validate_thumbprint_id(thumbprint_id)
    tdir = _thumbprint_dir(thumbprint_id)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Thumbprint not found: {thumbprint_id}")
    shutil.rmtree(tdir)
