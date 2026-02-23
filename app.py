"""Parseval — Flask application entry point.

Run with: python app.py
Access at: http://localhost:5000
"""

import logging
import os
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from parseval import corpus, storage, thumbprint as thumbprint_module, scorer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Configurable CORS origins (comma-separated); default localhost only
_cors_origins = os.environ.get("PARSEVAL_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
CORS_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()]

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app, origins=CORS_ORIGINS)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Input limits
MAX_THUMBPRINT_NAME_LEN = 200
MAX_FILES_PER_THUMBPRINT = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread-safe embedding model singleton
_model_lock = threading.Lock()
_model = None


def get_model():
    from parseval.embeddings import get_model as _get_model
    return _get_model()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(file_storage) -> tuple:
    """Save an uploaded FileStorage to the uploads dir. Returns (path, filename).
    On-disk path uses a unique name to avoid collisions; filename is original for display/type."""
    filename = secure_filename(file_storage.filename)
    if not corpus.allowed_file(filename):
        raise ValueError(f"Unsupported file type: '{filename}'. Allowed: .docx, .txt, .md")
    ext = os.path.splitext(filename)[1]
    unique_name = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    file_storage.save(path)
    return path, filename


def _cleanup(*paths):
    """Remove temporary upload files."""
    for path in paths:
        try:
            if path and os.path.isfile(path):
                os.unlink(path)
        except Exception:
            pass


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Static / SPA
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Health
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Return system status: spaCy model availability, embedding model cache."""
    spacy_available = False
    try:
        import spacy
        spacy.load("en_core_web_sm")
        spacy_available = True
    except Exception:
        pass

    embedding_cached = False
    try:
        model = get_model()
        embedding_cached = model.is_cached()
    except Exception:
        pass

    return jsonify({
        "spacy_available": spacy_available,
        "spacy_model": "en_core_web_sm",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_model_cached": embedding_cached,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Thumbprint CRUD
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/thumbprints", methods=["GET"])
def list_thumbprints():
    thumbprints = storage.list_thumbprints()
    return jsonify({"thumbprints": thumbprints})


@app.route("/api/thumbprints", methods=["POST"])
def create_thumbprint():
    name = request.form.get("name", "").strip()
    if not name:
        return _error("A name is required for the thumbprint.")
    if len(name) > MAX_THUMBPRINT_NAME_LEN:
        return _error(f"Thumbprint name must be at most {MAX_THUMBPRINT_NAME_LEN} characters.")

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return _error("At least one file is required.")
    if len([f for f in files if f.filename]) > MAX_FILES_PER_THUMBPRINT:
        return _error(f"At most {MAX_FILES_PER_THUMBPRINT} files are allowed per thumbprint.")

    saved_paths = []
    saved_names = []
    try:
        for f in files:
            if f.filename:
                try:
                    path, fname = _save_upload(f)
                    saved_paths.append(path)
                    saved_names.append(fname)
                except ValueError as e:
                    return _error(str(e))

        if not saved_paths:
            return _error("No valid files were uploaded.")

        meta = thumbprint_module.create_thumbprint(name, saved_paths, saved_names)
        return jsonify(meta), 201

    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        logging.exception("Failed to create thumbprint")
        return _error("An unexpected error occurred.", 500)
    finally:
        _cleanup(*saved_paths)


@app.route("/api/thumbprints/<thumbprint_id>", methods=["DELETE"])
def delete_thumbprint(thumbprint_id):
    try:
        storage.delete_thumbprint(thumbprint_id)
        return jsonify({"deleted": thumbprint_id})
    except ValueError:
        return _error("Invalid thumbprint id", 400)
    except FileNotFoundError:
        return _error("Thumbprint not found.", 404)
    except Exception:
        logging.exception("Failed to delete thumbprint")
        return _error("An unexpected error occurred.", 500)


@app.route("/api/thumbprints/<thumbprint_id>/regenerate", methods=["POST"])
def regenerate_thumbprint(thumbprint_id):
    try:
        if not storage.thumbprint_exists(thumbprint_id):
            return _error("Thumbprint not found.", 404)
    except ValueError:
        return _error("Invalid thumbprint id", 400)

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return _error("At least one file is required for regeneration.")
    if len([f for f in files if f.filename]) > MAX_FILES_PER_THUMBPRINT:
        return _error(f"At most {MAX_FILES_PER_THUMBPRINT} files are allowed per thumbprint.")

    saved_paths = []
    saved_names = []
    try:
        for f in files:
            if f.filename:
                try:
                    path, fname = _save_upload(f)
                    saved_paths.append(path)
                    saved_names.append(fname)
                except ValueError as e:
                    return _error(str(e))

        meta = thumbprint_module.regenerate_thumbprint(thumbprint_id, saved_paths, saved_names)
        return jsonify(meta)

    except ValueError as e:
        return _error(str(e))
    except FileNotFoundError:
        return _error("Thumbprint not found.", 404)
    except Exception:
        logging.exception("Failed to regenerate thumbprint")
        return _error("An unexpected error occurred.", 500)
    finally:
        _cleanup(*saved_paths)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze_corpus():
    """Analyze a document against a saved thumbprint (corpus mode)."""
    thumbprint_id = request.form.get("thumbprint_id", "").strip()
    if not thumbprint_id:
        return _error("thumbprint_id is required.")
    try:
        if not storage.thumbprint_exists(thumbprint_id):
            return _error("Thumbprint not found.", 404)
    except ValueError:
        return _error("Invalid thumbprint id", 400)

    file = request.files.get("file")
    if not file or file.filename == "":
        return _error("A document file is required.")

    saved_path = None
    try:
        saved_path, saved_name = _save_upload(file)

        meta, embedding_data = storage.load_thumbprint(thumbprint_id)

        text = corpus.extract_text(saved_path, saved_name)
        paragraphs = corpus.split_paragraphs_all(text)

        if not paragraphs:
            return _error("No readable content found in the document.")

        result = scorer.score_corpus_mode(paragraphs, meta, embedding_data)

        return jsonify({
            "mode": "corpus",
            "thumbprint_id": thumbprint_id,
            "thumbprint_name": meta.get("name"),
            "filename": saved_name,
            "overall_score": result["overall_score"],
            "paragraph_count": len(paragraphs),
            "paragraphs": result["paragraphs"],
        })

    except ValueError as e:
        return _error(str(e))
    except FileNotFoundError:
        return _error("Thumbprint not found.", 404)
    except Exception:
        logging.exception("Analysis failed")
        return _error("An unexpected error occurred.", 500)
    finally:
        _cleanup(saved_path)


@app.route("/api/analyze/self", methods=["POST"])
def analyze_self():
    """Analyze a document for internal style consistency (intra-document mode)."""
    file = request.files.get("file")
    if not file or file.filename == "":
        return _error("A document file is required.")

    saved_path = None
    try:
        saved_path, saved_name = _save_upload(file)

        text = corpus.extract_text(saved_path, saved_name)
        paragraphs = corpus.split_paragraphs_all(text)

        if not paragraphs:
            return _error("No readable content found in the document.")

        result = scorer.score_self_mode(paragraphs)

        return jsonify({
            "mode": "self",
            "filename": saved_name,
            "overall_consistency": result["overall_consistency"],
            "paragraph_count": len(paragraphs),
            "paragraphs": result["paragraphs"],
        })

    except ValueError as e:
        return _error(str(e))
    except Exception:
        logging.exception("Analysis failed")
        return _error("An unexpected error occurred.", 500)
    finally:
        _cleanup(saved_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("PARSEVAL_HOST", "127.0.0.1")
    port = 5000
    print(f"[parseval] Starting server at http://{host}:{port}")
    print("[parseval] Press Ctrl+C to stop.")
    # Use waitress instead of Werkzeug's dev server.
    # Werkzeug can drop long-running connections (thumbprint building takes 15-60s)
    # and its reloader interferes with torch/sentence-transformers file loads.
    # Waitress is a production-grade WSGI server with no such limitations.
    from waitress import serve
    serve(app, host=host, port=port, threads=4, channel_timeout=300)
