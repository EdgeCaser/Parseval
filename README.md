# Parseval

Writing thumbprint and authorship analysis tool. Helps identify whether parts of a document were written by a different author or with outside/AI assistance.

## Features

- **Corpus thumbprinting** — build a stylometric + semantic profile from any set of reference documents (.docx, .txt, .md)
- **Corpus analysis** — compare a target document against a saved thumbprint; each paragraph is colored green-to-red by similarity
- **Self analysis** — detect style shifts within a single document without any reference corpus

## Prerequisites

- Python 3.9–3.13
- **PyTorch** — must be installed separately for your platform before running `pip install -r requirements.txt`
  - Visit https://pytorch.org/get-started/locally/ and follow the instructions for your OS/CUDA version

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/parseval.git
cd parseval

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm
```

## Running

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

**First run note:** The embedding model (~90 MB) will download automatically to your HuggingFace cache on first use. This is a one-time operation.

## Usage

### 1. Corpus Manager tab
- Enter a name for your author/thumbprint
- Upload one or more reference documents (the author's known writing)
- Click **Build Thumbprint** — this may take a minute for large corpora

### 2. Corpus Analysis tab
- Select a saved thumbprint
- Upload the document you want to check
- Click **Analyze Document**
- Each paragraph is shown with a color overlay:
  - **Green** = stylistically similar to the reference author
  - **Red** = stylistically different (possible outside help or AI)

### 3. Self Analysis tab
- Upload any document (no thumbprint needed)
- Click **Analyze for Style Shifts**
- Paragraphs that are stylistically inconsistent with the rest of the document are highlighted in red

## Storage

Thumbprints are saved to:
- **Windows:** `C:\Users\<you>\.parseval\thumbprints\`
- **Linux/macOS:** `~/.parseval/thumbprints/`

Each thumbprint is stored as a subdirectory with a UUID name containing:
- `meta.json` — metadata and stylometric profile
- `embeddings.pkl` — semantic embedding vectors

## How it works

Each paragraph is scored using two complementary methods:

1. **Stylometric analysis** (40% weight) — measures 65 linguistic features: sentence length distributions, vocabulary richness (MATTR), function word frequencies, punctuation patterns, POS tag distributions, and syllable rates. Compares using normalized Euclidean distance.

2. **Semantic embeddings** (60% weight) — encodes each paragraph using `sentence-transformers/all-MiniLM-L6-v2` and compares cosine similarity to the corpus centroid.

For **self analysis**, each paragraph is scored against a leave-one-out profile built from all other paragraphs, detecting local style outliers without any external reference.

## Supported File Formats

| Format | Extension |
|--------|-----------|
| Word documents | `.docx` |
| Plain text | `.txt`, `.text` |
| Markdown | `.md` |

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. You may use and modify it for noncommercial purposes. Commercial use requires separate permission from the licensor. See [LICENSE](LICENSE) in this repository or [polyformproject.org/licenses/noncommercial/1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) for the full terms.
