"""Stylometric feature extraction.

Produces a 65-dimensional feature vector per paragraph:
  [0]     avg_sentence_length_words
  [1]     avg_word_length_chars
  [2]     mattr_ttr (moving-average type-token ratio)
  [3:53]  function_word_frequencies (50 words)
  [53:59] punctuation_frequencies (6 types)
  [59:63] pos_distribution (NOUN, VERB, ADJ, ADV)
  [63]    avg_paragraph_sentences
  [64]    avg_syllables_per_word

FEATURE_SIZE = 65
"""

import re
import numpy as np

from parseval.corpus import get_spacy_nlp, split_sentences

FEATURE_SIZE = 65

# Top-50 English function words (ordered for stable indexing)
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these", "those", "it",
    "its", "not", "as", "if", "then", "than", "so", "also", "just", "more",
]

# Punctuation types tracked (indices 53-58)
PUNCT_CHARS = [",", ";", ":", "-", "(", "!"]

# POS tags tracked (indices 59-62)
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV"]

# MATTR window size
MATTR_WINDOW = 100

# pyphen dictionary (lazy-loaded)
_pyphen_dic = None


def _get_pyphen():
    global _pyphen_dic
    if _pyphen_dic is None:
        try:
            import pyphen
            _pyphen_dic = pyphen.Pyphen(lang="en")
        except Exception:
            _pyphen_dic = False
    return _pyphen_dic


def _count_syllables(word: str) -> int:
    """Count syllables in a word using pyphen, fallback to vowel counting."""
    dic = _get_pyphen()
    if dic:
        positions = dic.positions(word.lower())
        return len(positions) + 1
    # Fallback: count vowel groups
    vowels = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(vowels))


def _tokenize(text: str) -> list:
    """Simple word tokenizer: extract alphabetic tokens."""
    return re.findall(r"[a-zA-Z']+", text)


def _mattr(tokens: list, window: int = MATTR_WINDOW) -> float:
    """Moving Average Type-Token Ratio."""
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    if len(tokens_lower) < window:
        return len(set(tokens_lower)) / len(tokens_lower)
    ttrs = []
    for i in range(len(tokens_lower) - window + 1):
        window_tokens = tokens_lower[i:i + window]
        ttrs.append(len(set(window_tokens)) / window)
    return float(np.mean(ttrs))


def _extract_features_from_text(text: str) -> np.ndarray:
    """Extract the 65-dimensional feature vector from a single text block."""
    vec = np.zeros(FEATURE_SIZE, dtype=np.float64)

    nlp = get_spacy_nlp()
    sentences = split_sentences(text)
    tokens = _tokenize(text)
    tokens_lower = [t.lower() for t in tokens]

    if not tokens:
        return vec  # All zeros for empty text

    # [0] avg sentence length (words)
    sent_lengths = []
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if sent_tokens:
            sent_lengths.append(len(sent_tokens))
    vec[0] = float(np.mean(sent_lengths)) if sent_lengths else 0.0

    # [1] avg word length (chars)
    word_lengths = [len(t) for t in tokens]
    vec[1] = float(np.mean(word_lengths)) if word_lengths else 0.0

    # [2] MATTR type-token ratio
    vec[2] = _mattr(tokens)

    # [3:53] function word frequencies
    total_tokens = len(tokens_lower)
    for i, fw in enumerate(FUNCTION_WORDS):
        vec[3 + i] = tokens_lower.count(fw) / total_tokens

    # [53:59] punctuation frequencies per sentence
    n_sents = max(1, len(sentences))
    for i, pchar in enumerate(PUNCT_CHARS):
        vec[53 + i] = text.count(pchar) / n_sents

    # [59:63] POS tag distribution (requires spaCy)
    if nlp:
        doc = nlp(text)
        pos_counts = {tag: 0 for tag in POS_TAGS}
        content_token_count = 0
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
                content_token_count += 1
        if content_token_count > 0:
            for i, tag in enumerate(POS_TAGS):
                vec[59 + i] = pos_counts[tag] / content_token_count
    # else: leave as zeros (graceful degradation)

    # [63] avg paragraph length in sentences
    vec[63] = float(len(sentences))

    # [64] avg syllables per word
    if tokens:
        syllable_counts = [_count_syllables(t) for t in tokens]
        vec[64] = float(np.mean(syllable_counts))

    return vec


def extract_paragraph_features(paragraph: str) -> np.ndarray:
    """Extract 65-dim feature vector from a single paragraph."""
    return _extract_features_from_text(paragraph)


def extract_corpus_profile(paragraphs: list) -> dict:
    """
    Compute corpus-level stylometric profile from a list of paragraphs.

    Returns a dict with:
      feature_mean: list[float] (65 values)
      feature_std:  list[float] (65 values, zeros replaced with 1.0)
      feature_names: list[str]
      n_paragraphs: int
      n_sentences:  int
      n_tokens:     int
    """
    if not paragraphs:
        raise ValueError("No paragraphs provided for corpus profile.")

    all_vectors = []
    total_sentences = 0
    total_tokens = 0

    for para in paragraphs:
        vec = _extract_features_from_text(para)
        all_vectors.append(vec)
        total_sentences += max(1, len(split_sentences(para)))
        total_tokens += len(_tokenize(para))

    matrix = np.array(all_vectors, dtype=np.float64)  # shape: (N, 65)
    feature_mean = np.mean(matrix, axis=0)
    feature_std = np.std(matrix, axis=0)

    # Replace zero stds with 1.0 to avoid division-by-zero during scoring
    feature_std = np.where(feature_std > 0, feature_std, 1.0)

    # Compute expected in-distribution z-score distance.
    # This is the mean ||z|| across all corpus paragraphs, and is used by the
    # scorer to calibrate the stylometric similarity scale — a paragraph scoring
    # at this distance is "as typical as an average corpus paragraph" and gets 1.0.
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z_matrix = (matrix - feature_mean) / safe_std
    dists = np.linalg.norm(z_matrix, axis=1)
    expected_dist = float(np.mean(dists))

    return {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "expected_dist": max(expected_dist, 1.0),  # floor at 1.0 to avoid divide-by-zero
        "feature_names": _feature_names(),
        "n_paragraphs": len(paragraphs),
        "n_sentences": total_sentences,
        "n_tokens": total_tokens,
    }


def build_profile_from_vectors(vectors: list) -> dict:
    """Build a stylometric profile from pre-computed feature vectors (numpy arrays).
    Used in the intra-document scorer to avoid re-extracting features."""
    if not vectors:
        return {
            "feature_mean": [0.0] * FEATURE_SIZE,
            "feature_std": [1.0] * FEATURE_SIZE,
        }
    matrix = np.array(vectors, dtype=np.float64)
    feature_mean = np.mean(matrix, axis=0)
    feature_std = np.std(matrix, axis=0)
    feature_std = np.where(feature_std > 0, feature_std, 1.0)
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z_matrix = (matrix - feature_mean) / safe_std
    dists = np.linalg.norm(z_matrix, axis=1)
    expected_dist = float(np.mean(dists))
    return {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "expected_dist": max(expected_dist, 1.0),
    }


def _feature_names() -> list:
    names = ["avg_sent_len", "avg_word_len", "mattr_ttr"]
    names += [f"fw_{w}" for w in FUNCTION_WORDS]
    names += [f"punct_{c}" for c in ["comma", "semicolon", "colon", "dash", "paren", "excl"]]
    names += [f"pos_{t.lower()}" for t in POS_TAGS]
    names += ["para_len_sents", "avg_syllables"]
    return names
