"""Stylometric feature extraction.

Feature vector layout (FEATURE_SIZE = 210):
  [0]       avg_sentence_length_words
  [1]       avg_word_length_chars
  [2]       mattr_ttr (moving-average type-token ratio)
  [3:53]    function_word_frequencies (50 words)
  [53:63]   punctuation_frequencies (10 types: , ; : - ( ! ? " ' .)
  [63:78]   pos_distribution (15 tags: NOUN, VERB, ADJ, ADV, ADP, DET, PRON, AUX, CCONJ, SCONJ, PART, INTJ, NUM, PROPN, PUNCT)
  [78]      avg_paragraph_sentences
  [79]      avg_syllables_per_word
  [80]      yule_k (lexical richness)
  [81]      honore_r (lexical richness)
  [82:210]  character_trigram_frequencies (128 fixed vocabulary)

Legacy: profiles with feature_dim=65 use only indices [0:65]; new profiles use full vector.
"""

import os
import re
import string
import numpy as np

from parseval.corpus import get_spacy_nlp, split_sentences

# ─── Dimensions ─────────────────────────────────────────────────────────────
NUM_FUNCTION_WORDS = 50
NUM_PUNCT = 10
NUM_POS = 15
NUM_CHAR_TRIGRAMS = 128
# 3 + 50 + 10 + 15 + 1 + 1 + 2 + 128 = 210
FEATURE_SIZE = 3 + NUM_FUNCTION_WORDS + NUM_PUNCT + NUM_POS + 1 + 1 + 2 + NUM_CHAR_TRIGRAMS  # 210

# Top-50 English function words (ordered for stable indexing)
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these", "those", "it",
    "its", "not", "as", "if", "then", "than", "so", "also", "just", "more",
]

# Punctuation types (indices 53:63)
PUNCT_CHARS = [",", ";", ":", "-", "(", "!", "?", '"', "'", "."]

# POS tags (indices 63:78) — spaCy universal POS
POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "AUX",
    "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "PROPN", "PUNCT",
]

# Fixed vocabulary of 128 character trigrams (common English; deterministic for reproducibility)
# Built from common trigrams + padding with alphabet combinations to reach 128.
_CHAR_TRIGRAM_BASE = (
    "the and ing ion tio for ent nde has nce edt tis oft sth men ati ted ere ers "
    "con ter ive all ble com pro rea thi wit hin our you are was had her his not were "
    "been have more some will with that from this they would could about there "
    "their what when which while after before other into over just only same "
    "than very also back each much must where right still through being "
)
_CHAR_TRIGRAMS = []
_seen = set()
for word in _CHAR_TRIGRAM_BASE.split():
    for j in range(max(0, len(word) - 2)):
        tg = word[j : j + 3].lower()
        if tg not in _seen and len(tg) == 3:
            _seen.add(tg)
            _CHAR_TRIGRAMS.append(tg)
# Pad to exactly NUM_CHAR_TRIGRAMS with deterministic trigrams (aaa, aab, ... or from alphabet)
_alphabet = " " + string.ascii_lowercase
for c1 in _alphabet:
    for c2 in _alphabet:
        for c3 in _alphabet:
            tg = c1 + c2 + c3
            if tg not in _seen:
                _seen.add(tg)
                _CHAR_TRIGRAMS.append(tg)
                if len(_CHAR_TRIGRAMS) >= NUM_CHAR_TRIGRAMS:
                    break
        if len(_CHAR_TRIGRAMS) >= NUM_CHAR_TRIGRAMS:
            break
    if len(_CHAR_TRIGRAMS) >= NUM_CHAR_TRIGRAMS:
        break
CHAR_TRIGRAMS = _CHAR_TRIGRAMS[:NUM_CHAR_TRIGRAMS]
del _seen, _CHAR_TRIGRAMS, _alphabet, _CHAR_TRIGRAM_BASE

# Index offsets
IDX_PUNCT = 53
IDX_POS = 63
IDX_PARA_LEN = 78
IDX_SYLLABLES = 79
IDX_YULE_K = 80
IDX_HONORE_R = 81
IDX_CHAR_TRIGRAMS = 82

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
        window_tokens = tokens_lower[i : i + window]
        ttrs.append(len(set(window_tokens)) / window)
    return float(np.mean(ttrs))


def _yule_k(tokens: list) -> float:
    """Yule's K (lexical diversity). K = 10^4 * (sum_s(s^2 * V_s) - N) / N^2."""
    if not tokens:
        return 0.0
    from collections import Counter
    counts = Counter(t.lower() for t in tokens)
    N = len(tokens)
    V_s = Counter(counts.values())  # number of types that occur s times
    term = sum(s * s * v for s, v in V_s.items())
    k = 1e4 * (term - N) / (N * N) if N > 0 else 0.0
    return float(k)


def _honore_r(tokens: list) -> float:
    """Honore's R (hapax legomena ratio). R = 100 * (1 - V(1)/V)."""
    if not tokens:
        return 0.0
    from collections import Counter
    counts = Counter(t.lower() for t in tokens)
    V = len(counts)
    V1 = sum(1 for c in counts.values() if c == 1)
    if V == 0:
        return 0.0
    return float(100.0 * (1.0 - V1 / V))


def _char_trigram_frequencies(text: str) -> np.ndarray:
    """Return frequencies of fixed vocabulary trigrams (length NUM_CHAR_TRIGRAMS)."""
    text_lower = text.lower()
    counts = [0.0] * NUM_CHAR_TRIGRAMS
    total = 0
    for i in range(len(text_lower) - 2):
        tg = text_lower[i : i + 3]
        if tg in CHAR_TRIGRAMS:
            idx = CHAR_TRIGRAMS.index(tg)
            counts[idx] += 1.0
            total += 1
    if total == 0:
        return np.zeros(NUM_CHAR_TRIGRAMS, dtype=np.float64)
    return np.array([c / total for c in counts], dtype=np.float64)


def _extract_features_from_text(text: str) -> np.ndarray:
    """Extract the full feature vector from a single text block."""
    vec = np.zeros(FEATURE_SIZE, dtype=np.float64)

    nlp = get_spacy_nlp()
    sentences = split_sentences(text)
    tokens = _tokenize(text)
    tokens_lower = [t.lower() for t in tokens]

    if not tokens:
        return vec

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

    # [2] MATTR
    vec[2] = _mattr(tokens)

    # [3:53] function word frequencies
    total_tokens = len(tokens_lower)
    for i in range(NUM_FUNCTION_WORDS):
        vec[3 + i] = tokens_lower.count(FUNCTION_WORDS[i]) / total_tokens

    # [53:63] punctuation frequencies per sentence
    n_sents = max(1, len(sentences))
    for i, pchar in enumerate(PUNCT_CHARS):
        vec[IDX_PUNCT + i] = text.count(pchar) / n_sents

    # [63:78] POS tag distribution
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
                vec[IDX_POS + i] = pos_counts[tag] / content_token_count

    # [78] avg paragraph length in sentences
    vec[IDX_PARA_LEN] = float(len(sentences))

    # [79] avg syllables per word
    if tokens:
        syllable_counts = [_count_syllables(t) for t in tokens]
        vec[IDX_SYLLABLES] = float(np.mean(syllable_counts))

    # [80] Yule's K, [81] Honore's R
    vec[IDX_YULE_K] = _yule_k(tokens)
    vec[IDX_HONORE_R] = _honore_r(tokens)

    # [82:210] character trigram frequencies
    vec[IDX_CHAR_TRIGRAMS : IDX_CHAR_TRIGRAMS + NUM_CHAR_TRIGRAMS] = _char_trigram_frequencies(text)

    return vec


def extract_paragraph_features(paragraph: str) -> np.ndarray:
    """Extract feature vector from a single paragraph."""
    return _extract_features_from_text(paragraph)


def extract_corpus_profile(paragraphs: list, use_mahal: bool | None = None) -> dict:
    """
    Compute corpus-level stylometric profile.

    Args:
        paragraphs: List of paragraph strings.
        use_mahal: If True, include covariance for Mahalanobis distance. If False, omit.
                   If None, fall back to env PARSEVAL_USE_MAHALANOBIS.

    Returns dict with feature_mean, feature_std, expected_dist, feature_names,
    n_paragraphs, n_sentences, n_tokens, feature_dim.
    Optionally covariance_inv and expected_mahal_dist when use_mahal is True.
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

    matrix = np.array(all_vectors, dtype=np.float64)
    feature_mean = np.mean(matrix, axis=0)
    feature_std = np.std(matrix, axis=0)
    feature_std = np.where(feature_std > 0, feature_std, 1.0)

    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z_matrix = (matrix - feature_mean) / safe_std
    dists = np.linalg.norm(z_matrix, axis=1)
    expected_dist = float(np.mean(dists))

    out = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "expected_dist": max(expected_dist, 1.0),
        "feature_names": _feature_names(),
        "n_paragraphs": len(paragraphs),
        "n_sentences": total_sentences,
        "n_tokens": total_tokens,
        "feature_dim": FEATURE_SIZE,
    }

    # Optional: covariance for Mahalanobis (GUI use_mahal or env PARSEVAL_USE_MAHALANOBIS)
    if use_mahal is None:
        use_mahal = os.environ.get("PARSEVAL_USE_MAHALANOBIS", "").strip().lower() in ("1", "true", "yes")
    if use_mahal:
        cov = np.cov(matrix.T)
        reg = 1e-5 * np.eye(cov.shape[0])
        try:
            cov_inv = np.linalg.inv(cov + reg)
            mahal_dists = np.array([
                np.sqrt(np.maximum(0, (matrix[k] - feature_mean) @ cov_inv @ (matrix[k] - feature_mean)))
                for k in range(matrix.shape[0])
            ])
            expected_mahal = float(np.mean(mahal_dists))
            out["covariance_inv"] = np.linalg.inv(cov + reg).tolist()
            out["expected_mahal_dist"] = max(expected_mahal, 1.0)
        except np.linalg.LinAlgError:
            pass  # fall back to Euclidean

    return out


def build_profile_from_vectors(vectors: list) -> dict:
    """Build a stylometric profile from pre-computed feature vectors.
    Used in the intra-document scorer. Includes feature_dim."""
    if not vectors:
        return {
            "feature_mean": [0.0] * FEATURE_SIZE,
            "feature_std": [1.0] * FEATURE_SIZE,
            "feature_dim": FEATURE_SIZE,
        }
    matrix = np.array(vectors, dtype=np.float64)
    feature_mean = np.mean(matrix, axis=0)
    feature_std = np.std(matrix, axis=0)
    feature_std = np.where(feature_std > 0, feature_std, 1.0)
    safe_std = np.where(feature_std > 0, feature_std, 1.0)
    z_matrix = (matrix - feature_mean) / safe_std
    dists = np.linalg.norm(z_matrix, axis=1)
    expected_dist = float(np.mean(dists))

    out = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "expected_dist": max(expected_dist, 1.0),
        "feature_dim": matrix.shape[1],
    }

    if os.environ.get("PARSEVAL_USE_MAHALANOBIS", "").strip().lower() in ("1", "true", "yes"):
        cov = np.cov(matrix.T)
        reg = 1e-5 * np.eye(cov.shape[0])
        try:
            cov_inv = np.linalg.inv(cov + reg)
            mahal_dists = np.array([
                np.sqrt(np.maximum(0, (matrix[k] - feature_mean) @ cov_inv @ (matrix[k] - feature_mean)))
                for k in range(matrix.shape[0])
            ])
            expected_mahal = float(np.mean(mahal_dists))
            out["covariance_inv"] = np.linalg.inv(cov + reg).tolist()
            out["expected_mahal_dist"] = max(expected_mahal, 1.0)
        except np.linalg.LinAlgError:
            pass

    return out


def _feature_names() -> list:
    names = ["avg_sent_len", "avg_word_len", "mattr_ttr"]
    names += [f"fw_{w}" for w in FUNCTION_WORDS]
    names += [f"punct_{c}" for c in ["comma", "semicolon", "colon", "dash", "paren", "excl", "quest", "dquote", "squote", "period"]]
    names += [f"pos_{t.lower()}" for t in POS_TAGS]
    names += ["para_len_sents", "avg_syllables", "yule_k", "honore_r"]
    names += [f"tri_{tg}" for tg in CHAR_TRIGRAMS]
    return names


def get_feature_names_for_profile(profile: dict) -> list:
    """Return feature names for a profile (from profile or default)."""
    return profile.get("feature_names", _feature_names()[: profile.get("feature_dim", 65)])
