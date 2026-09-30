import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILLERS = {"um", "uh", "like", "actually", "basically", "you know"}


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", (text or "").lower())


def keyword_coverage(answer, expected):
    answer_tokens = set(tokenize(answer))
    reference = (expected or "").replace(";", " ")
    concepts = [c.strip().lower() for c in reference.split() if c.strip()]
    if not concepts:
        return 0.0
    hits = sum(1 for c in concepts if all(t in answer_tokens for t in tokenize(c)))
    return hits / len(concepts)


def semantic_similarity(answer, expected):
    answer_text = (answer or "").strip()
    reference = (expected or "").replace(";", " ").strip()
    if not answer_text and not reference:
        return 0.0
    if not reference:
        reference = answer_text
    corpus = [answer_text, reference]
    vec = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    return float(cosine_similarity(vec[0:1], vec[1:2])[0, 0])


def extract_features(answer, expected=None, question=None):
    if expected is None and question is not None:
        expected = question
    expected = expected or ""
    tokens = tokenize(answer)
    filler_count = sum(1 for t in tokens if t in FILLERS)
    word_count = len(tokens)
    return {
        "word_count": word_count,
        "sentence_count": max(1, len(re.findall(r"[.!?]", answer or ""))),
        "filler_rate": filler_count / max(1, word_count),
        "keyword_coverage": keyword_coverage(answer, expected),
        "semantic_similarity": semantic_similarity(answer, expected),
    }


def vectorize_rows(rows):
    return np.array([[r[k] for k in ["word_count", "sentence_count", "filler_rate", "keyword_coverage", "semantic_similarity"]] for r in rows], dtype=float)
