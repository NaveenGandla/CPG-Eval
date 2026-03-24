"""Lightweight keyword extraction using TF-IDF."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract top-N keywords from text using TF-IDF on sentence-level pseudo-documents.

    Falls back to simple frequency-based noun extraction if the text is too short.
    """
    if not text or not text.strip():
        return []

    # Split text into sentence-level pseudo-documents for TF-IDF
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 2:
        # Fallback: frequency-based extraction for very short text
        return _frequency_keywords(text, top_n)

    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words="english",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Sum TF-IDF scores across all sentences to get global importance
        scores = tfidf_matrix.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [term for term, _ in ranked[:top_n]]
    except ValueError:
        return _frequency_keywords(text, top_n)


def _frequency_keywords(text: str, top_n: int) -> list[str]:
    """Simple frequency-based keyword extraction as fallback."""
    # Basic stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "that", "this", "these", "those", "it", "its",
        "also", "which", "who", "whom", "what", "when", "where", "how", "why",
        "there", "their", "they", "them", "he", "she", "we", "you", "your",
        "our", "his", "her",
    }

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())
    words = [w for w in words if w not in stopwords]

    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[:top_n]]
