"""Fonctions texte simples utilisees par le dataset et le RAG."""

import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_./:-]+")

STOP_WORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "la",
    "le",
    "les",
    "leur",
    "mais",
    "ou",
    "par",
    "pas",
    "pour",
    "que",
    "qui",
    "se",
    "si",
    "sur",
    "un",
    "une",
    "utiliser",
}


def normalize_text(text: str) -> str:
    """Nettoie les espaces et garde un texte lisible."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenisation volontairement simple, sans dependance externe."""
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def token_counts(text: str) -> dict[str, int]:
    """Compte les tokens d'un texte pour le scoring RAG."""
    return dict(Counter(tokenize(text)))


def split_text(text: str, max_words: int = 180, overlap: int = 35) -> list[str]:
    """Decoupe un document en morceaux avec leger chevauchement."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks
