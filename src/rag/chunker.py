"""
chunker.py
==========
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Découpe les documents en chunks optimisés pour le retrieval.

Stratégies disponibles :
  - fixed      : fenêtre glissante à taille fixe
  - recursive  : découpe hiérarchique (paragraphes → phrases → mots)
  - semantic   : regroupe les phrases par similarité (nécessite le modèle)

Référence : LangChain RecursiveCharacterTextSplitter (re-implémenté de zéro,
sans dépendance à LangChain).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Fragment de texte prêt à être encodé."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        self.text = self.text.strip()

    def __len__(self) -> int:
        return len(self.text.split())   # word count (proxy pour tokens)


# ─────────────────────────────────────────────────────────────
# Utilitaire : tokenisation légère (sans modèle)
# ─────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text)


# ─────────────────────────────────────────────────────────────
# Fixed chunker
# ─────────────────────────────────────────────────────────────

class FixedChunker:
    """
    Fenêtre glissante de taille fixe (en mots).
    Simple, rapide, sans biais de contenu.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        start = 0
        meta = metadata or {}

        while start < len(words):
            end = start + self.chunk_size
            fragment = " ".join(words[start:end])
            chunks.append(Chunk(text=fragment, metadata=dict(meta, chunk_index=len(chunks))))
            if end >= len(words):
                break
            start = end - self.chunk_overlap

        return [c for c in chunks if len(c) > 0]


# ─────────────────────────────────────────────────────────────
# Recursive chunker
# ─────────────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Découpe hiérarchique par séparateurs décroissants.
    Préserve la cohérence sémantique des paragraphes.

    Algorithme :
      1. Essayer de couper sur \n\n (paragraphes)
      2. Si un fragment est encore trop grand → couper sur \n
      3. Puis sur ". " (phrases)
      4. Puis sur " " (mots) en dernier recours
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
        length_fn: Callable[[str], int] = _word_count,
        min_chunk_size: int = 20,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_fn = length_fn
        self.min_chunk_size = min_chunk_size

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        raw = self._split_text(text, self.separators)
        raw = self._merge_splits(raw)
        meta = metadata or {}
        chunks = [
            Chunk(text=r, metadata=dict(meta, chunk_index=i))
            for i, r in enumerate(raw)
            if self.length_fn(r) >= self.min_chunk_size
        ]
        return chunks

    # ── Interne ──────────────────────────────────────────────

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        good: list[str] = []
        bad: list[str] = []

        for s in splits:
            if self.length_fn(s) <= self.chunk_size:
                good.append(s)
            else:
                if good:
                    bad.extend(good)
                    good = []
                bad.extend(self._split_text(s, remaining))

        if good:
            bad.extend(good)

        # Recolle les petits fragments avec le séparateur original
        result: list[str] = []
        for part in bad:
            if result and self.length_fn(result[-1]) + self.length_fn(part) <= self.chunk_size:
                result[-1] = result[-1] + sep + part
            else:
                result.append(part)

        return result

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Fusionne les petits fragments et applique le chevauchement."""
        merged: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_fn(split)

            if current_len + split_len > self.chunk_size and current_words:
                merged.append(" ".join(current_words))
                # Garde le chevauchement
                overlap_words: list[str] = []
                overlap_len = 0
                for w in reversed(current_words):
                    overlap_len += 1
                    overlap_words.insert(0, w)
                    if overlap_len >= self.chunk_overlap:
                        break
                current_words = overlap_words
                current_len = self.length_fn(" ".join(current_words))

            current_words.extend(split.split())
            current_len = self.length_fn(" ".join(current_words))

        if current_words:
            merged.append(" ".join(current_words))

        return [m.strip() for m in merged if m.strip()]


# ─────────────────────────────────────────────────────────────
# Sentence-based chunker (optionnel)
# ─────────────────────────────────────────────────────────────

class SentenceChunker:
    """
    Découpe en phrases puis regroupe jusqu'à chunk_size mots.
    Plus précis que RecursiveChunker pour les textes techniques.
    """

    _SENTENCE_ENDS = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 2):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  # nombre de phrases de chevauchement

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        sentences = self._SENTENCE_ENDS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        start = 0
        meta = metadata or {}

        while start < len(sentences):
            group: list[str] = []
            word_count = 0

            i = start
            while i < len(sentences):
                words = len(sentences[i].split())
                if word_count + words > self.chunk_size and group:
                    break
                group.append(sentences[i])
                word_count += words
                i += 1

            text_chunk = " ".join(group)
            chunks.append(Chunk(text=text_chunk, metadata=dict(meta, chunk_index=len(chunks))))

            if i >= len(sentences):
                break
            start = max(start + 1, i - self.chunk_overlap)

        return [c for c in chunks if c.text]


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
    min_chunk_size: int = 20,
) -> FixedChunker | RecursiveChunker | SentenceChunker:
    """
    Factory function — instancie le bon chunker selon la config.

    Parameters
    ----------
    strategy : "fixed" | "recursive" | "sentence"
    """
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "recursive":
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            min_chunk_size=min_chunk_size,
        )
    elif strategy == "sentence":
        return SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Stratégie inconnue : '{strategy}'. Choisir parmi fixed/recursive/sentence.")
