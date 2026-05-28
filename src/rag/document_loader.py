"""
document_loader.py
==================
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Charge des documents depuis le disque dans un format unifié.
Supporte : TXT, Markdown, PDF, JSON, JSONL, CSV.

Design :
  - Strategy pattern : chaque format a son propre loader
  - Sortie unifiée : liste de Document(content, metadata)
  - Aucune dépendance réseau (100 % offline)
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Dataclass unifié
# ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """Représente un fragment de document avec ses métadonnées."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.content = self.content.strip()

    def is_valid(self, min_chars: int = 30) -> bool:
        return len(self.content) >= min_chars


# ─────────────────────────────────────────────────────────────
# Loaders par format
# ─────────────────────────────────────────────────────────────

class TextLoader:
    """Charge les fichiers .txt et .md bruts."""

    SUPPORTED = {".txt", ".md"}

    def load(self, path: Path) -> list[Document]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [Document(
                content=text,
                metadata={"source": str(path), "format": path.suffix.lstrip(".")},
            )]
        except Exception as exc:
            logger.warning("TextLoader: impossible de lire %s — %s", path, exc)
            return []


class JsonLoader:
    """Charge les fichiers .json (objet unique ou liste)."""

    SUPPORTED = {".json"}

    _TEXT_KEYS = ("content", "text", "body", "instruction", "output",
                  "question", "answer", "passage", "description")

    def load(self, path: Path) -> list[Document]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("JsonLoader: %s — %s", path, exc)
            return []

        items = data if isinstance(data, list) else [data]
        docs: list[Document] = []
        for item in items:
            text = self._extract_text(item)
            if text:
                docs.append(Document(
                    content=text,
                    metadata={"source": str(path), "format": "json"},
                ))
        return docs

    def _extract_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in self._TEXT_KEYS:
                if key in item and isinstance(item[key], str):
                    return item[key]
            # Fallback : sérialise tout en texte clé: valeur
            return "\n".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, str))
        return ""


class JsonlLoader:
    """Charge les fichiers .jsonl (une entrée JSON par ligne)."""

    SUPPORTED = {".jsonl"}

    def __init__(self):
        self._json_loader = JsonLoader()

    def load(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = self._json_loader._extract_text(item)
                    if text:
                        docs.append(Document(
                            content=text,
                            metadata={"source": str(path), "format": "jsonl", "line": i},
                        ))
                except json.JSONDecodeError:
                    logger.debug("JsonlLoader: ligne %d ignorée dans %s", i, path)
        except Exception as exc:
            logger.warning("JsonlLoader: %s — %s", path, exc)
        return docs


class CsvLoader:
    """Charge les fichiers .csv (concatène les colonnes textuelles)."""

    SUPPORTED = {".csv"}

    def load(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = " | ".join(v for v in row.values() if v and isinstance(v, str))
                    if text.strip():
                        docs.append(Document(
                            content=text,
                            metadata={"source": str(path), "format": "csv", "row": i},
                        ))
        except Exception as exc:
            logger.warning("CsvLoader: %s — %s", path, exc)
        return docs


class PdfLoader:
    """
    Charge les fichiers .pdf.
    Nécessite `pypdf` (pip install pypdf).
    Fallback silencieux si non disponible.
    """

    SUPPORTED = {".pdf"}

    def load(self, path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            logger.warning("pypdf non installé. Ignoré : %s", path)
            return []

        docs: list[Document] = []
        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        content=text,
                        metadata={"source": str(path), "format": "pdf", "page": i + 1},
                    ))
        except Exception as exc:
            logger.warning("PdfLoader: %s — %s", path, exc)
        return docs


# ─────────────────────────────────────────────────────────────
# Registry & DocumentLoader public
# ─────────────────────────────────────────────────────────────

_LOADERS = [TextLoader(), JsonLoader(), JsonlLoader(), CsvLoader(), PdfLoader()]

_REGISTRY: dict[str, Any] = {}
for _loader in _LOADERS:
    for _ext in _loader.SUPPORTED:
        _REGISTRY[_ext] = _loader


class DocumentLoader:
    """
    Façade principale.

    Usage
    -----
    loader = DocumentLoader(extensions=[".txt", ".md", ".jsonl"])
    docs = loader.load_directory("data/raw", recursive=True)
    """

    def __init__(
        self,
        extensions: list[str] | None = None,
        min_chars: int = 30,
    ):
        self.extensions = set(extensions or list(_REGISTRY.keys()))
        self.min_chars = min_chars

    # ── API publique ─────────────────────────────────────────

    def load_file(self, path: Path | str) -> list[Document]:
        path = Path(path)
        loader = _REGISTRY.get(path.suffix.lower())
        if loader is None:
            logger.debug("Format non supporté : %s", path.suffix)
            return []
        docs = loader.load(path)
        valid = [d for d in docs if d.is_valid(self.min_chars)]
        logger.debug("%s → %d doc(s) valides", path.name, len(valid))
        return valid

    def load_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> list[Document]:
        directory = Path(directory)
        if not directory.exists():
            logger.warning("Répertoire introuvable : %s", directory)
            return []

        pattern = "**/*" if recursive else "*"
        all_docs: list[Document] = []
        files = sorted(directory.glob(pattern))

        logger.info("Scan de %s (%d fichier(s) trouvés)", directory, len(files))
        for file in files:
            if file.is_file() and file.suffix.lower() in self.extensions:
                docs = self.load_file(file)
                all_docs.extend(docs)

        logger.info("Total documents chargés : %d", len(all_docs))
        return all_docs

    def iter_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> Generator[Document, None, None]:
        """Version générateur pour les très gros corpus."""
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        for file in sorted(directory.glob(pattern)):
            if file.is_file() and file.suffix.lower() in self.extensions:
                yield from self.load_file(file)
