"""
embedder.py
===========
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Encapsule le modèle d'embeddings (sentence-transformers).
Conçu pour être 100 % offline : le modèle est chargé depuis le disque
ou téléchargé une seule fois puis mis en cache.

Modèles recommandés (du plus léger au plus précis)
---------------------------------------------------
  all-MiniLM-L6-v2          ~80 MB  384 dim  anglais
  all-mpnet-base-v2         ~420 MB 768 dim  anglais
  BAAI/bge-m3               ~570 MB 1024 dim multilingue ★ recommandé
  intfloat/multilingual-e5  ~560 MB 768 dim  multilingue
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Mapping nom modèle → dimension de sortie (cache statique)
_DIM_REGISTRY: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class Embedder:
    """
    Wrapper autour de SentenceTransformer.

    Usage
    -----
    emb = Embedder("BAAI/bge-m3", device="cpu")
    vecs = emb.encode(["texte 1", "texte 2"])   # np.ndarray (2, 1024)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int = 512,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self._model = None

        self._load(cache_dir)

    def _load(self, cache_dir: str | None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers non installé. "
                "Exécuter : pip install sentence-transformers"
            ) from e

        logger.info("Chargement du modèle d'embedding : %s (device=%s)", self.model_name, self.device)
        t0 = time.time()

        kwargs: dict = {"device": self.device}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir

        self._model = SentenceTransformer(self.model_name, **kwargs)
        self._model.max_seq_length = self.max_seq_length

        logger.info("Modèle chargé en %.1fs | dim=%d", time.time() - t0, self.dim)

    def encode(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs float32.

        Returns
        -------
        np.ndarray shape (len(texts), dim)
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        logger.info("Encodage de %d textes (batch_size=%d)…", len(texts), self.batch_size)
        t0 = time.time()

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        elapsed = time.time() - t0
        logger.info(
            "Encodage terminé en %.1fs (%.0f textes/s)",
            elapsed, len(texts) / max(elapsed, 1e-9),
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode une requête unique (shape (dim,))."""
        return self.encode([query], show_progress=False)[0]

    @property
    def dim(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return _DIM_REGISTRY.get(self.model_name, 768)

    @staticmethod
    def get_dim(model_name: str) -> int:
        """Retourne la dimension sans charger le modèle (depuis le registre)."""
        return _DIM_REGISTRY.get(model_name, 768)

    @classmethod
    def from_config(cls, cfg: dict) -> "Embedder":
        emb_cfg = cfg.get("embedding", {})
        return cls(
            model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=emb_cfg.get("batch_size", 32),
            normalize=emb_cfg.get("normalize_embeddings", True),
            max_seq_length=emb_cfg.get("max_seq_length", 512),
        )
