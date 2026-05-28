"""
retriever.py
============
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Interface de requêtage de l'index construit par build_index.py.
Utilisé par le module d'inférence (Task 8 — src/inference/chat.py).

Usage
-----
  from src.rag.retriever import Retriever

  ret = Retriever.load("data/index", "datacenter_rag", "configs/rag_config.yaml")
  results = ret.retrieve("comment redémarrer le serveur LDAP?")
  for r in results:
      print(r.score, r.text[:80])
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .embedder import Embedder
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Charge un index existant et répond aux requêtes textuelles.

    Paramètres
    ----------
    vector_store : VectorStore chargé
    embedder     : Embedder chargé (même modèle que lors de la construction)
    top_k        : nombre de résultats par défaut
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self._store = vector_store
        self._embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        index_name: str,
        config_path: str | Path = "configs/rag_config.yaml",
    ) -> "Retriever":
        """
        Charge le Retriever depuis un index persisté.

        Parameters
        ----------
        index_dir   : dossier contenant les fichiers .faiss / _meta.json / _bm25.pkl
        index_name  : préfixe des fichiers (ex : "datacenter_rag")
        config_path : fichier YAML utilisé lors de la construction
        """
        config_path = Path(config_path)
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        logger.info("Chargement du Retriever (index=%s, dir=%s)", index_name, index_dir)

        embedder = Embedder.from_config(cfg)
        store = VectorStore.from_config(cfg)
        store.load(Path(index_dir), index_name)

        return cls(
            vector_store=store,
            embedder=embedder,
            top_k=cfg.get("retrieval", {}).get("top_k", 5),
            score_threshold=cfg.get("retrieval", {}).get("score_threshold", 0.0),
        )

    # ── Retrieval ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Encode la requête et cherche les chunks les plus pertinents.

        Parameters
        ----------
        query  : question en langage naturel
        top_k  : surcharge le top_k par défaut

        Returns
        -------
        list[SearchResult] triés par score décroissant
        """
        k = top_k or self.top_k
        query_vec = self._embedder.encode_query(query)
        results = self._store.search(query_vec, query_text=query, top_k=k)
        results = [r for r in results if r.score >= self.score_threshold]
        logger.debug("Requête '%s' → %d résultats", query[:60], len(results))
        return results

    def retrieve_as_context(
        self,
        query: str,
        top_k: int | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Retourne les chunks concaténés prêts à être injectés dans un prompt LLM.
        """
        results = self.retrieve(query, top_k)
        return separator.join(r.text for r in results)

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[list[SearchResult]]:
        """Traite plusieurs requêtes en une passe."""
        return [self.retrieve(q, top_k) for q in queries]

    # ── Propriétés ───────────────────────────────────────────

    @property
    def n_chunks(self) -> int:
        return self._store.n_chunks

    def __repr__(self) -> str:
        return (
            f"Retriever(n_chunks={self.n_chunks}, top_k={self.top_k}, "
            f"model={self._embedder.model_name})"
        )
