"""
vector_store.py
===============
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Gère l'index vectoriel FAISS et (optionnellement) l'index BM25.
Offre une recherche hybride dense + sparse par score fusion (RRF).

Architecture
------------
  VectorStore
  ├── FAISSIndex      : dense search (embeddings)
  ├── BM25Index       : sparse search (term matching)
  └── HybridRetriever : fusion des deux scores (Reciprocal Rank Fusion)

Persistence
-----------
  data/index/
  ├── {name}.faiss      : index FAISS sérialisé
  ├── {name}_meta.json  : métadonnées + textes des chunks
  └── {name}_bm25.pkl   : index BM25 sérialisé (si activé)
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Résultat de recherche
# ─────────────────────────────────────────────────────────────

class SearchResult(NamedTuple):
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


# ─────────────────────────────────────────────────────────────
# Index FAISS
# ─────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Encapsule un index FAISS avec support IVF / HNSW / Flat.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "IVF",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
    ):
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None
        self._is_trained = False

        try:
            import faiss  # type: ignore
            self._faiss = faiss
        except ImportError as e:
            raise ImportError(
                "FAISS non installé. Exécuter : pip install faiss-cpu"
            ) from e

        self._build_index(hnsw_m, hnsw_ef_construction)

    def _build_index(self, hnsw_m: int, hnsw_ef: int):
        faiss = self._faiss

        if self.metric == "cosine":
            # Normalise avant → produit scalaire = cosine
            base = faiss.IndexFlatIP(self.dim)
        else:
            base = faiss.IndexFlatL2(self.dim)

        if self.index_type == "Flat":
            self._index = base
            self._is_trained = True

        elif self.index_type == "IVF":
            self._index = faiss.IndexIVFFlat(base, self.dim, self.nlist)
            self._index.nprobe = self.nprobe

        elif self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self.dim, hnsw_m)
            self._index.hnsw.efConstruction = hnsw_ef
            self._is_trained = True

        else:
            raise ValueError(f"index_type inconnu : {self.index_type}")

    def train(self, vectors: np.ndarray):
        if not self._is_trained:
            logger.info("Entraînement IVF sur %d vecteurs…", len(vectors))
            self._index.train(vectors)
            self._is_trained = True

    def add(self, vectors: np.ndarray):
        assert self._is_trained, "Index non entraîné — appeler .train() d'abord"
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        self._index.add(vectors)
        logger.debug("Ajout de %d vecteurs (total=%d)", len(vectors), self._index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (scores, indices)."""
        if self.metric == "cosine":
            query_vec = self._normalize(query_vec)
        scores, indices = self._index.search(query_vec, top_k)
        return scores, indices

    def save(self, path: Path):
        self._faiss.write_index(self._index, str(path))
        logger.info("Index FAISS sauvegardé : %s", path)

    def load(self, path: Path):
        self._index = self._faiss.read_index(str(path))
        self._is_trained = True
        logger.info("Index FAISS chargé : %s (%d vecteurs)", path, self._index.ntotal)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index else 0

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms


# ─────────────────────────────────────────────────────────────
# Index BM25
# ─────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 implémenté de zéro (sans dépendance rank_bm25).
    Compatible avec la recherche hybride.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._n: int = 0

    def build(self, texts: list[str]):
        self._docs = [t.lower().split() for t in texts]
        self._n = len(self._docs)
        self._avg_dl = sum(len(d) for d in self._docs) / max(1, self._n)
        self._compute_idf()
        logger.info("Index BM25 construit : %d documents, vocab=%d", self._n, len(self._idf))

    def _compute_idf(self):
        df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log((self._n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        terms = query.lower().split()
        scores = np.zeros(self._n, dtype=np.float32)

        for term in terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0:
                continue
            for i, doc in enumerate(self._docs):
                tf = doc.count(term)
                dl = len(doc)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"docs": self._docs, "idf": self._idf, "avg_dl": self._avg_dl, "n": self._n}, f)
        logger.info("Index BM25 sauvegardé : %s", path)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._docs = data["docs"]
        self._idf = data["idf"]
        self._avg_dl = data["avg_dl"]
        self._n = data["n"]
        logger.info("Index BM25 chargé : %s (%d docs)", path, self._n)


# ─────────────────────────────────────────────────────────────
# VectorStore principal
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Façade principale qui orchestre FAISS + BM25 + fusion.

    Usage rapide
    ------------
    store = VectorStore.from_config(cfg)
    store.build(chunks, embeddings)
    store.save("data/index", "datacenter_rag")
    results = store.search("quelle est la procédure de démarrage?", embedder)
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index | None = None,
        alpha: float = 0.6,
        score_threshold: float = 0.0,
    ):
        self._faiss = faiss_index
        self._bm25 = bm25_index
        self.alpha = alpha
        self.score_threshold = score_threshold

        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._ids: list[str] = []

    # ── Construction ─────────────────────────────────────────

    def build(self, chunks: list[Any], embeddings: np.ndarray):
        """
        Parameters
        ----------
        chunks     : liste de Chunk (chunker.py)
        embeddings : np.ndarray shape (N, dim) float32
        """
        assert len(chunks) == len(embeddings), "Mismatch chunks / embeddings"

        self._texts = [c.text for c in chunks]
        self._metadata = [c.metadata for c in chunks]
        self._ids = [c.chunk_id or f"chunk_{i}" for i, c in enumerate(chunks)]

        # FAISS
        logger.info("Construction FAISS sur %d vecteurs…", len(embeddings))
        t0 = time.time()
        self._faiss.train(embeddings)
        self._faiss.add(embeddings)
        logger.info("FAISS construit en %.1fs", time.time() - t0)

        # BM25
        if self._bm25 is not None:
            t0 = time.time()
            self._bm25.build(self._texts)
            logger.info("BM25 construit en %.1fs", time.time() - t0)

    # ── Recherche ────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Recherche hybride (dense + sparse).
        Si BM25 désactivé ou query_text vide → recherche dense seule.
        """
        if self._bm25 is not None and query_text:
            return self._hybrid_search(query_vec, query_text, top_k)
        return self._dense_search(query_vec, top_k)

    def _dense_search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        scores, indices = self._faiss.search(query_vec.reshape(1, -1), top_k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < self.score_threshold:
                continue
            results.append(SearchResult(
                chunk_id=self._ids[idx],
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadata[idx],
            ))
        return results

    def _hybrid_search(
        self, query_vec: np.ndarray, query_text: str, top_k: int
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion (RRF) avec pondération alpha."""
        k_rrf = 60  # constante RRF standard

        dense_results = self._dense_search(query_vec, top_k * 2)
        sparse_results = self._bm25.score(query_text, top_k * 2)

        rrf_scores: dict[int, float] = {}

        # Dense ranks
        for rank, res in enumerate(dense_results):
            idx = self._ids.index(res.chunk_id)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.alpha / (k_rrf + rank + 1)

        # Sparse ranks
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.alpha) / (k_rrf + rank + 1)

        sorted_idxs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx in sorted_idxs:
            results.append(SearchResult(
                chunk_id=self._ids[idx],
                text=self._texts[idx],
                score=rrf_scores[idx],
                metadata=self._metadata[idx],
            ))
        return results

    # ── Persistence ──────────────────────────────────────────

    def save(self, output_dir: Path | str, name: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._faiss.save(output_dir / f"{name}.faiss")

        meta_payload = {
            "ids": self._ids,
            "texts": self._texts,
            "metadata": self._metadata,
            "n_chunks": len(self._texts),
        }
        with open(output_dir / f"{name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        if self._bm25 is not None:
            self._bm25.save(output_dir / f"{name}_bm25.pkl")

        logger.info("Index complet sauvegardé dans %s/", output_dir)

    def load(self, output_dir: Path | str, name: str):
        output_dir = Path(output_dir)

        self._faiss.load(output_dir / f"{name}.faiss")

        with open(output_dir / f"{name}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self._ids = meta["ids"]
        self._texts = meta["texts"]
        self._metadata = meta["metadata"]

        bm25_path = output_dir / f"{name}_bm25.pkl"
        if bm25_path.exists() and self._bm25 is not None:
            self._bm25.load(bm25_path)

        logger.info("Index '%s' chargé : %d chunks", name, len(self._texts))

    @property
    def n_chunks(self) -> int:
        return len(self._texts)

    # ── Factory depuis config ─────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "VectorStore":
        from .embedder import Embedder
        dim = Embedder.get_dim(cfg["embedding"]["model_name"])

        faiss_cfg = cfg.get("faiss", {})
        faiss_idx = FAISSIndex(
            dim=dim,
            index_type=faiss_cfg.get("index_type", "IVF"),
            metric=faiss_cfg.get("metric", "cosine"),
            nlist=faiss_cfg.get("nlist", 100),
            nprobe=faiss_cfg.get("nprobe", 10),
            hnsw_m=faiss_cfg.get("hnsw_m", 32),
            hnsw_ef_construction=faiss_cfg.get("hnsw_ef_construction", 200),
        )

        bm25_idx = None
        if cfg.get("bm25", {}).get("enabled", False):
            bm25_cfg = cfg["bm25"]
            bm25_idx = BM25Index(k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75))

        hybrid_cfg = cfg.get("hybrid", {})
        return cls(
            faiss_index=faiss_idx,
            bm25_index=bm25_idx,
            alpha=hybrid_cfg.get("alpha", 0.6),
            score_threshold=cfg.get("retrieval", {}).get("score_threshold", 0.0),
        )
