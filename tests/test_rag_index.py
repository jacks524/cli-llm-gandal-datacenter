"""
tests/test_rag_index.py
=======================
Task 6 – RAG Index | Data Center LLM Chatbot
Tests unitaires — exécuter avec : pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Fixtures ─────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Le serveur LDAP doit être redémarré via systemctl restart slapd.",
    "La procédure de backup nécessite l'accès root sur le nœud primaire.",
    "Les logs Nginx se trouvent dans /var/log/nginx/access.log et error.log.",
    "Pour monitorer la RAM : free -h ou htop. Seuil critique à 90%.",
    "La configuration Kubernetes se trouve dans /etc/kubernetes/manifests/.",
    "En cas de failover, basculer vers le nœud secondaire avec la commande suivante.",
    "Le certificat SSL expire le 2025-12-31, renouvellement via certbot.",
]


# ── Tests document_loader ─────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt(self, tmp_path):
        from src.rag.document_loader import DocumentLoader, TextLoader
        f = tmp_path / "test.txt"
        f.write_text("Ceci est un document de test.\nLigne deux.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert len(docs) == 1
        assert "document de test" in docs[0].content
        assert docs[0].metadata["format"] == "txt"

    def test_load_jsonl(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"content": "Procédure de démarrage LDAP.", "id": 1}),
            json.dumps({"content": "Configuration réseau datacenter.", "id": 2}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert len(docs) == 2
        assert "LDAP" in docs[0].content

    def test_load_json_list(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        f = tmp_path / "data.json"
        data = [{"text": "Entrée 1"}, {"text": "Entrée 2"}, {"text": "Entrée 3"}]
        f.write_text(json.dumps(data), encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert len(docs) == 3

    def test_load_csv(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        f = tmp_path / "data.csv"
        f.write_text("titre,description\nServeur,Procédure de maintenance\nRéseau,Config VLAN", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert len(docs) >= 1

    def test_load_directory_recursive(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "a.txt").write_text("Document A " * 10, encoding="utf-8")
        (subdir / "b.txt").write_text("Document B " * 10, encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=True)
        assert len(docs) == 2

    def test_unsupported_format_ignored(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        f = tmp_path / "file.xyz"
        f.write_text("contenu quelconque")

        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert docs == []

    def test_min_chars_filter(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        f = tmp_path / "short.txt"
        f.write_text("hi", encoding="utf-8")

        loader = DocumentLoader(min_chars=30)
        docs = loader.load_file(f)
        assert docs == []


# ── Tests chunker ─────────────────────────────────────────────

class TestChunker:

    LONG_TEXT = " ".join(["mot"] * 1000)

    def test_fixed_chunker_produces_chunks(self):
        from src.rag.chunker import FixedChunker
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(self.LONG_TEXT)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text.split()) <= 110  # tolérance overlap

    def test_recursive_chunker_respects_size(self):
        from src.rag.chunker import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        chunks = chunker.split(self.LONG_TEXT)
        assert len(chunks) > 1

    def test_sentence_chunker(self):
        from src.rag.chunker import SentenceChunker
        text = "Première phrase. Deuxième phrase. Troisième phrase. " * 20
        chunker = SentenceChunker(chunk_size=20, chunk_overlap=1)
        chunks = chunker.split(text)
        assert len(chunks) > 1
        assert all(c.text for c in chunks)

    def test_get_chunker_factory(self):
        from src.rag.chunker import get_chunker
        for strategy in ["fixed", "recursive", "sentence"]:
            c = get_chunker(strategy=strategy)
            assert c is not None

    def test_get_chunker_invalid_raises(self):
        from src.rag.chunker import get_chunker
        with pytest.raises(ValueError):
            get_chunker(strategy="unknown")

    def test_metadata_preserved(self):
        from src.rag.chunker import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        meta = {"source": "test.txt", "page": 1}
        text = " ".join(["word"] * 200)
        chunks = chunker.split(text, metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "test.txt"
            assert "chunk_index" in c.metadata


# ── Tests BM25Index ───────────────────────────────────────────

class TestBM25Index:

    def test_build_and_score(self):
        from src.rag.vector_store import BM25Index
        bm25 = BM25Index()
        bm25.build(SAMPLE_TEXTS)
        results = bm25.score("serveur LDAP redémarrer", top_k=3)
        assert len(results) > 0
        top_idx, top_score = results[0]
        # Le doc 0 mentionne LDAP et serveur
        assert top_idx == 0 or top_score > 0

    def test_bm25_persistence(self, tmp_path):
        from src.rag.vector_store import BM25Index
        bm25 = BM25Index()
        bm25.build(SAMPLE_TEXTS)

        path = tmp_path / "bm25.pkl"
        bm25.save(path)

        bm25_loaded = BM25Index()
        bm25_loaded.load(path)
        results = bm25_loaded.score("backup root", top_k=3)
        assert len(results) > 0

    def test_bm25_empty_query(self):
        from src.rag.vector_store import BM25Index
        bm25 = BM25Index()
        bm25.build(SAMPLE_TEXTS)
        results = bm25.score("xyzabc123notaword", top_k=3)
        assert results == []


# ── Tests FAISSIndex ──────────────────────────────────────────

class TestFAISSIndex:

    DIM = 64

    def _random_vecs(self, n):
        return np.random.randn(n, self.DIM).astype(np.float32)

    def test_flat_index_add_and_search(self):
        from src.rag.vector_store import FAISSIndex
        idx = FAISSIndex(dim=self.DIM, index_type="Flat", metric="cosine")
        vecs = self._random_vecs(20)
        idx.add(vecs)
        query = self._random_vecs(1)
        scores, indices = idx.search(query, top_k=5)
        assert scores.shape == (1, 5)
        assert all(0 <= i < 20 for i in indices[0])

    def test_ivf_requires_training(self):
        from src.rag.vector_store import FAISSIndex
        idx = FAISSIndex(dim=self.DIM, index_type="IVF", nlist=4)
        vecs = self._random_vecs(100)
        idx.train(vecs)
        idx.add(vecs)
        scores, indices = idx.search(self._random_vecs(1), top_k=5)
        assert len(indices[0]) == 5

    def test_faiss_persistence(self, tmp_path):
        from src.rag.vector_store import FAISSIndex
        idx = FAISSIndex(dim=self.DIM, index_type="Flat")
        vecs = self._random_vecs(10)
        idx.add(vecs)

        path = tmp_path / "test.faiss"
        idx.save(path)

        idx2 = FAISSIndex(dim=self.DIM, index_type="Flat")
        idx2.load(path)
        assert idx2.ntotal == 10


# ── Tests VectorStore (intégration) ───────────────────────────

class TestVectorStore:

    DIM = 64

    def _make_chunks(self, texts):
        from src.rag.chunker import Chunk
        return [Chunk(text=t, metadata={"source": f"doc_{i}"}, chunk_id=f"chunk_{i:04d}")
                for i, t in enumerate(texts)]

    def _random_embeddings(self, n):
        return np.random.randn(n, self.DIM).astype(np.float32)

    def test_build_and_dense_search(self):
        from src.rag.vector_store import FAISSIndex, VectorStore
        faiss_idx = FAISSIndex(dim=self.DIM, index_type="Flat")
        store = VectorStore(faiss_index=faiss_idx)
        chunks = self._make_chunks(SAMPLE_TEXTS)
        embeds = self._random_embeddings(len(SAMPLE_TEXTS))
        store.build(chunks, embeds)

        query = self._random_embeddings(1)[0]
        results = store.search(query, top_k=3)
        assert len(results) == 3
        assert all(r.text for r in results)

    def test_hybrid_search(self):
        from src.rag.vector_store import BM25Index, FAISSIndex, VectorStore
        faiss_idx = FAISSIndex(dim=self.DIM, index_type="Flat")
        bm25 = BM25Index()
        store = VectorStore(faiss_index=faiss_idx, bm25_index=bm25, alpha=0.5)
        chunks = self._make_chunks(SAMPLE_TEXTS)
        embeds = self._random_embeddings(len(SAMPLE_TEXTS))
        store.build(chunks, embeds)

        query_vec = self._random_embeddings(1)[0]
        results = store.search(query_vec, query_text="LDAP serveur", top_k=3)
        assert len(results) >= 1

    def test_persistence_roundtrip(self, tmp_path):
        from src.rag.vector_store import FAISSIndex, VectorStore
        faiss_idx = FAISSIndex(dim=self.DIM, index_type="Flat")
        store = VectorStore(faiss_index=faiss_idx)
        chunks = self._make_chunks(SAMPLE_TEXTS)
        embeds = self._random_embeddings(len(SAMPLE_TEXTS))
        store.build(chunks, embeds)
        store.save(tmp_path, "test_index")

        # Recharge
        faiss_idx2 = FAISSIndex(dim=self.DIM, index_type="Flat")
        store2 = VectorStore(faiss_index=faiss_idx2)
        store2.load(tmp_path, "test_index")

        assert store2.n_chunks == len(SAMPLE_TEXTS)
        assert store2._texts[0] == SAMPLE_TEXTS[0]


# ── Tests IndexBuilder (smoke test sans modèle réel) ──────────

class TestIndexBuilderSmoke:

    def test_dry_run_no_crash(self, tmp_path):
        """Vérifie que le dry_run fonctionne sans GPU ni modèle réel."""
        import yaml
        from src.rag.document_loader import DocumentLoader

        # Créer de faux documents
        data_dir = tmp_path / "raw"
        data_dir.mkdir()
        for i in range(3):
            (data_dir / f"doc_{i}.txt").write_text(
                f"Ceci est le document numéro {i}. " * 20,
                encoding="utf-8",
            )

        cfg = {
            "embedding": {"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 8,
                          "normalize_embeddings": True, "max_seq_length": 128},
            "chunking": {"strategy": "recursive", "chunk_size": 50, "chunk_overlap": 5,
                         "min_chunk_size": 10},
            "faiss": {"index_type": "Flat", "metric": "cosine"},
            "bm25": {"enabled": False},
            "hybrid": {"alpha": 0.6},
            "retrieval": {"top_k": 3, "score_threshold": 0.0},
            "sources": {"data_dir": str(data_dir), "processed_dir": str(tmp_path / "processed"),
                        "supported_extensions": [".txt"], "recursive": True},
            "index": {"output_dir": str(tmp_path / "index"), "index_name": "test",
                      "overwrite": True},
            "logging": {"level": "WARNING"},
        }

        # Sauvegarder config
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        from src.rag.build_index import IndexBuilder
        builder = IndexBuilder.from_yaml(cfg_path)

        # Dry-run (ne charge pas de modèle)
        docs = builder._load_documents()
        assert len(docs) == 3
        chunks = builder._chunk_documents(docs)
        assert len(chunks) > 0
