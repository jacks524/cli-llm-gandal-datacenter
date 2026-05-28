"""
build_index.py
==============
Task 6 – RAG Index | Data Center LLM Chatbot
Auteur : Responsable RAG Index

Point d'entrée principal du pipeline RAG :

  1. Charge la configuration YAML
  2. Charge les documents (multi-format, récursif)
  3. Découpe en chunks (stratégie configurable)
  4. Encode en vecteurs (sentence-transformers)
  5. Construit l'index FAISS + BM25
  6. Sauvegarde l'index sur disque

Usage (CLI)
-----------
  python src/rag/build_index.py
  python src/rag/build_index.py --config configs/rag_config.yaml
  python src/rag/build_index.py --data-dir data/raw --output data/index --name my_index
  python src/rag/build_index.py --dry-run   # inspecte les documents sans construire

Usage (API Python)
------------------
  from src.rag.build_index import IndexBuilder
  builder = IndexBuilder.from_yaml("configs/rag_config.yaml")
  builder.run()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str | None = None):
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Réduire le bruit des bibliothèques tierces
    for noisy in ("transformers", "tokenizers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# IndexBuilder
# ─────────────────────────────────────────────────────────────

class IndexBuilder:
    """
    Orchestre le pipeline complet de construction d'index RAG.

    Paramètres
    ----------
    cfg : dict chargé depuis rag_config.yaml
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._validate_config()

    # ── Factories ────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/rag_config.yaml") -> "IndexBuilder":
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config introuvable : {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info("Config chargée depuis %s", config_path)
        return cls(cfg)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "IndexBuilder":
        builder = cls.from_yaml(args.config)
        # Surcharges CLI
        if args.data_dir:
            builder.cfg["sources"]["data_dir"] = args.data_dir
        if args.output:
            builder.cfg["index"]["output_dir"] = args.output
        if args.name:
            builder.cfg["index"]["index_name"] = args.name
        if args.model:
            builder.cfg["embedding"]["model_name"] = args.model
        if args.device:
            builder.cfg["embedding"]["device"] = args.device
        return builder

    # ── Pipeline principal ───────────────────────────────────

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Exécute le pipeline complet.

        Returns
        -------
        stats : dict avec les métriques de construction
        """
        t_start = time.time()
        setup_logging(
            level=self.cfg.get("logging", {}).get("level", "INFO"),
            log_file=self.cfg.get("logging", {}).get("log_file"),
        )

        logger.info("=" * 60)
        logger.info("Pipeline RAG Index — démarrage")
        logger.info("=" * 60)

        # ── Étape 1 : Chargement des documents ───────────────
        docs = self._load_documents()
        if not docs:
            logger.error("Aucun document chargé. Vérifier data_dir et supported_extensions.")
            sys.exit(1)

        # ── Étape 2 : Chunking ───────────────────────────────
        chunks = self._chunk_documents(docs)
        if not chunks:
            logger.error("Aucun chunk produit. Vérifier les paramètres de chunking.")
            sys.exit(1)

        # ── Étape 3 : Attribution des IDs uniques ────────────
        self._assign_chunk_ids(chunks)

        if dry_run:
            logger.info("[DRY-RUN] Documents : %d | Chunks : %d", len(docs), len(chunks))
            self._print_sample(chunks)
            return {"documents": len(docs), "chunks": len(chunks), "dry_run": True}

        # ── Étape 4 : Embeddings ─────────────────────────────
        embedder = self._load_embedder()
        texts = [c.text for c in chunks]
        embeddings = embedder.encode(texts)

        # ── Étape 5 : Construction de l'index ────────────────
        vector_store = self._build_store(embedder, chunks, embeddings)

        # ── Étape 6 : Sauvegarde ─────────────────────────────
        self._save(vector_store)

        # ── Résumé ───────────────────────────────────────────
        elapsed = time.time() - t_start
        stats = self._write_stats(docs, chunks, embeddings, elapsed)

        logger.info("=" * 60)
        logger.info("Index construit avec succès en %.1fs", elapsed)
        logger.info("  → Documents : %d", stats["n_documents"])
        logger.info("  → Chunks    : %d", stats["n_chunks"])
        logger.info("  → Dimension : %d", stats["embedding_dim"])
        logger.info("  → Output    : %s/", self.cfg["index"]["output_dir"])
        logger.info("=" * 60)
        return stats

    # ── Étapes détaillées ────────────────────────────────────

    def _load_documents(self):
        from .document_loader import DocumentLoader

        src = self.cfg.get("sources", {})
        loader = DocumentLoader(
            extensions=src.get("supported_extensions"),
            min_chars=self.cfg.get("chunking", {}).get("min_chunk_size", 30),
        )

        # Chercher dans data/raw ET data/processed
        all_docs = []
        for dir_key in ("data_dir", "processed_dir"):
            dir_path = Path(src.get(dir_key, ""))
            if dir_path.exists():
                docs = loader.load_directory(dir_path, recursive=src.get("recursive", True))
                all_docs.extend(docs)
                logger.info("Chargé %d doc(s) depuis %s", len(docs), dir_path)

        logger.info("Total : %d documents valides", len(all_docs))
        return all_docs

    def _chunk_documents(self, docs):
        from .chunker import get_chunker

        chk_cfg = self.cfg.get("chunking", {})
        chunker = get_chunker(
            strategy=chk_cfg.get("strategy", "recursive"),
            chunk_size=chk_cfg.get("chunk_size", 512),
            chunk_overlap=chk_cfg.get("chunk_overlap", 64),
            separators=chk_cfg.get("separators"),
            min_chunk_size=chk_cfg.get("min_chunk_size", 20),
        )

        all_chunks = []
        for doc in docs:
            chunks = chunker.split(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        logger.info("Chunking : %d documents → %d chunks", len(docs), len(all_chunks))
        return all_chunks

    def _assign_chunk_ids(self, chunks):
        for i, chunk in enumerate(chunks):
            fingerprint = hashlib.md5(chunk.text.encode()).hexdigest()[:8]
            chunk.chunk_id = f"chunk_{i:06d}_{fingerprint}"

    def _load_embedder(self):
        from .embedder import Embedder
        return Embedder.from_config(self.cfg)

    def _build_store(self, embedder, chunks, embeddings):
        import numpy as np
        from .vector_store import VectorStore

        # Patch dim dans config si nécessaire
        self.cfg["embedding"]["_dim"] = embedder.dim
        store = VectorStore.from_config(self.cfg)

        import numpy as np
        store.build(chunks, embeddings.astype(np.float32))
        return store

    def _save(self, vector_store):
        idx_cfg = self.cfg.get("index", {})
        output_dir = Path(idx_cfg.get("output_dir", "data/index"))
        name = idx_cfg.get("index_name", "datacenter_rag")

        if not idx_cfg.get("overwrite", False):
            faiss_path = output_dir / f"{name}.faiss"
            if faiss_path.exists():
                logger.warning(
                    "Index existant détecté (%s). Passer overwrite=true pour écraser.", faiss_path
                )
                return

        vector_store.save(output_dir, name)

    def _write_stats(self, docs, chunks, embeddings, elapsed) -> dict:
        stats = {
            "n_documents": len(docs),
            "n_chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
            "index_type": self.cfg.get("faiss", {}).get("index_type", "IVF"),
            "chunking_strategy": self.cfg.get("chunking", {}).get("strategy", "recursive"),
            "embedding_model": self.cfg.get("embedding", {}).get("model_name"),
            "bm25_enabled": self.cfg.get("bm25", {}).get("enabled", False),
            "build_time_s": round(elapsed, 2),
        }

        output_dir = Path(self.cfg["index"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = output_dir / "build_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("Stats sauvegardées : %s", stats_path)
        return stats

    def _print_sample(self, chunks, n: int = 3):
        logger.info("── Échantillon de chunks ──────────────────────")
        for c in chunks[:n]:
            preview = c.text[:120].replace("\n", " ")
            logger.info("  [%s] %s…", c.metadata.get("source", "?"), preview)

    # ── Validation ───────────────────────────────────────────

    def _validate_config(self):
        required = [
            ("sources", "data_dir"),
            ("index", "output_dir"),
            ("index", "index_name"),
        ]
        for section, key in required:
            if key not in self.cfg.get(section, {}):
                raise ValueError(f"Config manquante : [{section}].{key}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Construit l'index RAG vectoriel pour le chatbot LLM offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/rag/build_index.py
  python src/rag/build_index.py --config configs/rag_config.yaml
  python src/rag/build_index.py --data-dir data/raw --output data/index
  python src/rag/build_index.py --dry-run
  python src/rag/build_index.py --device cuda --model BAAI/bge-m3
        """,
    )
    parser.add_argument(
        "--config", default="configs/rag_config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument("--data-dir", default=None, help="Répertoire source des documents")
    parser.add_argument("--output", default=None, help="Répertoire de sortie de l'index")
    parser.add_argument("--name", default=None, help="Nom de l'index")
    parser.add_argument("--model", default=None, help="Nom du modèle d'embedding")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les documents/chunks sans construire l'index"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    try:
        builder = IndexBuilder.from_args(args)
        builder.run(dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("Erreur fatale : %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
