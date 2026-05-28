"""Construction de l'index RAG local.

Cette version utilise un index JSON tres simple, base sur les mots des
documents. Cela permet au projet de fonctionner sans GPU et sans dependance
lourde. Plus tard, l'equipe RAG pourra remplacer ce module par FAISS/Chroma.
"""

import argparse
import json
import math
from pathlib import Path

from src.common.paths import DEFAULT_INDEX_PATH, RAW_DATA_DIR
from src.common.text import split_text, token_counts


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(source_dir: Path):
    """Charge les documents texte qui serviront de base documentaire."""
    documents = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(split_text(text)):
            documents.append(
                {
                    "id": f"{path.relative_to(source_dir)}#{index}",
                    "source": str(path.relative_to(source_dir)),
                    "text": chunk,
                    "tokens": token_counts(chunk),
                }
            )
    return documents


def compute_idf(documents):
    """Calcule un IDF minimal pour donner plus de poids aux mots rares."""
    document_count = len(documents)
    frequencies = {}
    for document in documents:
        for token in document["tokens"]:
            frequencies[token] = frequencies.get(token, 0) + 1

    return {
        token: math.log((1 + document_count) / (1 + count)) + 1
        for token, count in frequencies.items()
    }


def build_index(source_dir: Path, output_path: Path):
    """Construit et sauvegarde l'index RAG."""
    documents = load_documents(source_dir)
    if not documents:
        raise RuntimeError(f"Aucun document .md/.txt trouve dans {source_dir}")

    index = {
        "version": 1,
        "description": "Index RAG local simple pour le chatbot data center.",
        "documents": documents,
        "idf": compute_idf(documents),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(documents)


def main():
    """Point d'entree du script de construction d'index."""
    parser = argparse.ArgumentParser(description="Construit l'index RAG local.")
    parser.add_argument("--source", default=str(RAW_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_INDEX_PATH))
    args = parser.parse_args()

    count = build_index(Path(args.source), Path(args.output))
    print(f"Index RAG construit: {count} chunks -> {args.output}")


if __name__ == "__main__":
    main()
