"""Recherche de contexte dans l'index RAG.

Le scoring est volontairement simple pour rester lancable partout. Le module
retourne les passages les plus proches d'une question utilisateur.
"""

import argparse
import json
from pathlib import Path

from src.common.paths import DEFAULT_INDEX_PATH
from src.common.text import token_counts


def load_index(index_path: Path = DEFAULT_INDEX_PATH):
    """Charge l'index RAG depuis le disque."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index introuvable: {index_path}. Lance d'abord: python -m src.rag.build_index"
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def score_document(query_tokens, document, idf):
    """Score un document selon le chevauchement pondere des tokens."""
    score = 0.0
    document_tokens = document["tokens"]
    for token, count in query_tokens.items():
        if token in document_tokens:
            score += min(count, document_tokens[token]) * idf.get(token, 1.0)
    return score


def retrieve_context(question: str, top_k: int = 5, index_path: Path = DEFAULT_INDEX_PATH):
    """Retourne les passages les plus utiles pour repondre a une question."""
    index = load_index(index_path)
    query_tokens = token_counts(question)
    scored = []

    for document in index["documents"]:
        score = score_document(query_tokens, document, index["idf"])
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "score": round(score, 4),
            "source": document["source"],
            "text": document["text"],
        }
        for score, document in scored[:top_k]
    ]


def format_context(passages):
    """Transforme les passages RAG en bloc de contexte pour le prompt."""
    if not passages:
        return "Aucun contexte documentaire pertinent n'a ete trouve."
    return "\n\n".join(
        f"[Source: {item['source']} | score={item['score']}]\n{item['text']}"
        for item in passages
    )


def main():
    """Petit outil CLI pour tester la recherche RAG."""
    parser = argparse.ArgumentParser(description="Recherche dans l'index RAG.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    args = parser.parse_args()

    passages = retrieve_context(args.question, top_k=args.top_k, index_path=Path(args.index))
    print(format_context(passages))


if __name__ == "__main__":
    main()
