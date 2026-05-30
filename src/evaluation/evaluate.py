"""Evaluation du chatbot data center.

Evaluation simple : on pose des questions connues au pipeline RAG + LLM et on
verifie si des mots attendus apparaissent dans la reponse.
"""

import argparse
import json
from pathlib import Path

from src.common.paths import DEFAULT_INDEX_PATH, DEFAULT_MODEL_PATH
from src.inference.chat import answer_question


def load_cases(path: Path):
    """Charge les cas de test."""
    return json.loads(path.read_text(encoding="utf-8"))


def contains_keywords(response: str, keywords: list[str]):
    """Verifie la presence des mots importants."""
    lowered = response.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def main():
    """Point d'entree du script d'evaluation."""
    parser = argparse.ArgumentParser(description="Evalue le chatbot data center.")
    parser.add_argument("--cases", default="src/evaluation/test_cases.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    passed = 0

    for index, case in enumerate(cases, start=1):
        response, warning = answer_question(
            question=case["question"],
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=3,
            max_tokens=250,
            n_ctx=2048,
        )
        ok = contains_keywords(response, case.get("expected_keywords", []))
        passed += int(ok)
        status = "OK" if ok else "ECHEC"
        print(f"\n[{status}] Cas {index}: {case['question']}")
        if warning:
            print(f"Info: {warning}")
        print(response[:800])

    print(f"\nScore: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
