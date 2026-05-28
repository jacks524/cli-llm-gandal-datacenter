"""Nettoyage du dataset du chatbot.

Le script supprime les doublons, normalise les espaces et ignore les exemples
incomplets. Il produit `data/processed/train_clean.jsonl`.
"""

import argparse
import json
from pathlib import Path

from src.common.paths import DEFAULT_CLEAN_DATASET_PATH, DEFAULT_DATASET_PATH
from src.common.text import normalize_text


REQUIRED_FIELDS = ("instruction", "input", "output")


def clean_examples(input_path):
    """Charge et nettoie un dataset JSONL."""
    seen = set()
    cleaned = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            normalized = {field: normalize_text(item.get(field, "")) for field in REQUIRED_FIELDS}

            if not normalized["instruction"] or not normalized["output"]:
                print(f"Ignore ligne {line_number}: instruction/output manquant")
                continue

            key = (normalized["instruction"].lower(), normalized["output"].lower())
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(normalized)
    return cleaned


def save_jsonl(examples, output_path):
    """Sauvegarde le dataset nettoye."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in examples:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    """Point d'entree du script de nettoyage."""
    parser = argparse.ArgumentParser(description="Nettoie le dataset JSONL.")
    parser.add_argument("--input", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_CLEAN_DATASET_PATH))
    args = parser.parse_args()

    examples = clean_examples(Path(args.input))
    save_jsonl(examples, Path(args.output))
    print(f"Dataset nettoye: {len(examples)} exemples -> {args.output}")


if __name__ == "__main__":
    main()
