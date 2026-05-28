"""Preparation du dataset pour le chatbot du data center.

Le script lit les exemples bruts dans `data/raw/qa_seed.jsonl` et produit un
dataset au format instruction/input/output dans `data/processed/train.jsonl`.
Ce format est pratique pour un futur fine-tuning LoRA/QLoRA.
"""

import argparse
import json
from pathlib import Path

from src.common.paths import DEFAULT_DATASET_PATH, RAW_DATA_DIR
from src.common.text import normalize_text


def load_seed_examples(path):
    """Charge les exemples JSONL fournis par l'equipe dataset."""
    examples = []
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalide ligne {line_number}: {exc}") from exc

            question = normalize_text(item.get("question", ""))
            answer = normalize_text(item.get("answer", ""))
            context = normalize_text(item.get("context", "Data center etudiant"))

            if not question or not answer:
                continue

            examples.append(
                {
                    "instruction": question,
                    "input": context,
                    "output": answer,
                }
            )
    return examples


def save_jsonl(examples, output_path):
    """Sauvegarde les exemples au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in examples:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    """Point d'entree du script de preparation."""
    parser = argparse.ArgumentParser(description="Prepare le dataset instruction/input/output.")
    parser.add_argument("--input", default=str(RAW_DATA_DIR / "qa_seed.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    args = parser.parse_args()

    examples = load_seed_examples(Path(args.input))
    save_jsonl(examples, Path(args.output))
    print(f"Dataset prepare: {len(examples)} exemples -> {args.output}")


if __name__ == "__main__":
    main()
