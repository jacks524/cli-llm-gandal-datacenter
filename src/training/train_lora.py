"""Fine-tuning LoRA/QLoRA du modele LLM.

Par defaut, le script tourne en `dry_run` pour verifier la configuration et le
dataset sans telecharger de modele. Pour entrainer vraiment, mettre
`training.dry_run: false` dans `src/training/config.yaml`.

Note importante : PEFT/LoRA s'applique au modele Transformers officiel
`Qwen/Qwen2.5-0.5B-Instruct`, pas directement au fichier GGUF quantifie.
"""

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: Path):
    """Charge la configuration YAML du fine-tuning."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    """Charge un dataset JSONL instruction/input/output."""
    examples = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_training_text(example):
    """Convertit un exemple en texte d'entrainement."""
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Contexte:\n"
        f"{example.get('input', '')}\n\n"
        "### Reponse:\n"
        f"{example['output']}"
    )


def dry_run(config, examples):
    """Verifie le pipeline sans lancer d'entrainement couteux."""
    print("Dry run LoRA actif.")
    print(f"Modele Transformers: {config['model_name']}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Sortie adapter: {config['output_dir']}")
    print(f"Nombre d'exemples: {len(examples)}")
    if examples:
        print("\nExemple formate:\n")
        print(format_training_text(examples[0]))


def train(config, examples):
    """Lance un fine-tuning LoRA minimal avec Hugging Face/PEFT/TRL."""
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Dependances manquantes. Installe les requirements puis relance: "
            "pip install -r requirements.txt"
        ) from exc

    dataset = Dataset.from_list([{"text": format_training_text(item)} for item in examples])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = config["lora"]
    training = config["training"]

    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=training["epochs"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training["max_seq_length"],
        peft_config=peft_config,
        args=args,
    )
    trainer.train()
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Adapter LoRA sauvegarde dans {config['output_dir']}")


def main():
    """Point d'entree du script d'entrainement."""
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA du modele data center.")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--train", action="store_true", help="Force un vrai entrainement.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    examples = load_jsonl(Path(config["dataset_path"]))

    if config["training"].get("dry_run", True) and not args.train:
        dry_run(config, examples)
        return

    train(config, examples)


if __name__ == "__main__":
    main()
