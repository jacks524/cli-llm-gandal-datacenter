"""Interface de chat locale pour le LLM du data center.

Lance une conversation offline avec le modele Qwen local,
en combinant le contexte recupere par le RAG et l'adaptateur LoRA si disponible.

Usage:
    python -m src.inference.chat
"""

import sys
import yaml
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.prompt_template import build_messages
from src.rag.retrieve import retrieve_context

CONFIG_PATH   = ROOT / "src" / "training" / "config.yaml"
BASE_MODEL_DIR = ROOT / "models" / "base"
ADAPTER_DIR   = ROOT / "models" / "adapters" / "datacenter-lora"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _model_source(config: dict) -> str:
    """Retourne le chemin local si le modele est telecharge, sinon le nom HuggingFace."""
    if BASE_MODEL_DIR.exists() and any(BASE_MODEL_DIR.iterdir()):
        return str(BASE_MODEL_DIR)
    return config["model_name"]


def load_model(config: dict):
    """Charge le tokenizer et le modele (+ LoRA si disponible).

    Returns:
        tuple: (tokenizer, model) prets pour l'inference.
    """
    source = _model_source(config)
    print(f"[INFO] Chargement du modele depuis : {source}")

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32)

    if ADAPTER_DIR.exists():
        print(f"[INFO] Chargement de l'adaptateur LoRA : {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    else:
        print("[INFO] Aucun adaptateur LoRA trouve — utilisation du modele de base.")

    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, question: str, max_new_tokens: int = 512) -> str:
    """Genere une reponse pour une question en integrant le contexte RAG.

    Args:
        tokenizer: Tokenizer du modele.
        model: Modele charge (base ou fine-tune).
        question: Question posee par l'utilisateur.
        max_new_tokens: Nombre maximum de tokens a generer.

    Returns:
        Reponse textuelle generee par le modele.
    """
    passages = retrieve_context(question)
    context = "\n".join(passages) if passages else ""

    messages = build_messages(question, context)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    config = _load_config()
    tokenizer, model = load_model(config)

    print("\n=== Chatbot Data Center (offline) ===")
    print("Tapez 'quitter' pour arreter.\n")

    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue

        if question.lower() in ("quitter", "exit", "quit"):
            print("Au revoir.")
            break

        print("Bot  : ", end="", flush=True)
        reponse = generate_response(tokenizer, model, question)
        print(reponse)
        print()


if __name__ == "__main__":
    main()
