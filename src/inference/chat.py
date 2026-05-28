"""Interface de chat locale pour le LLM du data center.

Le chat fonctionne en deux modes :
- avec `llama-cpp-python` + modele GGUF local, il genere une vraie reponse LLM ;
- sans modele, il reste utilisable en mode demo RAG et affiche une reponse
  extractive basee sur les documents internes.
"""

import argparse
from pathlib import Path

from src.common.paths import DEFAULT_INDEX_PATH, DEFAULT_MODEL_PATH
from src.inference.prompt_template import SYSTEM_PROMPT, build_prompt
from src.rag.retrieve import format_context, retrieve_context


def resolve_model_path(model_path: Path):
    """Retourne le modele demande ou le premier GGUF disponible dans models/base."""
    if model_path.exists():
        return model_path, None

    candidates = sorted(model_path.parent.glob("*.gguf"))
    if candidates:
        return candidates[0], f"modele par defaut introuvable, utilisation de {candidates[0].name}"

    return model_path, None


def load_llm(model_path: Path, n_ctx: int):
    """Charge le modele GGUF si llama-cpp-python est disponible."""
    resolved_path, fallback_warning = resolve_model_path(model_path)
    if not resolved_path.exists():
        return None, f"modele GGUF introuvable: {resolved_path}"

    try:
        from llama_cpp import Llama
    except ImportError:
        return None, "llama-cpp-python n'est pas installe"

    llm = Llama(model_path=str(resolved_path), n_ctx=n_ctx, verbose=False)
    return llm, fallback_warning


def generate_with_llm(llm, question: str, context: str, max_tokens: int):
    """Genere une reponse avec le modele local."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, context)},
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.2)
    return result["choices"][0]["message"]["content"].strip()


def generate_demo_answer(question: str, passages):
    """Reponse de secours quand le modele n'est pas encore branche."""
    if not passages:
        return (
            "Je n'ai pas trouve de contexte dans la base documentaire locale. "
            "Ajoute des documents dans data/raw/ puis relance l'indexation."
        )

    sources = ", ".join(sorted({item["source"] for item in passages}))
    best_text = passages[0]["text"]
    return (
        "Mode demo RAG actif: le modele GGUF n'est pas encore charge.\n\n"
        f"Question recue: {question}\n\n"
        f"Contexte le plus pertinent trouve dans {sources}:\n"
        f"{best_text}\n\n"
        "Pour obtenir une vraie reponse generative, installe llama-cpp-python "
        "et place le modele Qwen GGUF dans models/base/."
    )


def answer_question(question: str, model_path: Path, index_path: Path, top_k: int, max_tokens: int, n_ctx: int):
    """Pipeline complet : retrieval puis generation ou reponse demo."""
    passages = retrieve_context(question, top_k=top_k, index_path=index_path)
    context = format_context(passages)
    llm, error = load_llm(model_path, n_ctx=n_ctx)

    if llm is None:
        return generate_demo_answer(question, passages), error

    return generate_with_llm(llm, question, context, max_tokens=max_tokens), None


def main():
    """Point d'entree du chat local."""
    parser = argparse.ArgumentParser(description="Chatbot offline du data center.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    print("Assistant Data Center local. Tape 'exit' pour quitter.")
    while True:
        question = input("\nVous> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Assistant> Fin de session.")
            break
        if not question:
            continue

        try:
            response, warning = answer_question(
                question=question,
                model_path=Path(args.model),
                index_path=Path(args.index),
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                n_ctx=args.n_ctx,
            )
        except FileNotFoundError as exc:
            print(f"Assistant> {exc}")
            continue

        if warning:
            print(f"Info> {warning}")
        print(f"Assistant> {response}")


if __name__ == "__main__":
    main()
