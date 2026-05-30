"""Templates de prompt pour le chatbot data center."""

SYSTEM_PROMPT = """Tu es un assistant IA embarque dans un data center etudiant.
Tu reponds de facon claire, prudente et orientee exploitation.
Si une information manque dans le contexte fourni, tu le dis clairement au lieu d'inventer.
Reponds toujours en francais."""


def build_messages(question: str, context: str = "") -> list:
    """Construit la liste de messages pour le chat template du modele.

    Args:
        question: La question posee par l'utilisateur.
        context: Passages recuperes par le RAG (peut etre vide).

    Returns:
        Liste de dicts {role, content} compatible avec apply_chat_template.
    """
    if context:
        user_content = f"Contexte documentaire:\n{context}\n\nQuestion:\n{question}"
    else:
        user_content = question

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
