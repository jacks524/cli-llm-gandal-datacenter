"""Templates de prompt pour le chatbot.

Ce fichier centralise les instructions donnees au modele afin de garder un
comportement stable et coherent dans toute l'application.
"""

SYSTEM_PROMPT = """Tu es un assistant IA embarque dans un data center etudiant.
Tu reponds de facon claire, prudente et orientee exploitation.
Si une information manque, tu le dis au lieu d'inventer.
Tu ne proposes jamais une commande destructive sans avertissement clair.
"""


def build_prompt(question: str, context: str = "") -> str:
    """Construit le prompt final envoye au modele."""
    return f"{SYSTEM_PROMPT}\n\nContexte:\n{context}\n\nQuestion:\n{question}\n\nReponse:"
