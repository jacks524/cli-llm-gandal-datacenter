"""Recherche de contexte dans l'index RAG.

Ce fichier servira a retrouver les passages les plus pertinents pour une
question utilisateur avant d'appeler le LLM local.
"""


def retrieve_context(question: str, top_k: int = 5):
    """Retourne les passages les plus utiles pour repondre a une question."""
    # TODO: charger l'index local
    # TODO: chercher les top_k passages les plus proches de la question
    # TODO: retourner les passages sous forme de liste
    return []
