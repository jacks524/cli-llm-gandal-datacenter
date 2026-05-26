# LLM Data Center

Projet de chatbot LLM offline pour un data center etudiant.

## Objectif

Construire un assistant IA local capable d'aider les utilisateurs a comprendre, exploiter et diagnostiquer le data center a partir de documents internes.

## Structure

- `notebooks/` : exploration, tests rapides et demos.
- `src/data/` : preparation et nettoyage du dataset.
- `src/rag/` : construction et interrogation de l'index documentaire local.
- `src/training/` : fine-tuning LoRA/QLoRA.
- `src/inference/` : chat local et templates de prompt.
- `src/evaluation/` : cas de test et evaluation du chatbot.
- `data/` : donnees brutes, donnees traitees et donnees d'evaluation.
- `models/adapters/` : adapters LoRA entraines.
- `docs/` : documentation technique et guide d'utilisation.

## Note

Les notebooks servent a explorer et demontrer. Le code principal du projet doit rester dans les scripts Python de `src/`.
