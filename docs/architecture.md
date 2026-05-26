# Architecture du projet LLM Data Center

Ce document decrit l'architecture globale du chatbot embarque dans le data center.

## Objectif

Construire un assistant IA local capable de repondre aux questions liees au data center sans dependre d'une API externe.

## Composants prevus

- LLM offline leger : modele principal utilise pour generer les reponses.
- RAG local : systeme de recherche dans les documents internes du data center.
- Fine-tuning LoRA/QLoRA : adaptation legere du modele au style et aux cas d'usage du projet.
- Interface d'inference : script de chat qui connecte le modele, le prompt et le contexte RAG.
