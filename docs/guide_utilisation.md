# Guide d'utilisation

Ce document expliquera comment installer, configurer et lancer le chatbot local du data center.

## Etapes rapides

1. Creer un environnement Python.
2. Installer les dependances avec `pip install -r requirements.txt`.
3. Lancer `make demo` pour preparer les donnees, construire l'index et evaluer.
4. Lancer `make chat` pour discuter avec l'assistant.

## Commandes utiles

```bash
make clean-data
make index
make chat
make evaluate
make train-dry
make download-model
```

`make train-dry` ne lance pas encore un vrai entrainement. Il verifie que la
configuration LoRA et le dataset sont coherents.

## Modele GGUF

Le modele recommande est :

```text
models/base/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

S'il n'est pas present, le chatbot reste utilisable en mode demo RAG.
