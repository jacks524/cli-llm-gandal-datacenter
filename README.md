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

## Prerequis

- Python 3.10 ou 3.11.
- `make`, deja disponible sur la plupart des distributions Linux.
- Un terminal ouvert a la racine du projet.
- Le modele GGUF place dans `models/base/`.

Dans ce repo, le modele actuellement range est :

```text
models/base/qwen2.5-0.5b-instruct-fp16.gguf
```

Ce fichier est volontairement ignore par Git car il est volumineux.

## Lancement avec installation complete

Cette option installe toutes les dependances du projet : chat local, RAG,
experimentation, evaluation et futur fine-tuning LoRA.

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo
make chat
```

Avec Anaconda, utiliser plutot un environnement separe :

```bash
conda create -n llm-datacenter python=3.11 -y
conda activate llm-datacenter
pip install -r requirements.txt
make demo
make chat
```

Cette installation peut etre lourde, car les dependances de fine-tuning peuvent
installer `torch`, `transformers`, `datasets`, `peft` et `trl`.

## Lancement leger sans tout requirements.txt

Cette option est recommandee si l'objectif est seulement de lancer le chatbot
avec le modele GGUF local, sans installer tout le bloc fine-tuning.

Avec `venv` :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install llama-cpp-python pyyaml tqdm huggingface_hub
make demo
make chat
```

Avec Anaconda :

```bash
conda create -n llm-datacenter python=3.11 -y
conda activate llm-datacenter
pip install llama-cpp-python pyyaml tqdm huggingface_hub
make demo
make chat
```

Dans ce mode, les commandes dataset, RAG, evaluation et chat fonctionnent. Le
vrai entrainement LoRA n'est pas disponible tant que les dependances completes
ne sont pas installees.

## Lancement sans modele GGUF

Le projet fonctionne aussi en mode demo RAG sans modele. Dans ce cas, le
chatbot retrouve le contexte dans `data/raw/`, puis indique que le modele local
n'est pas encore branche.

```bash
make demo
make chat
```

Ce mode sert a verifier que le pipeline projet marche avant de tester le LLM.

## Brancher le vrai modele offline

Modele recommande :

```text
Qwen/Qwen2.5-0.5B-Instruct-GGUF
qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Telechargement automatique :

```bash
make download-model
```

Le fichier doit se trouver ici :

```text
models/base/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Le modele actuellement present dans ce projet peut aussi etre utilise :

```text
models/base/qwen2.5-0.5b-instruct-fp16.gguf
```

Le code detecte automatiquement un `.gguf` disponible dans `models/base/` si le
modele recommande `q4_k_m` n'est pas present.

Ensuite :

```bash
make index
make chat
```

## Commandes utiles

```bash
make demo          # prepare les donnees, construit l'index RAG, lance l'evaluation
make chat          # lance le chatbot dans le terminal
make clean-data    # prepare et nettoie le dataset
make index         # construit l'index RAG local
make evaluate      # lance les tests d'evaluation
make train-dry     # verifie le pipeline LoRA sans entrainer
make download-model # telecharge le modele Q4_K_M recommande
```

## Si le chat ne lance pas le modele

Verifier d'abord que le fichier existe :

```bash
ls -lh models/base/
```

Verifier ensuite que `llama-cpp-python` est installe :

```bash
python3 -c "import llama_cpp; print('llama-cpp-python OK')"
```

Si cette commande echoue :

```bash
pip install llama-cpp-python
```

## Role de l'equipe LoRA

La personne responsable du fine-tuning travaille surtout sur :

- `src/training/train_lora.py` : script d'entrainement.
- `src/training/config.yaml` : hyperparametres et chemins.
- `models/adapters/` : sortie de l'adapter LoRA entraine.

Elle n'entraine pas un nouveau cerveau complet. Elle adapte Qwen au style data
center : vocabulaire, formats de reponse, workflows internes, commandes et
comportement prudent.
