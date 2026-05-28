# 📚 Task 6 — Responsable RAG Index
## Data Center LLM Chatbot | Pipeline de construction d'index vectoriel

> **Rôle dans le projet :** Ce module est le cœur de la mémoire documentaire du chatbot.
> Il transforme des documents bruts (procédures, manuels, configurations) en un index
> vectoriel requêtable à haute vitesse — le moteur du RAG (*Retrieval-Augmented Generation*).

---

## 🗂 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture technique](#-architecture-technique)
3. [Structure des fichiers](#-structure-des-fichiers)
4. [Installation](#-installation)
5. [Configuration](#️-configuration)
6. [Utilisation](#-utilisation)
   - [Build de l'index (CLI)](#1-build-de-lindex-cli)
   - [Build de l'index (API Python)](#2-build-de-lindex-api-python)
   - [Requêtage du Retriever](#3-requêtage-du-retriever)
   - [Démo rapide](#4-démo-rapide)
7. [Pipeline détaillé](#-pipeline-détaillé)
8. [Stratégies de chunking](#-stratégies-de-chunking)
9. [Recherche hybride (Dense + BM25)](#-recherche-hybride-dense--bm25)
10. [Formats de documents supportés](#-formats-de-documents-supportés)
11. [Tests](#-tests)
12. [Intégration avec Task 8 (Inference)](#-intégration-avec-task-8-inference)
13. [Choix du modèle d'embedding](#-choix-du-modèle-dembedding)
14. [Troubleshooting](#-troubleshooting)
15. [Références techniques](#-références-techniques)

---

## 🎯 Vue d'ensemble

```
                     ┌─────────────────────────────────┐
    Documents bruts  │                                 │   Index vectoriel
   (txt, md, pdf,    │     RAG Index Builder (Task 6)  │  ──────────────►  data/index/
    json, jsonl,     │                                 │
    csv)             └─────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  1. Load docs     │  DocumentLoader (multi-format)
                    │  2. Chunk text    │  RecursiveChunker / FixedChunker
                    │  3. Embed vecs    │  SentenceTransformer (BAAI/bge-m3)
                    │  4. FAISS index   │  IVF / HNSW / Flat
                    │  5. BM25 index    │  Sparse BM25 (optionnel)
                    │  6. Save          │  .faiss + .json + .pkl
                    └───────────────────┘
```

Le module livre **3 artefacts** persistants dans `data/index/` :

| Fichier | Contenu | Usage |
|---|---|---|
| `datacenter_rag.faiss` | Index FAISS sérialisé | Recherche dense |
| `datacenter_rag_meta.json` | Textes + métadonnées | Renvoie le contexte |
| `datacenter_rag_bm25.pkl` | Index BM25 | Recherche sparse |
| `build_stats.json` | Statistiques de build | Monitoring |

---

## 🏗 Architecture technique

```
src/rag/
├── build_index.py      ← Point d'entrée CLI + classe IndexBuilder
├── document_loader.py  ← Loaders multi-format (Strategy pattern)
├── chunker.py          ← Chunkers (Fixed / Recursive / Sentence)
├── embedder.py         ← Wrapper SentenceTransformer
├── vector_store.py     ← FAISSIndex + BM25Index + VectorStore
└── retriever.py        ← Interface de requêtage (utilisé par Task 8)
```

### Diagramme de classes simplifié

```
IndexBuilder
  ├── uses → DocumentLoader → [TextLoader, JsonLoader, JsonlLoader, CsvLoader, PdfLoader]
  ├── uses → Chunker         → [FixedChunker, RecursiveChunker, SentenceChunker]
  ├── uses → Embedder        → SentenceTransformer
  └── uses → VectorStore
                ├── FAISSIndex   (dense search)
                └── BM25Index    (sparse search)

Retriever
  ├── uses → Embedder
  └── uses → VectorStore
```

---

## 📁 Structure des fichiers

```
rag_index_task6/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── build_index.py        # ← POINT D'ENTRÉE PRINCIPAL
│       ├── document_loader.py
│       ├── chunker.py
│       ├── embedder.py
│       ├── vector_store.py
│       └── retriever.py
├── data/
│   ├── raw/                      # Documents sources (fournis par Task 3)
│   ├── processed/                # Données traitées (Task 4/5)
│   └── index/                    # INDEX GÉNÉRÉ (sortie de Task 6)
│       ├── datacenter_rag.faiss
│       ├── datacenter_rag_meta.json
│       ├── datacenter_rag_bm25.pkl
│       ├── build_stats.json
│       └── build.log
├── configs/
│   └── rag_config.yaml           # Configuration centrale
├── tests/
│   └── test_rag_index.py         # 20+ tests unitaires
├── scripts/
│   └── demo_retrieval.py         # Démo interactive
├── requirements.txt
├── pytest.ini
└── README.md                     # ← CE FICHIER
```

---

## ⚙️ Installation

### Prérequis

- Python **3.10+**
- pip 23+
- RAM : min **4 Go** (8 Go recommandés pour BGE-M3)
- Disque : ~2 Go pour le modèle d'embedding

### Étapes

```bash
# 1. Cloner / placer le projet
cd /chemin/vers/projet

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Installer le support PDF
pip install pypdf

# 5. (Optionnel) GPU — remplacer faiss-cpu par faiss-gpu
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Vérification de l'installation

```bash
python -c "import faiss; import sentence_transformers; print('OK')"
```

---

## 🛠️ Configuration

Tout est piloté depuis **`configs/rag_config.yaml`**.

### Paramètres clés

```yaml
# Modèle d'embedding
embedding:
  model_name: "BAAI/bge-m3"    # Multilingue, 1024 dimensions
  device: "cpu"                  # "cuda" si GPU disponible
  batch_size: 32

# Chunking
chunking:
  strategy: "recursive"          # "fixed" | "recursive" | "sentence"
  chunk_size: 512                # Taille en mots
  chunk_overlap: 64              # Mots partagés entre chunks adjacents

# Index FAISS
faiss:
  index_type: "IVF"              # "Flat" (petit) | "IVF" (moyen) | "HNSW" (grand)
  metric: "cosine"
  nlist: 100                     # IVF : √N clusters (N = nombre de chunks)
  nprobe: 10                     # IVF : nb de clusters explorés à la requête

# BM25 (recherche hybride)
bm25:
  enabled: true
  k1: 1.5
  b: 0.75

# Fusion hybride
hybrid:
  alpha: 0.6                     # 0 = pure BM25, 1 = pure dense

# Retrieval
retrieval:
  top_k: 5
  score_threshold: 0.35

# Sources
sources:
  data_dir: "data/raw"
  supported_extensions: [".txt", ".md", ".pdf", ".json", ".jsonl", ".csv"]
```

### Recommandations selon la taille du corpus

| Corpus | `index_type` | `nlist` | `chunk_size` |
|--------|-------------|---------|--------------|
| < 10 000 chunks | `Flat` | — | 512 |
| 10K – 500K chunks | `IVF` | √N (ex : 200) | 512 |
| > 500K chunks | `HNSW` | — | 256 |

---

## 🚀 Utilisation

### 1. Build de l'index (CLI)

```bash
# Construction standard (utilise configs/rag_config.yaml)
python src/rag/build_index.py

# Avec chemin de config personnalisé
python src/rag/build_index.py --config configs/rag_config.yaml

# Surcharger la source de données et la sortie
python src/rag/build_index.py --data-dir data/raw --output data/index --name datacenter_rag

# Utiliser le GPU
python src/rag/build_index.py --device cuda

# Changer de modèle d'embedding
python src/rag/build_index.py --model intfloat/multilingual-e5-base

# Inspecter les documents sans construire l'index
python src/rag/build_index.py --dry-run

# Aide complète
python src/rag/build_index.py --help
```

**Exemple de sortie attendue :**
```
2024-01-15 10:23:01 [INFO] build_index — ============================================================
2024-01-15 10:23:01 [INFO] build_index — Pipeline RAG Index — démarrage
2024-01-15 10:23:01 [INFO] document_loader — Scan de data/raw (47 fichier(s) trouvés)
2024-01-15 10:23:02 [INFO] document_loader — Total documents chargés : 47
2024-01-15 10:23:02 [INFO] chunker — Chunking : 47 documents → 1243 chunks
2024-01-15 10:23:05 [INFO] embedder — Chargement du modèle : BAAI/bge-m3 (device=cpu)
2024-01-15 10:23:58 [INFO] embedder — Encodage terminé en 34.2s (36 textes/s)
2024-01-15 10:23:59 [INFO] vector_store — Construction FAISS sur 1243 vecteurs…
2024-01-15 10:23:59 [INFO] vector_store — Index complet sauvegardé dans data/index/
2024-01-15 10:23:59 [INFO] build_index — Index construit avec succès en 62.4s
2024-01-15 10:23:59 [INFO] build_index —   → Documents : 47
2024-01-15 10:23:59 [INFO] build_index —   → Chunks    : 1243
2024-01-15 10:23:59 [INFO] build_index —   → Dimension : 1024
```

---

### 2. Build de l'index (API Python)

```python
from src.rag.build_index import IndexBuilder

# Depuis YAML
builder = IndexBuilder.from_yaml("configs/rag_config.yaml")
stats = builder.run()

print(f"Chunks construits : {stats['n_chunks']}")
print(f"Temps de build    : {stats['build_time_s']}s")
```

```python
# Depuis un dict Python (sans fichier YAML)
from src.rag.build_index import IndexBuilder

cfg = {
    "embedding": {"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 32,
                  "normalize_embeddings": True, "max_seq_length": 512},
    "chunking":  {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64,
                  "min_chunk_size": 20},
    "faiss":     {"index_type": "Flat", "metric": "cosine"},
    "bm25":      {"enabled": True, "k1": 1.5, "b": 0.75},
    "hybrid":    {"alpha": 0.6},
    "retrieval": {"top_k": 5, "score_threshold": 0.35},
    "sources":   {"data_dir": "data/raw", "processed_dir": "data/processed",
                  "supported_extensions": [".txt", ".md"], "recursive": True},
    "index":     {"output_dir": "data/index", "index_name": "datacenter_rag", "overwrite": False},
    "logging":   {"level": "INFO"},
}
builder = IndexBuilder(cfg)
builder.run()
```

---

### 3. Requêtage du Retriever

```python
from src.rag.retriever import Retriever

# Charger l'index existant
ret = Retriever.load(
    index_dir="data/index",
    index_name="datacenter_rag",
    config_path="configs/rag_config.yaml"
)

print(ret)  # Retriever(n_chunks=1243, top_k=5, model=BAAI/bge-m3)

# Recherche simple
results = ret.retrieve("Comment redémarrer le serveur LDAP ?")
for r in results:
    print(f"Score: {r.score:.3f} | Source: {r.metadata['source']}")
    print(r.text[:200])
    print("---")

# Obtenir le contexte directement formaté pour le LLM
context = ret.retrieve_as_context(
    "Quelle est la procédure de backup ?",
    top_k=3
)
print(context)  # Prêt à injecter dans le prompt

# Recherche batch (évaluation)
queries = ["procédure LDAP", "configuration réseau", "monitoring alertes"]
all_results = ret.batch_retrieve(queries, top_k=3)
```

---

### 4. Démo rapide

```bash
# Lance une démo avec des données d'exemple intégrées (aucun document requis)
python scripts/demo_retrieval.py
```

---

## 🔄 Pipeline détaillé

```
┌─────────────────────────────────────────────────────────────────┐
│                    IndexBuilder.run()                           │
│                                                                 │
│  Step 1 │ DocumentLoader                                        │
│         │   data/raw/**  + data/processed/**                    │
│         │   → [Document(content, metadata), ...]               │
│         │                                                       │
│  Step 2 │ RecursiveChunker (ou Fixed / Sentence)               │
│         │   Document → [Chunk(text, metadata, chunk_id), ...]  │
│         │   Overlap assuré entre chunks consécutifs            │
│         │                                                       │
│  Step 3 │ Embedder (SentenceTransformer)                       │
│         │   [Chunk.text] → np.ndarray (N, dim) float32         │
│         │   Batch processing, normalisation L2                 │
│         │                                                       │
│  Step 4 │ FAISSIndex.build()                                    │
│         │   Train (IVF) + Add embeddings                       │
│         │                                                       │
│  Step 5 │ BM25Index.build() [si activé]                        │
│         │   Calcul IDF sur tout le corpus                      │
│         │                                                       │
│  Step 6 │ VectorStore.save()                                    │
│         │   → data/index/datacenter_rag.faiss                 │
│         │   → data/index/datacenter_rag_meta.json             │
│         │   → data/index/datacenter_rag_bm25.pkl              │
│         │   → data/index/build_stats.json                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✂️ Stratégies de chunking

### `recursive` (recommandé)

Découpe hiérarchiquement en respectant la structure du texte :

```
Texte complet
    ↓ Split sur \n\n (paragraphes)
    ↓ Si trop grand → Split sur \n
    ↓ Si trop grand → Split sur ". "
    ↓ Si trop grand → Split sur " "
    ↓ Merge des petits fragments avec overlap
```

**Idéal pour** : documentation technique, procédures, manuels.

### `fixed`

Fenêtre glissante de taille fixe (en mots).

```python
# chunk_size=512, chunk_overlap=64 signifie :
# chunk_0 : mots [0, 512]
# chunk_1 : mots [448, 960]   (512 - 64 = 448)
# chunk_2 : mots [896, 1408]
```

**Idéal pour** : données homogènes, datasets JSON structurés.

### `sentence`

Regroupe les phrases entières jusqu'à `chunk_size` mots.

**Idéal pour** : textes narratifs, articles, Q&A.

---

## 🔀 Recherche hybride (Dense + BM25)

Le module implémente la **Reciprocal Rank Fusion (RRF)** pour combiner les deux scores :

```
score_hybride(doc) = α × RRF_dense(doc) + (1-α) × RRF_bm25(doc)

avec RRF(rank) = 1 / (60 + rank)
```

| Valeur de `alpha` | Comportement |
|---|---|
| `alpha = 1.0` | Dense seul (sémantique pure) |
| `alpha = 0.6` | **Mix recommandé** (défaut) |
| `alpha = 0.0` | BM25 seul (termes exacts) |

**Quand activer le BM25 ?**
- Corpus avec termes techniques, acronymes, noms de commandes
- Requêtes avec mots-clés exacts (ex: `systemctl`, `kubectl rollout`)
- Recherche insensible aux reformulations sémantiques

---

## 📄 Formats de documents supportés

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.txt` | TextLoader | UTF-8, tous encodages |
| `.md` | TextLoader | Markdown brut (pas de parse) |
| `.json` | JsonLoader | Objet unique ou liste |
| `.jsonl` | JsonlLoader | 1 entrée JSON par ligne (train.jsonl) |
| `.csv` | CsvLoader | Concatène toutes les colonnes |
| `.pdf` | PdfLoader | Nécessite `pip install pypdf` |

### Clés JSON auto-détectées

Le JsonLoader extrait automatiquement le texte depuis les clés :
`content`, `text`, `body`, `instruction`, `output`, `question`, `answer`, `passage`, `description`

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=src/rag --cov-report=term-missing

# Un test spécifique
pytest tests/test_rag_index.py::TestBM25Index::test_build_and_score -v

# Tests rapides (sans modèle ML)
pytest tests/ -v -k "not Smoke"
```

### Couverture des tests

| Module | Tests |
|--------|-------|
| `document_loader.py` | TXT, JSONL, JSON, CSV, répertoire récursif, filtre min_chars |
| `chunker.py` | Fixed, Recursive, Sentence, factory, metadata |
| `vector_store.py` | BM25 build/search/persist, FAISS Flat/IVF/persist, VectorStore dense/hybrid/roundtrip |
| `build_index.py` | Smoke test dry-run, chargement config, chunking |

---

## 🔌 Intégration avec Task 8 (Inference)

Le module `Retriever` est conçu pour être importé directement par `src/inference/chat.py` :

```python
# Dans src/inference/chat.py (Task 8)
from src.rag.retriever import Retriever

class ChatBot:
    def __init__(self):
        # Charger le retriever une seule fois au démarrage
        self.retriever = Retriever.load(
            index_dir="data/index",
            index_name="datacenter_rag",
            config_path="configs/rag_config.yaml"
        )

    def answer(self, question: str) -> str:
        # Récupérer le contexte pertinent
        context = self.retriever.retrieve_as_context(question, top_k=5)

        # Injecter dans le prompt LLM
        prompt = f"""Contexte :
{context}

Question : {question}
Réponse :"""
        return self.llm.generate(prompt)
```

---

## 🧠 Choix du modèle d'embedding

| Modèle | Taille | Dim | Langues | Recommandé pour |
|--------|--------|-----|---------|-----------------|
| `all-MiniLM-L6-v2` | ~80 MB | 384 | EN | Tests rapides, faible RAM |
| `all-mpnet-base-v2` | ~420 MB | 768 | EN | Corpus anglais uniquement |
| `BAAI/bge-m3` ⭐ | ~570 MB | 1024 | Multi | **Corpus technique FR/EN** |
| `intfloat/multilingual-e5-base` | ~560 MB | 768 | Multi | Alternative à BGE-M3 |

> 💡 **Recommandation** : utiliser `BAAI/bge-m3` pour un Data Center avec documentation
> en français et en anglais. Ce modèle est en tête des benchmarks MTEB multilingues.

---

## 🛠 Troubleshooting

### `ImportError: faiss not found`
```bash
pip install faiss-cpu
```

### `ImportError: sentence_transformers not found`
```bash
pip install sentence-transformers
```

### L'index est trop lent à construire
```yaml
# Dans rag_config.yaml — réduire la batch size ou changer de modèle
embedding:
  model_name: "all-MiniLM-L6-v2"   # Modèle 8× plus rapide
  batch_size: 16
  device: "cuda"                     # Si GPU disponible
```

### `AssertionError: Index non entraîné`
```yaml
faiss:
  index_type: "Flat"   # Flat ne nécessite pas d'entraînement
```

Ou s'assurer que `nlist` < nombre de chunks :
```yaml
faiss:
  nlist: 50   # < nb_chunks / 10 recommandé
```

### L'index existant n'est pas écrasé
```yaml
index:
  overwrite: true   # Passer à true pour écraser
```

### Mémoire insuffisante (OOM)
```yaml
embedding:
  batch_size: 8      # Réduire le batch
  max_seq_length: 256  # Tronquer les chunks
chunking:
  chunk_size: 256    # Réduire la taille des chunks
```

---

## 📚 Références techniques

- **FAISS** : Johnson et al., *Billion-scale similarity search with GPUs*, TPAMI 2021
- **BGE-M3** : Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*, arXiv 2024
- **BM25** : Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009
- **RRF** : Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009
- **RAG** : Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020

---

## 👤 Auteur & Responsabilité

**Rôle** : Responsable RAG Index (Tâche 6 / 9)

**Fichiers produits** :
- `src/rag/` (6 modules Python)
- `configs/rag_config.yaml`
- `data/index/` (artefacts générés)
- `tests/test_rag_index.py`
- `scripts/demo_retrieval.py`

**Interface avec les autres membres** :

| Direction | Dépendance |
|-----------|-----------|
| ← Task 3 (Data acquisition) | `data/raw/` — documents sources |
| ← Task 4/5 (Data processing/cleaning) | `data/processed/train.jsonl`, `validation.jsonl` |
| → Task 8 (Inference) | `src/rag/retriever.py` — `Retriever.load()` |
| → Task 9 (Évaluation) | `build_stats.json`, métriques de retrieval |

---

*Projet : Chatbot LLM Offline pour Data Center | Architecture : RAG + LoRA*
