"""Chemins centraux du projet.

Ce module evite de dupliquer les chemins dans tous les scripts. Les chemins
sont calcules depuis la racine du repo, donc les commandes peuvent etre lancees
avec `python -m ...` depuis le dossier principal.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
ADAPTERS_DIR = MODELS_DIR / "adapters"

DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "train.jsonl"
DEFAULT_CLEAN_DATASET_PATH = PROCESSED_DATA_DIR / "train_clean.jsonl"
DEFAULT_INDEX_PATH = INDEX_DIR / "rag_index.json"
DEFAULT_MODEL_PATH = BASE_MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
