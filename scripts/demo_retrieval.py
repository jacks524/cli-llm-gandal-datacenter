#!/usr/bin/env python3
"""
scripts/demo_retrieval.py
=========================
Task 6 – RAG Index | Démo rapide du pipeline

Lance une démo interactive du retriever sans GPU.
Utilise des données d'exemple intégrées.

Usage
-----
  python scripts/demo_retrieval.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Données d'exemple (procédures Data Center) ───────────────

SAMPLE_DOCS = [
    ("ldap_procedures.txt", """
Procédure de redémarrage du serveur LDAP
=========================================
1. Se connecter en SSH sur dc01.internal avec l'utilisateur admin.
2. Vérifier l'état du service : systemctl status slapd
3. Arrêter le service : sudo systemctl stop slapd
4. Vider le cache : sudo rm -rf /var/cache/slapd/*
5. Redémarrer : sudo systemctl start slapd
6. Vérifier les logs : journalctl -u slapd -n 50
En cas d'échec, contacter l'équipe IAM sur #infra-ldap.
"""),
    ("backup_procedures.txt", """
Procédure de Backup Quotidien
==============================
Les backups sont automatisés via Cron (02:00 UTC).
Script : /opt/backup/run_backup.sh
Destination : NAS backup-nas01 -> /mnt/backups/daily/
Retention : 30 jours (daily), 12 mois (monthly)

Commandes manuelles :
  - Lancer un backup immédiat : sudo /opt/backup/run_backup.sh --now
  - Vérifier les derniers backups : ls -lh /mnt/backups/daily/
  - Restaurer un fichier : rsync -av backup-nas01:/mnt/backups/daily/<date>/ /restore/
"""),
    ("kubernetes_ops.txt", """
Gestion du cluster Kubernetes
================================
Namespace production : prod-ns
Namespace staging    : staging-ns

Commandes utiles :
  kubectl get pods -n prod-ns
  kubectl describe pod <nom> -n prod-ns
  kubectl logs <pod> -n prod-ns --tail=100
  kubectl rollout restart deployment/<nom> -n prod-ns

En cas de CrashLoopBackOff :
  1. kubectl logs <pod> --previous
  2. Vérifier les limites mémoire : kubectl top pod
  3. Redimensionner les resources dans le manifest YAML
"""),
    ("network_config.txt", """
Configuration Réseau du Data Center
=====================================
VLAN de management : 100 (192.168.100.0/24)
VLAN de production : 200 (10.0.0.0/16)
VLAN de backup     : 300 (172.16.0.0/12)

Passerelle par défaut : 192.168.100.1 (router-core01)
DNS primaire    : 192.168.100.10
DNS secondaire  : 192.168.100.11

Accès SSH restreint aux IPs du VLAN 100 uniquement.
Firewall géré par pfSense sur fw01.internal.
"""),
    ("monitoring_alerts.txt", """
Système de Monitoring et Alertes
==================================
Stack : Prometheus + Grafana + Alertmanager
URL Grafana : http://grafana.internal:3000

Seuils d'alerte :
  - CPU > 85% pendant 5 min → WARNING
  - CPU > 95% pendant 2 min → CRITICAL
  - RAM > 90%               → WARNING
  - Disk > 80%              → WARNING
  - Disk > 95%              → CRITICAL

Acquitter une alerte : se connecter à Alertmanager http://alertmanager:9093
Pour les alertes critiques, PagerDuty est automatiquement notifié.
"""),
]


def run_demo():
    print("\n" + "=" * 60)
    print("  DÉMO RAG INDEX — Data Center LLM Chatbot (Task 6)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "raw"
        data_dir.mkdir()
        index_dir = tmpdir / "index"

        # ── 1. Créer les fichiers d'exemple ──────────────────
        print("\n[1/4] Création des documents d'exemple…")
        for filename, content in SAMPLE_DOCS:
            (data_dir / filename).write_text(content.strip(), encoding="utf-8")
        print(f"      → {len(SAMPLE_DOCS)} fichiers créés dans {data_dir}")

        # ── 2. Charger et chunker ─────────────────────────────
        print("\n[2/4] Chargement et chunking des documents…")
        from src.rag.document_loader import DocumentLoader
        from src.rag.chunker import get_chunker

        loader = DocumentLoader(extensions=[".txt"])
        docs = loader.load_directory(data_dir)
        print(f"      → {len(docs)} document(s) chargés")

        chunker = get_chunker("recursive", chunk_size=80, chunk_overlap=10)
        all_chunks = []
        for doc in docs:
            chunks = chunker.split(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)
        print(f"      → {len(all_chunks)} chunks produits")

        # ── 3. Embeddings + Index ─────────────────────────────
        print("\n[3/4] Construction de l'index (sans GPU)…")
        print("      (Chargement du modèle sentence-transformers — peut prendre 30s)")

        import numpy as np
        from src.rag.embedder import Embedder
        from src.rag.vector_store import FAISSIndex, VectorStore

        try:
            embedder = Embedder(model_name="all-MiniLM-L6-v2", device="cpu", batch_size=16)
        except Exception:
            print("      ⚠️  sentence-transformers non disponible — utilisation de vecteurs aléatoires")
            # Fallback pour test sans dépendances
            _demo_fallback(all_chunks)
            return

        texts = [c.text for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress=False)

        faiss_idx = FAISSIndex(dim=embedder.dim, index_type="Flat", metric="cosine")
        store = VectorStore(faiss_index=faiss_idx)

        # Assigner les IDs
        for i, c in enumerate(all_chunks):
            c.chunk_id = f"chunk_{i:04d}"

        store.build(all_chunks, embeddings)
        store.save(index_dir, "demo_index")
        print(f"      → Index sauvegardé : {index_dir}/")

        # ── 4. Requêtes interactives ──────────────────────────
        print("\n[4/4] Test de retrieval…\n")
        queries = [
            "Comment redémarrer le serveur LDAP ?",
            "Quelle est la procédure de backup ?",
            "Que faire en cas de CrashLoopBackOff Kubernetes ?",
            "Quels sont les seuils d'alerte CPU ?",
        ]

        for query in queries:
            print(f"  ❓  {query}")
            query_vec = embedder.encode_query(query)
            results = store.search(query_vec, query_text=query, top_k=2)
            for i, r in enumerate(results, 1):
                source = Path(r.metadata.get("source", "?")).name
                snippet = r.text[:100].replace("\n", " ")
                print(f"      [{i}] ({source}, score={r.score:.3f}) {snippet}…")
            print()

    print("=" * 60)
    print("  Démo terminée avec succès ✓")
    print("=" * 60 + "\n")


def _demo_fallback(chunks):
    """Démo sans modèle d'embeddings réel."""
    from src.rag.vector_store import BM25Index
    print("      Mode BM25 seul (pas de modèle d'embedding disponible)")
    bm25 = BM25Index()
    bm25.build([c.text for c in chunks])

    queries = ["LDAP redémarrage", "backup procédure", "kubernetes pod crash"]
    for q in queries:
        results = bm25.score(q, top_k=2)
        print(f"\n  ❓  {q}")
        for idx, score in results:
            snippet = chunks[idx].text[:80].replace("\n", " ")
            print(f"      (score={score:.3f}) {snippet}…")


if __name__ == "__main__":
    run_demo()
