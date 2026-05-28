"""Nettoyage du dataset du chatbot.

Ce fichier servira a supprimer les doublons, corriger les formats invalides
et filtrer les exemples inutilisables avant l'entrainement.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
import re


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Metriques de qualite du dataset."""
    total_samples: int = 0
    valid_samples: int = 0
    duplicates_removed: int = 0
    empty_fields_removed: int = 0
    invalid_format_removed: int = 0
    too_short_removed: int = 0
    too_long_removed: int = 0
    final_samples: int = 0
    
    def __post_init__(self):
        self.total_samples = max(
            self.valid_samples 
            + self.duplicates_removed 
            + self.empty_fields_removed
            + self.invalid_format_removed
            + self.too_short_removed
            + self.too_long_removed,
            self.total_samples
        )


class DataCleaner:
    """Classe pour le nettoyage et validation du dataset."""
    
    # Constantes de validation
    MIN_QUESTION_LENGTH = 5
    MAX_QUESTION_LENGTH = 500
    MIN_ANSWER_LENGTH = 10
    MAX_ANSWER_LENGTH = 5000
    MIN_INSTRUCTION_LENGTH = 5
    MIN_OUTPUT_LENGTH = 5
    
    def __init__(self):
        """Initialise le nettoyeur de donnees."""
        self.metrics = QualityMetrics()
        self.seen_hashes = set()
        self.issues = []
    
    def hash_example(self, example: Dict) -> str:
        """Genere un hash pour identifier les doublons."""
        if "question" in example and "answer" in example:
            text = f"{example['question']}{example['answer']}"
        elif "instruction" in example and "output" in example:
            text = f"{example['instruction']}{example['output']}"
        else:
            text = json.dumps(example, sort_keys=True)
        
        return hashlib.md5(text.encode()).hexdigest()
    
    def is_empty_or_none(self, value: Any) -> bool:
        """Verifie si une valeur est vide ou None."""
        if value is None:
            return True
        if isinstance(value, str) and len(value.strip()) == 0:
            return True
        return False
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte (espaces superflus, caracteres invalides)."""
        if not isinstance(text, str):
            return str(text)
        
        # Supprime les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        # Supprime les caracteres de controle
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return text
    
    def validate_qa_pair(self, example: Dict) -> Tuple[bool, str]:
        """Valide une paire question-reponse."""
        # Verifie les champs requis
        if "question" not in example or "answer" not in example:
            return False, "Champs requis manquants (question/answer)"
        
        question = example.get("question")
        answer = example.get("answer")
        
        # Verifie les valeurs vides
        if self.is_empty_or_none(question) or self.is_empty_or_none(answer):
            return False, "Question ou reponse vide"
        
        # Verifie les types
        if not isinstance(question, str) or not isinstance(answer, str):
            return False, "Format invalide pour question ou reponse"
        
        # Verifie les longueurs
        q_len = len(question)
        a_len = len(answer)
        
        if q_len < self.MIN_QUESTION_LENGTH:
            return False, f"Question trop courte ({q_len} < {self.MIN_QUESTION_LENGTH})"
        if q_len > self.MAX_QUESTION_LENGTH:
            return False, f"Question trop longue ({q_len} > {self.MAX_QUESTION_LENGTH})"
        if a_len < self.MIN_ANSWER_LENGTH:
            return False, f"Reponse trop courte ({a_len} < {self.MIN_ANSWER_LENGTH})"
        if a_len > self.MAX_ANSWER_LENGTH:
            return False, f"Reponse trop longue ({a_len} > {self.MAX_ANSWER_LENGTH})"
        
        return True, "Valid"
    
    def validate_instruction_output(self, example: Dict) -> Tuple[bool, str]:
        """Valide une paire instruction-output."""
        # Verifie les champs requis
        if "instruction" not in example or "output" not in example:
            return False, "Champs requis manquants (instruction/output)"
        
        instruction = example.get("instruction")
        output = example.get("output")
        
        # Verifie les valeurs vides
        if self.is_empty_or_none(instruction) or self.is_empty_or_none(output):
            return False, "Instruction ou output vide"
        
        # Verifie les types
        if not isinstance(instruction, str) or not isinstance(output, str):
            return False, "Format invalide pour instruction ou output"
        
        # Verifie les longueurs
        i_len = len(instruction)
        o_len = len(output)
        
        if i_len < self.MIN_INSTRUCTION_LENGTH:
            return False, f"Instruction trop courte ({i_len})"
        if o_len < self.MIN_OUTPUT_LENGTH:
            return False, f"Output trop court ({o_len})"
        
        return True, "Valid"
    
    def clean_example(self, example: Dict, index: int) -> Tuple[Dict, bool, str]:
        """Nettoie un exemple et le valide."""
        if not isinstance(example, dict):
            return example, False, "Format invalide"
        
        cleaned = example.copy()
        
        # Determine le format (question/answer ou instruction/output)
        is_qa = "question" in example and "answer" in example
        is_io = "instruction" in example and "output" in example
        
        if not is_qa and not is_io:
            return cleaned, False, "Format non reconnu"
        
        # Nettoie le texte
        try:
            if is_qa:
                cleaned["question"] = self.clean_text(cleaned.get("question", ""))
                cleaned["answer"] = self.clean_text(cleaned.get("answer", ""))
                is_valid, reason = self.validate_qa_pair(cleaned)
            else:
                cleaned["instruction"] = self.clean_text(cleaned.get("instruction", ""))
                cleaned["output"] = self.clean_text(cleaned.get("output", ""))
                is_valid, reason = self.validate_instruction_output(cleaned)
            
            if not is_valid:
                return cleaned, False, reason
            
            # Verifie les doublons
            hash_val = self.hash_example(cleaned)
            if hash_val in self.seen_hashes:
                return cleaned, False, "Doublon"
            
            self.seen_hashes.add(hash_val)
            return cleaned, True, "Valid"
        
        except Exception as e:
            return cleaned, False, str(e)
    
    def clean_dataset(self, input_file: Path, output_file: Path, 
                     validation_file: Path) -> QualityMetrics:
        """Nettoie un dataset et genere des rapports de validation."""
        logger.info(f"Debut du nettoyage: {input_file}")
        
        cleaned_examples = []
        validation_records = []
        
        # Charge et traite les donnees
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.suffix == '.jsonl':
                    lines = f.readlines()
                    examples = [json.loads(line) for line in lines]
                else:
                    data = json.load(f)
                    examples = data if isinstance(data, list) else [data]
            
            self.metrics.total_samples = len(examples)
            logger.info(f"Nombre total d'exemples charges: {self.metrics.total_samples}")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier: {e}")
            return self.metrics
        
        # Traite chaque exemple
        for idx, example in enumerate(examples):
            cleaned_ex, is_valid, reason = self.clean_example(example, idx)
            
            # Cree un record de validation
            validation_record = {
                "index": idx,
                "original": example,
                "cleaned": cleaned_ex,
                "valid": is_valid,
                "reason": reason
            }
            validation_records.append(validation_record)
            
            # Met a jour les metriques
            if is_valid:
                cleaned_examples.append(cleaned_ex)
                self.metrics.valid_samples += 1
            else:
                if reason == "Doublon":
                    self.metrics.duplicates_removed += 1
                elif "vide" in reason:
                    self.metrics.empty_fields_removed += 1
                elif "invalide" in reason.lower():
                    self.metrics.invalid_format_removed += 1
                elif "courte" in reason.lower():
                    self.metrics.too_short_removed += 1
                elif "longue" in reason.lower():
                    self.metrics.too_long_removed += 1
        
        self.metrics.final_samples = len(cleaned_examples)
        
        # Sauvegarde le dataset nettoye
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.suffix == '.jsonl':
                    for example in cleaned_examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                else:
                    json.dump(cleaned_examples, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset nettoye sauvegarde: {output_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du dataset: {e}")
        
        # Sauvegarde les rapports de validation
        try:
            validation_file.parent.mkdir(parents=True, exist_ok=True)
            with open(validation_file, 'w', encoding='utf-8') as f:
                for record in validation_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Rapport de validation sauvegarde: {validation_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rapport: {e}")
        
        # Affiche les statistiques
        self._print_statistics()
        
        return self.metrics
    
    def _print_statistics(self):
        """Affiche les statistiques de nettoyage."""
        logger.info("\n" + "="*60)
        logger.info("STATISTIQUES DE NETTOYAGE")
        logger.info("="*60)
        logger.info(f"Exemples charges:              {self.metrics.total_samples}")
        logger.info(f"Exemples valides:             {self.metrics.valid_samples}")
        logger.info(f"Doublons supprimes:           {self.metrics.duplicates_removed}")
        logger.info(f"Champs vides supprimes:       {self.metrics.empty_fields_removed}")
        logger.info(f"Format invalide supprime:     {self.metrics.invalid_format_removed}")
        logger.info(f"Exemples trop courts:         {self.metrics.too_short_removed}")
        logger.info(f"Exemples trop longs:          {self.metrics.too_long_removed}")
        logger.info(f"Exemples finaux:              {self.metrics.final_samples}")
        
        if self.metrics.total_samples > 0:
            retention_rate = (self.metrics.final_samples / self.metrics.total_samples) * 100
            logger.info(f"Taux de retention:            {retention_rate:.1f}%")
        logger.info("="*60 + "\n")


def main():
    """Point d'entree du script de nettoyage."""
    # Chemin des donnees
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    
    # Fichiers d'entree/sortie
    input_file = processed_dir / "prepared.jsonl"
    output_file = processed_dir / "cleaned.jsonl"
    validation_file = processed_dir / "validation.jsonl"
    
    # Cree un example de test si le fichier n'existe pas
    if not input_file.exists():
        logger.warning(f"Fichier d'entree non trouve: {input_file}")
        logger.info("Creation d'un exemple de test...")
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        test_data = [
            {
                "question": "Quel est le role du data center ?",
                "answer": "Le data center est responsable du stockage et de la gestion des donnees critiques de l'organisation."
            },
            {
                "question": "Comment garantir la disponibilite du service ?",
                "answer": "Par la redondance des systemes, la surveillance continue et les plans de secours."
            },
            {
                "question": "Que signifie l'acronyme SLA ?",
                "answer": "SLA signifie Service Level Agreement. C'est un accord entre le fournisseur et le client defineissant les niveaux de service garantis."
            },
            {
                "question": "Q",  # Trop court
                "answer": "Cette question est trop courte et sera filtree."
            },
            {
                "question": "Quel est le role du data center ?",  # Doublon
                "answer": "Le data center est responsable du stockage et de la gestion des donnees critiques de l'organisation."
            },
            {
                "question": "Qu'est-ce que la virtualisation ?",
                "answer": ""  # Reponse vide
            }
        ]
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Exemple de test cree: {input_file}")
    
    # Lance le nettoyage
    cleaner = DataCleaner()
    metrics = cleaner.clean_dataset(input_file, output_file, validation_file)
    
    logger.info("Nettoyage termine avec succes!")


if __name__ == "__main__":
    main()
