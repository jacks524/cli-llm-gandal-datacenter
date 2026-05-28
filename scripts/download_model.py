"""Telechargement du modele GGUF recommande.

Le script telecharge Qwen2.5-0.5B-Instruct en quantification Q4_K_M dans
`models/base/`. Le dossier reste ignore par Git car le modele est volumineux.
"""

from pathlib import Path

from src.common.paths import BASE_MODELS_DIR


REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def main():
    """Telecharge le fichier GGUF depuis Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Installe d'abord huggingface_hub: pip install -r requirements.txt"
        ) from exc

    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BASE_MODELS_DIR / FILENAME
    if output_path.exists():
        print(f"Modele deja present: {output_path}")
        return

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(BASE_MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    final_path = Path(downloaded)
    print(f"Modele telecharge: {final_path}")


if __name__ == "__main__":
    main()
