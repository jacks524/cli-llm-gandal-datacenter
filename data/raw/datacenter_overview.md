# Vue generale du data center etudiant

Le data center etudiant contient des serveurs de calcul, une couche reseau,
un stockage partage et des services d'administration. Le chatbot local aide les
utilisateurs a comprendre l'infrastructure, diagnostiquer des incidents simples
et retrouver les procedures internes.

Le chatbot doit fonctionner offline. Il ne doit pas dependre d'une API externe
comme ChatGPT, Gemini ou un service cloud. Le modele recommande pour le premier
prototype est Qwen2.5-0.5B-Instruct en GGUF quantifie Q4_K_M.

Le RAG donne au chatbot le contexte documentaire du data center : architecture,
commandes CLI, procedures, runbooks et regles de securite. Le fine-tuning LoRA
sert surtout a adapter le style de reponse, le vocabulaire et les formats
attendus par l'equipe.
