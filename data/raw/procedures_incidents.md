# Procedures d'incident

## Service indisponible

Si un service ne repond pas, verifier d'abord son etat avec `systemctl status
<service>`. Ensuite, consulter les logs avec `journalctl -u <service> -n 100`.

Ne pas redemarrer un service critique sans verifier l'impact sur les
utilisateurs et sans prevenir le responsable de l'infrastructure.

## Stockage presque plein

Si le disque est presque plein, utiliser `df -h` pour identifier la partition,
puis `du -sh *` dans les dossiers suspects. Il faut eviter de supprimer des
fichiers sans validation. Les logs peuvent etre archives ou compresses si la
procedure interne l'autorise.

## Securite

Le chatbot doit refuser de donner une procedure dangereuse sans contexte clair.
Il doit avertir avant toute commande qui supprime, formate, redemarre ou modifie
fortement le systeme.
