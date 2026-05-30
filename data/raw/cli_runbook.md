# Runbook CLI du data center

## Verification rapide d'un serveur Linux

Pour verifier l'etat general d'un serveur :

1. Utiliser `uptime` pour voir la charge moyenne.
2. Utiliser `htop` ou `top` pour inspecter CPU et RAM.
3. Utiliser `df -h` pour verifier l'espace disque.
4. Utiliser `free -h` pour verifier la memoire disponible.
5. Utiliser `journalctl -xe` pour lire les erreurs recentes.

## Serveur lent

Si un serveur est lent, commencer par verifier :

- CPU avec `htop`.
- RAM avec `free -h`.
- Disque avec `df -h`.
- I/O wait avec `iostat` si disponible.
- Erreurs systeme avec `journalctl -xe`.

Le chatbot doit proposer une demarche progressive et eviter les commandes
destructives sans avertissement.

## Reseau

Pour diagnostiquer un probleme reseau :

1. `ip a` pour verifier les interfaces.
2. `ip route` pour verifier la passerelle.
3. `ping <ip>` pour tester la connectivite.
4. `ss -tulpn` pour voir les ports ecoutes.
5. `traceroute <ip>` si le paquet est installe.
