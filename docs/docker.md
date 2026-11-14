# Application complète avec Docker

La [validation locale et ses limites](docker-validation.md) détaille les contrôles
réellement exécutés, les résultats et l'emplacement des artefacts.

La pile comprend Nginx/React, l'API ONNX CPU, le worker TripoSR CUDA et MLflow
3.17.1. Le service ponctuel `model-init` vérifie et prépare les trois ONNX de
production. Le modèle expérimental Pix2Pix 512 n'est pas inclus. L'entraînement
et les campagnes de stress ne sont pas exécutés par cette pile.

## Prérequis et lancement Windows

Installer Docker Desktop avec le moteur WSL2 et un pilote NVIDIA compatible.
Le mobilier exige un GPU NVIDIA accessible aux conteneurs. Le lancement vérifie
un calcul CUDA réel avant toute interruption du lancement natif. Les premiers
builds téléchargent plusieurs Go ; prévoir suffisamment de disque pour les
images CUDA, les caches et les artefacts. Aucun Python, Node ou DVC hôte n'est
nécessaire sur une nouvelle machine.

Depuis la racine du dépôt :

```powershell
.\scripts\start_docker_stack.ps1
```

Si PowerShell bloque les scripts, utiliser une dérogation limitée au processus :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_docker_stack.ps1
```

L'application est sur **http://localhost:3000/#studio**, les docs API sur
http://localhost:8000/docs et MLflow sur http://localhost:5000.
Le worker n'expose aucun port sur le PC. Copier `.env.example` vers `.env` pour
personnaliser les ports, le préréglage ou désactiver le suivi avec
`ARCHIGEN_MLFLOW_ENABLED=0`. Conserver `high` pour la configuration validée.

## Remplacer les services natifs

```powershell
.\scripts\start_docker_stack.ps1 -ReplaceNative
```

Cette option exige seulement l'environnement `.venv` déjà utilisé par la pile
native et son module `psutil`. Elle vérifie les exécutables, arguments et dossiers
de travail avant d'arrêter les services du projet sur les ports standards.
Un processus étranger bloque la migration. Les images sont construites et CUDA
vérifié avant l'arrêt. Les commandes et réglages nécessaires au retour natif
sont conservés dans `outputs/docker/native-state.json`.

Le lanceur sauvegarde SQLite avec son API de sauvegarde dans
`mlartifacts/.backups/`, initialise les modèles et la bibliothèque, puis attend
la disponibilité de tous les services. En cas d'échec après suspension, il arrête
la pile Docker sans supprimer les volumes et rétablit les services natifs.
Les diagnostics sont dans `outputs/docker/startup-failure.log`.

Après une interruption brutale du lanceur, arrêter Docker avant le retour natif :

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
.\.venv\Scripts\python.exe scripts/docker_native_transition.py restore
```

## Commandes Compose directes

Sur une nouvelle copie, créer le fichier SQLite avant son montage comme fichier
(Compose ne doit pas le créer comme dossier). Sous Linux : `touch mlflow.db` et
`mkdir -p mlartifacts`. Sous PowerShell :

```powershell
if (!(Test-Path mlflow.db)) { New-Item -ItemType File mlflow.db | Out-Null }
New-Item -ItemType Directory -Force mlartifacts | Out-Null
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d --wait
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

Les commandes directes ne remplacent pas les processus natifs et ne déclenchent
pas la sauvegarde automatique du lanceur. Pour une base existante, la sauvegarder
avant une mise à jour. Ne jamais démarrer deux serveurs MLflow sur le même store.

## Données conservées

- `models/saved/` : ONNX vérifiés par taille et SHA-256, montés en lecture seule
  dans l'API. Les fichiers valides ne sont pas téléchargés de nouveau ; un fichier
  incorrect est remplacé atomiquement seulement après vérification du téléchargement.
- `mlflow.db` et `mlartifacts/` : mêmes fichiers que le lancement natif ; historique
  conservé. Les URI `mlflow-artifacts:/` restent accessibles via le serveur Docker.
- Volume `furniture_library` : meubles et catalogue personnalisés. Au premier
  lancement, import des GLB et de l'index de `frontend/public/models/`. Une
  bibliothèque Docker déjà remplie n'est jamais écrasée par cet import.
- Images TripoSR : source et poids épinglés, configuration et checkpoint vérifiés,
  caches DINO/rembg préchargés. L'inférence n'a pas besoin de télécharger les modèles.
- Projet courant : IndexedDB du navigateur, toujours sur **localhost:3000**.
  Changer d'adresse, de port ou de navigateur utilise un stockage différent.

`down` conserve les données. `down -v` supprime la bibliothèque dans le volume :
ne pas l'utiliser pour un arrêt normal. Sauvegarder aussi les fichiers GLB du volume
et le catalogue avant une migration de machine. Le projet IndexedDB reste une
donnée du navigateur et n'est pas inclus dans la sauvegarde serveur.

## Vérification reproductible

Depuis PowerShell, exécuter le script dans le conteneur API, sans Python hôte :

```powershell
Get-Content -Raw scripts/verify_docker_stack.py | docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec -T backend python -
docker cp archigen_backend:/tmp/docker-validation outputs/docker/validation
```

Il passe par le proxy frontend, segmente le plan témoin, génère une façade et une
chaise à 0,85 m avec détourage, relit le GLB, puis vérifie un seul run MLflow
terminé et le téléchargement de son artefact identique. Il crée donc une véritable
génération dans `furniture-reconstruction` ; ce n'est pas une campagne de stress.
Pour vérifier le redémarrage sans nouvelle reconstruction GPU, ajouter
`--skip-furniture` après `python -`.

Un contrôle navigateur facultatif est aussi fourni pour le développement :
`node scripts/verify_docker_browser.cjs`. Il nécessite les dépendances frontend
locales et Chrome/Edge, génère réellement une chaise dans un profil isolé, puis
vérifie l'aperçu, la navigation, le rafraîchissement et le téléchargement. Après
redémarrage Docker, `node scripts/verify_docker_browser.cjs --restore-only`
réouvre le même profil et vérifie le résultat conservé sans relancer le modèle.
Ces dépendances de validation ne sont pas nécessaires pour utiliser l'application.

Sans GPU, `-Gpu:$false` lance seulement plans/façades/MLflow ; le mobilier n'est
pas remplacé par une extrusion. En cas d'erreur CUDA, vérifier les journaux du
service `furniture`, le pilote NVIDIA et la configuration WSL2 de Docker Desktop.
