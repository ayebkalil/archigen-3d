# Jalon de mi-parcours - ArchiGen-3D

Date de préparation 11 novembre 2026.

## Documents

- Rapport français de cinq pages dans ../../output/pdf/Rapport_MiParcours_TrackX_ArchiGen3D.pdf.
- Pitch français de cinq diapositives dans ../../output/pdf/Pitch_MiParcours_TrackX_ArchiGen3D.pdf.
- Version PowerPoint modifiable dans ../../output/presentation/Pitch_MiParcours_TrackX_ArchiGen3D.pptx.
- Source du rapport dans Rapport_MiParcours_TrackX_ArchiGen3D.md.
- Notes orales dans Notes_Pitch_3min30.md.

La durée du pitch est de 3 min 30 s, répartie en 30, 45, 60, 45 et 30 secondes.

## Informations à compléter avant le dépôt

1. Confirmer le numéro de track et remplacer TrackX dans les documents et les noms de fichiers.
2. Confirmer le nom administratif du groupe. ArchiGen3D est le nom de projet utilisé.
3. Renseigner les trois emails institutionnels.
4. Valider les rôles, les responsabilités RACI et les estimations d'avancement avec le trinôme.
5. Confirmer la date finale et traduire S+1 à S+4 en semaines calendaires.

Le document propose 78 % d'avancement, soit la moyenne arrondie de 85, 80, 85, 85, 75 et 60 %. Il s'agit d'une estimation d'ingénierie, proposée pour validation.

## Environnement et preuve de validation

Le 11 novembre 2026, sur la copie de travail locale utilisée pour la préparation du dossier.

- python -m pytest tests/ backend/tests/ -q --disable-warnings --maxfail=3 a donné 119 passed et 69 warnings.
- npm test dans frontend a donné 27 réussites.
- npm run build dans frontend a réussi, avec l'avertissement de bundle supérieur à 500 kB.

Les métriques scientifiques proviennent des runs et fichiers de benchmark enregistrés. La préparation de ce dossier n'a pas relancé d'entraînement. Le rapport technique détaillé antérieur contient la matrice complète et l'historique MLflow en pièce jointe.

L'image de la maison exemple est une capture réelle de la référence tracée, identifiée comme telle dans le rapport et le pitch. Les métriques de généralisation proviennent d'autres plans.

## Démonstration native

Utiliser l'environnement Python natif CUDA déjà configuré. Dépendances de base dans backend/requirements.txt, entraînement dans requirements-train.txt et extensions dédiées dans requirements-facade512.txt et requirements-furniture-quality.txt.

Depuis la racine, dans une première console PowerShell.

```powershell
$env:ARCHIGEN_FURNITURE_DETAIL = 'high'
& .venv-furniture-quality/Scripts/python.exe -m uvicorn backend.app.furniture_worker:app --host 127.0.0.1 --port 8001
```

Dans une deuxième console.

```powershell
$env:ARCHIGEN_FURNITURE_SERVICE_URL = 'http://127.0.0.1:8001'
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Dans une troisième console.

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 3000 --strictPort
```

Le studio est accessible sur http://127.0.0.1:3000 et Swagger sur http://127.0.0.1:8000/docs.

## MLflow et reproduction des résultats

Depuis la racine.

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Ce lancement lit le store local existant. La base SQLite et les checkpoints doivent être disponibles sur le poste utilisé pour la séance. Les runs MLflow, leurs métriques et paramètres sont également exportés dans la pièce jointe du rapport technique détaillé.

Pour un entraînement court de vérification du pipeline de plans, dans un environnement disposant des données et d'un GPU libre.

```powershell
python -m src.floorplan.train --epochs 1 --batch_size 2 --img_size 512
```

Pour réévaluer le champion sur validation avec les poids correspondants et CubiCasa5K.

```powershell
python -m src.floorplan.evaluate --model multitask --split val --weights models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth
```

Pour rejouer le benchmark historique des façades après récupération des poids des variantes et du jeu CMP.

```powershell
python -m scripts.evaluate_facade_models
```

Ce dernier script réécrit models/saved/facade_authoritative_metrics.json. Exécuter les nouvelles évaluations sur une branche de travail et conserver le protocole, les hashes et les anciens résultats.

## DVC et Docker

Le remote OneDrive du poste de développement utilise un chemin Windows local. Chaque équipier doit configurer un remote vers le stockage DVC partagé et accessible depuis sa machine.

```powershell
dvc remote add --local -d team-weights CHEMIN_DU_STOCKAGE_PARTAGE
dvc pull
docker compose config
docker compose up --build -d --wait
```

Choisir le mode natif ou les conteneurs pour les ports 3000 et 8000. Docker exécute le plan et la façade sur ONNX CPU. Le worker mobilier reste natif CUDA sur 8001. La procédure scripts/start_local_stack.ps1 prépare le worker et Compose.

Le replay Docker depuis un clone indépendant demeure une action du planning. Le dossier ne prétend pas l'avoir rejoué le 11 novembre.

## Git et dépôt final

Le dossier est préparé depuis HEAD a30ef02572bf9bf7fa76c65c859ebf6340e2f78f et une copie de travail contenant des évolutions applicatives locales. Le commit du dossier de soumission contient les documents du jalon. La copie de travail testée comporte des modifications applicatives à revoir et intégrer avant de présenter un clone comme équivalent à la démo locale.

L'audit Git de HEAD avant ce jalon attribuait 88 commits à Khalil Ayeb. Les autres membres doivent intégrer et expliquer leurs propres contributions. Conserver les auteurs réels et les dates des commits.

Avant dépôt sur la plateforme.

```powershell
git status --short
git log --oneline -8
git shortlog -sn HEAD
```

Après revue des modifications applicatives et des informations administratives, intégrer la branche du jalon et pousser la version retenue. Le message de jalon attendu est `feat: jalon evaluation mi-parcours`.

Le tag ou commit doit être visible sur le dépôt partagé pour la séance. Le présent dossier constitue les fichiers et preuves préparés localement.
