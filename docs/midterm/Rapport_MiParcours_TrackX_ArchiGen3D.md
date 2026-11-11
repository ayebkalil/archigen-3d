# Rapport d’avancement de mi-parcours - ArchiGen-3D

Module Deep Learning Avancé et MLOps - Haythem Ghazouani - 11 novembre 2026

> Champs administratifs à compléter avant dépôt. TrackX, identifiant administratif du groupe et emails institutionnels.

## Fiche d’identité et état global

ArchiGen-3D - Plans 2D, reconstruction de maisons et synthèse de façades

| Champ | Informations du groupe |
| --- | --- |
| Sujet | Plateforme de vision et MLOps pour convertir un plan 2D en maison 3D interactive et générer des textures de façade. |
| Track métier | TrackX à confirmer. Track 4 Génératif pressenti pour la synthèse de façades. |
| Groupe | ArchiGen3D comme nom de projet. Identifiant administratif à confirmer. |
| Dépôt Git | <link href="https://github.com/ayebkalil/archigen-3d" color="#086659">https://github.com/ayebkalil/archigen-3d</link> |
| Workspace MLflow | Stockage local mlflow.db et mlruns/. Affichage avec mlflow ui, port 5000. |
| Avancement global | 78 % estimés par la moyenne simple des six composantes de la page 3. Estimation de charge, à valider par le trinôme. |

### Membres et rôles proposés

| Membre | Rôle principal proposé | Email institutionnel |
| --- | --- | --- |
| Khalil Ayeb | Modèles DL, entraînement et analyse des résultats | À compléter |
| Khawla Hawari | Données, DVC, MLflow et reproductibilité | À compléter |
| Amine Ben Fadhel | API, interface React et intégration Docker | À compléter |

Les noms proviennent du dépôt. Les rôles et la matrice RACI de la page 5 constituent une proposition de répartition à confirmer.

### Périmètre et démonstration

Deux modèles principaux constituent le cœur de l’évaluation. Le modèle de plan segmente les murs, ouvertures et types de pièces. Le modèle de façade synthétise une image conditionnée par les ouvertures. Le studio intègre aussi la reconstruction de mobilier TripoSR et un moteur géométrique d’aménagement.

Capture réelle du studio enregistrée dans outputs/house_geometry/roofed.png. La maison illustrée utilise le plan exemple tracé manuellement. Les benchmarks de la page 2 évaluent la prédiction sur d’autres plans.

## Travail réalisé - données et modèles

Sections 2.1 et 2.2 du gabarit officiel

### 2.1 Données et versioning DVC

CubiCasa5K contient 4 200 plans d’entraînement, 400 de validation et 400 de test, avec annotations SVG. Le pipeline conserve les proportions et recale images et masques sur un canevas 512 × 512. Il applique flips et rotations, puis les augmentations CAD lors du fine-tuning.

CMP Facades utilise 400 paires d’entraînement et 106 images de test dans le protocole historique. Les images et cartes sémantiques sont redimensionnées à 256 px, puis à 512 px pour l’expérience haute résolution. CMP ne fournit pas les quatre labels de styles architecturaux.

DVC versionne data/raw/facades.dvc, data/raw/cmp_original.dvc et data/processed/cmp512.dvc, ainsi que les poids. Le remote onedrive_backup pointe vers un dossier local OneDrive. Son accès doit être reconfiguré sur la machine de chaque équipier.

### 2.2 Architectures et pertes

Plans. UNet++ à encodeur ResNet-34 ImageNet et décodeur partagé. Deux têtes prédisent 5 classes structurelles et 12 classes de pièces. La perte combine cross-entropy pondérée et Dice. Les masques alimentent le vectoriseur, puis le viewer Three.js.

Façades. Pix2Pix U-Net 256 avec upsampling nearest + convolution et PatchGAN 70 × 70. La recette promue combine GAN, L1 ×30, LPIPS ×10 et feature matching ×10, avec EMA 0,999. Le prototype 512 ajoute un refiner FiLM, VGG-19 et deux PatchGAN. Entraînement local sur RTX 3050 Laptop, 4 Go, avec AMP et accumulation.

| Métrique et protocole | Baseline | Version retenue | Cible proposée |
| --- | --- | --- | --- |
| Pièces mIoU, validation CubiCasa N=400 | 36,95 % / 15 ép. | 62,39 % / fine-tune | ≥ 64 % |
| Portes IoU, même validation | 10,22 % | 61,13 % | ≥ 63 % |
| LPIPS, test CMP N=106, 256 px | 0,6327 | 0,3096 / EMA | &lt; 0,30 |
| KID ×1000, même test CMP | 150,357 | 63,552 ± 5,826 | &lt; 60 |

Cibles indicatives à approuver. Les choix de modèles utilisent la validation. Les scores CMP ci-dessus proviennent du benchmark de test figé. Le ± du KID est un écart-type de sous-échantillonnage.

Courbes réelles de MLflow, runs 6bec9eef et 112b8d10. Les configurations diffèrent en plus du recalage. Scores façades issus de models/saved/facade_authoritative_metrics.json.

## Travail réalisé - industrialisation

Sections 2.3 et 2.4, puis tableau synthétique d’avancement

### 2.3 Traçabilité MLflow

Le snapshot audité comprend 24 runs, 4 expériences non vides, 247 métriques finales et 4 980 points historiques. Les hyperparamètres incluent learning rate, batch, résolution, architecture et poids de pertes. Les poids sont référencés par DVC et SHA-256.

| Exécution | ID abrégé | État / preuve |
| --- | --- | --- |
| Plans alignés 25 époques | 112b8d10 | Terminé, courbes de validation et test |
| Fine-tuning CAD 12 époques | ebe8add5 | Terminé, validation mIoU pièces 62,39 % |
| Façade baseline 100 époques | e463b47e | Terminé, pertes d’entraînement |
| Façade perceptuelle avec EMA | 702efec3 | Terminé, LPIPS validation live et EMA |

Le store compte 17 FINISHED, 4 FAILED et 3 RUNNING. Ces derniers sont des états persistés. Certains runs importent des évaluations déjà calculées, dont la façade 512. Un import ne constitue pas un nouvel entraînement.

### 2.4 Services logiciels

FastAPI expose /health, /model/metadata, /predict/floorplan, /facade/generate et les routes de mobilier et d’aménagement. Pydantic valide les schémas. React/Vite propose upload, superposition 2D, édition de plan et visualisation Three.js. La pile native utilise les ports 8000 et 3000.

Vérification locale du 11/11. 119 tests Python réussis, 27 tests frontend réussis et build Vite réussi. Le build signale un bundle principal supérieur à 500 kB. Docker Compose et GitHub Actions sont présents. Le plan et la façade utilisent ONNX CPU en conteneur, avec un worker mobilier CUDA natif.

### Avancement estimé par composante

| Composant | Estimation | Éléments prêts et vérifiés | Action immédiate S+1 |
| --- | --- | --- | --- |
| Dataset et DVC | 85 % | Données, splits et poids versionnés | Tester un dvc pull sur une autre machine |
| Modèles baseline | 80 % | Deux familles entraînées, métriques sauvegardées | Améliorer la généralisation des plans |
| Tracking MLOps | 85 % | 24 runs, historique et registre | Clore les runs obsolètes et documenter les imports |
| API FastAPI | 85 % | Routes, validation et tests | Consolider la gestion des échecs |
| Frontend UI | 75 % | Studio 2D/3D et catalogue | Tester des plans variés avec utilisateurs |
| Docker et CI/CD | 60 % | Compose et workflows versionnés | Rejouer le déploiement depuis un clone propre |

Les statuts sont des estimations de l’équipe proposées pour ce document. Les tests vérifient le code local courant. La validation Docker sur une machine indépendante reste à réaliser.

## Verrous et travail restant

Sections 3 et 4 du gabarit officiel

### 3 Analyse des difficultés et actions correctives

| Verrou | Impact observé | Solution retenue ou prochaine validation |
| --- | --- | --- |
| Décalage de style des plans | Champion sur 12 plans externes, MAE de 5,25 pièces et 8,08 portes. 1/12 jugé directement exploitable dans le diagnostic de l’équipe. | Restylage CAD, fallback et correction interactive. Constituer un nouveau test externe verrouillé. |
| Régression après adaptation | Le challenger améliore les coques externes, mais le mIoU pièces sur validation baisse de 62,39 % à 56,94 %. | Conserver le champion et mesurer la régression avant promotion. Accepter un modèle sur plusieurs critères. |
| Façade floue et styles non supervisés | 512 px améliore LPIPS de 0,49887 à 0,49241 dans son protocole apparié. Les quatre styles restent sans labels. | Poursuivre le refiner et constituer des données de style. Présenter les limites des matériaux et du prototype diffusion. |
| Mémoire et portabilité | RTX 3050 4 Go. Environnement mobilier isolé et accès DVC lié à une machine. | AMP FP16, checkpointing, accumulation et garde 3 500 MiB. Rejouer le clone, les poids et les conteneurs sur un autre poste. |
| Traçabilité collective Git | L’historique HEAD a30ef02 audité attribue 88 commits à Khalil. Les contributions des deux autres membres demandent une preuve visible. | Chaque membre livre et explique ses propres changements sur une branche puis PR. Vérifier l’historique avant le jalon. |

Les 12 plans externes ont servi à diagnostiquer le restylage. Les résultats du challenger sur ce même ensemble sont exploratoires. Une nouvelle sélection indépendante est nécessaire pour la validation finale.

### 4 Travail restant priorisé

P1 - Modèles. Verrouiller de nouveaux plans externes. Comparer champion, restylage et routeur avec mIoU, comptes de pièces et exploitabilité. Évaluer la façade 512 dans un protocole apparié et documenter les styles supportés.

P2 - Production. Rejouer DVC et Docker depuis un clone propre. Vérifier ports, schémas, erreurs et parité ONNX. Exécuter la CI sur la branche du jalon puis sur main.

P3 - Surveillance. Définir une référence des statistiques d’images et un signal de dérive, puis mesurer les faux positifs. Les métriques d’API existantes couvrent la santé et la latence. La détection formelle de dérive reste à réaliser.

P4 - Livraison. Valider les rôles, compléter les informations administratives, consolider les contributions Git, répéter la démonstration et figer la version finale.

## Planning et responsabilités

Section 5 du gabarit officiel - proposition à valider par le trinôme

### Gantt pour les quatre semaines restantes

| Tâche / jalon | S+1 | S+2 | S+3 | S+4 | Responsable proposé |
| --- | --- | --- | --- | --- | --- |
| Test externe, adaptation et tuning | ● | ● |  |  | Khalil |
| Façade 512 et styles | ● | ● |  |  | Khalil |
| API, schémas et parcours UI | ● | ● | ● |  | Amine |
| DVC, Docker et clone propre | ● | ● | ● |  | Khawla + Amine |
| CI, parité et monitoring |  | ● | ● |  | Khawla |
| Tests et répétition finale |  |  | ● | ● | Toute l’équipe |

S+1 à S+4 sont des semaines relatives à la séance. La date de soutenance finale demande confirmation. Les cellules pleines indiquent les semaines de travail prévues.

### Matrice RACI proposée

| Tâche | Khalil | Khawla | Amine |
| --- | --- | --- | --- |
| Données et DVC | C | R/A | I |
| Entraînement DL | R/A | C | I |
| MLflow et registre | R | A | I |
| API FastAPI et inférence | C | C | R/A |
| Interface React / Three.js | C | I | R/A |
| Docker, CI et monitoring | C | R/A | R |
| Tests et démonstration | R | R | R/A |

R réalise, A valide, C conseille, I est informé. Les responsabilités indiquent la répartition de travail proposée, à confirmer avec chaque membre.

### Critères de livraison et preuves

Chaque candidat final doit disposer d’un checkpoint et d’un hash, d’un run ou d’une référence d’évaluation, d’un test de régression et d’un mode d’exécution documenté. Le dernier jalon inclut une démo de plan inédit, un essai de façade et la répétition du pitch de 3 min 30 s.

Sources techniques accessibles dans le dépôt. src/floorplan/models.py, src/models/architecture.py, models/saved, mlflow.db, .dvc/config, docker-compose.yml et .github/workflows/ci-tests.yml. Les benchmarks chiffrés sont détaillés dans le rapport d’évaluation technique joint au projet.

Champs à finaliser avant dépôt. TrackX, identifiant administratif du groupe, emails institutionnels, validation des estimations et du RACI. Le nom de fichier utilise ArchiGen3D comme nom de projet.
