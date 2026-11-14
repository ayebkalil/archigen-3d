# Campagne de stress ArchiGen 3D

La campagne est distincte de la comparaison CPU/GPU et des entraînements. Elle
utilise l'expérience MLflow **archigen-stress-testing** sur le serveur local existant.
Elle ne modifie ni les poids ni les API de production. Le service instrumenté est
privé, sur `127.0.0.1:18001`, et désactive le suivi automatique du mobilier pour
éviter un deuxième run par requête.

## Installation et lancement Windows

Utiliser les environnements et poids installés pour la pile native. Les dépendances
du harnais sont `httpx`, `psutil`, `Pillow`, `numpy`, `trimesh`, `pytest` et
`mlflow-skinny==3.17.1`, avec le moteur GPU existant. Aucune dépendance Win32
supplémentaire n'est nécessaire : les Job Objects utilisent `ctypes`.

Depuis la racine, avec la pile native déjà lancée :

```powershell
.\.venv-furniture-quality\Scripts\python.exe -m pip install -r requirements-stress.txt
.\scripts\start_mlflow.ps1
.\scripts\run_stress_campaign.ps1 -BudgetMinutes 30
```

Si le chargement neuronal est bloqué, le harnais peut poursuivre avec des essais
explicitement étiquetés `furniture_preprocessing`, sans reconstruction TripoSR.
Ils testent le décodage, le détourage CPU et la préparation des images. Pour les
lancer directement, puis mesurer le masque face à une référence alpha synthétique :

```powershell
.\scripts\run_stress_campaign.ps1 -BudgetMinutes 25 -ComponentsOnly
.\scripts\run_stress_campaign.ps1 -BudgetMinutes 10 -Manifest configs/stress_mask_validation.json -ComponentsOnly
```

Ces durées sont des plafonds et ne s'ajoutent pas nécessairement : les séries
peuvent finir rapidement. La courte endurance est limitée à 30 passages par défaut.
Le manifeste ciblé utilise `extends` pour conserver la même configuration de base.
La version actuelle échauffe aussi le détourage avec une image opaque avant les
mesures, indépendamment de la référence avec alpha.

Le lanceur vérifie l'identité des processus enregistrés dans
`outputs/native_gpu/processes.json`, suspend uniquement le backend et le worker
correspondants, puis les restaure et vérifie leur disponibilité dans `finally`.
Le frontend et MLflow restent ouverts. Ne pas utiliser le studio pendant les essais.
Un port de test déjà occupé provoque un refus, sans arrêt de son propriétaire.

En cas d'arrêt brutal du terminal, restaurer explicitement la pile :

```powershell
.\.venv-furniture-quality\Scripts\python.exe -m src.stress.manage_stack restore
```

Une suppression forcée du lanceur par Windows ne peut pas exécuter son bloc
`finally`. La fermeture du Job Object arrête ses enfants, mais la restauration
nécessite alors la commande ci-dessus. Le fichier `outputs/stress/stack-state.json`
conserve les commandes des services suspendus ; ne pas le supprimer avant restauration.

## Configuration et protection de la machine

Le manifeste versionné est `configs/stress_campaign.json`. Il fixe les graines,
transformations, répétitions, délais et limites. Le budget est un maximum : les
modèles absents ou un échec de démarrage peuvent terminer la campagne plus tôt.
Les phases suivent les proportions 5/18/7 minutes pour un budget de 30 minutes.
La réserve finale permet les sondes de récupération, le rapport et la restauration.

Les processus du service et des générateurs d'entrées sont attachés à des Job
Objects Windows avant de charger les modèles. Le plafond de mémoire engagée
du Job est le minimum de 6 Go et 60 % de la RAM disponible au lancement. Ce
plafond diffère du RSS échantillonné. Le garde-fou GPU TripoSR existant reste à
3 500 Mio. Les calculs GPU restent séquentiels sous verrou, même si plusieurs
clients attendent une réponse.
Le lancement crée le processus suspendu, l'attache au Job puis reprend son thread
initial ; cela couvre aussi les lanceurs venv Windows. Le PID réel annoncé par
le service est ensuite vérifié avec `IsProcessInJob` avant toute admission de charge.

Les entrées extrêmes sont fabriquées dans un enfant limité. Un watchdog termine
le Job si le délai expire ou si la RAM disponible descend sous 3 Go pendant les
mesures. Un simple timeout HTTP ne signifie jamais que le calcul est arrêté.
Les lots concurrents de référence peuvent utiliser un délai inférieur au plafond,
calculé depuis les latences de référence observées ; ce délai figure dans le cas.
Les limites de taille testées concernent les sources : la forme réellement reçue
par l'encodeur est observée séparément.

Le service de test initialise TripoSR sur le périphérique `meta`, puis affecte
strictement les poids aux précisions du moteur déployé. Cette initialisation évite
une copie vide du modèle et convertit les tenseurs en place ; les poids et les
précisions d'inférence restent ceux du moteur existant. Les tampons de normalisation
non persistants sont reconstruits aux mêmes valeurs FP16. Le moteur de production
n'est pas modifié. Les essais antérieurs d'initialisation restent des diagnostics
de démarrage, distincts des mesures d'inférence.

## Lire les résultats

Ouvrir **http://127.0.0.1:5000**, puis l'expérience **archigen-stress-testing**.
Le run parent contient `campaign.json`, `inventory.json`, le manifeste, les logs
du service et `report.md`. Ses enfants distinguent les mesures, échauffements,
sondes de récupération, disponibilités et échecs d'initialisation.
Les nouvelles campagnes archivent aussi les sources du backend, du harnais et de
TripoSR, leurs empreintes et le diff Git dans `provenance/`. Les premiers diagnostics
conservent les empreintes enregistrées à leur exécution ; aucun code ultérieur ne
doit être présenté comme leur snapshot original.

Chaque tentative conserve son entrée binaire exacte, son empreinte, sa réponse
sans duplication base64, ses sorties binaires, ses contrôles et sa trace d'étapes.
`model_inputs` et `onnx_inputs` décrivent les tenseurs observés. Les durées des
étapes englobantes et imbriquées se chevauchent : ne pas les additionner.
La journalisation MLflow et l'écriture des aperçus sont hors des chronométrages
des étapes concernées. Le temps HTTP comprend l'instrumentation et le transport.

Les échantillons mémoire concernent l'arbre de processus du service et le GPU
global. Les pics PyTorch, lorsqu'ils sont disponibles, concernent la génération
sous verrou. Les requêtes simultanées partagent les mesures globales ; elles
ne reçoivent pas une attribution artificielle de VRAM individuelle.
Les sorties et les métriques de grands masques sont elles-mêmes vérifiées dans
des enfants limités, séquentiellement, afin de ne pas saturer le contrôleur.

Pour les variantes JPEG ou à fond synthétique, `expected-alpha.png` conserve
l'alpha source avant composition. `raw-mask.png` est le masque obtenu avant
recadrage. Leur IoU compare donc les mêmes coordonnées, sans redimensionnement.
Le F1 des frontières utilise une tolérance de deux pixels à la résolution source :
il n'est pas comparable directement entre 512 px et 8 192 px. Cette référence
évalue le détourage synthétique d'une seule chaise, pas une reconstruction 3D.

```powershell
.\.venv-furniture-quality\Scripts\python.exe scripts/report_stress_campaign.py --run-id <ID>
```

Cette commande télécharge le résumé depuis MLflow. Les fichiers locaux sont
conservés sous `outputs/stress/<ID>/`. En cas de panne MLflow, ils restent disponibles,
avec avertissements explicites. Aucun rattrapage automatique n'est effectué.

## Interprétation de la qualité

`success` indique seulement un succès technique. Les sorties sans anomalie
géométrique restent `unreviewed` tant qu'elles n'ont pas été examinées. Un GLB
valide et dense ne prouve pas la fidélité d'un meuble. Les images uniformes qui
produisent un objet sont signalées comme un échec sémantique.

Les métriques avec vérité terrain ne sont pertinentes que si les annotations
correspondantes sont réellement présentes. Les références DVC et anciens rapports
ne constituent ni des poids disponibles ni de nouvelles mesures. Les moteurs sans
poids ou simulés sont exclus des résultats neuronaux. Une source 8 192 px réduite
à 512 px teste surtout le décodage et le prétraitement, pas une inférence 8 192 px.

Les essais non exécutés, les limites de données et les étapes non instrumentées
restent explicitement visibles. Trois répétitions donnent médiane/minimum/maximum,
sans estimation fiable de p95. Une courte endurance ne démontre pas l'absence de fuite.

Les revues visuelles sont des artefacts séparés `review.json`, avec empreinte de
l'image réellement examinée et portée de l'observation. Elles ne réécrivent pas
les mesures originales. Les résultats locaux de cette campagne et ses blocages
sont décrits dans [stress-testing-results.md](stress-testing-results.md).

## Tests du harnais

```powershell
.\.venv-furniture-quality\Scripts\python.exe -m pytest tests/test_stress_campaign.py -q --basetemp=outputs/stress-pytest
```

Les tests couvrent notamment la détermination des entrées, les interfaces façades,
les traces concurrentes, l'attribution d'erreur, l'arrêt des descendants, le suivi
indisponible, l'absence de run automatique dupliqué et la prudence des rapports.

## Campagne avec les poids de production restaurés

Le rapport [stress-testing-neural-results.md](stress-testing-neural-results.md)
documente la nouvelle série réellement exécutée : 436 mesures neuronales/API,
avec les sorties, les erreurs et les limites d'évaluation. Les anciens résultats
de prétraitement restent conservés dans le rapport historique.

```powershell
.\scripts\run_stress_campaign.ps1 -BudgetMinutes 30 -Manifest configs/stress_neural_restored.json
```
