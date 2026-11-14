# Stress testing avec les poids restaurés — 14 novembre 2026

Cette série utilise les vrais poids de production. Elle complète les résultats
historiques de `stress-testing-results.md`, sans les remplacer et sans refaire la
comparaison CPU/GPU. Les API publiques et le frontend n'ont pas été modifiés.

## Résultats et traçabilité

**436 mesures** : 388 succès techniques, 18 rejets attendus et 30 erreurs HTTP.
Les **9 échauffements et 48 sondes de récupération** sont exclus de ces statistiques.
Chaque sonde de récupération a réussi sans redémarrage du service testé.
Deux échecs initiaux du harnais sont comptés séparément des 493 tentatives HTTP.

Expérience : [archigen-stress-testing](http://127.0.0.1:5000/#/experiments/2).
Run récapitulatif : [c8e86025ea554905b4ee9d51c0effc18](http://127.0.0.1:5000/#/experiments/2/runs/c8e86025ea554905b4ee9d51c0effc18).
Le récapitulatif a été publié après 1697 secondes, soit 28 min 17 s depuis le
premier run, dans le budget de 30 minutes. Il regroupe les phases sans créer de
second run pour une génération.
Les résultats sont répartis en quatre phases pour corriger le harnais et couvrir
les sondes encore manquantes dans le budget global de 30 minutes :

| Phase | Run parent | Tentatives HTTP, échauffements et sondes inclus |
|---|---|---:|
| Mobilier, dégradations et concurrence ; diagnostic de routage | [e9e3454da59a4491be7731b349daa7c9](http://127.0.0.1:5000/#/experiments/2/runs/e9e3454da59a4491be7731b349daa7c9) | 66 |
| Plans et façades, harnais corrigé | [cafe26342782492d87da0b027b62daac](http://127.0.0.1:5000/#/experiments/2/runs/cafe26342782492d87da0b027b62daac) | 338 |
| Façades extrêmes et invalides mobilier | [7206d0fdb49e43489dcbca564a5c6cc3](http://127.0.0.1:5000/#/experiments/2/runs/7206d0fdb49e43489dcbca564a5c6cc3) | 78 |
| Mobilier 4096 et 8192 px | [5fddb1077ebf496997cfd8dc3b6dec23](http://127.0.0.1:5000/#/experiments/2/runs/5fddb1077ebf496997cfd8dc3b6dec23) | 11 |

Les fichiers locaux `outputs/stress/restored-series/cases.csv` et `summary.json`
contiennent les minimums, médianes et maximums par phase/modèle/cas, les effectifs,
la provenance et les métriques disponibles. `not-executed.json` recense les cas
sans trois mesures. Chaque phase contient `tracking-verification.json` : statuts
terminaux et présence des artefacts vérifiés via l'API MLflow, sans avertissement
de suivi. Les vérifications n'assimilent pas un fichier vide à un artefact absent.

Matériel : RTX 4050 Laptop 6 Go. Job Objects plafonnés à 6144 Mio pendant ces phases,
arrêt si RAM disponible < 3072 Mio, garde GPU de 3500 Mio conservé. Aucun crash,
timeout, dépassement mémoire ou arrêt par garde-fou n'a été observé dans les essais
neuronaux exécutés. Ce constat ne donne pas la limite absolue de saturation.

## Tailles réellement calculées et durées

Les plans et Pix2Pix tournent sur **ONNX Runtime CPU**, TripoSR sur **CUDA**.
Les poids, leurs SHA-256, les fournisseurs ONNX, versions, révision Git et diff du
code sont conservés dans les inventaires et les paramètres de chaque phase.
Le checkpoint TripoSR est chargé strictement ; le harnais utilise son initialisation
en mémoire bornée, explicitement enregistrée, sans remplacer la reconstruction.

| Chaîne | Tenseur observé | HTTP témoin, médiane [min ; max], 3 mesures | HTTP source PNG 8192, 3 mesures |
|---|---|---|---|
| Plans | `[1,3,512,512]`, champion et parfois secours | 0,729 s [0,669 ; 0,744] | 1,019 s [1,010 ; 1,023] |
| Façades | `[1,3,256,256]` | 0,150 s [0,137 ; 0,171] | 0,463 s [0,461 ; 0,466], entrée sémantique directe |
| TripoSR | tokenizer `[1,1,3,512,512]` | 4,645 s [4,625 ; 4,666] | 6,668 s [6,601 ; 6,777] |

Ces lignes décrivent des moteurs et des entrées différents, sans rapport de vitesse
entre eux. Les essais 8192 mobilier appartiennent à une autre phase que le témoin
principal ; son propre témoin médian vaut 4,770 s. Le prétraitement mobilier passe
de 0,043 s à 1,842 s ; l'inférence observée reste voisine de 0,50–0,55 s dans cette
phase. L'extraction 384³ et l'atlas 2048 px restent identiques. Le détourage ONNX,
lorsqu'il est utilisé sur l'image opaque, reçoit `[1,3,320,320]`.

Une source 8192 ne devient donc pas une inférence 8192. Les plus gros fichiers
effectivement testés valent 1 260 911 octets pour les plans, 292 910 pour les
façades et 5 948 885 pour le mobilier. Aucun fichier d'entrée de plusieurs centaines
de Mo n'a été testé. Les temps d'inférence de l'API plan agrègent les appels ONNX
de segmentation/secours et, lorsqu'elle est produite, de façade ; ils ne sont pas
des temps isolés du seul champion. Les écritures MLflow suivent les mesures HTTP.
L'instrumentation et la conservation des images peuvent ajouter du coût au service.

## Plans : réussite technique et pièces manquantes

159 mesures : **141 succès**, 3 rejets attendus, 15 erreurs. La résolution PNG a
été testée trois fois à chaque niveau 512, 1024, 2048, 4096 et 8192. Les croquis,
annotations, plans colorés, bruit, flou et plusieurs autres styles ont été envoyés.

| Plan synthétique | Pièces attendues | Pièces retournées, chacune sur 3 essais | IoU moyen de correspondance des polygones |
|---|---:|---:|---:|
| Grille 4 | 4 | 0 | 0,000 |
| Grille 16 | 16 | 8 | 0,451 |
| Grille 64 | 64 | 54 | 0,680 |
| Grille 128 | 128 | 126 | 0,719 |

La dégradation n'est **pas monotone avec le nombre de pièces** : on ne peut pas
annoncer une capacité fiable de 128 pièces. La grille 4 est inutilisable pour
extraire les pièces malgré HTTP 200. Les autres perdent des pièces. Ces résultats
concernent la chaîne segmentation → vectorisation ; l'origine exacte des pièces
manquantes n'est pas isolée entre ces deux étapes. Les masques neuronaux et les
polygones finaux sont conservés séparément. Les IoU ci-dessus ne mesurent ni les
classes sémantiques ni la précision sur de vrais plans annotés.

Un témoin simple conserve quatre régions visuellement cohérentes ; le croquis
examiné en conserve trois. Ce sont des exemples examinés, pas une validation de
tous les plans. Les images blanches/transparents sont acceptées avec zéro pièce
et zéro mur : elles nécessitent une validation de contenu côté produit.

Les fichiers tronqués, faux formats, dimensions nulles et en-têtes géants donnent
HTTP 500 au décodage ou avant l'inférence. Le cas `16384 × 1` échoue au redimensionnement :
la hauteur arrondie devient zéro. Cela relève du prétraitement, pas du réseau.
Pour le fichier tronqué, la trace marque parfois seulement `pipeline_total` ; le
message Pillow indique un décodage probable, sans attribution plus fine certaine.

## Pix2Pix EMA 256 : détails comprimés et erreurs de décodage

183 mesures : **168 succès et 15 erreurs HTTP 500** sur les images sémantiques
invalides. Les façades API sont testées avec murs et ouvertures, et non avec des
photographies. Les sources sémantiques jusqu'à 8192 px et les dégradations sévères
ont été exécutées. Les sorties sont en 256 × 256.

Le témoin examiné garde une porte et deux fenêtres reconnaissables dans sa région
UV visible. Les géométries denses et 64 niveaux restent acceptées, avec forte perte
de détails. Certaines entrées de 64 ouvertures se chevauchent déjà : fenêtres de
0,60 m espacées de 0,50 m. Leur dégradation ne doit pas être attribuée uniquement
au modèle. Les images uniformes et le ratio extrême produisent des textures
arbitraires ou étirées, sans erreur HTTP.

L'évaluation visuelle porte sur la région UV effectivement affichée ; le remplissage
extérieur du carré de texture n'est pas traité comme un défaut du mur. Le modèle
annonce `style_conditioned=false` : un style inconnu accepté ne prouve pas qu'il a
été appris. Les essais directs du générateur sont séparés de la finition de l'API.
Pour ces essais directs, `sketch.png` est un alias de la sortie ; l'entrée exacte
à consulter est `input.bin`, comme indiqué dans `response.json`.

## TripoSR high : GLB valides, échecs sémantiques

94 mesures : **79 succès et 15 rejets attendus**, aucun échec CUDA observé.
Les GLB réussis sont lisibles et non vides, sans face dégénérée détectée. Leur
hauteur vaut environ 0,85 m ; cela vérifie la mise à l'échelle demandée, pas la
fidélité à l'objet réel. Ils contiennent entre 130 528 et 1 130 036 faces et pèsent
entre 5 079 652 et 35 309 360 octets. Plus de faces ne signifie pas plus de précision.

| Condition examinée | Résultat technique | Qualité selon les contrôles et vues disponibles |
|---|---|---|
| Chaise complète, fond simple | GLB valide | Silhouette de chaise exploitable dans les vues examinées |
| Fond aléatoire synthétique | GLB valide | Chaise encore reconnaissable dans l'exemple examiné |
| Deux chaises | GLB valide | Deux objets dans un seul meuble : dégradé pour l'usage mono-objet |
| Recadrage ne gardant que le haut | GLB valide | Volume en bloc, chaise complète non exploitable |
| Image blanche | GLB valide | Volume arbitraire inutilisable ; un exemple atteint 151 composantes |
| Image transparente, vide, tronquée ou faux format | HTTP 422 | Rejet et sonde témoin suivante réussie |

Les vues de maillage sont des projections de triangles échantillonnés, sans
texture ; les points/trous d'affichage ne prouvent pas des trous du GLB. Les atlas
sont conservés séparément. `preview.png` représente l'entrée préparée, pas un rendu
3D. Aucun scan de référence ni autre catégorie visuelle réelle n'a été testé :
`unusual_category` change seulement le libellé de la chaise.

Concurrence GPU : 3 requêtes à concurrence 1, 6 à concurrence 2 et 12 à concurrence
4, toutes réussies. Les plages de latence sont respectivement **4,50–4,65 s**,
**4,67–9,17 s**, **4,62–18,28 s**. La file et le calcul sont distincts dans les traces.
Le verrou séquentiel est conservé ; quatre clients ne signifient pas quatre calculs
neuronaux simultanés. Plans et façades ont également trois lots aux concurrences
1, 2 et 4, sans échec sur leurs témoins.

Pics observés, en Mio :

| Chaîne | RSS échantillonnée du service et descendants | VRAM globale échantillonnée | Pic alloué PyTorch | Pic réservé PyTorch |
|---|---:|---:|---:|---:|
| Plans | 2028,4 | 620 | — | — |
| Façades | 982,6 | 617 | — | — |
| Mobilier | 3862,4 | 1813 | 995,9 | 1084 |

La VRAM globale comprend les autres utilisateurs du GPU ; ce n'est pas une
allocation attribuable aux modèles CPU. Les pics RSS sont échantillonnés, pas
une mesure exhaustive du pic de mémoire engagée Windows. Les répétitions courtes
ne démontrent pas l'absence de fuite mémoire.

## Limites restantes et recommandations

- Aucun seuil absolu d'OOM, de saturation GPU ou de concurrence au-delà de 4 n'est
  établi. Le Pix2Pix expérimental 512 n'est pas disponible et reste exclu.
- Cinq cas plans et vingt cas mobilier n'ont pas trois mesures ; leur liste exacte
  est dans `not-executed.json`. Elle comprend les dégradations les plus sévères du
  mobilier, plusieurs sources JPEG et ses dimensions invalides extrêmes. Toutes
  les familles ont des exemples, mais toute la matrice n'a pas été exécutée.
- Aucun jeu réel annoté n'a été matérialisé ; pas de mIoU/F1 réel ni de fidélité 3D
  mesurée. Vingt-deux exemples ont une revue visuelle explicite ; les autres sorties
  restent `unreviewed` ou évaluées seulement par les contrôles structurels.
- Séparer taille binaire, pixels décodés et ratio d'aspect. Une politique initiale
  prudente peut limiter le flux courant à 4096 px et 16 mégapixels ; garder le mode
  8192 sous les protections mémoire testées. Ce sont des politiques proposées,
  pas une garantie pour toute image sous ces dimensions. Les gros fichiers à forte
  entropie restent à tester ; ne pas déduire une limite en Mo des seuls pixels.
- Rejeter les fichiers corrompus par HTTP 400/415/422, les images sans contenu et
  les géométries d'ouvertures incohérentes. Borner les dimensions redimensionnées
  à au moins un pixel ou rejeter les ratios extrêmes avant calcul.
- Conserver un calcul GPU actif, une file bornée et au maximum quatre clients dans
  le périmètre testé ; privilégier un ou deux clients pour contenir l'attente.
  Aucun gain de débit matériel n'est inféré de cette concurrence.
- Un watchdog initial de 30 s pour les cas chauds testés constitue un réglage
  prudent, pas un SLA général. Séparer démarrage, attente en file et calcul ; un
  timeout HTTP client ne doit jamais être assimilé à un arrêt du serveur.
- Signaler « résultat à vérifier » pour zéro pièce, une grille incomplète ou un
  objet mal reconstruit. Le nombre de faces et HTTP 200 ne sont pas des scores de qualité.

## Harnais, restauration et reproduction

Le premier lancement a révélé une incompatibilité de routage FastAPI avec le
montage statique du frontend. Les poids étaient chargés avant l'échec. Les deux
échecs sont conservés avec `diagnosed_stage=test_harness_api_routing`, et non comme
des pannes de modèles. Le harnais importe maintenant seulement les routes API.
Il conserve l'environnement ONNX lors de la restauration, réserve le temps des
autres modèles et utilise le bon plan source pour les transformations. Les débits
HTTP et ceux incluant la validation sont distingués dans les phases corrigées ;
la première phase utilisait un débit de lot incluant la validation locale.

Les services ont été restaurés après chaque phase. La vérification finale via les
API publiques a produit 18 pièces et 40 murs sur `sample_floorplan.png`, une texture
256 × 256, et un worker sain annonçant CUDA. Frontend et MLflow répondent HTTP 200.
Les 33 tests ciblés du harnais, de distribution des modèles et du suivi mobilier passent.

```powershell
.\scripts\start_mlflow.ps1
.\scripts\run_stress_campaign.ps1 -BudgetMinutes 30 -Manifest configs/stress_neural_restored.json
.\.venv-furniture-quality\Scripts\python.exe scripts/report_stress_campaign.py --run-id <ID>
.\.venv-furniture-quality\Scripts\python.exe scripts/verify_stress_tracking.py --run-id <ID>
```

Pour reproduire les phases ciblées, utiliser les manifestes `stress_architecture_restored.json`
(15 minutes), `stress_final_boundaries.json` (9 minutes) et
`stress_furniture_large_images.json` (4 minutes). Ces plafonds ne s'additionnent pas
aux durées réellement consommées : chaque phase s'arrête quand ses essais admissibles
sont terminés. Ne pas les lancer simultanément. Les sources, graines, commandes et
versions exactes de chaque phase figurent dans ses artefacts de provenance.

```powershell
.\.venv\Scripts\python.exe scripts/summarize_stress_series.py --run-ids <ID1> <ID2> --output outputs/stress/my-series
.\.venv\Scripts\python.exe scripts/render_stress_mesh.py --directory outputs/stress/<ID>/<CHILD_ID>
.\.venv\Scripts\python.exe scripts/render_stress_floorplan.py --directory outputs/stress/<ID>/<CHILD_ID>
```

Les rendus utilisent l'environnement principal avec matplotlib/Pillow/trimesh.
Les anciennes expériences d'entraînement et de comparaison n'ont pas été utilisées
pour enregistrer cette campagne.
