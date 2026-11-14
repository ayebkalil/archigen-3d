# Résultats observés — stress ArchiGen 3D, 14 novembre 2026

## Portée réelle

**La campagne neuronale complète est bloquée.** Les poids de segmentation et de
façades sont absents localement, avec seulement leurs références DVC ; les caches
et le remote local configuré ne sont pas disponibles. TripoSR possède son checkpoint,
mais son démarrage dans le Job Object limité a échoué avant toute reconstruction.
Les résultats exécutés portent donc sur le **décodage, le détourage u2netp CPU et
le prétraitement mobilier**. Aucune limite de qualité géométrique TripoSR, de
segmentation neuronale des plans ou de génération neuronale de façades n'a été mesurée.

Les services natifs d'origine ont été restaurés et leur santé HTTP vérifiée après
les campagnes. Cette vérification ne remplace pas une inférence de récupération
sur le worker GPU. La comparaison CPU/GPU antérieure n'a pas été répétée.

Configuration : RTX 4050 Laptop 6 Go, 32 Go de RAM ; plafond du Job calculé à
60 % de la RAM disponible, limité à 6 Go. Le garde-fou GPU existant de 3 500 Mio
est conservé. Les empreintes, versions, diff Git et fichiers du harnais sont
enregistrés dans les inventaires MLflow. Le code était explicitement non commité.

## Où consulter les essais

Expérience **[archigen-stress-testing](http://127.0.0.1:5000/#/experiments/2)**.

| Série | Run parent | Exécution |
|---|---|---|
| Dernier diagnostic de démarrage neuronal | [96ed4cf8a3794b15a9ab8c5ea1f4e159](http://127.0.0.1:5000/#/experiments/2/runs/96ed4cf8a3794b15a9ab8c5ea1f4e159) | Disponibilité plans/façades et échec d'initialisation GPU |
| Matrice de prétraitement | [eac5575df46f48f983b6e0527b87e02e](http://127.0.0.1:5000/#/experiments/2/runs/eac5575df46f48f983b6e0527b87e02e) | 195 tentatives, dont 174 mesures ; 115,4 s pour le contrôleur |
| Validation des masques | [ece55778185f486bb55b76cc8db4d8a7](http://127.0.0.1:5000/#/experiments/2/runs/ece55778185f486bb55b76cc8db4d8a7) | 44 tentatives, dont 42 mesures ; 60,4 s pour le contrôleur |
| Contrôle final du harnais | [6c756466aee640cb8649321d4e0c7b70](http://127.0.0.1:5000/#/experiments/2/runs/6c756466aee640cb8649321d4e0c7b70) | 32 tentatives, dont 24 mesures ; 23,6 s pour le contrôleur |

Les deux séries principales totalisent **239 tentatives de composants, dont 216 mesures**.
Avec le contrôle final, cela représente **271 tentatives, dont 240 mesures**.
Les 31 autres tentatives sont des échauffements et sondes de récupération. Les diagnostics
d'initialisation et les restaurations ne sont pas inclus dans ces nombres.
Ces durées excluent la restauration de la pile ; le budget de 30 minutes est un
plafond, pas une obligation de prolonger des essais déjà terminés ou bloqués.

La matrice principale comporte 175 succès techniques, 15 rejets attendus, trois
erreurs HTTP 500, un arrêt par garde-fou RAM et une sonde échouée avant redémarrage.
La série des masques comporte 44 succès techniques ; le contrôle final comporte
26 succès, trois rejets attendus et trois erreurs 500 de dimensions extrêmes.
Des succès techniques sont
néanmoins dégradés ou inutilisables selon la revue de leurs entrées préparées.

Les runs contiennent les fichiers binaires exacts, réponses, images préparées,
aperçus alpha, traces, échantillons mémoire et revues visuelles `review.json`.
Les masques de référence, masques obtenus et cartes d'erreur sont aussi présents
dans la seconde série. Le contrôleur a vérifié les statuts et artefacts via l'API
MLflow ; aucun run des deux séries n'était encore RUNNING à cette vérification.

## Segmentation des plans : bloquée

Les checkpoints attendus du MultiTask UNet++ et des variantes ONNX sont absents.
Les plans externes sont disponibles, mais les images/annotations CubiCasa référencées
par DVC ne sont pas matérialisées. Aucun IoU ou F1 de segmentation n'est produit.
Les plans synthétiques et leurs ouvertures figurent dans le harnais pour une
campagne ultérieure ; leur détection et la vectorisation n'ont pas été mesurées ici.

Le code effectue un letterboxing à 512 × 512 et peut produire un masque nul en
l'absence de modèle. L'API active annonçait `model_loaded=true`, tandis que les
métadonnées indiquaient un modèle principal et une empreinte nuls. **L'état healthy
ne constitue donc pas une preuve de chargement neuronal.** Ce constat provient de
l'inventaire, pas d'un benchmark de qualité de segmentation.

## Façades : bloquées

Les poids Pix2Pix/ONNX attendus sont absents, ainsi que les poids des variantes
expérimentales référencées par DVC. Aucun résultat neuronal de façade n'est mesuré.
Le mode simulé disponible dans le code est exclu des résultats neuronaux.

L'API publique utilise des murs, étages et ouvertures, puis synthétise une image
sémantique ; elle n'accepte pas une photographie de façade comme entrée. Les tests
de géométrie API et les tests directs du générateur sémantique sont séparés dans
le harnais. La résolution standard prévue est 256 × 256 ; les variantes
expérimentales ne doivent pas être confondues avec ce moteur.

## Mobilier : reconstruction bloquée, prétraitement mesuré

### Démarrage TripoSR sous isolation

Les premières tentatives plafonnées ont quitté avec une exception native Windows
`0xc0000005`. Les traces instrumentées situent ces sorties au chargement/conversion
des poids. Une adaptation d'initialisation sur `meta` a aussi révélé une erreur du
**harnais** liée aux imports torchvision ; elle a été corrigée et conservée comme
diagnostic historique, sans être attribuée au modèle.

Le dernier diagnostic atteint le transfert sur CUDA puis lève
`torch.OutOfMemoryError`. Le message mentionne encore de la VRAM libre : il ne
permet pas d'isoler une insuffisance de VRAM d'un effet de la limite mémoire du
Job et du chargement CUDA. Les plafonds RAM des essais variaient avec la mémoire
disponible, autour de 3,4 à 4,1 Go.

**Ces échecs d'amorçage ne démontrent ni une limite intrinsèque du réseau, ni une
limite de taille d'image, ni l'impossibilité de fonctionner sur une RTX 4050.**
Aucune inférence TripoSR n'a eu lieu dans cette campagne isolée. La pile native
ordinaire peut redémarrer et annonce CUDA ; son chargement n'est pas soumis au
même plafond Job. Les anciens résultats GPU ne sont pas réimportés comme de
nouvelles mesures.

### Tableau des conditions observées

Les résultats ci-dessous concernent exclusivement la préparation d'une seule
chaise de référence et ses variantes synthétiques. Ils ne généralisent pas à toutes
les photos de mobilier.

| Condition | Observation technique | Qualité/limite observée |
|---|---|---|
| PNG/JPEG de 512 à 4 096 px | Trois succès par variante | Fonctionnement du prétraitement sur cette matrice ; pas de fidélité 3D évaluée |
| PNG 8 192 × 8 192, 5 948 885 octets | Deux succès, un arrêt RAM | Niveau non fiable avec la RAM disponible lors des essais |
| JPEG 8 192 × 8 192, 1 587 394 octets | Trois succès | Moins d'octets ne supprime pas le coût des pixels décodés |
| JPEG qualité 5 | Trois succès | Artefacts en blocs conservés et erreurs de masque visibles |
| Fond synthétique bruité | Trois succès | Bruit résiduel dans certains interstices de la chaise |
| Deux objets, objet incomplet, masquage | Entrées acceptées | Plusieurs objets ou éléments manquants restent présents dans l'entrée préparée |
| Image blanche uniforme | Trois succès | Aperçu sans objet exploitable : échec silencieux du prétraitement |
| Image 16 384 × 1 | Trois succès | Entrée préparée visuellement grise et vide : inutilisable |
| Fichier vide, transparent, tronqué, texte, dimensions nulles | Trois rejets 422 par variante | Rejets attendus |
| En-tête 1 000 000 × 1 000 000 | Trois HTTP 500 | `DecompressionBombError` au décodage ; rejet API mal classé |
| Concurrence 1 / 2 / 4 | 3 / 6 / 12 succès | Petits bursts de prétraitement seulement ; aucune capacité GPU déduite |
| Endurance de 30 passages | 30 succès | Essai court, insuffisant pour exclure une fuite à long terme |

L'arrêt RAM a interrompu la connexion ; une sonde avant redémarrage a échoué,
puis une sonde après redémarrage a réussi. Dix-huit autres sondes ont confirmé
une récupération sans redémarrage après des rejets ou erreurs de décodage.
Le pic RSS échantillonné de l'arbre du service atteint **2 605,6 Mio**. Le watchdog
est plus fréquent que l'échantillonnage mémoire : le creux transitoire ayant
déclenché sa décision n'apparaît pas nécessairement dans ces échantillons.

Pour 4 096 px, les maxima HTTP observés sont 0,579 s en PNG et 1,007 s en JPEG.
Pour les succès à 8 192 px, ils sont 2,063 s en PNG et 3,936 s en JPEG.
Le RSS échantillonné en fin d'endurance passe d'environ 724,92 à 724,88 Mio ; cela
ne prouve pas l'absence de fuite sur une utilisation prolongée.

### Taille réellement reçue et masque

La sortie préparée est **512 × 512**, y compris pour les sources 8 192 px.
Le tenseur u2netp observé sur les entrées opaques est **[1, 3, 320, 320]**.
La forme d'entrée de l'encodeur TripoSR n'a pas été mesurée puisque son démarrage
était bloqué ; sa configuration prévoit 512 px. Aucun pic VRAM propre à une
reconstruction neuronale n'est disponible ; les échantillons GPU globaux ne lui
sont pas attribués.

La seconde série échauffe explicitement le détourage avant trois mesures par
variante. La première série n'échauffait que la référence avec alpha : son premier
JPEG inclut l'initialisation du détourage et ne doit pas servir de mesure chaude.

| Variante synthétique | IoU du premier plan | F1 frontière, tolérance 2 px source |
|---|---:|---:|
| JPEG 512 px, qualité 90 | 0,9744 | 0,9285 |
| JPEG 8 192 px, qualité 90 | 0,9740 | 0,3239 |
| JPEG qualité 5, 512 px | 0,9377 | 0,6957 |
| Fond synthétique bruité | 0,9687 | 0,9018 |

Ces métriques sont identiques sur les trois répétitions de chaque variante.
La référence est l'alpha conservé avant composition/compression, dans les mêmes
coordonnées que le masque obtenu. Il ne s'agit pas d'une vérité terrain 3D.
Le F1 à deux pixels est beaucoup plus strict relativement à l'objet en 8 192 px :
sa baisse ne démontre pas à elle seule une dégradation de qualité entre résolutions.

Les revues visuelles portent sur les artefacts explicitement identifiés par leur
empreinte, pas sur tous les résultats. Les sorties non examinées restent marquées
comme telles. Les dégradations injectées dans l'image et conservées dans l'aperçu
ne sont pas attribuées à une reconstruction qui n'a pas été exécutée.

## Recommandations et essais restants

- Pour le prétraitement mobilier, adopter provisoirement un côté maximal de
  **4 096 px**, **16 777 216 pixels décodés** et **3 Mio par fichier**. Les trois
  répétitions réussies jusqu'à 4 096 px motivent cette protection prudente, sans
  garantir tous les contenus à ces tailles. La limite en octets complète celle
  en pixels : elle ne la remplace pas.
- Prévoir un timeout de prétraitement de **10 s**, provisoire et supérieur au
  maximum observé de 1,007 s à 4 096 px. Les délais neuronaux de 120/180 s restent
  des plafonds d'essai non validés comme SLA de production.
- Conserver **un calcul TripoSR à la fois** et une file bornée. Les bursts CPU
  observés ne permettent pas de choisir un débit ou une concurrence GPU supérieurs.
- Classer les images corrompues et les protections anti-bombe de décompression
  en erreurs client explicites ; contrôler octets, dimensions et pixels avant le
  travail coûteux. Ne pas assimiler une réponse 500 à un crash du modèle.
- Ajouter une validation du premier plan et une vérification de l'aperçu pour
  les images uniformes, ratios extrêmes, objets multiples ou incomplets. Ne pas
  déclarer automatiquement une reconstruction correcte sur la seule validité du GLB.
- Pour reprendre les essais neuronaux, matérialiser les poids et annotations via
  un remote DVC accessible, vérifier leurs empreintes, puis libérer assez de RAM
  pour le démarrage isolé. Environ 10 Gio disponibles permettent au plafond de
  60 % d'atteindre son maximum de 6 Gio ; ce seuil est une règle du harnais, pas
  une exigence mémoire mesurée du réseau.
- Restent à mesurer : plans complexes, qualité de segmentation/vectorisation,
  façades et styles inconnus, GLB/textures TripoSR sous dégradations, limites
  neuronales de concurrence, récupération après OOM en inférence et endurance longue.

Les scripts et commandes reproductibles sont dans [stress-testing.md](stress-testing.md).
Les 13 tests du harnais passent ; les sept tests existants de suivi mobilier passent
dans l'environnement principal avec MLflow complet. Le client skinny de
l'environnement GPU ne prend pas en charge les fixtures SQLite de ces sept tests.
