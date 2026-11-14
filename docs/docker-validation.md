# Validation locale de la pile Docker

La pile complète a été construite et exécutée sur Docker Desktop/WSL2 avec la
RTX 4050. Les quatre services sont `healthy`. Le worker est uniquement exposé
sur le réseau Compose ; les ports 3000, 8000 et 5000 écoutent sur 127.0.0.1.

| Contrôle réellement exécuté | Résultat |
|---|---|
| Calcul CUDA dans le conteneur | RTX 4050 détectée, multiplication de matrices réussie |
| Trois ONNX de production | Empreintes vérifiées ; segmentation principale, secours et façade chargés |
| Segmentation du plan témoin via Nginx | 18 pièces ; 1,38 s au premier contrôle, 1,44 s après redémarrage |
| Façade via `/facade/generate` | Génération réelle 256 px, texture PNG produite |
| Chaise via l'API publique | TripoSR CUDA, chargement strict, `high`, extraction 384³, texture 2048 px |
| Maillage GLB | 106 066 sommets, 189 064 faces, hauteur 0,85 m, GLB de 6 710 032 octets lisible |
| Suivi MLflow de l'appel API | Exactement un run `FINISHED`, artefact téléchargé avec empreinte identique |
| Génération depuis le navigateur | Aperçu visible ; navigation mobilier/façades et plan/studio conservée |
| Rafraîchissement et réouverture après redémarrage Docker | Même chaise restaurée depuis IndexedDB, téléchargement vérifié, aucune nouvelle inférence |
| Bibliothèque importée | Deux meubles et leur index conservés ; GLB accessibles via le proxy |
| Historique MLflow | Les 815 anciens identifiants de runs sont tous conservés ; 2 nouveaux runs mobilier |
| Poids ONNX absents | API refusée explicitement dans un conteneur isolé |
| GPU absent | Worker refusé avant chargement des poids dans un conteneur isolé |
| Téléchargement incorrect | Test de refus SHA-256, préservation du fichier existant, puis réparation valide |
| Propriétaire de port non natif | Refus sans arrêt du processus |

Les 797 runs `archigen-stress-testing` sont inchangés. L'expérience
`furniture-reconstruction` contient désormais 20 runs au lieu de 18.
Les emplacements d'artefacts des expériences sont inchangés.

Le premier contrôle API a mesuré **6,82 s de reconstruction hors suivi** et
**11,38 s HTTP**, avec un pic GPU global échantillonné de **1 718 Mio** et
**971,54 Mio alloués par PyTorch**. La génération depuis le navigateur a pris
15,63 s selon le chronométrage du moteur. Ces deux essais ne constituent pas
un benchmark et ne permettent aucune conclusion statistique de performance.

Le run API est consultable dans
[MLflow](http://localhost:5000/#/experiments/1/runs/844b74cd123c476c93c6767aa8268313).
Les mesures, sorties JSON, GLB et captures sont dans `outputs/docker/`, notamment
`validation/`, `validation-restart/`, `browser/` et les états d'historique.
Les sauvegardes SQLite sont dans `mlartifacts/.backups/`.

Les tests ciblés représentent **35 tests Python** et **35 tests frontend**
réussis. Ils couvrent notamment la restauration des poids, le suivi mobilier,
la géométrie, la sauvegarde SQLite et la protection de la bibliothèque. Le retour
automatique après une erreur partielle d'arrêt natif est testé par simulation ;
une panne réelle après arrêt des quatre services n'a pas été volontairement
provoquée. La bascule native vers Docker a, elle, réellement réussi.

La validation navigateur utilise un profil isolé. La capture confirme la présence
de la chaise, sans erreur JavaScript collectée. Elle ne mesure pas sa fidélité
géométrique. Le test après redémarrage ne relance pas TripoSR.

La pile native est arrêtée ; la pile Docker reste active. Les commandes
reproductibles et le retour natif sont décrits dans [le guide Docker](docker.md).
Une installation sur un second PC entièrement vierge et le fonctionnement sans
connexion réseau de toute l'interface n'ont pas été testés.
