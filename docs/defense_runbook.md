# ArchiGen 3D — Guide Pratique de Soutenance (100% Technique, 0 Slides)

**Module :** Deep Learning Avancé & MLOps  
**Enseignant :** Prof. Haythem Ghazouani  
**Format :** 15 minutes chrono — 5 min Démo Live + 5 min MLOps & Reproductibilité + 5 min Code Review & Q&A Individuel.

---

## 🕒 Chronométrage & Rôles des Intervenants

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DÉROULÉ STRICT DE LA SOUTENANCE (15 MIN)                        │
├─────────────────────┬───────────────────────────┬──────────────────────────────────────┤
│ Temps               │ Thématique                │ Intervenants                         │
├─────────────────────┼───────────────────────────┼──────────────────────────────────────┤
│ 00:00 - 05:00       │ 1. Démonstration Live     │ Amine Ben Fadhel                     │
│ 05:00 - 10:00       │ 2. Revue MLOps & Modèles  │ Khawla Hawari & Khalil Ayeb          │
│ 10:00 - 15:00       │ 3. Code Review & Q&A      │ Khalil Ayeb, Amine Ben Fadhel, Khawla│
└─────────────────────┴───────────────────────────┴──────────────────────────────────────┘
```

---

## Étape 1 (00:00 - 05:00) : Démonstration Applicative en Direct

> **Orateur : Amine Ben Fadhel**

### Actions en direct sur l'écran :
1. **Lancement en 1 clic :**
   ```bash
   docker-compose up --build
   ```
   *Montrer les deux conteneurs qui démarrent : `archigen_backend` (FastAPI sur 8000) et `archigen_frontend` (Nginx sur 3000).*
2. **Ouverture du navigateur sur `http://localhost:3000` :**
   * Montrer le plan par défaut qui s'analyse automatiquement en moins de 400 ms.
   * Pointer du doigt l'en-tête de latence dynamique `X-Inference-Latency-ms` affiché dans la barre de navigation.
3. **Manipulation de la Scène 3D (Three.js WebGL) :**
   * Faire tourner l'orbite 3D avec la souris : montrer l'extrusion verticale des cloisons, les percements nets des portes et fenêtres.
   * Basculer entre les textures procédurales : *Modern Minimalist*, *Mediterranean Stone*, *Traditional Brick*.
4. **Démonstration de l'Outil de Calibration à 2 Points :**
   * Activer le mode calibration métrique : cliquer sur les deux extrémités d'une porte (0.90 m), valider. Montrer le recalcul instantané de l'échelle et des surfaces ($m^2$).
5. **Alerte du Filtre de Plausibilité Architecturale :**
   * Montrer comment l'application alerte l'utilisateur si une pièce présente une surface anormale (ex. chambre $< 5\text{ m}^2$).

---

## Étape 2 (05:00 - 10:00) : Revue MLOps, Données & Traçabilité

> **Orateurs : Khawla Hawari, Khalil Ayeb, Amine Ben Fadhel**

### Actions en direct sur l'écran :
1. **Démonstration DVC (Khawla Hawari) :**
   ```bash
   dvc status
   ```
   * Montrer que les gros fichiers (`.pth`, `.onnx`, `data/raw/`, `data/CubiCasa5k/`) sont strictement exclus de Git via `.gitignore` et indexés dans les pointeurs `.dvc`.
   * Expliquer que n'importe quel clone peut répliquer l'environnement avec `dvc pull`.
2. **Démonstration MLflow (Khawla Hawari) :**
   ```bash
   mlflow ui --port 5000
   ```
   * Ouvrir `http://localhost:5000` devant le jury.
   * Montrer l'expérience `cubicasa_floorplan_segmentation` :
     * La courbe de convergence de la perte d'entraînement.
     * La progression du Room Pixel Accuracy atteignant $87.50\%$ (et $56.40\%$ de mIoU sur les 10 classes réelles) sur le split de test complet non vu ($N=400$).
     * Les artefacts visuels : ouvrir l'un des fichiers `val_grid_epoch_025.png` pour prouver le logging automatique des masques de validation.
3. **Revue de la Synthèse de Façades cGAN (Khalil Ayeb) :**
   * Expliquer l'ablation expérimentale : Baseline vs Enhanced vs Perceptual.
   * Expliquer pourquoi les métriques pixel (SSIM/L1) récompensent le flou moyen tandis que la perte perceptuelle LPIPS (SqueezeNet) et Spectral Normalization suppriment visiblement les artefacts d'échiquier et restituent les textures architecturales.
   * Montrer l'export ONNX et le test de parité numérique stricte ($\Delta_{\max} = 6.31 \times 10^{-6}$).

---

## Étape 3 (10:00 - 15:00) : Revue de Code & Questions Techniques

> **Orateurs : Trinôme au complet (Khalil Ayeb, Amine Ben Fadhel, Khawla Hawari)**

### 1. Questions potentielles sur la Génération de Façades :
* **Q : "Pourquoi votre FID de façade est autour de 350-370 ?"**  
  * **R :** *"Le jeu de test CMP ne compte que 106 images. Dans un espace Inception à 2048 dimensions, la matrice de covariance empirique est sous-dimensionnée (rang déficient), ce qui fausse le calcul classique de Fréchet. C'est pourquoi nous avons calculé le KID (Kernel Inception Distance), qui est un estimateur polynomial cubique sans biais particulièrement adapté aux petits échantillons, classant notre modèle perceptuel en tête à $67.65 \times 10^{-3}$ (contre $150.36 \times 10^{-3}$ pour la baseline)."*
* **Q : "Pourquoi le modèle Enhanced a un meilleur SSIM que le Perceptual ?"**  
  * **R :** *"La perte $\mathcal{L}_1$ pénalise les déviations pixel par pixel en forçant le réseau à prédire la moyenne conditionnelle de la distribution, ce qui produit une image lissée statistiquement récompensée par le SSIM (0.3345 vs 0.2792) et le $\mathcal{L}_1$ (0.3379 vs 0.3526). La perte perceptuelle LPIPS force au contraire la correspondance des activations haute-fréquence, divisant par deux la distance perceptuelle (LPIPS: 0.3169 vs 0.4940 pour Enhanced et 0.6327 pour Baseline)."*

### 2. Questions potentielles sur la Segmentation de Plans :
* **Q : "Comment avez-vous résolu le problème d'IoU initial et atteint 74.16% sur les murs et 87.47% en exactitude des pièces ?"**  
  * **R :** *"Une inspection visuelle minutieuse par superposition a révélé que les coordonnées SVG initiales n'étaient pas recalées avec le facteur d'échelle letterbox du canevas $512 \times 512$. Dès la correction du rasterizer dans `src/floorplan/dataset.py` et la génération du cache aligné, l'apprentissage a convergé immédiatement : l'IoU des murs a bondi de 22.44% à 74.32%, les portes de 10.71% à 62.38%, et les fenêtres à 73.28%. Puis, notre phase de fine-tuning régularisé avec augmentations CAD-spécifiques a porté l'exactitude des pièces à 87.47% (IC 95%: [85.8, 89.1], p < 0.001 au test bootstrap) et le mIoU des 10 classes réelles à 56.47% (IC 95%: [54.0, 58.8], p < 0.001), avec 74.16% sur les murs. En toute honnêteté scientifique, ce gain résulte conjointement du budget d'entraînement étendu (12 époques additionnelles) et des augmentations. La comparaison avec les 82.7% du papier officiel CubiCasa5K reste indicative en raison des différences de protocoles d'annotation."*
* **Q : "Comment vos tests unitaires sont-ils automatisés ?"**  
  * **R :** *"Notre fichier `.github/workflows/ci-tests.yml` déclenche à chaque push ou PR l'exécution de `pytest -v`. Les 22 tests automatisés valident la cohérence du dataset, le flux de gradients du générateur/discriminateur, la vectorisation géométrique hybride, l'absorption de fragments, la classification NO_ROOMS_DETECTED, les runtimes CUDA/CPU et les schémas Pydantic des endpoints FastAPI."*

### 3. Questions potentielles sur la Géométrie 3D & l'Ingénierie Système :
* **Q : "Comment gérez-vous la séparation entre murs intérieurs et extérieurs dans l'extrusion 3D ?"**  
  * **R :** *"Le vectoriseur utilise la dominance de classe entre les canaux `Wall_External` et `Wall_Internal` prédits par notre UNet++ multi-tâches, avec une vérification de l'enveloppe extérieure comme repli de secours uniquement si le masque structurel est absent. Sur notre modèle de test, cela extrait rigoureusement 7 murs extérieurs et 7 cloisons intérieures avec 91.07% d'exactitude contre la vérité terrain."*
* **Q : "Quelle est la latence réelle de votre chaîne de traitement ?"**  
  * **R :** *"Sur notre station GPU locale (NVIDIA RTX 3050 Laptop 4 Go VRAM, CPU AMD Ryzen 5), l'inférence neuronale UNet++ s'exécute en 56.0 ms (médiane), la vectorisation polygonale hybride en 34.0 ms, la génération de textures de façade en 164.6 ms (accélérée nativement sous PyTorch CUDA pour l'ensemble des murs extérieurs), et l'extrusion 3D en 0.7 ms. La latence médiane totale du pipeline de bout en bout est de **297.3 ms** (moyenne: 342.1 ms, P90: 480.8 ms, P95: 780.1 ms). La traîne du P95 s'explique par les plans comptant un nombre élevé de façades extérieures générées intégralement en amont ; en production, synthétiser uniquement la façade principale à la demande élimine cette traîne pour maintenir l'ensemble sous 350 ms. En mode dégradé exclusivement CPU (fallback ONNX Runtime), le pipeline complet s'exécute en 1 358.8 ms."*

---

## 🚀 Commande de Lancement Rapide Pré-Soutenance

```bash
# 1. Vérifier que les tests sont verts
pytest tests/ backend/tests/ -v

# 2. Lancer l'application complète
docker-compose up --build

# 3. Ouvrir l'application
# Web UI : http://localhost:3000
# Swagger API : http://localhost:8000/docs
# MLflow Dashboard : http://localhost:5000 (si démarré)
```
