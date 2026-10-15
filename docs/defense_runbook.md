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
│ Temps               │ Thématique                │ Intervenant Principal                │
├─────────────────────┼───────────────────────────┼──────────────────────────────────────┤
│ 00:00 - 05:00       │ 1. Démonstration Live     │ Amine (Lead Full-Stack)              │
│ 05:00 - 10:00       │ 2. Revue MLOps & Modèles  │ Khawla (Lead MLOps) & Khalil (Lead DL│
│ 10:00 - 15:00       │ 3. Code Review & Q&A      │ Trinôme au complet (Chacun sa part)  │
└─────────────────────┴───────────────────────────┴──────────────────────────────────────┘
```

---

## Étape 1 (00:00 - 05:00) : Démonstration Applicative en Direct

> **Orateur : Amine (Lead Full-Stack)**

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
     * La progression du Room Pixel Accuracy atteignant $85.85\%$ sur le split de test complet.
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
  * **R :** *"Le jeu de test CMP ne compte que 106 images. Dans un espace Inception à 2048 dimensions, la matrice de covariance empirique est sous-dimensionnée (rang déficient), ce qui fausse le calcul classique de Fréchet. C'est pourquoi nous avons calculé le KID (Kernel Inception Distance), qui est un estimateur polynomial cubique sans biais particulièrement adapté aux petits échantillons, classant notre modèle perceptuel en tête à $60.40 \times 10^{-3}$."*
* **Q : "Pourquoi le modèle Enhanced a un meilleur SSIM que le Perceptual ?"**  
  * **R :** *"La perte $\mathcal{L}_1$ pénalise les déviations pixel par pixel en forçant le réseau à prédire la moyenne conditionnelle de la distribution, ce qui produit une image lissée statistiquement récompensée par le SSIM (0.1430 vs 0.1058). La perte perceptuelle LPIPS force au contraire la correspondance des activations haute-fréquence, restituant la netteté visuelle et les détails architecturaux."*

### 2. Questions potentielles sur la Segmentation de Plans :
* **Q : "Comment avez-vous résolu le problème d'IoU initial et atteint 74.32% sur les murs et 62.38% sur les portes ?"**  
  * **R :** *"Une inspection visuelle minutieuse par superposition a révélé que les polygones vectoriels SVG n'étaient pas recalés avec le facteur d'échelle letterbox du canevas $512 \times 512$. Dès la correction du rasterizer dans `src/floorplan/dataset.py` et la génération du cache aligné, l'apprentissage a convergé immédiatement : l'IoU des murs a bondi de 22.44% à 74.32%, les portes de 10.71% à 62.38%, les fenêtres à 73.28%, et l'exactitude des pièces à 85.85% (dépassant les 82.7% du papier officiel CubiCasa5K). Le mIoU pièces atteint 54.72% (proche des 57.5% publiés). Ce gain provient d'une correction rigoureuse de la vérité terrain, pas d'un modèle sur-paramétré."*
* **Q : "Comment vos tests unitaires sont-ils automatisés ?"**  
  * **R :** *"Notre fichier `.github/workflows/ci-tests.yml` déclenche à chaque push ou PR l'exécution de `pytest tests/ backend/tests/ -v`. Les 18 tests valident la cohérence du dataset, le flux de gradients du générateur/discriminateur, la vectorisation géométrique et les schémas Pydantic des endpoints FastAPI."*

### 3. Questions potentielles sur la Géométrie 3D & l'Ingénierie Système :
* **Q : "Comment gérez-vous la séparation entre murs intérieurs et extérieurs dans l'extrusion 3D ?"**  
  * **R :** *"Le vectoriseur utilise la dominance de classe entre les canaux `Wall_External` et `Wall_Internal` prédits par notre UNet++ multi-tâches, avec une vérification de l'enveloppe extérieure comme repli de secours uniquement si le masque structurel est absent. Sur notre modèle de test, cela extrait rigoureusement 7 murs extérieurs et 7 cloisons intérieures avec 91.07% d'exactitude contre la vérité terrain."*
* **Q : "Quelle est la latence réelle de votre chaîne de traitement ?"**  
  * **R :** *"Sur notre machine (GPU NVIDIA RTX 3050 Laptop 4 Go VRAM, CPU AMD Ryzen 5), l'inférence tensorielle UNet++ seule ne prend que 45 ms. Le pipeline complet de bout en bout (segmentation, vectorisation Shapely, synthèse de texture de façade cGAN sous ONNX Runtime et génération du fichier OBJ 3D) requiert en moyenne 2.03 secondes (2 031 ms), mesuré sur 20 itérations complètes."*

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
