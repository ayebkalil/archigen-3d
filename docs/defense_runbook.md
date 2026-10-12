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

> **Orateurs : Khawla (Lead MLOps) & Khalil (Lead DL)**

### Actions en direct sur l'écran :
1. **Démonstration DVC (Khawla) :**
   ```bash
   dvc status
   ```
   * Montrer que les gros fichiers (`.pth`, `.onnx`, `data/raw/`, `data/CubiCasa5k/`) sont strictement exclus de Git via `.gitignore` et indexés dans les pointeurs `.dvc`.
   * Expliquer que n'importe quel clone peut répliquer l'environnement avec `dvc pull`.
2. **Démonstration MLflow (Khawla) :**
   ```bash
   mlflow ui --port 5000
   ```
   * Ouvrir `http://localhost:5000` devant le professeur.
   * Montrer l'expérience `cubicasa_floorplan_segmentation` :
     * La courbe de convergence de la perte d'entraînement ($2.83 \to 1.80$).
     * La progression du Room mIoU atteignant $40.97\%$.
     * Les artefacts visuels : ouvrir l'un des fichiers `val_grid_epoch_060.png` pour prouver le logging automatique des masques de validation toutes les 5 époques.
3. **Revue de la Synthèse de Façades cGAN (Khalil) :**
   * Expliquer l'ablation expérimentale : Baseline vs Enhanced vs Perceptual.
   * Expliquer pourquoi les métriques pixel (SSIM/L1) récompensent le flou moyen tandis que la perte perceptuelle LPIPS (SqueezeNet) et Spectral Normalization génèrent les textures de pierre les plus réalistes sans artefacts d'échiquier.
   * Montrer l'export ONNX et le test de parité numérique stricte ($\Delta_{\max} = 6.31 \times 10^{-6}$).

---

## Étape 3 (10:00 - 15:00) : Revue de Code & Questions Techniques

> **Orateurs : Trinôme au complet (selon la question de l'enseignant)**

### 1. Questions potentielles pour Khalil (Lead DL) :
* **Q : "Pourquoi votre FID de façade est autour de 350-370 ?"**  
  * **R :** *"Le jeu de test CMP ne compte que 106 images. Dans un espace Inception à 2048 dimensions, la matrice de covariance empirique est sous-dimensionnée (rang déficient), ce qui fausse le calcul classique de Fréchet. C'est pourquoi nous avons calculé le KID (Kernel Inception Distance), qui est un estimateur polynomial cubique sans biais particulièrement adapté aux petits échantillons, classant notre modèle perceptuel en tête à $44.12 \times 10^{-3}$."*
* **Q : "Pourquoi le modèle Enhanced a un meilleur SSIM que le Perceptual ?"**  
  * **R :** *"La perte $\mathcal{L}_1$ pénalise les déviations pixel par pixel en forçant le réseau à prédire la moyenne conditionnelle de la distribution, ce qui produit une image lissée statistiquement récompensée par le SSIM. La perte perceptuelle force au contraire la correspondance des gradients haute-fréquence, restituant la netteté visuelle au détriment de l'alignement pixel-exact."*

### 2. Questions potentielles pour Khawla (Lead MLOps) :
* **Q : "Quel était le bug initial du rasterizer SVG de CubiCasa5K ?"**  
  * **R :** *"Une détection de sous-chaîne naïve `'Wall' in class_name` incluait des placards hauts (`WallCabinet`) parmi les murs porteurs, et la traversée récursive de l'arbre SVG dessinait les murs par-dessus les portes et fenêtres enfants. Nous l'avons corrigé par filtrage exact et hiérarchie directe."*
* **Q : "Comment vos tests unitaires sont-ils automatisés ?"**  
  * **R :** *"Notre fichier `.github/workflows/ci-tests.yml` déclenche à chaque push ou PR l'exécution de `pytest tests/ backend/tests/ -v`. Les 18 tests valident la cohérence du dataset, le flux de gradients du générateur/discriminateur, la vectorisation géométrique et les schémas Pydantic des endpoints FastAPI."*

### 3. Questions potentielles pour Amine (Lead Full-Stack) :
* **Q : "Pourquoi l'IoU des murs n'est que de 22.44% mais la coque 3D est étanche à 99.75% ?"**  
  * **R :** *"Sur une grille de $512 \times 512$, les cloisons intérieures ne font que 2 pixels d'épaisseur. Un simple décalage d'un pixel dégrade mathématiquement l'IoU à $33\%$. En revanche, pour la reconstruction 3D, ce qui compte est l'axe médian des murs : notre vectoriseur simplifie le contour par Douglas-Peucker et snapping orthogonal, garantissant des volumes polygonaux étanches et exploitables."*
* **Q : "Comment gérez-vous la fuite topologique sur les plans non texturés ?"**  
  * **R :** *"Sur les plans au trait pur où le modèle sémantique ne détecte pas de texture de pièce, les ouvertures de portes connectent l'intérieur avec l'extérieur. Nous avons mis en place une fermeture morphologique ainsi qu'un outil de calibration interactif permettant à l'utilisateur de contraindre l'échelle métrique."*

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
