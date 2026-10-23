# ArchiGen 3D (AI-Powered BIM & Floorplan-to-3D Platform)

[![CI Pipeline](https://github.com/ayebk/archigen-3d/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ayebk/archigen-3d/actions)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Cycle Ingénieur en Intelligence Artificielle & Data Science — 3ème Année**  
**Module :** Deep Learning Avancé & MLOps  
**Enseignant :** Haythem Ghazouani  

---

## 👥 Équipe de Développement

* **Khalil Ayeb**
* **Amine Ben Fadhel**
* **Khawla Hawari**

---

## 🏛️ Architectures des Modèles & Données

1. **Modèle 1 : Segmentation Multi-Tâches de Plans (MultiTask UNet++)**
   * **Jeu de données :** CubiCasa5K (4 200 plans train / 400 val / 400 test, licence CC BY-NC 4.0).
   * **Architecture :** UNet++ avec encodeur ResNet-34 pré-entraîné ImageNet et deux têtes de décodage simultanées :
     * Tête structurelle (5 classes) : Fond, Mur Intérieur, Mur Extérieur, Porte, Fenêtre.
     * Tête sémantique (12 classes) : 11 types de pièces fonctionnelles + fond.
   * **Résultats vérifiés sur 400 plans de test non vus :**
     * **Wall Merged IoU :** **74.32%** (External : 73.89%, Internal : 62.34%)
     * **Door IoU :** **62.38%** | **Window IoU :** **73.28%**
     * **Room Pixel Accuracy :** **85.85%** (surpasse le benchmark officiel de 82.7%)
     * **Room Semantic mIoU :** **54.72%** (proche des 57.5% publiés)
     * **Taux de Coque 3D Étanche :** **96.25%**
     * **Exactitude Classification Murs Extérieurs :** **91.07%** (Précision : 94.21%, Rappel : 92.41%)

2. **Modèle 2 : Synthèse Générative de Façades (Pix2Pix cGAN Perceptuel Promu)**
   * **Jeu de données :** CMP Facades (400 paires train / 106 test).
   * **Architecture :** Générateur UNet-256 (upsampling Nearest+Conv sans damier), Discriminateur PatchGAN $70 \times 70$ stabilisé par Spectral Normalization.
   * **Entraînement :** 100 époques avec lissage exponentiel des poids (EMA $\alpha = 0.999$) et décroissance linéaire du learning rate dès l'époque 50.
   * **Résultats vérifiés (Split Test CMP Facades) :**
     * **KID :** $\mathbf{63.55 \times 10^{-3} \pm 5.83 \times 10^{-3}}$ (Protocole standard seed 42, subset 50, B=100 ; meilleur score vs $67.65 \times 10^{-3}$ v1.0 et $150.36 \times 10^{-3}$ Baseline).
     * **Test Apparié par Image ($N=106$, $B=2\,000$) :** KID de $63.21 \pm 5.76$ vs $66.90 \pm 6.22$, gains statistiquement significatifs sur 72% à 75% des images individuelles ($p < 10^{-6}$).
     * **LPIPS :** **0.3096** (gain perceptuel majeur vs 0.6327 Baseline).
     * **Export ONNX Runtime :** Parité numérique stricte confirmée ($\Delta_{\max} = 4.13 \times 10^{-6} \ll 10^{-4}$).

---

## 🚪 Détection d'Objets & Évaluation à l'Aveugle (Domain Shift)

1. **Détection des Portes en Espace Pixel (CubiCasa5K Test Split, $N=400$) :**
   * **Rayon $\le 24\text{ px}$ :** Précision 91.24%, Rappel 89.68%, **F1-Score : 90.45%** (3 397 TP, 326 FP, 391 FN).
   * **Rayon $\le 12\text{ px}$ :** Précision 90.14%, Rappel 88.60%, **F1-Score : 89.36%**.
   * **Localisation Centroïde :** **Médiane = 1.12 pixel** (Moyenne = 1.77 px, $\sigma = 2.72\text{ px}$).
   * **IoU Empreinte Bâtie Pleine :** **82.16%** (coque extérieure étanche fermée).

2. **Évaluation à l'Aveugle sur 12 Plans Externes Verrouillés ($N=12$, SHA256 scellés) :**
   * **Triage d'Exploitabilité Réaliste (Diagnostic Visuel de l'Équipe) :**
     * **Directement Exploitable :** **1 / 12 plans (8.3%)** (`fresh_04`, Ladowski : 6/6 pièces, 7/8 portes).
     * **Partiellement Exploitable / Retouches Requises :** **4 / 12 plans (33.3%)** (`fresh_05`, `fresh_06`, `fresh_07`, `fresh_09`).
     * **Inexploitable / Échec Complet :** **7 / 12 plans (58.3%)** (`fresh_01`, `fresh_02`, `fresh_03`, `fresh_08`, `fresh_10`, `fresh_11`, `fresh_12`).
   * **Erreurs Découvertes Hors Distribution :**
     * **MAE Portes :** **8.08 portes** (écart extrême $-15$ à $+21$ ; le "biais +0.08" était une annulation statistique fortuite).
     * **MAE Pièces :** **5.25 pièces**.
   * **Constat Majeur :** L'étanchéité extérieure (75% coques valides) ne garantit pas l'exploitabilité intérieure ; le décalage de distribution (domain shift) sur les plans CAD teintés ou croquis historiques constitue la limitation première du modèle.

3. **Branche Expérimentale (`experiment/floorplan-style-adaptation`) :**
   * **Adaptation de Domaine Procédurale :** Moteur de restylage vectoriel SVG en 6 styles architecturaux (`cad_blueprint`, `hatched_walls`, `annotated_dimensions`, etc.) garantissant zéro fuite d'annotations.
   * **Banc de Développement ($N=20$ plans réels variés) :** Le filtre heuristique d'exploitabilité géométrique (`rooms >= 3 and area >= 25 m2`) passe de **55.0% à 85.0%** (plans vides réduits de 30% à 10%). On observe cependant une tendance au sur-partitionnement des cloisons intérieures (+70 pièces détectées au total).
   * **Contrôle de Régression In-Distribution (CubiCasa Val, $N=400$) :**
     * **Wall Merged IoU :** **73.90% vs 73.90%** (parité parfaite).
     * **Door IoU :** **60.88% vs 61.13%** (baisse minime de 0.25%).
     * **Window IoU :** **73.70% vs 74.72%** (baisse de 1.02%).
     * **Room Semantic mIoU :** **56.94% vs 62.39%** (-5.45% de régression sémantique, motivant le maintien de v1.1 en production).
   * **Diagnostic Hors-Piste sur les 12 Plans de Soutenance ($N=12$) :**
     * Coques étanches valides : **75.0% $\to$ 100.0%** (12/12 coques fermées).
     * Plans en échec total (0 pièce) : **3/12 $\to$ 0/12** (récupération de `fresh_01`, `fresh_03`, `fresh_12`).
     * Erreur absolue pièces : **5.25 $\to$ 4.42 pièces** ; Portes : **8.08 $\to$ 6.92 portes** (variations modestes dans la marge de bruit statistique).
     * *Note méthodologique :* Les 12 plans ne constituent plus un test aveugle pur car les styles ont été conçus en réponse directe aux modes d'échec observés sur ceux-ci.
   * **Studio d'Édition Interactive CAD :** Outils de retouche directe 2D/3D (Select, + Door, Delete, Relabel, Scale) intégrés au canevas React avec synchronisation temps réel Three.js.

---

## ⏱️ Métriques de Performance & Latence Réelle

Mesuré sur station locale (GPU NVIDIA GeForce RTX 3050 Laptop 4 Go VRAM, CPU AMD Ryzen 5, 16 Go RAM) :
* **1. Prétraitement d'Image (Letterbox 512×512 & Tenseurs) :** **73.3 ms**
* **2. Inférence Tensorielle UNet++ (GPU CUDA) :** **64.9 ms**
* **3. Vectorisation Polygonale Rectilinéaire (CPU) :** **76.6 ms**
* **4. Synthèse de Façade cGAN PyTorch (GPU CUDA, mode `front_only=True`) :** **30.4 ms**
* **5. Génération Scène 3D Three.js & Métadonnées (CPU) :** **1.3 ms**
* **Latence Totale Bout-en-Bout (Moyenne mesurée) :** **246.5 ms** (P95 : 282.7 ms) — conforme au budget interactif (< 400 ms).

---

## ⚡ Démarrage Rapide (Production Docker Compose)

```bash
# 1. Cloner le projet
git clone https://github.com/ayebk/archigen-3d.git
cd archigen-3d

# 2. Récupérer les poids des modèles via DVC (depuis le stockage cloud synchronisé ou remote local)
# Exemple de configuration d'un remote de restauration :
dvc remote add -d -f cloud_backup "C:\Users\ayebk\OneDrive\archigen_dvc_remote_backup"
dvc pull

# 3. Lancer la pile complète (FastAPI + React 18 + Nginx)
docker compose up --build -d
```

* **Application Web :** `http://localhost:3000`
* **Documentation Interactive API Swagger :** `http://localhost:8000/docs`
* **Health Check Endpoint :** `http://localhost:8000/health` (Conteneur validé en statut `healthy`)

---

## 🧪 Tests Automatisés & Reproductibilité

```bash
# Lancer les tests unitaires et d'intégration
pytest tests/ backend/tests/ -v

# Lancer le benchmark quantitatif complet sur les 400 plans de test
python src/floorplan/evaluate.py --model multitask --split test

# Générer un modèle 3D complet (OBJ + MTL + Textures) depuis un plan 2D
python scripts/floorplan_to_3d.py --input data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/10611/F1_scaled.png
```
