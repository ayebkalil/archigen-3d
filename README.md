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

2. **Modèle 2 : Synthèse Générative de Façades (Pix2Pix cGAN Perceptuel)**
   * **Jeu de données :** CMP Facades (400 paires train / 106 test).
   * **Architecture :** Générateur UNet-256 (upsampling Nearest+Conv sans damier), Discriminateur PatchGAN $70 \times 70$ stabilisé par Spectral Normalization.
   * **Fonction de coût :** Perte hybride LSGAN + LPIPS (SqueezeNet) + Feature Matching + $\mathcal{L}_1$.
   * **Résultats vérifiés (Split Test CMP Facades) :**
     * **KID :** $\mathbf{60.40 \times 10^{-3}}$ (meilleur score du benchmark vs 149.89 Baseline)
     * **LPIPS :** **0.324** (gain de 49.3% en similarité perceptuelle)
     * **Export ONNX Runtime :** Parité numérique stricte confirmée ($\Delta_{\max} = 6.31 \times 10^{-6}$).

---

## ⏱️ Métriques de Performance & Latence Réelle

Mesuré sur station locale (GPU NVIDIA GeForce RTX 3050 Laptop 4 Go VRAM, CPU AMD Ryzen 5, 16 Go RAM) :
* **Inférence Tensorielle UNet++ (GPU) :** **~45 ms**
* **Vectorisation Polygonale Rectilinéaire (CPU) :** **~180 ms**
* **Synthèse de Façade cGAN ONNX Runtime (GPU) :** **~1 650 ms**
* **Extrusion 3D & Exportation OBJ/MTL (CPU) :** **~150 ms**
* **Latence Totale de Bout en Bout :** **~2.03 s (2 031.9 ms)** mesurée sur 20 itérations complètes.

---

## ⚡ Démarrage Rapide (Production Docker Compose)

```bash
# Cloner le projet
git clone https://github.com/ayebk/archigen-3d.git
cd archigen-3d

# Récupérer les poids des modèles via DVC
dvc pull

# Lancer la pile complète (FastAPI + React 18 + Nginx)
docker-compose up --build
```

* **Application Web :** `http://localhost:3000`
* **Documentation Interactive API Swagger :** `http://localhost:8000/docs`
* **Health Check Endpoint :** `http://localhost:8000/health`

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
