# ArchiGen 3D (AI-Powered HomeByMe)

[![CI Pipeline](https://github.com/ayebk/archigen-3d/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ayebk/archigen-3d/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

**Cycle Ingénieur en Intelligence Artificielle & Data Science — 3ème Année**  
**Module :** Deep Learning Avancé & MLOps  
**Enseignant :** Haythem Ghazouani  

---

## 👥 Équipe & Rôles Spécifiques

| Étudiant | Rôle Majeur | Branche Git Dédiée |
| :--- | :--- | :--- |
| **Khalil Ayeb** | **Lead Deep Learning / Research Scientist** (Modèle Génératif Pix2Pix, FID/LPIPS, ONNX) | `khalil/dl-generative-lead` |
| **Khawla** | **Lead MLOps & Computer Vision Engineer** (Segmentation CubiCasa5K, DVC, MLflow, CI/CD) | `khawla/mlops-vision-lead` |
| **Amine** | **Lead Full-Stack Software Engineer** (API FastAPI, Frontend React + Three.js, Docker Compose) | `amine/fullstack-lead` |

---

## 🏛️ Architecture du Projet

1. **Phase 1 (Parsing 2D) :** Analyse et segmentation sémantique d'un plan 2D via SegFormer/YOLOv11-seg.
2. **Phase 2 (3D Interactive) :** Extrusion 3D instantanée des cloisons via React et Three.js (`@react-three/fiber`).
3. **Phase 3 (Rendu Génératif) :** Synthèse d'une façade/pièce photoréaliste via Pix2Pix (cGAN) sous ONNX Runtime en $<400\text{ ms}$.

---

## ⚡ Démarrage Rapide en 1 Clic (Production)

L'ensemble de la solution est conteneurisé et orchestré via Docker Compose :

```bash
# Cloner le projet
git clone https://github.com/ayebk/archigen-3d.git
cd archigen-3d

# Lancer tous les services (Backend + Frontend)
docker-compose up --build
```

* **Frontend Web :** `http://localhost:3000`
* **API Documentation Swagger :** `http://localhost:8000/docs`
* **Health Check Endpoint :** `http://localhost:8000/health`

---

## 🧪 Tests Automatisés

```bash
cd backend
pytest tests/ -v
```

---

## 📦 Versioning des Données (DVC)

Les jeux de données bruts et checkpoints ne sont jamais commités sur Git. Ils sont gérés via DVC :

```bash
dvc pull
```
