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

## 🛋️ Système d'Aménagement Intérieur (Stage 3 : Placement Basé sur des Règles)

Le module d'aménagement intérieur (`src/design/`) est un **système déterministe basé sur des règles géométriques** pour le placement de mobilier 3D sous licence libre CC0 (modèles Kenney.nl). 

> **Clarification méthodologique :** Ce module n'est **en aucun cas un modèle de Deep Learning ou d'IA générative**. Il n'effectue aucune revendication de "qualité de design" ou d'expertise décorative. Il s'agit strictement d'un moteur géométrique vérifiant l'absence de collisions, la continuité de passage et l'ancrage mural.

### Principes Algorithmiques & Contraintes Vérifiées :
1. **Classification des Arêtes par Transformée de Distance (Méthode B) :** Utilise directement la carte de distance euclidienne du masque de mur prédit par le réseau de neurones (`scipy.ndimage.distance_transform_edt`) pour identifier les faces de murs réelles.
2. **Règle Stricte d'Ancrage Mural ($\ge 80\%$ à $\le 0.10\text{ m}$) :** Tout meuble adossé au mur (`back_to_wall`) est échantillonné en 15 points le long de son arête arrière. Il est strictement rejeté si moins de 80% de ses points d'appui sont à moins de 0.10 m d'un pixel de mur prédit. Les éléments flottants (`rug`, `coffee_table`, `dining_table` centrée) sont exemptés.
3. **Zéro Collision & Dégagement des Portes :** Aucun chevauchement entre polygones de meubles ($> 1\text{ cm}^2$), exclusion stricte des zones de battement de portes ($0.85\text{ m}$ de dégagement).
4. **Accessibilité BFS :** Validation par flood-fill BFS sur grille de 10 cm vérifiant l'existence d'un chemin libre entre la porte d'entrée et la face avant de chaque meuble principal.
5. **Filtrage des Fragments de Sur-Segmentation :** 
   > **Note sur le choix des seuils :** Les seuils de surface minimale plausible (**salle de bain : $2.00\text{ m}^2$**, **chambre : $5.00\text{ m}^2$**, **cuisine : $3.00\text{ m}^2$**, **salon : $6.00\text{ m}^2$**) ont été **choisis après analyse directe des échecs sur le jeu de diagnostic** (plans 0 à 100 et 100 à 200), où $97.4\%$ des salles de bain en échec étaient de minuscules fragments de bruit UNet++ ($1.1 - 1.6\text{ m}^2$) incapables d'accueillir un équipement standard et un dégagement de porte. Les pièces sous ces seuils reçoivent le statut `"not a room (fragment)"` et le bouton "Générer" est désactivé dans l'UI.

### Benchmark Final sur 100 Plans Non Vus Réservés (Plans 200 à 300, Exécuté 1 Seule Fois Sans Tuning) :

* **Jeu de données :** CubiCasa5k validation plans 200 à 300 ($N = 100$ plans, $N = 542$ pièces détectées).
* **Fragments exclus (< seuil plausible) :** **13.28% (72 / 542 pièces)** (58 salles de bain, 10 chambres, 4 salons, 0 cuisine).
* **Pièces Plausibles Évaluées :** **$N = 470$ pièces**.

#### 1. Validité Comparée : Pièces Plausibles vs. Toutes Pièces Détectées
| Type de Pièce | Pièces Détectées | Fragments Exclus | Pièces Plausibles | Validité Sans Collision (Plausibles) | Validité Sans Collision (Toutes Détectées) | Tous Meubles Requis (Plausibles) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Salle de bain** | 161 | 58 (36.0%) | 103 | **101 / 103 (98.06%)** | 101 / 161 (62.73%) | **100 / 103 (97.09%)** |
| **Cuisine** | 97 | 0 (0.0%) | 97 | **95 / 97 (97.94%)** | 95 / 97 (97.94%) | **65 / 97 (67.01%)** |
| **Salon** | 93 | 4 (4.3%) | 89 | **80 / 89 (89.89%)** | 80 / 93 (86.02%) | **67 / 89 (75.28%)** |
| **Chambre** | 191 | 10 (5.2%) | 181 | **140 / 181 (77.35%)** | 140 / 191 (73.30%) | **111 / 181 (61.33%)** |
| **TOTAL** | **542** | **72 (13.28%)** | **470** | **416 / 470 (88.51%)** | **416 / 542 (76.75%)** | **343 / 470 (72.98%)** |

*Note sur la complétude des cuisines :* La baisse à 67.01% pour "tous meubles requis" est causée par les cuisines ouvertes (partitions sans mur solide où l'îlot/comptoir secondaire ne peut s'ancrer sans flotter ; le moteur omet le meuble secondaire plutôt que de le laisser flotter).

#### 2. Contact Mural Arrière : Masque Prédit vs. Masque Vérité Terrain (Côte à Côte)
Mesuré sur l'ensemble des $N = 849$ meubles adossés placés dans les pièces valides des plans 200 à 300 :

| Métrique de Contact Mural | Masque de Mur Prédit (Modèle) | Masque Vérité Terrain (`struct_mask == 1 \| 2`) |
| :--- | :---: | :---: |
| **Part des meubles avec contact $\ge 80\%$** | **818 / 849 (96.35%)** | **630 / 849 (74.20%)** |
| **Contact moyen le long de l'arête arrière** | **96.58%** | **74.97%** |
| **Contact médian** | **100.00%** | **100.00%** |
| **Meubles obstruant une fenêtre GT (`struct_mask == 4`)** | — | **10 / 1,277 meubles (0.78%)** |
| **Meubles obstruant une fenêtre prédite** | **52 / 1,277 meubles (4.07%)** | — |

* **Temps de Calcul Moyen :** **76.0 ms / pièce** (100% hors-ligne, déterministe).

---

## 🎨 Rendu Visuel Photoréaliste (Stage 4 : Matériaux PBR, Éclairage HDRI & Performances)

Le moteur de visualisation 3D WebGL (Three.js dans [`frontend/src/components/Viewer3D.jsx`](frontend/src/components/Viewer3D.jsx)) a bénéficié d'une mise à niveau photoréaliste complète sans aucune modification de la logique de placement de mobilier (`src/design/`).

### Améliorations Graphiques Intégrées :
1. **Matériaux PBR Multi-Textures 1K :**
   - Sols pièces sèches (chambre, salon, bureau) : `oak_wood_planks` (Albedo sRGB, Normal map, ARM Roughness/AO/Metalness).
   - Sols pièces humides (salle de bain, cuisine) : `marble_01` (Albedo sRGB, Normal map, ARM).
   - Murs architecturaux : `white_stucco` (Albedo sRGB, Normal map, ARM).
2. **Illumination Basée sur l'Image (IBL / HDRI) :**
   - Carte d'environnement Radiance HDR `qwantani_late_afternoon_puresky_1k.hdr` (Poly Haven CC0) cuite par `PMREMGenerator` pour des reflets ambiants et spéculaires réalistes.
3. **Ombres Douces PCF & Plan de Contact :**
   - Carte d'ombres douces `THREE.PCFSoftShadowMap` (atlas 2048×2048, radius 2.4, bias $-0.00015$).
   - Plan de contact au sol semi-transparent (`ShadowMaterial`, opacité 0.22) pour ancrer les volumes 3D.
4. **Tone Mapping & Gestion des Couleurs :**
   - Tone mapping `THREE.ACESFilmicToneMapping` (exposition 1.05) et espace colorimétrique `THREE.SRGBColorSpace`.
5. **Préconfigurations de Caméra (Presets) :**
   - 4 angles interactifs : **Perspective 3D** (FOV 45°), **Axonométrique 45°**, **Plan Vue du Dessus (Top-Down)**, et **Focus Pièce Sélectionnée**.
6. **Atténuation et Traitement du Z-Fighting (Sols & Tapis) :**
   - La dalle de sol est placée à $y = 0.02\text{ m}$. Les meubles standards reposent sur le sol à $y = 0.02\text{ m}$.
   - Les tapis (`rug_living`) sont surélevés à $y = 0.023\text{ m}$ ($+3\text{ mm}$ au-dessus du sol) avec application d'un décalage matériel GPU Three.js (`polygonOffset = true`, `factor = -2.0`, `units = -4.0`).
   - Testé et confirmé sans scintillement visible sur l'ensemble des 4 angles de caméra inspectés (perspective, axonométrique, vue du dessus, et gros plan rasoir).
7. **Mode Basse Qualité Dédié (Low Quality Mode) :**
   - Un commutateur interactif dans l'interface WebGL (`PBR` vs `Low`) permet de basculer instantanément vers un mode diffus sans ombres (désactivation de l'atlas d'ombres 2048, suppression des calculs d'ombres PCF, désactivation de l'environnement IBL et des textures normales/rugosité). Ce mode offre un gain de performance de près de 5× sur processeur (SwiftShader), garantissant une navigation 3D réactive même sur matériel modeste.

### Mesure des Taux de Rafraîchissement (Grand Plan de 16 Pièces / 24 Murs) :
Mesuré par échantillonnage continu `requestAnimationFrame` sur 3 000 ms :

| Mode d'Exécution & Qualité | Matériel & Drapeaux | Fréquence (FPS) | Temps par Trame | Trame P95 |
| :--- | :--- | :---: | :---: | :---: |
| **GPU Matériel (NVIDIA - PBR High)** | GPU NVIDIA GeForce RTX 3050 Laptop (D3D11 ANGLE, écran 144 Hz) | **Soutient le taux de rafraîchissement 144 Hz** (144.2 FPS, VSync) | **6.93 ms** | **7.00 ms** |
| **Rendu Logiciel (CPU - PBR High)** | CPU SwiftShader WebGL (`--use-angle=swiftshader --disable-gpu`) | **9.3 FPS** | **107.68 ms** | **133.40 ms** |
| **Rendu Logiciel (CPU - Mode Basse Qualité)** | CPU SwiftShader WebGL (mode diffus sans ombres ni textures PBR) | **43.1 FPS** (gain 4.6×) | **23.19 ms** | **33.40 ms** |

### Description Précise des Captures Avant / Après :
Les captures comparatives ont été réalisées sous la **même pose caméra verrouillée** (`position: [1.5, 4.0, 4.8]`, `cible: [0.0, 0.2, -0.5]`) dans le **même salon meublé à l'identique** (canapé 3 places, table basse, fauteuil, meuble TV, lampadaire, tapis) :
- **Image Avant (`stage4_before_same_angle.png`) :** Rendu en mode ombrage plat standard (couleurs diffuses unies, éclairage directionnel basique, aucune texture PBR, aucune carte d'environnement HDR, ombres dures sans filtrage doux).
- **Image Après (`stage4_after_same_angle.png`) :** Rendu en mode PBR photoréaliste (parquet en chêne et murs en stuc avec relief et rugosité, reflets ambiants HDR Qwantani, tone mapping ACES et ombres de contact adoucies PCF).
- **Gros Plan Rasoir (`stage4_rug_floor_closeup.png`) :** Vérification détaillée de l'interface tapis/sol confirmant l'absence de z-fighting coplanaire.

### Vérification de l'Image Docker Autonome & Latences Conteneur :
- **Image Autonome (Zero Volume Mounts) :** Construit intégralement depuis zéro avec `docker compose up --build -d` sans aucun montage de volume du répertoire de travail. Toutes les dépendances sont strictement gérées dans `requirements.txt` sans intervention manuelle.
- **Servitude des Modèles :** Les 33 modèles GLB sous licence CC0 (Kenney.nl) sont servis localement avec **33/33 réponses HTTP 200 validées**.
- **Routeur de Repli & Transparence Cryptographique (Conteneur ONNX CPU) :**
  - Le conteneur embarque le modèle champion ONNX (`floorplan_multitask.onnx`) ainsi que le routeur de repli autonome ONNX (`floorplan_fallback.onnx`, exporté depuis `multitask_resnet34_blend_alpha_0_3.pth`).
  - Les points `/health` et `/model/metadata` exposent `fallback_loaded: true` et les empreintes SHA-256 complètes :
    - Modèle Principal : `floorplan_multitask.onnx` (`538ade1935a9624e19047c910ed595f0e9a62aa08d872715216d3b323dfa034c`)
    - Modèle Repli : `floorplan_fallback.onnx` (`7188af6fc15f11a8fc1979a02656e2bb32d200e7d2d7bb8b8d07b43296fff809`)
    - Modèle Façade cGAN : `generator_perceptual_ema.onnx` (`46d95ec2b937af63ef36e0f8c7d1255105e8546d0c7dab42919ae8e6d2fc25ab`)

### Latences Conteneur CPU Mesurées sur 20 Exécutions (`/predict/floorplan`) :
Mesuré via requêtes HTTP directes vers le conteneur Docker autonome (Pipeline : **ONNX Runtime CPU**) :

| Configuration du Plan & Inférence | TTA (Test-Time Augmentation) | Repli Déclenché | Médiane (ms) | P95 (ms) | Moyenne (ms) | Min / Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plan Normal (`modern_06`)** | OFF | Non | **636.09 ms** | **1,168.14 ms** | 695.32 ms | 543.08 / 1,249.36 ms |
| **Plan Normal (`modern_06`)** | ON (4-fold) | Non | **2,083.10 ms** | **2,375.15 ms** | 2,094.14 ms | 1,825.28 / 2,494.70 ms |
| **Plan Vide Effondré (512×512)** | OFF | Oui (Fallback Router) | **1,128.82 ms** | **1,307.92 ms** | 1,138.03 ms | 909.35 / 1,428.18 ms |
| **Plan Vide Effondré (512×512)** | ON (4-fold) | Oui (Fallback Router) | **2,679.59 ms** | **3,121.55 ms** | 2,723.78 ms | 2,291.28 / 3,196.69 ms |

*Note sur `/design` (moteur de placement déterministe) :* Médiane conteneur de **1.67 ms** (Moyenne : 3.19 ms).

### Parité des Agencements ONNX CPU vs. PyTorch GPU (Banc Réévalué sur 50 Plans Réels) :
Après identification de l'écart initial (l'ancien ONNX provenait du checkpoint intermédiaire `multitask_resnet34_best.pth` au lieu du champion final `multitask_resnet34_finetuned_best.pth`), l'export a été réaligné avec une parité de logits sub-$10^{-4}$ ($1.79 \times 10^{-5}$ struct, $2.67 \times 10^{-5}$ room).

Le benchmark sur 50 plans réels (architecturaux, CubiCasa5k, plans externes) montre une concordance quasi-parfaite :
- **IoU Géométrique Moyen des Polygones de Pièces :** **98.39%** (Pipeline PyTorch GPU vs. Pipeline ONNX CPU).
- **Écart Moyen du Nombre de Pièces Détectées :** **0.00** (50 / 50 plans ont exactement le même nombre de pièces).
- **Écart Moyen du Nombre de Murs :** **0.02** (49 / 50 plans identiques à l'arête près).
- **Concordance Sémantique des Types de Pièces :** **100.00%**.
- **Latence Moyenne d'Inférence par Plan :** **344.4 ms** (PyTorch GPU) vs. **868.9 ms** (ONNX CPU).
- Détail complet sauvegardé dans [`outputs/design_stage4/onnx_vs_pytorch_comparison.json`](outputs/design_stage4/onnx_vs_pytorch_comparison.json).

- **Flux Étendu en Conteneur :** Testé avec succès sur conteneur Docker avec aménagement complet :
  - Chambre (`docker_flow_bedroom.png`)
  - Salon (`docker_flow_living_room.png`)
  - Salle de bain (`docker_flow_bathroom.png`)
  - Pièce fragmentée sous le seuil plausible (`docker_flow_fragment_room.png` avec badge et désactivation).

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
