# 3D Asset Licenses, Provenance, and Verification

This document details the licensing, provenance, and verification of all 3D assets used by the rule-based interior placement engine in ArchiGen 3D.

---

## 1. Upstream Official Source & License

All 3D models are curated from the **Furniture Kit (v2.0)** created by **Kenney** ([kenney.nl](https://kenney.nl/assets/furniture-kit)).

- **Official Web Page**: [https://kenney.nl/assets/furniture-kit](https://kenney.nl/assets/furniture-kit)
- **Official Direct ZIP Package**: [https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip](https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip)
- **License**: **Creative Commons Zero (CC0 1.0 Universal) — Public Domain Dedication**
- **License URL**: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

### License Statement
> The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law. You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

---

## 2. Byte-for-Byte Binary Verification

To ensure asset integrity and provenance, the official release ZIP (`kenney_furniture-kit.zip`, 5,130,729 bytes) was downloaded directly from `kenney.nl` and compared against the raw repository files:
- **Total files in official ZIP archive**: 1,557
- **Total `.glb` models in official ZIP**: 140
- **Curated models compared**: 33 (23 core + 10 style variants)
- **Verification outcome**: **33 of 33 files match the official release with identical SHA-256 hashes byte-for-byte.**

---

## 3. Offline Operation Guarantee

All normalized `.glb` models are stored locally in [`frontend/public/models/furniture/`](frontend/public/models/furniture/) and registered in [`src/design/catalog.json`](src/design/catalog.json).
At runtime, the application serves all assets statically from the local bundle with **zero external network requests**.

---

## 4. Curated Model Inventory (33 Models)

### Core Furniture (23 Models)
| ID | Category | Raw Model (Official ZIP) | Uniform Scale | Placement Rule | Clearances [F / S / B] |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `bed_double` | `bed` | `bedDouble.glb` | 1.67 | `back_to_wall` | 0.80m / 0.60m / 0.00m |
| `bed_single` | `bed_single` | `bedSingle.glb` | 1.67 | `back_to_wall` | 0.80m / 0.50m / 0.00m |
| `nightstand` | `nightstand` | `cabinetBedDrawerTable.glb` | 1.90 | `adjacent_anchor` | 0.50m / 0.05m / 0.00m |
| `wardrobe` | `wardrobe` | `bookcaseClosedDoors.glb` | 2.45 | `back_to_wall` | 0.80m / 0.05m / 0.00m |
| `sofa_3seater` | `sofa` | `loungeSofa.glb` | 2.04 | `back_to_wall` | 0.80m / 0.30m / 0.00m |
| `armchair` | `armchair` | `loungeDesignChair.glb` | 1.50 | `facing_anchor` | 0.60m / 0.20m / 0.10m |
| `coffee_table` | `coffee_table` | `tableCoffee.glb` | 1.67 | `in_front_of_anchor` | 0.40m / 0.40m / 0.40m |
| `tv_unit` | `tv_unit` | `cabinetTelevision.glb` | 2.00 | `back_to_wall` | 1.00m / 0.10m / 0.00m |
| `rug_living` | `rug` | `rugRectangle.glb` | 1.60 | `under_anchor` | 0.00m / 0.00m / 0.00m |
| `potted_plant` | `plant` | `pottedPlant.glb` | 2.20 | `corner` | 0.10m / 0.10m / 0.10m |
| `dining_table` | `dining_table` | `table.glb` | 1.95 | `center` | 0.80m / 0.80m / 0.80m |
| `dining_chair` | `dining_chair` | `chairModernCushion.glb` | 2.30 | `around_anchor` | 0.40m / 0.10m / 0.40m |
| `sideboard` | `sideboard` | `bookcaseClosedWide.glb` | 1.85 | `back_to_wall` | 0.80m / 0.10m / 0.00m |
| `desk` | `desk` | `desk.glb` | 1.85 | `back_to_wall` | 0.90m / 0.20m / 0.00m |
| `office_chair` | `office_chair` | `chairDesk.glb` | 1.80 | `facing_anchor` | 0.20m / 0.20m / 0.60m |
| `bookcase` | `bookcase` | `bookcaseOpen.glb` | 2.10 | `back_to_wall` | 0.60m / 0.05m / 0.00m |
| `floor_lamp` | `lamp` | `lampRoundFloor.glb` | 2.00 | `corner` | 0.10m / 0.10m / 0.10m |
| `kitchen_counter` | `kitchen_counter` | `kitchenCabinet.glb` | 2.00 | `back_to_wall` | 0.90m / 0.00m / 0.00m |
| `kitchen_sink_unit`| `kitchen_sink` | `kitchenSink.glb` | 2.00 | `back_to_wall` | 0.90m / 0.00m / 0.00m |
| `refrigerator` | `refrigerator` | `kitchenFridgeLarge.glb` | 1.70 | `back_to_wall` | 0.90m / 0.05m / 0.05m |
| `toilet` | `toilet` | `toilet.glb` | 1.50 | `back_to_wall` | 0.60m / 0.25m / 0.00m |
| `bathroom_vanity` | `bathroom_sink` | `bathroomSink.glb` | 1.80 | `back_to_wall` | 0.70m / 0.15m / 0.00m |
| `bathtub` | `bathtub` | `bathtub.glb` | 1.40 | `corner_or_wall` | 0.70m / 0.00m / 0.00m |

### Style Variants (10 Models)
| ID | Category | Style Tag | Raw Model (Official ZIP) | Scale | Placement Rule |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `sofa_modern` | `sofa` | `modern` | `loungeDesignSofa.glb` | 1.96 | `back_to_wall` |
| `sofa_classic` | `sofa` | `classic` | `loungeSofaLong.glb` | 2.14 | `back_to_wall` |
| `armchair_scandi` | `armchair` | `scandinavian`| `chairModernFrameCushion.glb`| 2.50 | `facing_anchor` |
| `armchair_classic`| `armchair` | `classic` | `loungeChair.glb` | 1.80 | `facing_anchor` |
| `coffee_table_glass`| `coffee_table`| `modern` | `tableCoffeeGlassSquare.glb` | 1.70 | `in_front_of_anchor` |
| `dining_table_round`| `dining_table`| `scandinavian`| `tableRound.glb` | 1.80 | `center` |
| `dining_table_cloth`| `dining_table`| `classic` | `tableCloth.glb` | 1.95 | `center` |
| `dining_chair_scandi`| `dining_chair`| `scandinavian`| `chairRounded.glb` | 2.20 | `around_anchor` |
| `dining_chair_classic`| `dining_chair`| `classic` | `chair.glb` | 2.20 | `around_anchor` |
| `desk_corner` | `desk` | `classic` | `deskCorner.glb` | 1.65 | `back_to_wall` |

---

## 5. Original License Notice (from ZIP `License.txt`)

```text
Furniture Kit (2.0)
Created/distributed by Kenney (www.kenney.nl)
Creation date: 20-10-2018 16:21

License: (Creative Commons Zero, CC0)
http://creativecommons.org/publicdomain/zero/1.0/

This content is free to use in personal, educational and commercial projects.
Support us by crediting Kenney or www.kenney.nl (this is not mandatory)
```

---

## 6. PBR Textures & HDRI Environment Map Provenance

All photorealistic PBR material texture maps and the high-dynamic-range environment map used in the Stage 4 visual quality upgrade are curated from **Poly Haven** ([polyhaven.com](https://polyhaven.com)).

- **License**: **Creative Commons Zero (CC0 1.0 Universal) — Public Domain Dedication**
- **License URL**: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

### Curated Asset Inventory

| Asset Name | Asset Type | Local Path | Upstream Source & Authors | License |
| :--- | :--- | :--- | :--- | :--- |
| **Qwantani Late Afternoon Pure Sky** | Radiance HDR (1K) | `frontend/public/showcase/sky/qwantani_late_afternoon_puresky_1k.hdr` | [polyhaven.com/a/qwantani_late_afternoon_puresky](https://polyhaven.com/a/qwantani_late_afternoon_puresky) by Sergej Majboroda & Greg Zaal | CC0 1.0 |
| **Oak Wood Planks** | PBR Texture Set (Diff, Nor, ARM) | `frontend/public/showcase/tex-1k/oak_wood_planks/` | [polyhaven.com/a/wood_floor](https://polyhaven.com/a/wood_floor) by Rob Tuytel | CC0 1.0 |
| **Marble 01 (Tiles)** | PBR Texture Set (Diff, Nor, ARM) | `frontend/public/showcase/tex-1k/marble_01/` | [polyhaven.com/a/marble_01](https://polyhaven.com/a/marble_01) by Rob Tuytel | CC0 1.0 |
| **White Stucco (Plaster)** | PBR Texture Set (Diff, Nor, ARM) | `frontend/public/showcase/tex-1k/white_stucco/` | [polyhaven.com/a/painted_plaster_wall](https://polyhaven.com/a/painted_plaster_wall) by Rob Tuytel | CC0 1.0 |

### Offline Bundling & Zero-Network Guarantee
- All texture files and the `.hdr` sky map are stored locally in the Git repository under [`frontend/public/showcase/`](frontend/public/showcase/).
- During frontend build (`npm run build`), Vite copies these assets into `dist/showcase/`.
- The production Docker image (`archigen-3d-frontend`) copies `dist/` directly into `/usr/share/nginx/html`.
- **Zero runtime network requests**: No external CDN, asset server, or remote API is queried for 3D textures, models, or lighting. All requests resolve to relative container endpoints `/showcase/...` and `/models/furniture/...`.



## Roof Geometry Libraries

- `straight-skeleton` **1.1.0**: MIT, copyright 2021 vHawk. Pure JavaScript/TypeScript implementation (no CGAL/WASM runtime). Package license: `frontend/node_modules/straight-skeleton/LICENSE`.
- `clipper-lib` **6.4.2**: Boost Software License 1.0, Angus Johnson / JavaScript port by Timo; includes Tom Wu's JSBN attribution. Copyright and notices are preserved in the installed library and production notices at `/third-party/roof-libraries.txt`.
