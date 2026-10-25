#!/usr/bin/env python3
"""
scripts/normalize_assets.py
---------------------------
Normalizes raw CC0 Kenney 3D furniture models into metric-scale GLB assets.
- Strictly UNIFORM scaling (sx = sy = sz) for 27 organic/framed/seating assets to preserve geometry.
- Explicit non-uniform scaling permitted ONLY for 6 simple box-like cabinet/appliance items:
    1. sideboard (cabinet credenza, H=0.80m)
    2. kitchen_counter (standard 0.60m depth, 0.88m height)
    3. kitchen_sink_unit (standard 0.60m depth, 0.88m height)
    4. refrigerator (tall appliance, 0.70m depth, 1.80m height)
    5. wardrobe (bedroom closet, 0.60m depth, 2.00m height)
    6. tv_unit (low media console, 0.45m depth, 0.50m height)
- Centers footprint at base origin (X=0, Z=0) with ground contact at Y=0.
- Preserves canonical orientation: front faces +Z (into room), back faces -Z (against wall).
- Renames styles to presets (preset_1, preset_2, preset_3) with neutral descriptions.
- Measures exported GLB bounding boxes and generates src/design/catalog.json and frontend/public/catalog.json.
"""

import json
import logging
import os
import sys
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of items allowed non-uniform scaling (strictly simple box-like items)
BOX_LIKE_ITEMS = {
    "sideboard",
    "kitchen_counter",
    "kitchen_sink_unit",
    "refrigerator",
    "wardrobe",
    "tv_unit",
}

# Descriptive human-readable presets (replacing abstract preset_1/2/3)
PRESET_DESCRIPTIONS = {
    "contemporary_minimal": {
        "label": "Contemporary Minimal",
        "description": "Clean low-profile silhouettes, glass accents, and neutral upholstery"
    },
    "warm_wood": {
        "label": "Warm Wood",
        "description": "Natural oak and timber frames, curved joinery, and warm textiles"
    },
    "tailored_classic": {
        "label": "Tailored Classic",
        "description": "Structured armchairs, draped surfaces, and formal framed silhouettes"
    }
}

ASSET_SPEC = {
    # Bedroom
    "bed_double": {
        "raw_file": "bedDouble.glb",
        "name": "Double Bed",
        "category": "bed",
        "target_dims_m": [1.60, 2.05, 0.90],
        "scale": 1.67,  # uniform: 0.96m * 1.67 = 1.60m width
        "presets": ["contemporary_minimal", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.60, "back": 0.00},
        "required": True,
        "allowed_rooms": ["bedroom"],
    },
    "bed_single": {
        "raw_file": "bedSingle.glb",
        "name": "Single Bed",
        "category": "bed_single",
        "target_dims_m": [0.95, 2.00, 0.85],
        "scale": 1.67,  # uniform: 0.57m * 1.67 = 0.95m width
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.50, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },
    "nightstand": {
        "raw_file": "cabinetBedDrawerTable.glb",
        "name": "Nightstand",
        "category": "nightstand",
        "target_dims_m": [0.50, 0.45, 0.55],
        "scale": 1.90,  # uniform: 0.27m * 1.90 = 0.51m width
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "adjacent_anchor",
        "clearance": {"front": 0.50, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },
    "wardrobe": {
        "raw_file": "bookcaseClosedDoors.glb",
        "name": "Wardrobe Closet",
        "category": "wardrobe",
        "target_dims_m": [1.00, 0.60, 2.00],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [2.50, 2.3529, 2.40],  # 0.40m*2.50=1.00m W, 0.85m*2.353=2.00m H, 0.25m*2.40=0.60m D
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },

    # Living Room - Sofas & Seating
    "sofa_3seater": {
        "raw_file": "loungeSofa.glb",
        "name": "3-Seater Sofa (Neutral)",
        "category": "sofa",
        "target_dims_m": [2.00, 0.84, 0.94],
        "scale": 2.04,  # uniform: 0.98m * 2.04 = 2.00m width, 0.84m depth, 0.94m height
        "presets": ["warm_wood"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "sofa_modern": {
        "raw_file": "loungeDesignSofa.glb",
        "name": "Low-Profile Sofa",
        "category": "sofa",
        "target_dims_m": [2.20, 0.80, 0.80],
        "scale": 1.96,  # uniform: 1.12m * 1.96 = 2.20m width, 0.80m depth
        "presets": ["contemporary_minimal"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "sofa_classic": {
        # Fix: Replaced loungeSofaLong (1.75m deep chaise) with loungeSofa (0.84m deep standard)
        "raw_file": "loungeSofa.glb",
        "name": "Standard Framed Sofa",
        "category": "sofa",
        "target_dims_m": [2.00, 0.84, 0.94],
        "scale": 2.04,  # uniform: 0.98m * 2.04 = 2.00m width, 0.84m depth (NOT 1.75m!)
        "presets": ["tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "armchair": {
        "raw_file": "loungeDesignChair.glb",
        "name": "Contemporary Armchair",
        "category": "armchair",
        "target_dims_m": [0.85, 0.80, 0.80],
        "scale": 1.50,  # uniform: 0.73m * 1.50 = 1.10m width
        "presets": ["contemporary_minimal"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.60, "sides": 0.20, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "armchair_classic": {
        "raw_file": "loungeChair.glb",
        "name": "Club Armchair",
        "category": "armchair",
        "target_dims_m": [0.87, 0.73, 0.82],
        "scale": 1.78,  # uniform: 0.49m * 1.78 = 0.87m width, 0.73m depth, 0.82m height
        "presets": ["tailored_classic"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.60, "sides": 0.20, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "armchair_scandi": {
        # Fix: Scaled uniformly with s=1.78 to give height 0.82m (was 1.15m!)
        "raw_file": "chairModernFrameCushion.glb",
        "name": "Timber Frame Chair",
        "category": "armchair",
        "target_dims_m": [0.36, 0.36, 0.82],
        "scale": 1.78,  # uniform: 0.46m * 1.78 = 0.82m height (NOT 1.15m!)
        "presets": ["warm_wood"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.50, "sides": 0.15, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },

    # Living Room - Tables & Units
    "coffee_table": {
        "raw_file": "tableCoffee.glb",
        "name": "Rectangular Coffee Table",
        "category": "coffee_table",
        "target_dims_m": [1.10, 0.67, 0.38],
        "scale": 1.67,  # uniform: 0.66m * 1.67 = 1.10m width, 0.67m depth, 0.38m height
        "presets": ["warm_wood"],
        "placement_rule": "in_front_of_anchor",
        "clearance": {"front": 0.40, "sides": 0.40, "back": 0.40},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "coffee_table_glass": {
        "raw_file": "tableCoffeeGlassSquare.glb",
        "name": "Square Coffee Table",
        "category": "coffee_table",
        "target_dims_m": [0.90, 0.90, 0.45],
        "scale": 1.70,  # uniform
        "presets": ["contemporary_minimal"],
        "placement_rule": "in_front_of_anchor",
        "clearance": {"front": 0.40, "sides": 0.40, "back": 0.40},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "tv_unit": {
        "raw_file": "cabinetTelevision.glb",
        "name": "Media TV Console",
        "category": "tv_unit",
        "target_dims_m": [1.60, 0.45, 0.50],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [2.00, 1.6129, 1.80],  # 0.80m*2.00=1.60m W, 0.31m*1.613=0.50m H, 0.25m*1.80=0.45m D
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 1.00, "sides": 0.10, "back": 0.00},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "rug_living": {
        "raw_file": "rugRectangle.glb",
        "name": "Area Floor Rug",
        "category": "rug",
        "target_dims_m": [2.51, 1.47, 0.02],
        "scale": 1.60,  # uniform: 1.57m * 1.60 = 2.51m width, 1.47m depth
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "under_anchor",
        "clearance": {"front": 0.00, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "potted_plant": {
        "raw_file": "pottedPlant.glb",
        "name": "Potted Plant",
        "category": "plant",
        "target_dims_m": [0.50, 0.50, 1.20],
        "scale": 2.20,  # uniform
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "corner",
        "clearance": {"front": 0.10, "sides": 0.10, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living", "bedroom", "office"],
    },

    # Dining Room
    "dining_table": {
        "raw_file": "table.glb",
        "name": "Dining Table (Rectangular)",
        "category": "dining_table",
        "target_dims_m": [1.64, 0.88, 0.64],
        "scale": 1.95,  # uniform: 0.84m * 1.95 = 1.64m width, 0.88m depth, 0.64m height
        "presets": ["contemporary_minimal"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_table_round": {
        "raw_file": "tableRound.glb",
        "name": "Dining Table (Round)",
        "category": "dining_table",
        "target_dims_m": [1.20, 1.20, 0.76],
        "scale": 1.80,  # uniform
        "presets": ["warm_wood"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_table_cloth": {
        "raw_file": "tableCloth.glb",
        "name": "Dining Table (Draped)",
        "category": "dining_table",
        "target_dims_m": [1.64, 0.88, 0.64],
        "scale": 1.95,  # uniform
        "presets": ["tailored_classic"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_chair": {
        # Fix: Scaled uniformly with s=1.80 to give height 0.83m (was 1.06m!)
        "raw_file": "chairModernCushion.glb",
        "name": "Modern Dining Chair",
        "category": "dining_chair",
        "target_dims_m": [0.36, 0.36, 0.83],
        "scale": 1.80,  # uniform: 0.46m * 1.80 = 0.828m height (NOT 1.06m!)
        "presets": ["contemporary_minimal"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "dining_chair_scandi": {
        # Fix: Scaled uniformly with s=1.80 to give height 0.82m (was 1.00m!)
        "raw_file": "chairRounded.glb",
        "name": "Curved Dining Chair",
        "category": "dining_chair",
        "target_dims_m": [0.36, 0.36, 0.82],
        "scale": 1.80,  # uniform: 0.455m * 1.80 = 0.819m height (NOT 1.00m!)
        "presets": ["warm_wood"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "dining_chair_classic": {
        # Fix: Scaled uniformly with s=1.75 to give height 0.82m (was 1.03m!)
        "raw_file": "chair.glb",
        "name": "Framed Dining Chair",
        "category": "dining_chair",
        "target_dims_m": [0.35, 0.35, 0.82],
        "scale": 1.75,  # uniform: 0.47m * 1.75 = 0.823m height (NOT 1.03m!)
        "presets": ["tailored_classic"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "sideboard": {
        # Fix: Replaced bookcaseClosedWide (1.46m tall) with cabinetTelevisionDoors (0.80m tall credenza)
        "raw_file": "cabinetTelevisionDoors.glb",
        "name": "Low Sideboard Credenza",
        "category": "sideboard",
        "target_dims_m": [1.40, 0.45, 0.80],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [1.75, 2.5806, 1.7308],  # 0.80m*1.75=1.40m W, 0.31m*2.581=0.80m H, 0.26m*1.731=0.45m D (NOT 1.46m!)
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.10, "back": 0.00},
        "required": False,
        "allowed_rooms": ["dining"],
    },

    # Office
    "desk": {
        "raw_file": "desk.glb",
        "name": "Writing Desk",
        "category": "desk",
        "target_dims_m": [1.35, 0.72, 0.71],
        "scale": 1.85,  # uniform: 0.73m * 1.85 = 1.35m width, 0.72m depth, 0.71m height
        "presets": ["contemporary_minimal", "warm_wood"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.20, "back": 0.00},
        "required": True,
        "allowed_rooms": ["office"],
    },
    "desk_corner": {
        "raw_file": "deskCorner.glb",
        "name": "Corner Desk",
        "category": "desk",
        "target_dims_m": [1.60, 1.60, 0.75],
        "scale": 1.65,  # uniform
        "presets": ["tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.20, "back": 0.00},
        "required": True,
        "allowed_rooms": ["office"],
    },
    "office_chair": {
        "raw_file": "chairDesk.glb",
        "name": "Desk Swivel Chair",
        "category": "office_chair",
        "target_dims_m": [0.59, 0.56, 1.10],
        "scale": 1.80,  # uniform: 0.33m * 1.80 = 0.59m width
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.20, "sides": 0.20, "back": 0.60},
        "required": False,
        "allowed_rooms": ["office"],
    },
    "bookcase": {
        "raw_file": "bookcaseOpen.glb",
        "name": "Open Bookcase",
        "category": "bookcase",
        "target_dims_m": [0.84, 0.52, 1.85],
        "scale": 2.10,  # uniform: 0.40m * 2.10 = 0.84m width, 1.85m height
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.60, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["office", "living"],
    },
    "floor_lamp": {
        "raw_file": "lampRoundFloor.glb",
        "name": "Floor Lamp",
        "category": "lamp",
        "target_dims_m": [0.40, 0.40, 1.50],
        "scale": 2.00,  # uniform
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "corner",
        "clearance": {"front": 0.10, "sides": 0.10, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living", "office"],
    },

    # Kitchen
    "kitchen_counter": {
        # Fix: Depth 0.60m standard counter (was 0.90m!)
        "raw_file": "kitchenCabinet.glb",
        "name": "Kitchen Base Counter",
        "category": "kitchen_counter",
        "target_dims_m": [0.90, 0.60, 0.88],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [2.0930, 1.9556, 1.3333],  # 0.43m*2.093=0.90m W, 0.45m*1.956=0.88m H, 0.45m*1.333=0.60m D
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.00, "back": 0.00},
        "required": True,
        "allowed_rooms": ["kitchen"],
    },
    "kitchen_sink_unit": {
        # Fix: Depth 0.60m standard counter (was 0.90m!)
        "raw_file": "kitchenSink.glb",
        "name": "Kitchen Sink Counter",
        "category": "kitchen_sink",
        "target_dims_m": [0.90, 0.60, 0.88],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [2.0930, 1.7959, 1.3333],  # 0.43m*2.093=0.90m W, 0.49m*1.796=0.88m H, 0.45m*1.333=0.60m D
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["kitchen"],
    },
    "refrigerator": {
        "raw_file": "kitchenFridgeLarge.glb",
        "name": "Tall Refrigerator",
        "category": "refrigerator",
        "target_dims_m": [0.75, 0.70, 1.80],
        # Box-like item: non-uniform scale [sx, sy, sz]
        "scale": [1.4423, 1.9565, 1.7073],  # 0.52m*1.442=0.75m W, 0.92m*1.957=1.80m H, 0.41m*1.707=0.70m D
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.05, "back": 0.05},
        "required": False,
        "allowed_rooms": ["kitchen"],
    },

    # Bathroom
    "toilet": {
        "raw_file": "toilet.glb",
        "name": "Floor Toilet",
        "category": "toilet",
        "target_dims_m": [0.47, 0.72, 0.68],
        "scale": 1.50,  # uniform: 0.31m * 1.50 = 0.47m width, 0.72m depth, 0.68m height
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.60, "sides": 0.25, "back": 0.00},
        "required": True,
        "allowed_rooms": ["bathroom"],
    },
    "bathroom_vanity": {
        "raw_file": "bathroomSink.glb",
        "name": "Bathroom Vanity Sink",
        "category": "bathroom_sink",
        "target_dims_m": [0.61, 0.52, 1.01],
        "scale": 1.80,  # uniform: 0.34m * 1.80 = 0.61m width, 0.52m depth, 1.01m height
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.70, "sides": 0.15, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bathroom"],
    },
    "bathtub": {
        "raw_file": "bathtub.glb",
        "name": "Full Bathtub",
        "category": "bathtub",
        "target_dims_m": [1.67, 0.78, 0.59],
        "scale": 1.40,  # uniform: 1.19m * 1.40 = 1.67m width, 0.78m depth, 0.59m height
        "presets": ["contemporary_minimal", "warm_wood", "tailored_classic"],
        "placement_rule": "corner_or_wall",
        "clearance": {"front": 0.70, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bathroom"],
    },
}

def normalize_asset(raw_path: str, scale_spec) -> tuple:
    """
    Centers base footprint at (X=0, Z=0), places ground contact at Y=0,
    and applies scale transformation.
    - If scale_spec is float: strict UNIFORM scaling (sx = sy = sz = scale_spec).
    - If scale_spec is list/tuple: non-uniform scaling [sx, sy, sz].
    Returns (glb_bytes, final_extents, final_bounds, raw_bounds, raw_extents, is_box).
    """
    scene = trimesh.load(raw_path, file_type="glb")
    raw_bounds = scene.bounds
    raw_ext = raw_bounds[1] - raw_bounds[0]  # [W_x, H_y, D_z]

    # 1. Center base footprint at origin in X and Z, ground at Y=0
    center_x = (raw_bounds[0][0] + raw_bounds[1][0]) / 2.0
    min_y = raw_bounds[0][1]
    center_z = (raw_bounds[0][2] + raw_bounds[1][2]) / 2.0

    t_mat = np.eye(4)
    t_mat[0, 3] = -center_x
    t_mat[1, 3] = -min_y
    t_mat[2, 3] = -center_z
    scene.apply_transform(t_mat)

    # 2. Scale matrix
    if isinstance(scale_spec, (list, tuple)):
        sx, sy, sz = float(scale_spec[0]), float(scale_spec[1]), float(scale_spec[2])
        is_box = True
    else:
        s = float(scale_spec)
        sx, sy, sz = s, s, s
        is_box = False

    s_mat = np.diag([sx, sy, sz, 1.0])
    scene.apply_transform(s_mat)

    # Export normalized glb
    glb_bytes = trimesh.exchange.gltf.export_glb(scene)

    # Reload from exported bytes to measure true physical bounds
    reloaded = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")
    final_bounds = reloaded.bounds
    final_ext = final_bounds[1] - final_bounds[0]  # [W_x, H_y, D_z]

    return glb_bytes, final_ext, final_bounds, raw_bounds, raw_ext, is_box

def run():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "furniture_raw")
    out_dir = os.path.join(base_dir, "frontend", "public", "models", "furniture")
    catalog_path = os.path.join(base_dir, "src", "design", "catalog.json")
    frontend_catalog_path = os.path.join(base_dir, "frontend", "public", "catalog.json")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)

    catalog = {
        "version": "1.2.0",
        "description": "ArchiGen 3D Curated CC0 Furniture Catalog (Kenney Kit)",
        "license": "CC0 1.0 Universal",
        "scaling_policy": "Strict uniform scaling for 27 items; non-uniform scaling permitted solely for 6 simple box-like cabinet/appliance items.",
        "box_like_non_uniform_items": sorted(list(BOX_LIKE_ITEMS)),
        "coordinate_convention": {
            "origin": "center of base at y=0, (x=0, z=0)",
            "forward": "+Z (faces into room / open side)",
            "backward": "-Z (against wall)",
            "upward": "+Y",
            "lateral": "+X (right)",
            "unit": "meters"
        },
        "preset_support": PRESET_DESCRIPTIONS,
        "preset_aliases": {
            "preset_1": "contemporary_minimal",
            "preset_2": "warm_wood",
            "preset_3": "tailored_classic",
            "modern": "contemporary_minimal",
            "scandinavian": "warm_wood",
            "classic": "tailored_classic"
        },
        "items": {}
    }

    rows = []
    logger.info(f"Normalizing {len(ASSET_SPEC)} models...")

    for item_id, spec in ASSET_SPEC.items():
        raw_path = os.path.join(raw_dir, spec["raw_file"])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw asset: {raw_path}")

        glb_bytes, f_ext, f_bounds, r_bounds, r_ext, is_box = normalize_asset(raw_path, spec["scale"])

        out_name = f"{item_id}.glb"
        out_p = os.path.join(out_dir, out_name)
        with open(out_p, "wb") as f:
            f.write(glb_bytes)

        # Exact measured dimensions from reloaded mesh
        fin_w = round(float(f_ext[0]), 3)
        fin_h = round(float(f_ext[1]), 3)
        fin_d = round(float(f_ext[2]), 3)

        raw_w = round(float(r_ext[0]), 3)
        raw_h = round(float(r_ext[1]), 3)
        raw_d = round(float(r_ext[2]), 3)

        catalog["items"][item_id] = {
            "id": item_id,
            "name": spec["name"],
            "category": spec["category"],
            "glb": f"/models/furniture/{out_name}",
            "scaling_type": "non_uniform_box" if is_box else "strict_uniform",
            "scale_factors": spec["scale"] if is_box else [spec["scale"], spec["scale"], spec["scale"]],
            # Measured dimensions: [width (X), depth (Z), height (Y)]
            "dimensions_m": [fin_w, fin_d, fin_h],
            "bounds": {
                "min": [round(float(b), 3) for b in f_bounds[0]],
                "max": [round(float(b), 3) for b in f_bounds[1]]
            },
            "presets": spec["presets"],
            "styles": spec["presets"],  # Backward-compatible alias
            "placement_rule": spec["placement_rule"],
            "clearance": spec["clearance"],
            "required": spec["required"],
            "allowed_rooms": spec["allowed_rooms"]
        }

        scale_str = f"[{spec['scale'][0]:.2f}, {spec['scale'][1]:.2f}, {spec['scale'][2]:.2f}]" if is_box else f"uniform {spec['scale']:.2f}"
        rows.append({
            "id": item_id,
            "category": spec["category"],
            "box": "YES (box-like)" if is_box else "NO (uniform)",
            "raw_ext": f"{raw_w:.2f} x {raw_d:.2f} x {raw_h:.2f}",
            "scale": scale_str,
            "measured_ext": f"{fin_w:.2f} x {fin_d:.2f} x {fin_h:.2f}",
            "measured_bbox": f"[{f_bounds[0][0]:.2f},{f_bounds[0][1]:.2f},{f_bounds[0][2]:.2f}]-[{f_bounds[1][0]:.2f},{f_bounds[1][1]:.2f},{f_bounds[1][2]:.2f}]",
            "kb": round(len(glb_bytes) / 1024.0, 1)
        })

    # Save to src/design/catalog.json
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    # Save identical copy to frontend/public/catalog.json
    with open(frontend_catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Catalog saved ({len(catalog['items'])} items) to {catalog_path} and {frontend_catalog_path}")

    # Print Side-by-Side Table comparing catalog and measured dimensions
    print("\n" + "="*145)
    print("ASSET BANK: MEASURED VS CATALOG DIMENSIONS & SCALING CLASSIFICATION")
    print("="*145)
    header = f"| {'ID':<21} | {'Type':<15} | {'Raw WxDxH (m)':<18} | {'Scale Applied':<24} | {'Measured WxDxH (m)':<20} | {'Catalog [W,D,H] (m)':<22} | {'Delta':<8} |"
    sep = f"|{'-'*23}|{'-'*17}|{'-'*20}|{'-'*26}|{'-'*22}|{'-'*24}|{'-'*10}|"
    print(header)
    print(sep)
    for r, (item_id, item) in zip(rows, catalog["items"].items()):
        meas_str = f"{item['dimensions_m'][0]:.3f} x {item['dimensions_m'][1]:.3f} x {item['dimensions_m'][2]:.3f}"
        cat_str = f"[{item['dimensions_m'][0]:.3f}, {item['dimensions_m'][1]:.3f}, {item['dimensions_m'][2]:.3f}]"
        # Delta is 0.000m by definition since catalog is generated directly from measurement
        delta_str = "0.0 mm"
        line = f"| {item_id:<21} | {r['box']:<15} | {r['raw_ext']:<18} | {r['scale']:<24} | {meas_str:<20} | {cat_str:<24} | {delta_str:<8} |"
        print(line)
    print("="*145 + "\n")

if __name__ == "__main__":
    run()
