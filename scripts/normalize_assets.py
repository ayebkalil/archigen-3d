#!/usr/bin/env python3
"""
scripts/normalize_assets.py
---------------------------
Normalizes raw CC0 Kenney 3D furniture models into metric-scale GLB assets.
- Strictly UNIFORM scaling (sx = sy = sz = s) to preserve geometry proportions.
- Centers footprint at base origin (X=0, Z=0) with ground contact at Y=0.
- Preserves canonical orientation: front faces +Z (into room), back faces -Z (against wall).
- Computes raw bbox, per-axis scale factors vs reference standards, flags >10% discrepancies.
- Exports clean GLBs to frontend/public/models/furniture/.
- Generates src/design/catalog.json with style variants for key categories.
"""

import json
import logging
import os
import sys
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base asset specifications:
# ref_dims_m: [width_x, depth_z, height_y] in real-world metric meters for aspect ratio comparison
# uniform_s: uniform scale factor applied to raw model (sx = sy = sz = uniform_s)
ASSET_SPEC = {
    # Bedroom
    "bed_double": {
        "raw_file": "bedDouble.glb",
        "name": "Double Bed",
        "category": "bed",
        "ref_dims_m": [1.60, 2.05, 0.90],
        "uniform_s": 1.67,  # 0.96m * 1.67 = 1.60m width
        "styles": ["modern", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.60, "back": 0.00},
        "required": True,
        "allowed_rooms": ["bedroom"],
    },
    "bed_single": {
        "raw_file": "bedSingle.glb",
        "name": "Single Bed",
        "category": "bed_single",
        "ref_dims_m": [0.95, 2.00, 0.85],
        "uniform_s": 1.67,  # 0.57m * 1.67 = 0.95m width
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.50, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },
    "nightstand": {
        "raw_file": "cabinetBedDrawerTable.glb",
        "name": "Nightstand",
        "category": "nightstand",
        "ref_dims_m": [0.50, 0.45, 0.55],
        "uniform_s": 1.90,  # 0.27m * 1.90 = 0.51m width
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "adjacent_anchor",
        "clearance": {"front": 0.50, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },
    "wardrobe": {
        "raw_file": "bookcaseClosedDoors.glb",
        "name": "Wardrobe Closet",
        "category": "wardrobe",
        "ref_dims_m": [1.00, 0.60, 2.00],
        "uniform_s": 2.45,  # 0.40m * 2.45 = 0.98m width, 2.08m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bedroom"],
    },

    # Living Room - Sofas & Seating
    "sofa_3seater": {
        "raw_file": "loungeSofa.glb",  # Standard 3-seater sofa (clean 2:1 proportions)
        "name": "3-Seater Sofa (Standard)",
        "category": "sofa",
        "ref_dims_m": [2.00, 0.85, 0.85],
        "uniform_s": 2.04,  # 0.98m * 2.04 = 2.00m width, 0.84m depth, 0.94m height
        "styles": ["scandinavian"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "sofa_modern": {
        "raw_file": "loungeDesignSofa.glb",
        "name": "Modern Designer Sofa",
        "category": "sofa",
        "ref_dims_m": [2.20, 0.85, 0.80],
        "uniform_s": 1.96,  # 1.12m * 1.96 = 2.20m width, 0.80m depth
        "styles": ["modern"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "sofa_classic": {
        "raw_file": "loungeSofaLong.glb",  # Sectional sofa with chaise
        "name": "Classic Sectional Sofa",
        "category": "sofa",
        "ref_dims_m": [2.10, 1.70, 0.90],
        "uniform_s": 2.14,  # 0.98m * 2.14 = 2.10m width, 1.75m depth
        "styles": ["classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.30, "back": 0.00},
        "required": True,
        "allowed_rooms": ["living"],
    },
    "armchair": {
        "raw_file": "loungeDesignChair.glb",
        "name": "Modern Armchair",
        "category": "armchair",
        "ref_dims_m": [0.85, 0.80, 0.80],
        "uniform_s": 1.50,  # 0.73m * 1.50 = 1.10m width
        "styles": ["modern"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.60, "sides": 0.20, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "armchair_classic": {
        "raw_file": "loungeChair.glb",
        "name": "Classic Club Armchair",
        "category": "armchair",
        "ref_dims_m": [0.85, 0.85, 0.85],
        "uniform_s": 1.80,
        "styles": ["classic"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.60, "sides": 0.20, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "armchair_scandi": {
        "raw_file": "chairModernFrameCushion.glb",
        "name": "Scandinavian Frame Chair",
        "category": "armchair",
        "ref_dims_m": [0.65, 0.65, 0.85],
        "uniform_s": 2.50,
        "styles": ["scandinavian"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.50, "sides": 0.15, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living"],
    },

    # Living Room - Tables & Units
    "coffee_table": {
        "raw_file": "tableCoffee.glb",
        "name": "Wood Coffee Table",
        "category": "coffee_table",
        "ref_dims_m": [1.10, 0.60, 0.45],
        "uniform_s": 1.67,  # 0.66m * 1.67 = 1.10m width, 0.67m depth, 0.38m height
        "styles": ["scandinavian"],
        "placement_rule": "in_front_of_anchor",
        "clearance": {"front": 0.40, "sides": 0.40, "back": 0.40},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "coffee_table_glass": {
        "raw_file": "tableCoffeeGlassSquare.glb",
        "name": "Modern Glass Coffee Table",
        "category": "coffee_table",
        "ref_dims_m": [0.90, 0.90, 0.45],
        "uniform_s": 1.70,
        "styles": ["modern"],
        "placement_rule": "in_front_of_anchor",
        "clearance": {"front": 0.40, "sides": 0.40, "back": 0.40},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "tv_unit": {
        "raw_file": "cabinetTelevision.glb",
        "name": "Media TV Unit",
        "category": "tv_unit",
        "ref_dims_m": [1.60, 0.45, 0.50],
        "uniform_s": 2.00,  # 0.80m * 2.00 = 1.60m width, 0.50m depth, 0.62m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 1.00, "sides": 0.10, "back": 0.00},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "rug_living": {
        "raw_file": "rugRectangle.glb",
        "name": "Living Area Rug",
        "category": "rug",
        "ref_dims_m": [2.40, 1.60, 0.02],
        "uniform_s": 1.60,  # 1.57m * 1.60 = 2.51m width, 1.47m depth
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "under_anchor",
        "clearance": {"front": 0.00, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["living"],
    },
    "potted_plant": {
        "raw_file": "pottedPlant.glb",
        "name": "Indoor Plant",
        "category": "plant",
        "ref_dims_m": [0.50, 0.50, 1.20],
        "uniform_s": 2.20,
        "styles": ["modern", "scandinavian", "classic"],
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
        "ref_dims_m": [1.60, 0.90, 0.76],
        "uniform_s": 1.95,  # 0.84m * 1.95 = 1.64m width, 0.88m depth, 0.64m height
        "styles": ["modern"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_table_round": {
        "raw_file": "tableRound.glb",
        "name": "Round Dining Table",
        "category": "dining_table",
        "ref_dims_m": [1.20, 1.20, 0.76],
        "uniform_s": 1.80,
        "styles": ["scandinavian"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_table_cloth": {
        "raw_file": "tableCloth.glb",
        "name": "Classic Dining Table (Cloth)",
        "category": "dining_table",
        "ref_dims_m": [1.60, 0.90, 0.76],
        "uniform_s": 1.95,
        "styles": ["classic"],
        "placement_rule": "center",
        "clearance": {"front": 0.80, "sides": 0.80, "back": 0.80},
        "required": True,
        "allowed_rooms": ["dining"],
    },
    "dining_chair": {
        "raw_file": "chairModernCushion.glb",
        "name": "Modern Dining Chair",
        "category": "dining_chair",
        "ref_dims_m": [0.50, 0.50, 0.85],
        "uniform_s": 2.30,  # 0.20m * 2.30 = 0.46m width, 0.46m depth, 1.06m height
        "styles": ["modern"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "dining_chair_scandi": {
        "raw_file": "chairRounded.glb",
        "name": "Scandinavian Dining Chair",
        "category": "dining_chair",
        "ref_dims_m": [0.48, 0.50, 0.85],
        "uniform_s": 2.20,
        "styles": ["scandinavian"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "dining_chair_classic": {
        "raw_file": "chair.glb",
        "name": "Classic Dining Chair",
        "category": "dining_chair",
        "ref_dims_m": [0.48, 0.50, 0.85],
        "uniform_s": 2.20,
        "styles": ["classic"],
        "placement_rule": "around_anchor",
        "clearance": {"front": 0.40, "sides": 0.10, "back": 0.40},
        "required": False,
        "allowed_rooms": ["dining"],
    },
    "sideboard": {
        "raw_file": "bookcaseClosedWide.glb",
        "name": "Dining Sideboard",
        "category": "sideboard",
        "ref_dims_m": [1.50, 0.45, 0.85],
        "uniform_s": 1.85,  # 0.80m * 1.85 = 1.48m width, 0.46m depth, 1.46m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.80, "sides": 0.10, "back": 0.00},
        "required": False,
        "allowed_rooms": ["dining"],
    },

    # Office
    "desk": {
        "raw_file": "desk.glb",
        "name": "Office Desk",
        "category": "desk",
        "ref_dims_m": [1.40, 0.70, 0.75],
        "uniform_s": 1.85,  # 0.73m * 1.85 = 1.35m width, 0.72m depth, 0.71m height
        "styles": ["modern", "scandinavian"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.20, "back": 0.00},
        "required": True,
        "allowed_rooms": ["office"],
    },
    "desk_corner": {
        "raw_file": "deskCorner.glb",
        "name": "Executive Corner Desk",
        "category": "desk",
        "ref_dims_m": [1.60, 1.60, 0.75],
        "uniform_s": 1.65,
        "styles": ["classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.20, "back": 0.00},
        "required": True,
        "allowed_rooms": ["office"],
    },
    "office_chair": {
        "raw_file": "chairDesk.glb",
        "name": "Desk Swivel Chair",
        "category": "office_chair",
        "ref_dims_m": [0.62, 0.62, 0.95],
        "uniform_s": 1.80,  # 0.33m * 1.80 = 0.59m width, 0.56m depth, 1.10m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "facing_anchor",
        "clearance": {"front": 0.20, "sides": 0.20, "back": 0.60},
        "required": False,
        "allowed_rooms": ["office"],
    },
    "bookcase": {
        "raw_file": "bookcaseOpen.glb",
        "name": "Open Bookcase",
        "category": "bookcase",
        "ref_dims_m": [0.85, 0.35, 1.85],
        "uniform_s": 2.10,  # 0.40m * 2.10 = 0.84m width, 0.52m depth, 1.85m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.60, "sides": 0.05, "back": 0.00},
        "required": False,
        "allowed_rooms": ["office", "living"],
    },
    "floor_lamp": {
        "raw_file": "lampRoundFloor.glb",
        "name": "Floor Reading Lamp",
        "category": "lamp",
        "ref_dims_m": [0.40, 0.40, 1.50],
        "uniform_s": 2.00,
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "corner",
        "clearance": {"front": 0.10, "sides": 0.10, "back": 0.10},
        "required": False,
        "allowed_rooms": ["living", "office"],
    },

    # Kitchen
    "kitchen_counter": {
        "raw_file": "kitchenCabinet.glb",
        "name": "Kitchen Base Cabinet",
        "category": "kitchen_counter",
        "ref_dims_m": [0.90, 0.65, 0.90],
        "uniform_s": 2.00,  # 0.43m * 2.00 = 0.86m width, 0.90m depth, 0.90m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.00, "back": 0.00},
        "required": True,
        "allowed_rooms": ["kitchen"],
    },
    "kitchen_sink_unit": {
        "raw_file": "kitchenSink.glb",
        "name": "Kitchen Sink Unit",
        "category": "kitchen_sink",
        "ref_dims_m": [0.90, 0.65, 0.90],
        "uniform_s": 2.00,
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.90, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["kitchen"],
    },
    "refrigerator": {
        "raw_file": "kitchenFridgeLarge.glb",
        "name": "Tall Refrigerator",
        "category": "refrigerator",
        "ref_dims_m": [0.75, 0.70, 1.85],
        "uniform_s": 1.70,  # 0.52m * 1.70 = 0.88m width, 0.70m depth, 1.56m height
        "styles": ["modern", "scandinavian", "classic"],
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
        "ref_dims_m": [0.45, 0.70, 0.80],
        "uniform_s": 1.50,  # 0.31m * 1.50 = 0.47m width, 0.72m depth, 0.68m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.60, "sides": 0.25, "back": 0.00},
        "required": True,
        "allowed_rooms": ["bathroom"],
    },
    "bathroom_vanity": {
        "raw_file": "bathroomSink.glb",
        "name": "Bathroom Vanity Sink",
        "category": "bathroom_sink",
        "ref_dims_m": [0.70, 0.50, 0.85],
        "uniform_s": 1.80,  # 0.34m * 1.80 = 0.61m width, 0.52m depth, 1.01m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "back_to_wall",
        "clearance": {"front": 0.70, "sides": 0.15, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bathroom"],
    },
    "bathtub": {
        "raw_file": "bathtub.glb",
        "name": "Full Bathtub",
        "category": "bathtub",
        "ref_dims_m": [1.70, 0.75, 0.55],
        "uniform_s": 1.40,  # 1.19m * 1.40 = 1.67m width, 0.78m depth, 0.59m height
        "styles": ["modern", "scandinavian", "classic"],
        "placement_rule": "corner_or_wall",
        "clearance": {"front": 0.70, "sides": 0.00, "back": 0.00},
        "required": False,
        "allowed_rooms": ["bathroom"],
    },
}

def normalize_uniform(raw_path: str, uniform_scale: float) -> tuple:
    """
    Applies strict uniform scaling (sx = sy = sz = uniform_scale),
    centers base footprint at (X=0, Z=0), and places ground base at Y=0.
    Returns (glb_bytes, final_extents, final_bounds, raw_bounds, raw_extents).
    """
    scene = trimesh.load(raw_path, file_type="glb")
    raw_bounds = scene.bounds
    raw_ext = raw_bounds[1] - raw_bounds[0] # [W_x, H_y, D_z]
    
    # 1. Center base footprint at origin in X and Z, ground at Y=0
    center_x = (raw_bounds[0][0] + raw_bounds[1][0]) / 2.0
    min_y = raw_bounds[0][1]
    center_z = (raw_bounds[0][2] + raw_bounds[1][2]) / 2.0
    
    t_mat = np.eye(4)
    t_mat[0, 3] = -center_x
    t_mat[1, 3] = -min_y
    t_mat[2, 3] = -center_z
    scene.apply_transform(t_mat)
    
    # 2. Strict UNIFORM scale matrix
    s = float(uniform_scale)
    s_mat = np.diag([s, s, s, 1.0])
    scene.apply_transform(s_mat)
    
    # Export normalized glb
    glb_bytes = trimesh.exchange.gltf.export_glb(scene)
    
    # Re-verify exported data by reloading
    reloaded = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")
    final_bounds = reloaded.bounds
    final_ext = final_bounds[1] - final_bounds[0] # [W_x, H_y, D_z]
    
    return glb_bytes, final_ext, final_bounds, raw_bounds, raw_ext

def run():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "furniture_raw")
    out_dir = os.path.join(base_dir, "frontend", "public", "models", "furniture")
    catalog_path = os.path.join(base_dir, "src", "design", "catalog.json")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    
    catalog = {
        "version": "1.1.0",
        "description": "ArchiGen 3D Curated CC0 Furniture Catalog (Kenney Kit)",
        "license": "CC0 1.0 Universal",
        "scaling_method": "strict_uniform (sx = sy = sz, zero mesh distortion)",
        "coordinate_convention": {
            "origin": "center of base at y=0, (x=0, z=0)",
            "forward": "+Z (faces into room / open side)",
            "backward": "-Z (against wall)",
            "upward": "+Y",
            "lateral": "+X (right)",
            "unit": "meters"
        },
        "style_support": {
            "modern": "sleek minimalist models & contemporary finishes",
            "scandinavian": "light wood, organic curved forms & textiles",
            "classic": "formal framed furniture & draped upholstery"
        },
        "items": {}
    }
    
    rows = []
    logger.info(f"Normalizing {len(ASSET_SPEC)} models with STRICT UNIFORM SCALING...")
    
    for item_id, spec in ASSET_SPEC.items():
        raw_path = os.path.join(raw_dir, spec["raw_file"])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw asset: {raw_path}")
            
        s = spec["uniform_s"]
        glb_bytes, f_ext, f_bounds, r_bounds, r_ext = normalize_uniform(raw_path, s)
        
        # Save to frontend public
        out_name = f"{item_id}.glb"
        out_p = os.path.join(out_dir, out_name)
        with open(out_p, "wb") as f:
            f.write(glb_bytes)
            
        raw_w, raw_h, raw_d = float(r_ext[0]), float(r_ext[1]), float(r_ext[2])
        fin_w, fin_h, fin_d = float(f_ext[0]), float(f_ext[1]), float(f_ext[2])
        
        # Compute ideal per-axis factors vs reference standard
        ref_w, ref_d, ref_h = spec["ref_dims_m"]
        fx = ref_w / raw_w
        fz = ref_d / raw_d
        fy = ref_h / raw_h
        
        factors = [fx, fz, fy]
        f_min, f_max = min(factors), max(factors)
        f_mean = np.mean(factors)
        diff_pct = (f_max - f_min) / f_mean * 100.0
        flag = diff_pct > 10.0
        
        catalog["items"][item_id] = {
            "id": item_id,
            "name": spec["name"],
            "category": spec["category"],
            "glb": f"/models/furniture/{out_name}",
            "uniform_scale": s,
            "dimensions_m": [round(fin_w, 3), round(fin_d, 3), round(fin_h, 3)],
            "bounds": {
                "min": [round(float(b), 3) for b in f_bounds[0]],
                "max": [round(float(b), 3) for b in f_bounds[1]]
            },
            "styles": spec["styles"],
            "placement_rule": spec["placement_rule"],
            "clearance": spec["clearance"],
            "required": spec["required"],
            "allowed_rooms": spec["allowed_rooms"]
        }
        
        rows.append({
            "id": item_id,
            "category": spec["category"],
            "raw_bbox": f"[{r_bounds[0][0]:.2f},{r_bounds[0][1]:.2f},{r_bounds[0][2]:.2f}]-[{r_bounds[1][0]:.2f},{r_bounds[1][1]:.2f},{r_bounds[1][2]:.2f}]",
            "raw_ext": f"{raw_w:.2f}x{raw_d:.2f}x{raw_h:.2f}",
            "axis_factors": f"X:{fx:.2f} Z:{fz:.2f} Y:{fy:.2f}",
            "diff_pct": diff_pct,
            "flag": "FLAG (>10%)" if flag else "OK (<=10%)",
            "s": s,
            "final_bbox": f"[{f_bounds[0][0]:.2f},{f_bounds[0][1]:.2f},{f_bounds[0][2]:.2f}]-[{f_bounds[1][0]:.2f},{f_bounds[1][1]:.2f},{f_bounds[1][2]:.2f}]",
            "final_ext": f"{fin_w:.2f}x{fin_d:.2f}x{fin_h:.2f}m",
            "kb": round(len(glb_bytes) / 1024.0, 1)
        })
        
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    logger.info(f"Catalog saved ({len(catalog['items'])} items) to {catalog_path}")
    
    # Print formatted detailed table for item 1
    print("\n" + "="*145)
    print("ITEM 1: ASSET BANK NORMALIZATION — RAW BBOX, PER-AXIS SCALE FACTORS, UNIFORM SCALE, FINAL BBOX")
    print("="*145)
    header = f"| {'ID':<21} | {'Raw Extents (W x D x H)':<24} | {'Per-Axis Factors':<22} | {'Diff %':<8} | {'Status':<12} | {'Scale s':<7} | {'Final Extents (m)':<20} | {'Final BBox [min] - [max]':<20} |"
    sep = f"|{'-'*23}|{'-'*26}|{'-'*24}|{'-'*10}|{'-'*14}|{'-'*9}|{'-'*22}|{'-'*28}|"
    print(header)
    print(sep)
    for r in rows:
        line = f"| {r['id']:<21} | {r['raw_ext']:<24} | {r['axis_factors']:<22} | {r['diff_pct']:>6.1f}% | {r['flag']:<12} | {r['s']:<7.2f} | {r['final_ext']:<20} | {r['final_bbox']:<26} |"
        print(line)
    print("="*145 + "\n")

if __name__ == "__main__":
    run()
