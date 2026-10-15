"""
ArchiGen 3D - Standalone 2D Floorplan to 3D Model Generator
==========================================================
End-to-end pipeline:
1. Ingests any 2D architectural blueprint (PNG/JPEG).
2. Runs the retrained Multi-Task UNet++ (ResNet-34) to segment:
   - Walls (internal/external), doors, windows (5-class structural head)
   - Semantic room types (12-class room head)
3. Vectorizes binary masks to clean rectilinear polygons with Shapely.
4. Identifies exterior walls and synthesizes CMP semantic elevation sketches.
5. Runs the Pix2Pix cGAN generator (ONNX) to produce photorealistic facade textures.
6. Extrudes 3D walls and exports:
   - Wavefront .OBJ 3D Mesh
   - Accompanying .MTL Material Library with UV texture mappings
   - Photorealistic Facade Texture PNG
   - Three.js Interactive Scene JSON
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer
from src.floorplan.extruder_3d import Floorplan3DExtruder
from src.bridge.facade_bridge import FacadeBridge

def run_floorplan_to_3d(
    input_image_path: str,
    output_dir: str = "outputs/3d_model",
    num_floors: int = 1,
    wall_height_m: float = 2.80,
    model_checkpoint: str = "models/saved/floorplan_unetpp/multitask_resnet34_best.pth",
    facade_onnx_path: str = "models/saved/generator_perceptual.onnx"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f"      ArchiGen 3D : 2D Blueprint -> 3D Mesh       ")
    print(f"==================================================")
    print(f"[*] Input floorplan: {input_image_path}")
    print(f"[*] Output directory: {out_path}")
    print(f"[*] Device: {device}")

    # 1. Load MultiTask UNet++ Model
    print(f"[*] Loading trained MultiTask UNet++ from {model_checkpoint}...")
    model = MultiTaskFloorplanUNet(
        encoder_name="resnet34",
        encoder_weights=None,
        num_struct_classes=5,
        num_room_classes=12
    ).to(device)

    ckpt = torch.load(model_checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 2. Preprocess Input Image (Aspect-ratio preserving letterbox)
    raw_img = Image.open(input_image_path).convert("RGB")
    iw, ih = raw_img.size
    size = 512
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2

    resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, (pad_x, pad_y))

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    tensor = TF.to_tensor(canvas).to(device)
    tensor = ((tensor - mean) / std).unsqueeze(0)

    # 3. Deep Learning Multi-Task Inference
    print(f"[*] Running multi-task segmentation inference...")
    with torch.no_grad():
        outputs = model(tensor)
        struct_pred = torch.argmax(outputs["struct"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        room_pred = torch.argmax(outputs["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Map 5-class structure (bg, wall_int, wall_ext, door, win) to 4-class for vectorizer
    mask_4class = np.zeros_like(struct_pred)
    mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1  # Walls
    mask_4class[struct_pred == 3] = 2                         # Doors
    mask_4class[struct_pred == 4] = 3                         # Windows

    # 4. Geometric Vectorization
    print(f"[*] Vectorizing discrete masks to rectilinear polygons...")
    vectorizer = FloorplanVectorizer()
    layout = vectorizer.vectorize(
        mask=mask_4class,
        struct_mask=struct_pred,
        room_mask=room_pred,
        scale_override_m_per_px=0.03
    )

    walls = layout.get("walls", [])
    rooms = layout.get("rooms", [])
    doors = layout.get("doors", [])
    windows = layout.get("windows", [])
    metadata = layout.get("metadata", {})

    print(f"    -> Extracted {len(walls)} walls ({sum(1 for w in walls if w.get('is_exterior'))} exterior)")
    print(f"    -> Extracted {len(rooms)} semantic rooms (Total area: {metadata.get('total_area_m2', 0):.1f} m²)")
    for r in rooms:
        print(f"       • {r['room_type']}: {r['surface_m2']:.1f} m²")
    print(f"    -> Extracted {len(doors)} doors and {len(windows)} windows")

    # 5. Facade Bridge & Photorealistic Texture Synthesis
    print(f"[*] Running Facade Bridge with Pix2Pix generator...")
    bridge = FacadeBridge(onnx_model_path=facade_onnx_path)
    facade_meta = bridge.process_layout_facades(layout=layout, num_floors=num_floors)
    front_wall_id = facade_meta.get("front_wall_id")

    texture_rel_path = None
    # Save the synthesized facade texture if available
    for wall in walls:
        if wall.get("facade_texture") and "texture_base64" in wall["facade_texture"]:
            b64_str = wall["facade_texture"]["texture_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            import base64
            img_bytes = base64.b64decode(b64_str)
            tex_file = out_path / "facade_front_texture.png"
            with open(tex_file, "wb") as f_tex:
                f_tex.write(img_bytes)
            texture_rel_path = tex_file.name
            print(f"    -> Saved synthesized facade texture: {tex_file}")
            break

    # 6. Procedural 3D Mesh Extrusion
    print(f"[*] Extruding 3D architecture...")
    extruder = Floorplan3DExtruder(wall_height_m=wall_height_m)
    layout["metadata"]["num_floors"] = num_floors

    # 6A. Export Three.js Scene JSON
    threejs_scene = extruder.generate_threejs_scene_data(layout)
    scene_json_path = out_path / "threejs_scene.json"
    import json
    with open(scene_json_path, "w", encoding="utf-8") as f_json:
        json.dump(threejs_scene, f_json, indent=2)
    print(f"    -> Saved Three.js Scene Data: {scene_json_path}")

    # 6B. Export Wavefront .OBJ + .MTL
    obj_path = out_path / "building_model.obj"
    extruder.export_obj_with_mtl(
        layout=layout,
        output_obj_path=str(obj_path),
        texture_relative_path=texture_rel_path
    )

    print(f"\n[COMPLETE SUCCESS] 3D Model Generation Complete!")
    print(f"Files ready for Blender, Three.js, or WebGL viewer in: {out_path.resolve()}\n")

    return {
        "obj_path": str(obj_path),
        "scene_json_path": str(scene_json_path),
        "total_rooms": len(rooms),
        "total_area_m2": metadata.get("total_area_m2", 0.0)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D building mesh from 2D floorplan blueprint.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input 2D floorplan image.")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/3d_model", help="Directory to save 3D files.")
    parser.add_argument("--floors", "-f", type=int, default=1, help="Number of building floors (default: 1).")
    args = parser.parse_args()

    # Find a default test plan if no input provided
    input_img = args.input
    if input_img is None:
        cubicasa_test_plans = list(Path("data/CubiCasa5k").rglob("F1_scaled.png"))
        if cubicasa_test_plans:
            input_img = str(cubicasa_test_plans[10])
            print(f"[NOTE] No --input specified, using sample plan: {input_img}")
        else:
            raise FileNotFoundError("Please provide a floorplan image path using --input <path_to_image>.")

    run_floorplan_to_3d(
        input_image_path=input_img,
        output_dir=args.output_dir,
        num_floors=args.floors
    )
