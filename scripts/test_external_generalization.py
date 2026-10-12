"""
External Floorplan Generalization Evaluation Suite.
Tests the production FloorplanPipeline on 10 diverse real-world floorplans outside CubiCasa5k.
Outputs:
- Metric summary table (rooms detected, surface area, shell validity, door/window detection).
- Detailed failure mode diagnostic for each test category.
- Visual side-by-side prediction gallery saved to models/samples/external_generalization/.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.app.floorplan_engine import FloorplanPipeline

def run_external_benchmark():
    input_dir = Path("data/external_test_plans")
    output_dir = Path("models/samples/external_generalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("      ARCHIGEN 3D: 10-PLAN EXTERNAL GENERALIZATION BENCHMARK")
    print("=" * 80)

    pipeline = FloorplanPipeline.get_instance()
    
    plan_files = sorted(list(input_dir.glob("*.png")))
    if not plan_files:
        print("[ERROR] No external test plans found!")
        return

    results = []

    # Palette for room semantic overlays (RGB)
    ROOM_COLORS = {
        "LivingRoom": (255, 179, 186),
        "Bedroom": (186, 225, 255),
        "Kitchen": (255, 255, 186),
        "Bathroom": (186, 255, 201),
        "Entry": (226, 204, 240),
        "Storage": (240, 230, 210),
        "Balcony": (200, 240, 240),
        "Other": (220, 220, 220),
        "Undefined": (200, 200, 200)
    }

    for idx, p_path in enumerate(plan_files, 1):
        name = p_path.stem
        print(f"\n[{idx}/10] Testing: {name}...")

        with open(p_path, "rb") as f:
            img_bytes = f.read()

        # Run pipeline
        try:
            res = pipeline.process_image(img_bytes)
            layout = res["layout"]
            meta = layout.get("metadata", {})
            walls = layout.get("walls", [])
            rooms = layout.get("rooms", [])
            doors = layout.get("doors", [])
            windows = layout.get("windows", [])
            threejs = res.get("threejs_scene", {})

            # Key metrics
            num_walls = len(walls)
            ext_walls = sum(1 for w in walls if w.get("is_exterior", False))
            num_doors = len(doors)
            num_windows = len(windows)
            num_rooms = len(rooms)
            total_m2 = meta.get("total_area_m2", meta.get("total_surface_m2", 0.0))
            scale_m = meta.get("scale_m_per_px", 0.03)
            shell_valid = meta.get("is_building_shell_valid", False)
            room_types = [r.get("room_type", "Unknown") for r in rooms]
            unique_types = list(set(room_types))

            # Failure diagnosis
            diagnostics = []
            if shell_valid:
                diagnostics.append("Watertight Shell: Valid")
            else:
                diagnostics.append("Watertight Shell: Incomplete/Open boundary")

            if num_rooms == 0:
                diagnostics.append("CRITICAL: Zero functional rooms enclosed")
            elif num_rooms > 12:
                diagnostics.append("Over-segmentation: Wall fragmentation into spurious cells")

            if num_doors == 0 and "cad" in name:
                diagnostics.append("Door opening missed: Swing arc unrecognized as gap")

            if total_m2 < 10.0 or total_m2 > 500.0:
                diagnostics.append(f"Scale anomaly: Predicted {total_m2:.1f} m2 deviates from residential norms")

            diag_str = "; ".join(diagnostics)

            # Record
            record = {
                "id": idx,
                "name": name,
                "walls": num_walls,
                "ext_walls": ext_walls,
                "doors": num_doors,
                "windows": num_windows,
                "rooms": num_rooms,
                "room_types": ", ".join(unique_types) if unique_types else "None",
                "area_m2": round(total_m2, 1),
                "scale": round(scale_m, 4),
                "shell_valid": "Yes" if shell_valid else "No",
                "diagnostic": diag_str
            }
            results.append(record)

            # Generate visual overlay image
            orig_img = cv2.imread(str(p_path))
            overlay = orig_img.copy()

            # Draw rooms
            for r in rooms:
                pts = r.get("polygon_meters", [])
                if len(pts) >= 3:
                    # Convert meters back to pixels for visualization
                    px_pts = (np.array(pts) / scale_m).astype(np.int32)
                    col = ROOM_COLORS.get(r.get("room_type"), (200, 200, 200))
                    # Fill semi-transparent
                    sub_overlay = overlay.copy()
                    cv2.fillPoly(sub_overlay, [px_pts], col)
                    cv2.addWeighted(sub_overlay, 0.45, overlay, 0.55, 0, overlay)
                    cv2.polylines(overlay, [px_pts], True, (80, 80, 80), 1)

            # Draw walls
            for w in walls:
                pts = w.get("polygon_meters", [])
                if len(pts) >= 2:
                    px_pts = (np.array(pts) / scale_m).astype(np.int32)
                    col = (0, 0, 220) if w.get("is_exterior", False) else (220, 0, 0)
                    cv2.polylines(overlay, [px_pts], True, col, 2)

            # Draw doors & windows
            for d in doors:
                pos = d.get("position", [0, 0])
                c_px = (int(pos[0] / scale_m), int(pos[1] / scale_m))
                cv2.circle(overlay, c_px, 6, (0, 200, 200), -1)

            for w in windows:
                pos = w.get("position", [0, 0])
                c_px = (int(pos[0] / scale_m), int(pos[1] / scale_m))
                cv2.circle(overlay, c_px, 6, (200, 200, 0), -1)

            # Header info strip
            status_text = f"{name} | Rooms: {num_rooms} | Area: {total_m2:.1f}m2 | Shell: {'VALID' if shell_valid else 'FAIL'}"
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (30, 30, 30), -1)
            cv2.putText(overlay, status_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Concatenate side by side: [Original | Overlay]
            side_by_side = np.hstack([orig_img, overlay])
            cv2.imwrite(str(output_dir / f"eval_{name}.png"), side_by_side)

        except Exception as e:
            print(f"  [ERROR processing {name}]: {e}")
            results.append({
                "id": idx,
                "name": name,
                "walls": 0, "ext_walls": 0, "doors": 0, "windows": 0, "rooms": 0,
                "room_types": "N/A", "area_m2": 0.0, "scale": 0.0, "shell_valid": "Error",
                "diagnostic": f"Exception: {str(e)}"
            })

    # Save JSON results
    with open(output_dir / "external_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print Markdown Summary Table
    print("\n" + "=" * 95)
    print("           EXTERNAL GENERALIZATION BENCHMARK SUMMARY (10 TEST CASES)")
    print("=" * 95)
    header = f"{'#':<3} | {'Test Case':<26} | {'Rooms':<5} | {'Area(m2)':<8} | {'Shell':<6} | {'Failure Mode / Behavioral Diagnostic'}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['id']:<3} | {r['name']:<26} | {r['rooms']:<5} | {r['area_m2']:<8} | {r['shell_valid']:<6} | {r['diagnostic']}")
    print("=" * 95 + "\n")

    return results

if __name__ == "__main__":
    run_external_benchmark()
