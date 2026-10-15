"""
Synthetic Stress Tests Floorplan Generalization Evaluation Suite.
Tests the production FloorplanPipeline on 10 synthetic stress test blueprints
spanning distinct architectural drafting and domain-shift challenges:
1. clean_bw_cad: Minimalist black and white CAD blueprint.
2. marketing_brochure: Real-estate style colored rooms with pastel fills.
3. hand_drawn_sketch: Rough pencil sketch with wobbly lines and textured noise.
4. furniture_cluttered: Floorplan densely populated with interior furniture.
5. dimension_text_annotated: Floorplan covered in dimension lines, arrows, and metric text.
6. low_res_screenshot: Blurry, heavily downscaled low-resolution thumbnail (140x140).
7. open_concept_loft: Modern undivided open layout with kitchen island.
8. narrow_railroad_layout: High aspect-ratio apartment with central hallway.
9. dark_mode_cad: Inverted scheme (black background, bright white lines).
10. diagonal_angled_walls: Non-Manhattan layout with 45-degree angled walls.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from backend.app.floorplan_engine import FloorplanPipeline

SYNTHETIC_STRESS_GROUND_TRUTH = {
    "01_clean_bw_cad": {
        "challenge": "Minimalist B&W CAD (thin 4px lines, swing arcs)",
        "gt_rooms": 4,
        "gt_doors": 2,
        "gt_windows": 2,
        "gt_area_m2": 136.9,
    },
    "02_marketing_brochure": {
        "challenge": "Real-Estate Brochure (colored pastel room fills)",
        "gt_rooms": 3,
        "gt_doors": 2,
        "gt_windows": 0,
        "gt_area_m2": 123.2,
    },
    "03_hand_drawn_sketch": {
        "challenge": "Hand-Drawn Sketch (wobbly pencil lines, texture noise)",
        "gt_rooms": 3,
        "gt_doors": 0,
        "gt_windows": 0,
        "gt_area_m2": 104.0,
    },
    "04_furniture_cluttered": {
        "challenge": "Cluttered Layout (interior furniture: bed, table, sofa)",
        "gt_rooms": 4,
        "gt_doors": 2,
        "gt_windows": 2,
        "gt_area_m2": 136.9,
    },
    "05_dimension_text_annotated": {
        "challenge": "Technical Annotation (dimension lines, arrows, text)",
        "gt_rooms": 4,
        "gt_doors": 2,
        "gt_windows": 2,
        "gt_area_m2": 136.9,
    },
    "06_low_res_screenshot": {
        "challenge": "Degraded Resolution (140px blurry downscale)",
        "gt_rooms": 4,
        "gt_doors": 2,
        "gt_windows": 2,
        "gt_area_m2": 136.9,
    },
    "07_open_concept_loft": {
        "challenge": "Open-Concept Loft (1 bath partition, open living space)",
        "gt_rooms": 2,
        "gt_doors": 1,
        "gt_windows": 1,
        "gt_area_m2": 151.3,
    },
    "08_narrow_railroad_layout": {
        "challenge": "Railroad Apartment (high aspect ratio, corridor partitions)",
        "gt_rooms": 5,
        "gt_doors": 2,
        "gt_windows": 0,
        "gt_area_m2": 104.5,
    },
    "09_dark_mode_cad": {
        "challenge": "Inverted Scheme (black background, bright white lines)",
        "gt_rooms": 4,
        "gt_doors": 2,
        "gt_windows": 2,
        "gt_area_m2": 136.9,
    },
    "10_diagonal_angled_walls": {
        "challenge": "Non-Manhattan Geometry (45-degree diagonal walls)",
        "gt_rooms": 2,
        "gt_doors": 0,
        "gt_windows": 0,
        "gt_area_m2": 101.7,
    }
}


def run_synthetic_stress_benchmark():
    input_dir = Path("data/external_test_plans")
    output_dir = Path("models/samples/external_generalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print("      ARCHIGEN 3D: SYNTHETIC STRESS TESTS GENERALIZATION BENCHMARK (10 TEST CASES)")
    print("=" * 95)

    pipeline = FloorplanPipeline.get_instance()
    
    plan_files = sorted(list(input_dir.glob("*.png")))
    if not plan_files:
        print("[ERROR] No stress test plans found in data/external_test_plans!")
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

    room_count_errors = []

    for idx, p_path in enumerate(plan_files, 1):
        name = p_path.stem
        gt_info = SYNTHETIC_STRESS_GROUND_TRUTH.get(name, {
            "challenge": "Custom Stress Test",
            "gt_rooms": 4,
            "gt_doors": 0,
            "gt_windows": 0,
            "gt_area_m2": 100.0
        })
        gt_r = gt_info["gt_rooms"]

        with open(p_path, "rb") as f:
            img_bytes = f.read()

        try:
            res = pipeline.process_image(img_bytes)
            layout = res["layout"]
            meta = layout.get("metadata", {})
            walls = layout.get("walls", [])
            rooms = layout.get("rooms", [])
            doors = layout.get("doors", [])
            windows = layout.get("windows", [])

            num_walls = len(walls)
            ext_walls = sum(1 for w in walls if w.get("is_exterior", False))
            num_doors = len(doors)
            num_windows = len(windows)
            num_rooms = len(rooms)
            total_m2 = meta.get("total_area_m2", meta.get("total_surface_m2", 0.0))
            scale_m = meta.get("scale_meters_per_pixel", meta.get("scale_m_per_px", 0.03))
            shell_valid = meta.get("is_building_shell_valid", False)
            shell_status = meta.get("shell_status", "VALID" if shell_valid else "INVALID")
            
            delta_rooms = num_rooms - gt_r
            room_count_errors.append(abs(delta_rooms))

            # Failure diagnosis and rigorous status categorization
            diagnostics = []

            if num_rooms == 0:
                status = "FAIL: NO_ROOMS"
                diagnostics.append("Zero functional rooms detected (boundary closure failure)")
            elif num_walls < 4:
                status = "FAIL: INCOMPLETE_SHELL"
                diagnostics.append("Fewer than 4 outer walls vectorized")
            elif not shell_valid:
                status = "FAIL: INVALID_GEOMETRY"
                diagnostics.append(f"Geometric shell integrity check failed ({shell_status})")
            elif delta_rooms >= 4:
                status = f"FAIL: OVER_SEGMENTED (+{delta_rooms})"
                diagnostics.append(f"Spurious wall fragmentation creates {num_rooms} tiny cells (GT: {gt_r})")
            elif delta_rooms <= -2:
                status = f"FAIL: UNDER_SEGMENTED ({delta_rooms})"
                diagnostics.append(f"Partitions missed: only {num_rooms} enclosed rooms (GT: {gt_r})")
            elif abs(delta_rooms) <= 1:
                status = f"PASS: QUALIFIED ({delta_rooms:+d})"
                diagnostics.append(f"Plausible layout ({num_rooms} rooms vs GT {gt_r})")
            else:
                status = f"PARTIAL ({delta_rooms:+d})"
                diagnostics.append(f"Detected {num_rooms} rooms vs GT {gt_r}")

            # Contextual failure observations
            if "dark_mode" in name and num_rooms == 0:
                diagnostics.append("Domain shift: Inverted contrast breaks UNet encoder activations")
            if "low_res" in name and delta_rooms >= 3:
                diagnostics.append("Domain shift: Blur artifacts mistaken for interior wall edges")
            if "hand_drawn" in name:
                diagnostics.append("Domain shift: Wobbly line contours deviate from rectilinear grid")

            diag_str = "; ".join(diagnostics)

            record = {
                "id": idx,
                "name": name,
                "challenge": gt_info["challenge"],
                "gt_rooms": gt_r,
                "pred_rooms": num_rooms,
                "delta_rooms": delta_rooms,
                "walls": num_walls,
                "ext_walls": ext_walls,
                "doors": num_doors,
                "windows": num_windows,
                "area_m2": round(total_m2, 1),
                "scale": round(scale_m, 4),
                "shell_status": shell_status,
                "status": status,
                "diagnostic": diag_str
            }
            results.append(record)

            # Generate visual side-by-side overlay
            orig_img = cv2.imread(str(p_path))
            overlay = orig_img.copy()

            for r in rooms:
                pts = r.get("polygon_meters", [])
                if len(pts) >= 3:
                    px_pts = (np.array(pts) / max(scale_m, 1e-4)).astype(np.int32)
                    col = ROOM_COLORS.get(r.get("room_type"), (200, 200, 200))
                    sub_overlay = overlay.copy()
                    cv2.fillPoly(sub_overlay, [px_pts], col)
                    cv2.addWeighted(sub_overlay, 0.45, overlay, 0.55, 0, overlay)
                    cv2.polylines(overlay, [px_pts], True, (80, 80, 80), 1)

            for w in walls:
                pts = w.get("polygon_meters", [])
                if len(pts) >= 2:
                    px_pts = (np.array(pts) / max(scale_m, 1e-4)).astype(np.int32)
                    col = (0, 0, 220) if w.get("is_exterior", False) else (220, 0, 0)
                    cv2.polylines(overlay, [px_pts], True, col, 2)

            for d in doors:
                pos = d.get("position", [0, 0])
                c_px = (int(pos[0] / max(scale_m, 1e-4)), int(pos[1] / max(scale_m, 1e-4)))
                cv2.circle(overlay, c_px, 6, (0, 200, 200), -1)

            for win in windows:
                pos = win.get("position", [0, 0])
                c_px = (int(pos[0] / max(scale_m, 1e-4)), int(pos[1] / max(scale_m, 1e-4)))
                cv2.circle(overlay, c_px, 6, (200, 200, 0), -1)

            status_text = f"{name} | Rooms: {num_rooms} (GT: {gt_r}) | Status: {status}"
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (30, 30, 30), -1)
            cv2.putText(overlay, status_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            side_by_side = np.hstack([orig_img, overlay])
            cv2.imwrite(str(output_dir / f"stress_{name}.png"), side_by_side)

        except Exception as e:
            print(f"  [ERROR processing {name}]: {e}")
            results.append({
                "id": idx,
                "name": name,
                "challenge": gt_info["challenge"],
                "gt_rooms": gt_r, "pred_rooms": 0, "delta_rooms": -gt_r,
                "walls": 0, "ext_walls": 0, "doors": 0, "windows": 0,
                "area_m2": 0.0, "scale": 0.0, "shell_status": "EXCEPTION",
                "status": f"FAIL: EXCEPTION",
                "diagnostic": f"Pipeline crashed: {str(e)}"
            })
            room_count_errors.append(gt_r)

    # Save JSON results
    with open(output_dir / "synthetic_stress_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print Formatted Benchmark Table
    print("\n" + "=" * 115)
    print("                   SYNTHETIC STRESS TESTS GENERALIZATION BENCHMARK (10 BLUEPRINTS)")
    print("=" * 115)
    header = f"{'#':<3} | {'Stress Test Case':<26} | {'GT':<3} | {'Pred':<4} | {'Diff':<5} | {'Area(m2)':<8} | {'Shell Status':<12} | {'Evaluation Outcome'}"
    print(header)
    print("-" * 115)
    for r in results:
        diff_str = f"{r['delta_rooms']:+d}"
        print(f"{r['id']:<3} | {r['name']:<26} | {r['gt_rooms']:<3} | {r['pred_rooms']:<4} | {diff_str:<5} | {r['area_m2']:<8} | {r['shell_status']:<12} | {r['status']}")
    print("-" * 115)

    stress_mae = float(np.mean(room_count_errors)) if room_count_errors else 0.0
    qual_success = sum(1 for r in results if "PASS: QUALIFIED" in r["status"])
    zero_rooms = sum(1 for r in results if r["pred_rooms"] == 0)

    print(f"  Average Room Count MAE vs Synthetic Ground Truth: {stress_mae:.2f}")
    print(f"  Qualified Plausible Success Rate (Diff within +/-1): {qual_success}/10 ({qual_success*10.0}%)")
    print(f"  Zero-Detection Complete Failures:                  {zero_rooms}/10 ({zero_rooms*10.0}%)")
    print("=" * 115 + "\n")

    return results


if __name__ == "__main__":
    run_synthetic_stress_benchmark()
