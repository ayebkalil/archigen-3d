"""
generate_bridge_before_after_comparison.py
Renders a side-by-side Before / After comparison grid of the Facade Bridge:
Old Minimal Bridge vs New Enriched Bridge on 4 realistic architectural layouts.
Outputs high-resolution visual grid to:
  outputs/bridge_comparison/facade_bridge_before_after_grid.png
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge.facade_bridge import FacadeBridge, CMP_PALETTE

OUT_DIR = PROJECT_ROOT / "outputs/bridge_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 Representative Test Wall Layouts
TEST_LAYOUTS = [
    {
        "name": "Layout 1: 1-Story Front Entrance",
        "wall": {
            "id": "wall_front",
            "is_exterior": True,
            "is_front": True,
            "polygon_meters": [[0.0, 0.0], [8.0, 0.0], [8.0, 0.25], [0.0, 0.25]],
            "openings": [
                {"type": "window", "position_along_wall_m": 2.0},
                {"type": "door", "position_along_wall_m": 4.0},
                {"type": "window", "position_along_wall_m": 6.0}
            ]
        },
        "num_floors": 1
    },
    {
        "name": "Layout 2: 2-Story Balcony Townhouse",
        "wall": {
            "id": "wall_townhouse",
            "is_exterior": True,
            "is_front": True,
            "polygon_meters": [[0.0, 0.0], [6.5, 0.0], [6.5, 0.25], [0.0, 0.25]],
            "openings": [
                {"type": "window", "position_along_wall_m": 1.8},
                {"type": "door", "position_along_wall_m": 3.25},
                {"type": "window", "position_along_wall_m": 4.8}
            ]
        },
        "num_floors": 2
    },
    {
        "name": "Layout 3: Multi-Bay Exterior Side Wall",
        "wall": {
            "id": "wall_side",
            "is_exterior": True,
            "is_front": False,
            "polygon_meters": [[0.0, 0.0], [10.0, 0.0], [10.0, 0.25], [0.0, 0.25]],
            "openings": [
                {"type": "window", "position_along_wall_m": 2.0},
                {"type": "window", "position_along_wall_m": 5.0},
                {"type": "window", "position_along_wall_m": 8.0}
            ]
        },
        "num_floors": 1
    },
    {
        "name": "Layout 4: Compact 2-Story Urban Villa",
        "wall": {
            "id": "wall_villa",
            "is_exterior": True,
            "is_front": True,
            "polygon_meters": [[0.0, 0.0], [5.5, 0.0], [5.5, 0.25], [0.0, 0.25]],
            "openings": [
                {"type": "door", "position_along_wall_m": 2.75}
            ]
        },
        "num_floors": 2
    }
]

def synthesize_old_minimal_sketch(bridge, wall, num_floors=1):
    """Old minimal synthesis: only wall, plain windows, plain door, cornice, base."""
    wall_len_m = bridge.calculate_wall_length_m(wall)
    total_height_m = num_floors * 2.8
    scale = min(232 / wall_len_m, 232 / total_height_m)
    draw_w = int(round(wall_len_m * scale))
    draw_h = int(round(total_height_m * scale))
    off_x = (256 - draw_w) // 2
    off_y = (256 - draw_h) // 2

    canvas = Image.new("RGB", (256, 256), CMP_PALETTE["background"])
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([off_x, off_y, off_x + draw_w, off_y + draw_h], fill=CMP_PALETTE["facade"])
    # Cornice & Base only
    cornice_h = max(int(round(0.25 * scale)), 4)
    base_h = max(int(round(0.30 * scale)), 5)
    draw.rectangle([off_x, off_y, off_x + draw_w, off_y + cornice_h], fill=CMP_PALETTE["cornice"])
    draw.rectangle([off_x, off_y + draw_h - base_h, off_x + draw_w, off_y + draw_h], fill=CMP_PALETTE["base"])

    # Openings without lintels or balcony
    attached_openings = wall.get("openings", [])
    doors = [op for op in attached_openings if op.get("type") == "door"]
    windows = [op for op in attached_openings if op.get("type") == "window"]
    has_door = len(doors) > 0 or wall.get("is_front", False)

    for fl in range(num_floors):
        fl_bot = off_y + draw_h - int(round(fl * 2.8 * scale))
        fl_top = off_y + draw_h - int(round((fl + 1) * 2.8 * scale))
        if fl == 0 and has_door:
            d_x = off_x + draw_w // 2
            d_w = max(int(round(1.15 * scale)), 12)
            d_h = max(int(round(2.20 * scale)), 20)
            draw.rectangle([d_x - d_w // 2, fl_bot - base_h - d_h, d_x + d_w // 2, fl_bot - base_h], fill=CMP_PALETTE["door"])

        for w_op in windows:
            wx_m = w_op.get("position_along_wall_m", 2.0)
            wx_px = off_x + int(round(wx_m * scale))
            win_w = max(int(round(1.10 * scale)), 10)
            win_h = max(int(round(1.45 * scale)), 14)
            wy1 = fl_bot - int(round(0.90 * scale))
            wy0 = wy1 - win_h
            draw.rectangle([wx_px - win_w // 2, wy0, wx_px + win_w // 2, wy1], fill=CMP_PALETTE["window"])
            # Simple sill
            draw.rectangle([wx_px - win_w // 2, wy1, wx_px + win_w // 2, wy1 + 3], fill=CMP_PALETTE["sill"])

    return canvas

def main():
    bridge = FacadeBridge()
    print("Generating Before / After visual comparison grid for Facade Bridge...")

    rows = []
    for item in TEST_LAYOUTS:
        wall = item["wall"]
        num_fl = item["num_floors"]

        # 1. Old Minimal
        old_sketch = synthesize_old_minimal_sketch(bridge, wall, num_fl)
        old_gen = bridge.inference_engine.generate_batch([old_sketch])[0]

        # 2. New Enriched
        new_sketch, _ = bridge.synthesize_facade_sketch(wall, num_floors=num_fl, is_front=wall.get("is_front", False))
        new_gen = bridge.inference_engine.generate_batch([new_sketch])[0]

        rows.append({
            "title": item["name"],
            "old_sketch": old_sketch,
            "old_gen": old_gen,
            "new_sketch": new_sketch,
            "new_gen": new_gen
        })

    # Assemble into a clean grid: 4 rows x 4 columns
    # Col 1: Old Sketch | Col 2: Old Generated | Col 3: New Sketch | Col 4: New Generated
    col_w = 256
    row_h = 256
    header_h = 50
    label_h = 30

    grid_w = col_w * 4 + 60
    grid_h = header_h + len(rows) * (row_h + label_h + 15) + 30
    grid = Image.new("RGB", (grid_w, grid_h), (245, 247, 250))
    draw = ImageDraw.Draw(grid)

    # Title Banner
    draw.rectangle([0, 0, grid_w, header_h], fill=(15, 41, 74))
    draw.text((20, 15), "ARCHIGEN 3D - FACADE BRIDGE COMPARISON: OLD MINIMAL vs NEW ENRICHED", fill=(255, 255, 255))

    col_headers = ["(A) Old Minimal Sketch", "(B) Old Facade Generated", "(C) New Enriched Sketch", "(D) New Facade Generated"]
    col_x_offsets = [20, 20 + col_w + 10, 20 + (col_w + 10) * 2 + 10, 20 + (col_w + 10) * 3 + 10]

    for col_idx, (h_text, cx) in enumerate(zip(col_headers, col_x_offsets)):
        draw.text((cx + 10, header_h + 8), h_text, fill=(30, 41, 59))

    curr_y = header_h + 35
    for r_idx, r_data in enumerate(rows):
        # Row title
        draw.text((20, curr_y - 2), f"[{r_idx+1}] {r_data['title']}", fill=(15, 41, 74))
        img_y = curr_y + 18

        grid.paste(r_data["old_sketch"], (col_x_offsets[0], img_y))
        grid.paste(r_data["old_gen"], (col_x_offsets[1], img_y))
        grid.paste(r_data["new_sketch"], (col_x_offsets[2], img_y))
        grid.paste(r_data["new_gen"], (col_x_offsets[3], img_y))

        # Borders
        for cx in col_x_offsets:
            draw.rectangle([cx - 1, img_y - 1, cx + col_w, img_y + row_h], outline=(200, 210, 220), width=1)

        curr_y = img_y + row_h + 20

    out_file = OUT_DIR / "facade_bridge_before_after_grid.png"
    grid.save(out_file)
    print(f"\n[+] Successfully generated Before/After comparison grid: {out_file}")

if __name__ == "__main__":
    main()
