"""
External Floorplan Dataset Generator and Generalization Benchmark Suite.
Creates 10 diverse external floorplan test samples outside CubiCasa5k to test real-world robustness:
1. clean_bw_cad: Minimalist black and white CAD blueprint.
2. marketing_brochure: Real-estate style colored rooms with pastel fills.
3. hand_drawn_sketch: Rough pencil sketch with wobbly lines and imperfect corners.
4. furniture_cluttered: Floorplan densely populated with beds, sofas, tables, fixtures.
5. dimension_text_annotated: Floorplan covered in dimension lines, arrows, and metric text ("4.20m").
6. low_res_screenshot: Blurry, heavily compressed low-resolution thumbnail.
7. open_concept_loft: Modern undivided open layout with kitchen island.
8. narrow_railroad_layout: High aspect-ratio apartment with central hallway.
9. dark_mode_cad: Inverted scheme (black background, bright white lines).
10. diagonal_angled_walls: Non-Manhattan layout with 45-degree angled walls.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = Path("data/external_test_plans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_external_plans():
    print("[DATASET] Generating 10 diverse external test floorplans...")

    # Base canvas size 512x512
    W, H = 512, 512

    # -------------------------------------------------------------
    # 1. Clean B&W CAD Blueprint
    # -------------------------------------------------------------
    img1 = np.ones((H, W, 3), dtype=np.uint8) * 255
    # Outer walls
    cv2.rectangle(img1, (60, 60), (450, 450), (0, 0, 0), thickness=6)
    # Internal partitions
    cv2.line(img1, (240, 60), (240, 450), (0, 0, 0), thickness=4)
    cv2.line(img1, (240, 260), (450, 260), (0, 0, 0), thickness=4)
    cv2.line(img1, (60, 220), (240, 220), (0, 0, 0), thickness=4)
    # Door openings (white gaps with swing arcs)
    cv2.line(img1, (240, 120), (240, 160), (255, 255, 255), thickness=6)
    cv2.ellipse(img1, (240, 160), (40, 40), 0, 180, 270, (0, 0, 0), 1)
    cv2.line(img1, (180, 220), (220, 220), (255, 255, 255), thickness=6)
    cv2.ellipse(img1, (220, 220), (40, 40), 0, 90, 180, (0, 0, 0), 1)
    # Window openings (parallel thin lines)
    cv2.line(img1, (120, 60), (180, 60), (255, 255, 255), thickness=6)
    cv2.line(img1, (120, 58), (180, 58), (0, 0, 0), 1)
    cv2.line(img1, (120, 62), (180, 62), (0, 0, 0), 1)
    cv2.line(img1, (320, 60), (380, 60), (255, 255, 255), thickness=6)
    cv2.line(img1, (320, 58), (380, 58), (0, 0, 0), 1)
    cv2.line(img1, (320, 62), (380, 62), (0, 0, 0), 1)
    cv2.imwrite(str(OUTPUT_DIR / "01_clean_bw_cad.png"), img1)

    # -------------------------------------------------------------
    # 2. Marketing Brochure (Pastel Fills)
    # -------------------------------------------------------------
    img2 = np.ones((H, W, 3), dtype=np.uint8) * 248
    # Pastel fills for rooms (BGR)
    cv2.rectangle(img2, (70, 70), (230, 250), (235, 245, 255), -1)   # Bedroom (soft blue)
    cv2.rectangle(img2, (70, 250), (230, 440), (240, 255, 240), -1)  # Kitchen (soft green)
    cv2.rectangle(img2, (230, 70), (440, 440), (245, 240, 255), -1)  # Living room (soft warm)
    # Dark modern walls
    cv2.rectangle(img2, (70, 70), (440, 440), (60, 50, 50), 7)
    cv2.line(img2, (230, 70), (230, 440), (80, 70, 70), 5)
    cv2.line(img2, (70, 250), (230, 250), (80, 70, 70), 5)
    # Door openings
    cv2.line(img2, (230, 140), (230, 180), (245, 240, 255), 7)
    cv2.line(img2, (130, 250), (170, 250), (240, 255, 240), 7)
    cv2.imwrite(str(OUTPUT_DIR / "02_marketing_brochure.png"), img2)

    # -------------------------------------------------------------
    # 3. Hand-Drawn Sketch (Wobbly lines & Noise)
    # -------------------------------------------------------------
    img3 = np.ones((H, W, 3), dtype=np.uint8) * 250
    # Add subtle paper texture
    noise = np.random.normal(0, 5, (H, W, 3)).astype(np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def draw_wobbly_line(canvas, p1, p2, color=(40, 40, 40), thickness=3):
        pts = []
        n_steps = 15
        for i in range(n_steps + 1):
            t = i / n_steps
            x = int(p1[0] * (1 - t) + p2[0] * t + np.random.randint(-3, 4))
            y = int(p1[1] * (1 - t) + p2[1] * t + np.random.randint(-3, 4))
            pts.append([x, y])
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), color, thickness)

    draw_wobbly_line(img3, (80, 80), (420, 80), thickness=4)
    draw_wobbly_line(img3, (420, 80), (420, 420), thickness=4)
    draw_wobbly_line(img3, (420, 420), (80, 420), thickness=4)
    draw_wobbly_line(img3, (80, 420), (80, 80), thickness=4)
    draw_wobbly_line(img3, (250, 80), (250, 420), thickness=3)
    draw_wobbly_line(img3, (80, 250), (250, 250), thickness=3)
    cv2.imwrite(str(OUTPUT_DIR / "03_hand_drawn_sketch.png"), img3)

    # -------------------------------------------------------------
    # 4. Furniture Cluttered Floorplan
    # -------------------------------------------------------------
    img4 = img1.copy()
    # Bed in room 1 (pillow + mattress)
    cv2.rectangle(img4, (80, 80), (160, 180), (180, 180, 180), 2)
    cv2.rectangle(img4, (90, 85), (120, 115), (200, 200, 200), -1)
    cv2.rectangle(img4, (125, 85), (155, 115), (200, 200, 200), -1)
    # Dining table with 4 chairs in living room
    cv2.rectangle(img4, (300, 120), (380, 200), (160, 160, 160), 2)
    cv2.circle(img4, (340, 105), 10, (160, 160, 160), 2)
    cv2.circle(img4, (340, 215), 10, (160, 160, 160), 2)
    cv2.circle(img4, (285, 160), 10, (160, 160, 160), 2)
    cv2.circle(img4, (395, 160), 10, (160, 160, 160), 2)
    # Sofa in bottom living
    cv2.rectangle(img4, (280, 320), (410, 370), (140, 140, 140), 2)
    cv2.rectangle(img4, (280, 320), (300, 370), (140, 140, 140), 1)
    cv2.rectangle(img4, (390, 320), (410, 370), (140, 140, 140), 1)
    cv2.imwrite(str(OUTPUT_DIR / "04_furniture_cluttered.png"), img4)

    # -------------------------------------------------------------
    # 5. Dimension Text & Leader Lines Annotated
    # -------------------------------------------------------------
    img5 = img1.copy()
    # Dimension line on top
    cv2.line(img5, (60, 35), (450, 35), (100, 100, 100), 1)
    cv2.line(img5, (60, 25), (60, 45), (100, 100, 100), 1)
    cv2.line(img5, (450, 25), (450, 45), (100, 100, 100), 1)
    cv2.putText(img5, "11.50 m", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    # Dimension line on left
    cv2.line(img5, (35, 60), (35, 450), (100, 100, 100), 1)
    cv2.line(img5, (25, 60), (45, 60), (100, 100, 100), 1)
    cv2.line(img5, (25, 450), (45, 450), (100, 100, 100), 1)
    cv2.putText(img5, "9.80 m", (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    # Room text labels inside
    cv2.putText(img5, "BEDROOM 1", (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img5, "14.2 m2", (105, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(img5, "LIVING ROOM", (280, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img5, "32.0 m2", (310, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(img5, "KITCHEN", (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUTPUT_DIR / "05_dimension_text_annotated.png"), img5)

    # -------------------------------------------------------------
    # 6. Low-Resolution Compressed Screenshot
    # -------------------------------------------------------------
    pil_img = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # Downscale heavily to 140x140 and re-upscale with blur
    low_res = pil_img.resize((140, 140), Image.Resampling.BILINEAR)
    low_res = low_res.filter(ImageFilter.GaussianBlur(radius=0.8))
    re_upscaled = low_res.resize((512, 512), Image.Resampling.BICUBIC)
    re_upscaled.save(str(OUTPUT_DIR / "06_low_res_screenshot.png"), "JPEG", quality=45)

    # -------------------------------------------------------------
    # 7. Open-Concept Loft Layout
    # -------------------------------------------------------------
    img7 = np.ones((H, W, 3), dtype=np.uint8) * 255
    cv2.rectangle(img7, (50, 50), (460, 460), (0, 0, 0), 6)
    # Only one bathroom enclosed
    cv2.rectangle(img7, (50, 50), (180, 180), (0, 0, 0), 4)
    # Kitchen island in open area
    cv2.rectangle(img7, (220, 100), (360, 140), (0, 0, 0), 2)
    # Door for bathroom
    cv2.line(img7, (180, 100), (180, 140), (255, 255, 255), 6)
    cv2.ellipse(img7, (180, 140), (40, 40), 0, 180, 270, (0, 0, 0), 1)
    # Large facade windows on bottom and right
    cv2.line(img7, (150, 460), (360, 460), (255, 255, 255), 6)
    cv2.line(img7, (150, 458), (360, 458), (0, 0, 0), 1)
    cv2.line(img7, (150, 462), (360, 462), (0, 0, 0), 1)
    cv2.imwrite(str(OUTPUT_DIR / "07_open_concept_loft.png"), img7)

    # -------------------------------------------------------------
    # 8. Narrow Railroad Layout (High Aspect Ratio)
    # -------------------------------------------------------------
    img8 = np.ones((H, W, 3), dtype=np.uint8) * 255
    # Long rectangular perimeter
    cv2.rectangle(img8, (120, 40), (390, 470), (0, 0, 0), 6)
    # Central longitudinal hallway
    cv2.line(img8, (230, 40), (230, 470), (0, 0, 0), 4)
    # Compartments off hallway
    cv2.line(img8, (120, 180), (230, 180), (0, 0, 0), 4)
    cv2.line(img8, (120, 320), (230, 320), (0, 0, 0), 4)
    cv2.line(img8, (230, 240), (390, 240), (0, 0, 0), 4)
    # Doors into hallway
    cv2.line(img8, (230, 110), (230, 150), (255, 255, 255), 6)
    cv2.line(img8, (230, 270), (230, 310), (255, 255, 255), 6)
    cv2.imwrite(str(OUTPUT_DIR / "08_narrow_railroad_layout.png"), img8)

    # -------------------------------------------------------------
    # 9. Dark-Mode Inverted CAD
    # -------------------------------------------------------------
    img9 = 255 - img1
    cv2.imwrite(str(OUTPUT_DIR / "09_dark_mode_cad.png"), img9)

    # -------------------------------------------------------------
    # 10. Angled 45-Degree Diagonal Walls
    # -------------------------------------------------------------
    img10 = np.ones((H, W, 3), dtype=np.uint8) * 255
    poly_outer = np.array([[120, 60], [420, 60], [450, 340], [320, 450], [60, 450], [60, 180]], np.int32)
    cv2.polylines(img10, [poly_outer], isClosed=True, color=(0, 0, 0), thickness=6)
    # Internal angled divider
    cv2.line(img10, [60, 180], [260, 260], (0, 0, 0), 4)
    cv2.line(img10, [260, 260], [320, 450], (0, 0, 0), 4)
    cv2.imwrite(str(OUTPUT_DIR / "10_diagonal_angled_walls.png"), img10)

    print(f"[DATASET] Successfully saved 10 diverse test plans to {OUTPUT_DIR}")

if __name__ == "__main__":
    create_external_plans()
