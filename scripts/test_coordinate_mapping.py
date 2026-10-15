import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

sample_path = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/6044")
img_scaled_p = sample_path / "F1_scaled.png"
img_orig_p = sample_path / "F1_original.png"
svg_p = sample_path / "model.svg"

tree = ET.parse(svg_p)
root = tree.getroot()
vb = [float(x) for x in root.attrib["viewBox"].split()]
print("viewBox:", vb)
svg_w, svg_h = vb[2], vb[3]

im_scaled = cv2.imread(str(img_scaled_p))
sh_h, sh_w = im_scaled.shape[:2]
print(f"F1_scaled: W={sh_w}, H={sh_h}, aspect={sh_w/sh_h:.4f}")

im_orig = cv2.imread(str(img_orig_p))
oh_h, oh_w = im_orig.shape[:2]
print(f"F1_original: W={oh_w}, H={oh_h}, aspect={oh_w/oh_h:.4f}")

print(f"SVG viewBox: W={svg_w}, H={svg_h}, aspect={svg_w/svg_h:.4f}")

# Extract all wall polygons
polys = []
for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
    c = g.attrib.get("class", "")
    if "Wall" in c and "Cabinet" not in c:
        for poly in g.findall("{http://www.w3.org/2000/svg}polygon"):
            pts_str = poly.attrib.get("points", "")
            clean = pts_str.replace(",", " ").split()
            pts = []
            for i in range(0, len(clean), 2):
                if i + 1 < len(clean):
                    pts.append((float(clean[i]), float(clean[i+1])))
            if len(pts) >= 3:
                polys.append(np.array(pts))

print(f"Extracted {len(polys)} wall polygons.")

# Hypothesis A: SVG coordinates map directly 1:1 to F1_scaled (no scale)
canvas_a = im_scaled.copy()
for p in polys:
    cv2.polylines(canvas_a, [p.astype(np.int32)], True, (0, 0, 255), 3)
cv2.imwrite("scripts/map_hyp_A_1to1.png", canvas_a)

# Hypothesis B: SVG coordinates map to F1_scaled with sx = sh_w / svg_w, sy = sh_h / svg_h (anisotropic)
canvas_b = im_scaled.copy()
sx_b = sh_w / svg_w
sy_b = sh_h / svg_h
for p in polys:
    scaled_pts = p.copy()
    scaled_pts[:, 0] *= sx_b
    scaled_pts[:, 1] *= sy_b
    cv2.polylines(canvas_b, [scaled_pts.astype(np.int32)], True, (0, 255, 0), 3)
cv2.imwrite("scripts/map_hyp_B_anisotropic.png", canvas_b)

# Hypothesis C: SVG coordinates map to F1_original with sx = oh_w / svg_w, sy = oh_h / svg_h
canvas_c = im_orig.copy()
sx_c = oh_w / svg_w
sy_c = oh_h / svg_h
for p in polys:
    scaled_pts = p.copy()
    scaled_pts[:, 0] *= sx_c
    scaled_pts[:, 1] *= sy_c
    cv2.polylines(canvas_c, [scaled_pts.astype(np.int32)], True, (255, 0, 0), 5)
cv2.imwrite("scripts/map_hyp_C_orig_anisotropic.png", canvas_c)

# Hypothesis D: Check if F1_scaled was cropped or if model.svg corresponds to a sub-region!
print("Hypotheses generated. Checking match quality...")
