import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

sample_path = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/6044")
img_p = sample_path / "F1_scaled.png"
svg_p = sample_path / "model.svg"

img = cv2.imread(str(img_p))
ih, iw = img.shape[:2]

tree = ET.parse(svg_p)
root = tree.getroot()
vb = [float(x) for x in root.attrib["viewBox"].split()]
svg_w, svg_h = vb[2], vb[3]

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

size = 512

# 1. OLD BROKEN PIPELINE
scale_old = min(size / svg_w, size / svg_h)
nw_old, nh_old = int(svg_w * scale_old), int(svg_h * scale_old)
pad_x_old = (size - nw_old) // 2
pad_y_old = (size - nh_old) // 2
img_resized_old = cv2.resize(img, (nw_old, nh_old))
canvas_old = np.full((size, size, 3), 255, dtype=np.uint8)
canvas_old[pad_y_old:pad_y_old+nh_old, pad_x_old:pad_x_old+nw_old] = img_resized_old
overlay_old = canvas_old.copy()
for p in polys:
    scaled_p = p * scale_old + [pad_x_old, pad_y_old]
    cv2.fillPoly(overlay_old, [scaled_p.astype(np.int32)], (0, 0, 255))
blended_old = cv2.addWeighted(canvas_old, 0.6, overlay_old, 0.4, 0)
cv2.putText(blended_old, "OLD: ViewBox Misaligned", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 2. NEW CORRECT PIPELINE
scale_new = min(size / iw, size / ih)
nw_new, nh_new = int(iw * scale_new), int(ih * scale_new)
pad_x_new = (size - nw_new) // 2
pad_y_new = (size - nh_new) // 2
img_resized_new = cv2.resize(img, (nw_new, nh_new))
canvas_new = np.full((size, size, 3), 255, dtype=np.uint8)
canvas_new[pad_y_new:pad_y_new+nh_new, pad_x_new:pad_x_new+nw_new] = img_resized_new
overlay_new = canvas_new.copy()
for p in polys:
    scaled_p = p * scale_new + [pad_x_new, pad_y_new]
    cv2.fillPoly(overlay_new, [scaled_p.astype(np.int32)], (0, 255, 0)) # Green
blended_new = cv2.addWeighted(canvas_new, 0.6, overlay_new, 0.4, 0)
cv2.putText(blended_new, "NEW: Pixel-Aligned", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 0), 2)

# Side-by-side
comparison = np.hstack([blended_old, blended_new])
cv2.imwrite("scripts/alignment_comparison_old_vs_new.png", comparison)
print("Saved comparison to scripts/alignment_comparison_old_vs_new.png")
