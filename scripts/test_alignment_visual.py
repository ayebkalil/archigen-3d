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
scale = min(size / iw, size / ih)
nw, nh = int(iw * scale), int(ih * scale)
pad_x = (size - nw) // 2
pad_y = (size - nh) // 2

img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
canvas = np.full((size, size, 3), 255, dtype=np.uint8)
canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized

# Draw overlay with transparency
overlay = canvas.copy()
mask = np.zeros((size, size), dtype=np.uint8)
for p in polys:
    scaled_p = p * scale + [pad_x, pad_y]
    cv2.fillPoly(overlay, [scaled_p.astype(np.int32)], (0, 0, 255))
    cv2.fillPoly(mask, [scaled_p.astype(np.int32)], 255)

blended = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)
cv2.imwrite("scripts/visual_aligned_6044.png", blended)
cv2.imwrite("scripts/mask_aligned_6044.png", mask)

print(f"Sample 6044 visual alignment saved to scripts/visual_aligned_6044.png")
print(f"Mask non-zero: {np.count_nonzero(mask)} pixels")
