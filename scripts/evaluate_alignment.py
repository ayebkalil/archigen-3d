import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

sample_path = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/6044")
im_scaled = cv2.imread(str(sample_path / "F1_scaled.png"), cv2.IMREAD_GRAYSCALE)
im_orig = cv2.imread(str(sample_path / "F1_original.png"), cv2.IMREAD_GRAYSCALE)
tree = ET.parse(sample_path / "model.svg")
root = tree.getroot()
vb = [float(x) for x in root.attrib["viewBox"].split()]
svg_w, svg_h = vb[2], vb[3]

sh_h, sh_w = im_scaled.shape
oh_h, oh_w = im_orig.shape

# Collect polygons
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

def evaluate_alignment(img, polys, sx, sy, dx=0, dy=0):
    mask = np.zeros(img.shape, dtype=np.uint8)
    for p in polys:
        scaled = p.copy()
        scaled[:, 0] = scaled[:, 0] * sx + dx
        scaled[:, 1] = scaled[:, 1] * sy + dy
        cv2.fillPoly(mask, [scaled.astype(np.int32)], 255)
    
    wall_pixels = img[mask > 0]
    if len(wall_pixels) == 0:
        return 255.0, 0
    mean_val = np.mean(wall_pixels)
    dark_ratio = np.mean(wall_pixels < 128)
    return mean_val, dark_ratio

print("Evaluating Hypotheses on Sample 6044:")
# 1. Direct 1:1 on scaled
m1, d1 = evaluate_alignment(im_scaled, polys, 1.0, 1.0)
print(f"1. 1:1 on F1_scaled: mean={m1:.1f}, dark_ratio={d1*100:.2f}%")

# 2. Anisotropic on scaled
m2, d2 = evaluate_alignment(im_scaled, polys, sh_w / svg_w, sh_h / svg_h)
print(f"2. Anisotropic on F1_scaled: mean={m2:.1f}, dark_ratio={d2*100:.2f}%")

# 3. Anisotropic on original
m3, d3 = evaluate_alignment(im_orig, polys, oh_w / svg_w, oh_h / svg_h)
print(f"3. Anisotropic on F1_original: mean={m3:.1f}, dark_ratio={d3*100:.2f}%")

# 4. What about other scale factors?
# Let's search over scale and translation!
print("\nTesting multiple samples across the dataset...")
