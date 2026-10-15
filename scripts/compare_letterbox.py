import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k")
with open(root / "train.txt", "r") as f:
    samples = [line.strip().lstrip("/") for line in f if line.strip()][:25]

print(f"Testing 1:1 mapping on {len(samples)} samples:")

for s_rel in samples:
    folder = root / s_rel
    svg_p = folder / "model.svg"
    img_p = folder / "F1_scaled.png"
    if not svg_p.exists() or not img_p.exists():
        continue
        
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    ih, iw = img.shape
    
    tree = ET.parse(svg_p)
    r = tree.getroot()
    vb = [float(x) for x in r.attrib.get("viewBox", f"0 0 {iw} {ih}").split()]
    svg_w, svg_h = vb[2], vb[3]
    
    # Collect wall polygons
    polys = []
    for g in r.findall(".//{http://www.w3.org/2000/svg}g"):
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
                    
    if not polys:
        continue
        
    # Test 1: 1:1 on F1_scaled
    mask_1to1 = np.zeros(img.shape, dtype=np.uint8)
    for p in polys:
        # clip to img bounds
        clipped = p.copy()
        cv2.fillPoly(mask_1to1, [clipped.astype(np.int32)], 255)
    px_1to1 = img[mask_1to1 > 0]
    dark_1to1 = np.mean(px_1to1 < 128) if len(px_1to1) > 0 else 0
    
    # Test 2: Current dataset.py approach (letterbox based on svg_w, svg_h)
    size = 512
    scale = min(size / svg_w, size / svg_h)
    nw, nh = int(svg_w * scale), int(svg_h * scale)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    img_resized = cv2.resize(img, (nw, nh))
    canvas_img = np.full((size, size), 255, dtype=np.uint8)
    canvas_img[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized
    
    mask_current = np.zeros((size, size), dtype=np.uint8)
    for p in polys:
        scaled_p = p * scale + [pad_x, pad_y]
        cv2.fillPoly(mask_current, [scaled_p.astype(np.int32)], 255)
    px_current = canvas_img[mask_current > 0]
    dark_current = np.mean(px_current < 128) if len(px_current) > 0 else 0
    
    # Test 3: Letterbox based on IMAGE SIZE (iw, ih), where polygons are in image coordinates!
    scale_img = min(size / iw, size / ih)
    nw_i, nh_i = int(iw * scale_img), int(ih * scale_img)
    pad_xi = (size - nw_i) // 2
    pad_yi = (size - nh_i) // 2
    img_resized_i = cv2.resize(img, (nw_i, nh_i))
    canvas_img_i = np.full((size, size), 255, dtype=np.uint8)
    canvas_img_i[pad_yi:pad_yi+nh_i, pad_xi:pad_xi+nw_i] = img_resized_i
    
    mask_correct = np.zeros((size, size), dtype=np.uint8)
    for p in polys:
        scaled_p = p * scale_img + [pad_xi, pad_yi]
        cv2.fillPoly(mask_correct, [scaled_p.astype(np.int32)], 255)
    px_correct = canvas_img_i[mask_correct > 0]
    dark_correct = np.mean(px_correct < 128) if len(px_correct) > 0 else 0
    
    print(f"Sample {s_rel[-10:]} | Img: {iw}x{ih} | SVG: {svg_w:.0f}x{svg_h:.0f} | 1:1 on Raw: {dark_1to1*100:5.1f}% | Current dataset.py: {dark_current*100:5.1f}% | Correct Img-Letterbox: {dark_correct*100:5.1f}%")
