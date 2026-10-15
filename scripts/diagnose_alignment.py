import os
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import cv2

root = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k")
split_file = root / "train.txt"

with open(split_file, "r") as f:
    samples = [line.strip().lstrip("/") for line in f if line.strip()][:10]

print(f"Inspecting {len(samples)} samples from train.txt:")

for s_rel in samples:
    folder = root / s_rel
    svg_p = folder / "model.svg"
    if not svg_p.exists():
        continue
    
    print(f"\n================ Sample: {s_rel} ================")
    tree = ET.parse(svg_p)
    r = tree.getroot()
    vb_str = r.attrib.get("viewBox", "")
    w_attr = r.attrib.get("width", "")
    h_attr = r.attrib.get("height", "")
    print(f"SVG root tag: {r.tag}")
    print(f"SVG attribs: viewBox='{vb_str}', width='{w_attr}', height='{h_attr}'")
    
    # Check images in folder
    for fname in ["F1_scaled.png", "F1_original.png"]:
        ip = folder / fname
        if ip.exists():
            im = Image.open(ip)
            print(f"  {fname}: size={im.size} (W={im.size[0]}, H={im.size[1]}), aspect={im.size[0]/im.size[1]:.4f}")
            
    # Check polygon points range in SVG
    all_pts = []
    for poly in r.findall(".//{http://www.w3.org/2000/svg}polygon"):
        pts_str = poly.attrib.get("points", "")
        clean = pts_str.replace(",", " ").split()
        for i in range(0, len(clean), 2):
            if i + 1 < len(clean):
                all_pts.append((float(clean[i]), float(clean[i+1])))
                
    if all_pts:
        all_pts = np.array(all_pts)
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)
        print(f"  Polygon points bounding box: min=({min_x:.1f}, {min_y:.1f}), max=({max_x:.1f}, {max_y:.1f})")
        print(f"  Polygon span: W={max_x - min_x:.1f}, H={max_y - min_y:.1f}")
        
    # Check if there are <image> elements in SVG!
    for img_tag in r.findall(".//{http://www.w3.org/2000/svg}image"):
        href = img_tag.attrib.get("{http://www.w3.org/1999/xlink}href", img_tag.attrib.get("href", ""))
        iw = img_tag.attrib.get("width", "")
        ih = img_tag.attrib.get("height", "")
        itrans = img_tag.attrib.get("transform", "")
        ix = img_tag.attrib.get("x", "")
        iy = img_tag.attrib.get("y", "")
        print(f"  SVG <image> tag: href='{href}', x='{ix}', y='{iy}', width='{iw}', height='{ih}', transform='{itrans}'")
